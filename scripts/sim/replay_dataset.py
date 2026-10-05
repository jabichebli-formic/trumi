"""Replay an exported LeRobot dataset in the digital twin, straight from the stored values: the twin UR5e is driven by
each frame's observation.state (6 joints + gripper command) next to the dataset's own wrist image for that frame.
What you watch is exactly what the policy trains on. Also reports, from the stored joints: contacts in the twin and
joint speed above the UR5e's 180 deg/s.

First dump the dataset (LeRobot environment): scripts/tools/lerobot_dump_states.py --root data/lerobot/<name>
Usage (from ~/trumi):
    MUJOCO_GL=egl data/sim/.venv/bin/python scripts/sim/replay_dataset.py --dump data/lerobot/<name>_review [--episodes 0 5 12]
Output: <dump>/ep<E>_twin.mp4
"""

import argparse
import json
import pathlib
import subprocess
import sys

import av
import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from plan_cell import Planner  # noqa: E402
from playback_touches import FONT, calibrated_placement  # noqa: E402
from view_twin import REPO  # noqa: E402


def dataset_frames(video, t0, t1, fps):
    """Frames of one episode from the dataset's video file (RGB), at t0 + k/fps."""
    out = []
    with av.open(str(video)) as c:
        s = c.streams.video[0]
        c.seek(int(max(t0 - 1.0, 0) / s.time_base), stream=s)
        for fr in c.decode(s):
            t = float(fr.pts * s.time_base)
            if t < t0 - 0.5 / fps:
                continue
            if t > t1 - 0.5 / fps:
                break
            out.append(fr.to_ndarray(format="rgb24"))
    return out


def main(a):
    d = json.load(open(a.dump / "episodes.json"))
    cell = json.load(open(a.cell))
    yaw, _, _, Twm = calibrated_placement(cell, np.array(json.load(open(a.calibration))["T_base_marker"]))
    cell["marker"]["centre"] = Twm[:3, 3].tolist()
    cell["marker"]["yaw_deg"] = float(np.degrees(np.arctan2(Twm[1, 0], Twm[0, 0])))
    pl = Planner(cell, controller_yaw_deg=yaw, tcp_z_mm=a.tcp_z_mm)
    r, r_close = mujoco.Renderer(pl.m, 480, 640), mujoco.Renderer(pl.m, 200, 260)
    episodes = [e for e in d["episodes"] if a.episodes is None or e["episode"] in a.episodes]
    for ep in episodes:
        e, fps = ep["episode"], ep["fps"]
        S = np.load(a.dump / f"ep{e}_state.npy")
        q, g = S[:, :6].astype(float), S[:, 6] * 255
        imgs = dataset_frames(REPO / ep["video"], ep["from_timestamp"], ep["to_timestamp"], fps)
        sp = np.degrees(np.abs(np.diff(q, axis=0))) * fps
        hits = [pl.contacts(qq) for qq in q]
        print(f"episode {e} ({ep['source_video']}): {len(S)} states, {len(imgs)} video frames | above 180 deg/s for "
              f"{(sp.max(1) > 180).sum() / fps:.2f} s (max {sp.max():.0f}) | twin contact frames {sum(map(bool, hits))}"
              + (f" ({', '.join(sorted({h for hh in hits for h in hh}))})" if any(hits) else ""))
        if len(imgs) != len(S):
            print(f"   WARNING: {len(imgs)} video frames for {len(S)} states")
        dd = mujoco.MjData(pl.m)
        cam, close = mujoco.MjvCamera(), mujoco.MjvCamera()
        dd.qpos[:6] = q[0]
        mujoco.mj_forward(pl.m, dd)
        cam.lookat[:] = dd.site_xpos[pl.site] + [0, 0, -0.1]
        cam.distance, cam.azimuth, cam.elevation = 1.6, 110.0, -30.0
        close.distance, close.azimuth, close.elevation = 0.45, 110.0, -20.0
        tmp = a.dump / f"ep{e}_twin_tmp.mp4"
        ff = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x480", "-r", str(fps),
                               "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(tmp)], stdin=subprocess.PIPE)
        substeps = max(1, round((1 / fps) / pl.m.opt.timestep))
        for k in range(len(S)):
            dd.ctrl[:6], dd.ctrl[6] = q[k], g[k]
            for _ in range(substeps):  # arm set from the dataset; the gripper linkage is simulated from its command
                dd.qpos[:6] = q[k]
                dd.qvel[:6] = 0
                mujoco.mj_step(pl.m, dd)
            r.update_scene(dd, camera=cam)
            twin = r.render().copy()
            close.lookat[:] = dd.site_xpos[pl.site]
            r_close.update_scene(dd, camera=close)
            twin[-204:-4, -264:-4] = r_close.render()
            if hits[k]:
                twin[:8] = (255, 0, 0)  # red bar: contact in the twin on this frame
            img = imgs[min(k, len(imgs) - 1)] if imgs else np.zeros((480, 640, 3), np.uint8)
            ff.stdin.write(np.hstack([twin, img]).tobytes())
        ff.stdin.close()
        ff.wait()
        label = (f"drawtext=fontfile={FONT}:text='twin driven by the dataset joints (episode {e}, {ep['source_video']})':x=10:y=12:fontsize=18:"
                 f"fontcolor=white:box=1:boxcolor=black@0.5,drawtext=fontfile={FONT}:text='dataset wrist image (what the policy sees)':"
                 "x=650:y=12:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.5")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(tmp), "-vf", label, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        str(a.dump / f"ep{e}_twin.mp4")], check=True)
        tmp.unlink()
    print(f"saved {a.dump}/ep<E>_twin.mp4")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Replay an exported LeRobot dataset in the digital twin.")
    ap.add_argument("--dump", type=pathlib.Path, required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--cell", type=pathlib.Path, default=REPO / "data" / "sim" / "cell.json")
    ap.add_argument("--calibration", type=pathlib.Path, default=REPO / "data" / "robot" / "marker_in_robot_base.json")
    ap.add_argument("--tcp_z_mm", type=float, default=257.2)
    main(ap.parse_args())

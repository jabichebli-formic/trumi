"""Play back the calibration touches in the digital twin, using the robot's own recorded joint angles.

The robot is placed at its scan-fitted base, turned to the heading implied by the calibration, with the TCP at the
closed fingertip ends. For every touch it compares the twin's fingertip with the position the robot itself reported
(should agree within ~1-2 mm), then renders a video: overview (left) and close-up of the marker (right).

Usage (from ~/trumi):
    MUJOCO_GL=egl data/sim/.venv/bin/python scripts/sim/playback_touches.py [--calibration data/robot/marker_in_robot_base.json]
Outputs: data/robot/calibration_playback.mp4, data/robot/calibration_playback.png
"""

import argparse
import json
import pathlib
import subprocess
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from plan_cell import Planner  # noqa: E402
from view_twin import REPO  # noqa: E402

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def yaw_T(yaw_deg, t):
    T = np.eye(4)
    T[:3, :3] = R.from_euler("z", yaw_deg, degrees=True).as_matrix()
    T[:3, 3] = t
    return T


def calibrated_placement(cell, T_ctrl_marker):
    """Robot heading in the scan from the calibration + photo-based marker estimate; returns (yaw, base gap, T_world_ctrl, T_world_marker)."""
    base = np.array(cell["robot_base"]["pos"], float)
    mk = cell["marker"]
    derived = yaw_T(mk["yaw_deg"], mk["centre"]) @ np.linalg.inv(T_ctrl_marker)
    yaw = float(np.degrees(np.arctan2(derived[1, 0], derived[0, 0])))
    T_world_ctrl = yaw_T(yaw, base)
    return yaw, float(np.linalg.norm(derived[:2, 3] - base[:2])), T_world_ctrl, T_world_ctrl @ T_ctrl_marker


def main(a):
    calib = json.load(open(a.calibration))
    cell = json.load(open(a.cell))
    T_cm = np.array(calib["T_base_marker"])
    yaw, gap, Twc, Twm = calibrated_placement(cell, T_cm)
    cell["marker"]["centre"] = Twm[:3, 3].tolist()
    cell["marker"]["yaw_deg"] = float(np.degrees(np.arctan2(Twm[1, 0], Twm[0, 0])))
    scan_marker_z = cell["lower_table"]["top_z"] + 0.003
    cell["lower_table"]["top_z"] = float(Twm[2, 3] - 0.003)  # trust the robot's measurement of the table height
    cell["box"] = None  # the box is not on the table during calibration
    pl = Planner(cell, controller_yaw_deg=yaw, tcp_z_mm=a.tcp_z_mm)
    for g in range(pl.m.ngeom):  # hide the collision boxes except the marker
        nm = mujoco.mj_id2name(pl.m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if pl.m.geom_bodyid[g] == 0 and nm and not nm.startswith(("marker", "scan")):
            pl.m.geom_rgba[g][3] = 0.0
    opt_close = mujoco.MjvOption()
    opt_close.geomgroup[0] = 0  # close-up: hide the scan so the marker and fingertips are visible
    print(f"robot heading in the scan: {yaw:.1f} deg | base implied by the photo-based marker spot is {gap*100:.1f} cm from the scan-fitted base")
    print(f"marker height: robot says {Twm[2,3]:.3f} m, scan table says {scan_marker_z:.3f} m -> difference {(Twm[2,3]-scan_marker_z)*100:+.1f} cm")

    joints = [(n, np.array(q)) for n, q in calib["joints_rad"]]
    tcps = [np.array(p) for _, p in calib["measured_tcp_m"]]
    print("\ntwin fingertip (from the robot's joint angles) vs fingertip position the robot reported:")
    errs = []
    for (n, q), p_robot in zip(joints, tcps):
        pl.d.qpos[:] = 0
        pl.d.qpos[:6] = q
        mujoco.mj_forward(pl.m, pl.d)
        p_ctrl = np.linalg.inv(Twc) @ np.r_[pl.d.site_xpos[pl.site], 1]
        e = np.linalg.norm(p_ctrl[:3] - p_robot) * 1000
        errs.append(e)
        hits = pl.contacts(q, allowed=("marker_board",))
        print(f"   {n:18s} {e:5.1f} mm" + (f"   (twin touches: {', '.join(sorted(set(hits)))})" if hits else ""))
    print(f"   -> median {np.median(errs):.1f} mm, max {max(errs):.1f} mm")

    # --- animation: move between the recorded touch poses, pausing at each
    seq = []
    for i, (n, q) in enumerate(joints):
        if i:
            q0 = joints[i - 1][1]
            for s in (1 - np.cos(np.linspace(0, np.pi, 45))) / 2:
                seq.append((q0 + s * (q - q0), ""))
        seq += [(q, n.split()[0])] * 25
    r = mujoco.Renderer(pl.m, 480, 640)
    over, close = mujoco.MjvCamera(), mujoco.MjvCamera()
    mc = Twm[:3, 3]
    over.lookat[:] = (mc + pl.base) / 2 + [0, 0, 0.15]
    over.distance, over.azimuth, over.elevation = 1.5, 110.0 + 0, -25.0
    close.lookat[:] = mc
    close.distance, close.azimuth, close.elevation = 0.55, 110.0, -40.0
    out_mp4 = a.calibration.with_name("calibration_playback.mp4")
    tmp = out_mp4.with_suffix(".tmp.mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x480", "-r", "30",
                           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(tmp)], stdin=subprocess.PIPE)
    stills, labels = [], []
    for q, label in seq:
        pl.d.qpos[:] = 0
        pl.d.qpos[:6] = q
        mujoco.mj_forward(pl.m, pl.d)
        frame = []
        for cam in (over, close):
            r.update_scene(pl.d, camera=cam, scene_option=opt_close if cam is close else None)
            sc = r.scene
            if cam is over:  # outline the robot-measured marker on the scanned table surface
                corners = [Twm @ np.r_[x, y, 0, 1] for x, y in [(-.08, .08), (.08, .08), (.08, -.08), (-.08, -.08)]]
                for c0, c1 in zip(corners, corners[1:] + corners[:1]):
                    a0, a1 = c0[:3].copy(), c1[:3].copy()
                    a0[2] = a1[2] = scan_marker_z + 0.002
                    mujoco.mjv_initGeom(sc.geoms[sc.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).flatten(),
                                        np.array([0, 1, 0.3, 1], dtype=np.float32))
                    mujoco.mjv_connector(sc.geoms[sc.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, 0.004, a0, a1)
                    sc.ngeom += 1
            for p in tcps:  # where the robot said each touch was (orange)
                p_w = Twc @ np.r_[p, 1]
                mujoco.mjv_initGeom(sc.geoms[sc.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, [0.004, 0, 0], p_w[:3],
                                    np.eye(3).flatten(), np.array([1, 0.5, 0, 1], dtype=np.float32))
                sc.ngeom += 1
            frame.append(r.render())
        ff.stdin.write(np.hstack(frame).tobytes())
        if label and (not labels or labels[-1] != label or len(stills) < len(labels)):
            if len(labels) == len(stills):
                stills.append(frame[1])
                labels.append(label)
    ff.stdin.close()
    ff.wait()
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(tmp), "-vf",
                    f"drawtext=fontfile={FONT}:text='left - the cell, green square = marker as measured by the robot     right - close-up of the touches':x=10:y=10:fontsize=20:"
                    "fontcolor=white:box=1:boxcolor=black@0.5", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_mp4)], check=True)
    tmp.unlink()
    # still grid of the close-ups, one per touch
    k = len(stills)
    cols = 5
    rows = int(np.ceil(k / cols))
    grid = np.zeros((rows * 480, cols * 640, 3), np.uint8)
    for i, im in enumerate(stills):
        grid[(i // cols) * 480:(i // cols + 1) * 480, (i % cols) * 640:(i % cols + 1) * 640] = im
    txt = ",".join(f"drawtext=fontfile={FONT}:text='{i + 1}. {l}':x={10 + 640 * (i % cols)}:y={10 + 480 * (i // cols)}:fontsize=28:"
                   "fontcolor=white:box=1:boxcolor=black@0.6" for i, l in enumerate(labels))
    out_png = a.calibration.with_name("calibration_playback.png")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{cols*640}x{rows*480}", "-i", "-",
                    "-vf", f"{txt},scale=1600:-1", "-frames:v", "1", str(out_png)], input=grid.tobytes(), check=True)
    print(f"\nsaved {out_mp4} and {out_png}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Play back the calibration touches in the digital twin.")
    ap.add_argument("--calibration", type=pathlib.Path, default=REPO / "data" / "robot" / "marker_in_robot_base.json")
    ap.add_argument("--cell", type=pathlib.Path, default=REPO / "data" / "sim" / "cell.json")
    ap.add_argument("--tcp_z_mm", type=float, default=173.8)
    main(ap.parse_args())

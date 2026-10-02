"""Pre-flight check of a TRumi episode on the real UR5e cell, in the digital twin, before running it on hardware.

Chain of frames:
    TRumi fingertip pose (mapping-marker frame, from the session's dataset_plan.pkl)
    -> UR controller base frame, via the measured marker calibration (scripts/robot/calibrate_marker.py)
    -> twin / scan frame, via the robot base position fitted in the scan and the heading implied by the calibration
For one episode and arm it solves IK for the twin's UR5e + Robotiq 2F-85, checks joint limits, joint speeds and
collisions with the cell (tables, belt, rails, box from data/sim/cell.json), renders the replay next to the GoPro
view, and writes a robot-ready joint trajectory (slowed down to a joint-speed cap) for scripts/robot/replay_on_robot.py.

Usage (from ~/trumi):
    MUJOCO_GL=egl data/sim/.venv/bin/python scripts/sim/preflight.py --session data/<session> --episode 1 \
        [--arm right] [--calibration data/robot/marker_in_robot_base.json] [--tcp_z_mm 155.8] [--speed_cap_deg_s 45]
Without a calibration file it uses the marker position estimated from the photo (data/sim/cell.json), clearly
marked as an ESTIMATE in the report.

Outputs in <session>/preflight/: ep<N>_<arm>_preflight.mp4, ep<N>_<arm>_report.txt, ep<N>_<arm>_robot_trajectory.json
"""

import argparse
import json
import pathlib
import pickle
import subprocess
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from plan_cell import Planner  # noqa: E402
from replay_ur5e import (  # noqa: E402
    OPENING_CTRL, OPENING_M, SLAM_STRIDE, TRUMI_TO_ROBOTIQ, clean_trajectory, read_video_frames, solve_ik)
from view_twin import REPO  # noqa: E402

ARMS = {"right": 0, "left": 1}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
OUT_FPS = 30


def T_from(Rm, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = Rm, t
    return T


def yaw_T(yaw_deg, t):
    return T_from(R.from_euler("z", yaw_deg, degrees=True).as_matrix(), t)


def main(a):
    session = a.session.resolve()
    cell = json.load(open(a.cell))
    base = np.array(cell["robot_base"]["pos"], dtype=float)
    mk = cell["marker"]
    T_world_marker_est = yaw_T(mk["yaw_deg"], mk["centre"])
    notes = []

    # --- calibration: marker pose in the UR controller base frame
    if a.calibration and a.calibration.is_file():
        T_ctrl_marker = np.array(json.load(open(a.calibration))["T_base_marker"])
        calib_src = f"MEASURED ({a.calibration})"
        # robot heading in the scan = the one that puts the calibrated marker where the photo-based estimate is
        T_world_ctrl_derived = T_world_marker_est @ np.linalg.inv(T_ctrl_marker)
        ctrl_yaw = float(np.degrees(np.arctan2(T_world_ctrl_derived[1, 0], T_world_ctrl_derived[0, 0])))
        gap = np.linalg.norm(T_world_ctrl_derived[:2, 3] - base[:2])
        notes.append(f"robot heading in the scan from calibration: {ctrl_yaw:.1f} deg; base position implied by the "
                     f"photo-based marker estimate is {gap*100:.1f} cm from the scan-fitted base"
                     + ("  (consistent)" if gap < 0.05 else "  (LARGE: check the marker position in data/sim/cell.json)"))
    else:
        ctrl_yaw = 0.0
        T_ctrl_marker = np.linalg.inv(yaw_T(ctrl_yaw, base)) @ T_world_marker_est
        calib_src = "ESTIMATE (no calibration file; marker position from the photo)"
    T_world_ctrl = yaw_T(ctrl_yaw, base)
    T_world_marker = T_world_ctrl @ T_ctrl_marker
    cell = json.loads(json.dumps(cell))
    cell["marker"]["centre"] = T_world_marker[:3, 3].tolist()
    cell["marker"]["yaw_deg"] = float(np.degrees(np.arctan2(T_world_marker[1, 0], T_world_marker[0, 0])))

    # --- episode: TRumi fingertip -> Robotiq fingertip targets
    ep = pickle.load(open(session / "dataset_plan.pkl", "rb"))[a.episode - 1]
    g = ARMS[a.arm] if len(ep["grippers"]) > 1 else 0
    ts = np.asarray(ep["episode_timestamps"], float)
    t = ts - ts[0]
    pose = np.asarray(ep["grippers"][g]["tcp_pose"], float)
    width = np.asarray(ep["grippers"][g]["gripper_width"], float)
    pos, rot, n_jumps, n_floor = clean_trajectory(t, pose[:, :3], R.from_rotvec(pose[:, 3:]))
    rot_r = rot * R.from_matrix(TRUMI_TO_ROBOTIQ)
    pos = pos + [0, 0, a.z_offset_mm / 1000]  # "in the air" rehearsal: lift the whole path (marker frame z = up)
    p_ctrl = pos @ T_ctrl_marker[:3, :3].T + T_ctrl_marker[:3, 3]
    R_ctrl = R.from_matrix(T_ctrl_marker[:3, :3]) * rot_r
    p_w = p_ctrl @ T_world_ctrl[:3, :3].T + T_world_ctrl[:3, 3]
    R_w = R.from_matrix(T_world_ctrl[:3, :3]) * R_ctrl
    grip = np.interp(np.clip(width, OPENING_M[0], OPENING_M[-1]), OPENING_M, OPENING_CTRL)

    # --- IK through the trajectory, starting from the best collision-free configuration
    pl = Planner(cell, controller_yaw_deg=ctrl_yaw, tcp_z_mm=a.tcp_z_mm)
    starts = pl.solutions(p_w[0], R_w[0].as_matrix())
    if not starts:
        raise SystemExit("no collision-free robot configuration reaches the episode's first pose")
    qs, e_pos, e_rot, hits = [starts[0]], [], [], []
    for i in range(len(t)):
        q, ep_, er_ = solve_ik(pl.m, pl.d, pl.site, p_w[i], R_w[i].as_matrix(), qs[-1])
        qs.append(q)
        e_pos.append(ep_)
        e_rot.append(er_)
        hits.append(pl.contacts(q))
    qs = np.array(qs[1:])
    e_pos, e_rot = np.array(e_pos), np.array(e_rot)
    speed = np.degrees(np.abs(np.diff(qs, axis=0)) / np.diff(t)[:, None])
    slow = max(1.0, speed.max() / a.speed_cap_deg_s)
    lim = np.degrees(pl.m.jnt_range[:6])
    near_lim = (np.degrees(qs) < lim[:, 0] + 5) | (np.degrees(qs) > lim[:, 1] - 5)
    collisions = {}
    for i, h in enumerate(hits):
        for name in h:
            collisions.setdefault(name, []).append(t[i])

    out = session / "preflight"
    out.mkdir(exist_ok=True)
    stem = f"ep{a.episode}_{a.arm}" + (f"_air{a.z_offset_mm:.0f}mm" if a.z_offset_mm else "")
    ok = e_pos.max() < 0.002 and e_rot.max() < 1.0 and not near_lim.any() and not collisions
    rep = [f"PRE-FLIGHT {session.name} episode {a.episode} ({a.arm} arm): {'PASS' if ok else 'NEEDS ATTENTION'}",
           f"calibration: {calib_src}", *notes,
           f"trajectory: {len(t)} steps, {t[-1]:.1f} s; glitch jumps smoothed: {n_jumps}; floor clamps: {n_floor}",
           f"IK: position error max {e_pos.max()*1000:.1f} mm, rotation error max {e_rot.max():.2f} deg",
           f"joint limits: {'OK' if not near_lim.any() else f'{near_lim.any(1).sum()} frames within 5 deg of a limit'}",
           f"joint speed: max {speed.max():.0f} deg/s at recorded speed -> replay {slow:.1f}x slower "
           f"({t[-1]*slow:.0f} s) to stay under {a.speed_cap_deg_s:.0f} deg/s",
           "collisions: " + ("none" if not collisions else
                             "; ".join(f"{k} at {v[0]:.1f}-{v[-1]:.1f} s ({len(v)} frames)" for k, v in collisions.items())),
           f"TCP (fingertip) offset used: {a.tcp_z_mm if a.tcp_z_mm else 155.8} mm from the flange (set the same on the pendant)",
           f"path lifted by {a.z_offset_mm:.0f} mm (in-the-air rehearsal)" if a.z_offset_mm else "path at recorded height"]
    (out / f"{stem}_report.txt").write_text("\n".join(rep) + "\n")

    traj = {"description": "joint-space trajectory for the real UR5e (UR controller base frame); times already slowed down",
            "session": str(session), "episode": a.episode, "arm": a.arm, "calibration": calib_src,
            "speed_cap_deg_s": a.speed_cap_deg_s, "slowdown": slow, "tcp_z_mm": a.tcp_z_mm or 155.8,
            "t_s": (t * slow).round(4).tolist(), "q_rad": qs.round(6).tolist(),
            "tcp_pose_ctrl": [list(p) + list(r) for p, r in zip(p_ctrl.round(5), R_ctrl.as_rotvec().round(5))],
            "gripper_0open_255closed": grip.round(1).tolist(), "preflight_pass": bool(ok), "z_offset_mm": a.z_offset_mm}
    json.dump(traj, open(out / f"{stem}_robot_trajectory.json", "w"))

    # --- video: twin replay (left) + GoPro (right)
    steps = list(range(0, len(t), 60 // OUT_FPS))
    cam_meta = ep["cameras"][g]
    if a.no_video:
        gopro = [np.zeros((480, 640, 3), np.uint8)] * len(steps)
    else:
        gopro = read_video_frames(session / "demos" / cam_meta["video_path"],
                                  [cam_meta["video_start_end"][0] + SLAM_STRIDE * s for s in steps])
    r = mujoco.Renderer(pl.m, 480, 640)
    vc = mujoco.MjvCamera()
    vc.lookat[:] = p_w.mean(0)
    vc.distance, vc.azimuth, vc.elevation = 1.6, 110.0, -30.0
    tmp = out / f"{stem}_tmp.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x480", "-r",
                           str(OUT_FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(tmp)], stdin=subprocess.PIPE)
    for n, s in enumerate(steps):
        pl.d.qpos[:] = 0
        pl.d.qpos[:6] = qs[s]
        mujoco.mj_forward(pl.m, pl.d)
        r.update_scene(pl.d, camera=vc)
        sc = r.scene
        for k in range(0, len(p_w), 6):
            if sc.ngeom >= sc.maxgeom:
                break
            col = [1, 0.1, 0.1, 0.9] if hits[k] else [1, 0.6, 0.1, 0.35 if k > s else 0.9]
            mujoco.mjv_initGeom(sc.geoms[sc.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, [0.004, 0, 0], p_w[k],
                                np.eye(3).flatten(), np.array(col, dtype=np.float32))
            sc.ngeom += 1
        ff.stdin.write(np.hstack([r.render(), gopro[n]]).tobytes())
    ff.stdin.close()
    ff.wait()
    label = (f"drawtext=fontfile={FONT}:text='twin pre-flight ({'PASS' if ok else 'CHECK REPORT'})':x=10:y=10:fontsize=20:"
             f"fontcolor=white:box=1:boxcolor=black@0.5,drawtext=fontfile={FONT}:text='TRumi camera (episode {a.episode})':"
             f"x=650:y=10:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.5")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(tmp), "-vf", label, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(out / f"{stem}_preflight.mp4")], check=True)
    tmp.unlink()
    print("\n".join(rep))
    print(f"saved {out/(stem + '_preflight.mp4')}, {out/(stem + '_report.txt')}, {out/(stem + '_robot_trajectory.json')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pre-flight check of a TRumi episode on the real UR5e cell, in the twin.")
    ap.add_argument("--session", required=True, type=pathlib.Path)
    ap.add_argument("--episode", required=True, type=int, help="episode number in the dataset plan (1-based)")
    ap.add_argument("--arm", choices=list(ARMS), default="right", help="which arm, for two-gripper sessions")
    ap.add_argument("--calibration", type=pathlib.Path, default=REPO / "data" / "robot" / "marker_in_robot_base.json")
    ap.add_argument("--cell", type=pathlib.Path, default=REPO / "data" / "sim" / "cell.json")
    ap.add_argument("--tcp_z_mm", type=float, default=None, help="real TCP offset from the flange (pendant); default = model")
    ap.add_argument("--speed_cap_deg_s", type=float, default=45.0, help="joint speed cap for the hardware replay")
    ap.add_argument("--z_offset_mm", type=float, default=0.0, help="lift the whole path by this much (in-the-air rehearsal)")
    ap.add_argument("--no_video", action="store_true", help="skip the GoPro panel (e.g. for synthetic test episodes)")
    main(ap.parse_args())

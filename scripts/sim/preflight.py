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
from plan_cell import SIM_TCP_Z_MM, Planner  # noqa: E402
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


def smooth_pose(t, pos, rot, window_s):
    """Savitzky-Golay smoothing (2nd order) of positions and of orientation quaternions (sign-continuous, renormalised).
    Removes frame-to-frame tracking jitter, which otherwise shows up as very short joint-speed spikes."""
    from scipy.signal import savgol_filter

    n = int(round(window_s / np.median(np.diff(t)))) | 1  # odd number of samples
    if n < 5 or n > len(t):
        return pos, rot
    quat = rot.as_quat()
    for i in range(1, len(quat)):  # q and -q are the same rotation: keep neighbours on the same side
        if quat[i] @ quat[i - 1] < 0:
            quat[i] = -quat[i]
    quat = savgol_filter(quat, n, 2, axis=0)
    return savgol_filter(pos, n, 2, axis=0), R.from_quat(quat / np.linalg.norm(quat, axis=1, keepdims=True))


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
    if a.no_box:
        cell["box"] = None

    # --- episode: TRumi fingertip -> Robotiq fingertip targets
    ep = pickle.load(open(session / "dataset_plan.pkl", "rb"))[a.episode - 1]
    g = ARMS[a.arm] if len(ep["grippers"]) > 1 else 0
    ts = np.asarray(ep["episode_timestamps"], float)
    t = ts - ts[0]
    pose = np.asarray(ep["grippers"][g]["tcp_pose"], float)
    width = np.asarray(ep["grippers"][g]["gripper_width"], float)
    raw_pos = pose[:, :3] + np.array(a.correction_mm) / 1000 + [0, 0, a.z_offset_mm / 1000]  # for review: before clean-up/smoothing
    pos, rot, n_jumps, n_floor = clean_trajectory(t, pose[:, :3], R.from_rotvec(pose[:, 3:]))
    if a.smooth_s > 0:
        pos, rot = smooth_pose(t, pos, rot, a.smooth_s)
        notes.append(f"smoothing: Savitzky-Golay over {a.smooth_s:.2f} s on fingertip position and orientation")
    rot_r = rot * R.from_matrix(TRUMI_TO_ROBOTIQ)
    pos = pos + np.array(a.correction_mm) / 1000  # systematic TRumi offset measured by a touch test (marker frame)
    pos = pos + [0, 0, a.z_offset_mm / 1000]  # "in the air" rehearsal: lift the whole path (marker frame z = up)
    p_ctrl = pos @ T_ctrl_marker[:3, :3].T + T_ctrl_marker[:3, 3]
    R_ctrl = R.from_matrix(T_ctrl_marker[:3, :3]) * rot_r
    p_w = p_ctrl @ T_world_ctrl[:3, :3].T + T_world_ctrl[:3, 3]
    R_w = R.from_matrix(T_world_ctrl[:3, :3]) * R_ctrl
    grip = np.interp(np.clip(width, OPENING_M[0], OPENING_M[-1]), OPENING_M, OPENING_CTRL)
    setback = np.zeros(len(t))  # mm the real fingertips sit back from the closed TCP (Robotiq arc), per step
    if a.gripper_tables:  # measured on the real gripper (scripts/tools/gripper_sweep_table.py)
        gt = json.load(open(a.gripper_tables))
        grip = np.interp(width, gt["width_to_cmd"]["trumi_width_m"], gt["width_to_cmd"]["robotiq_cmd"])
        setback = np.interp(grip, gt["arc"]["robotiq_pos"], gt["arc"]["fingertip_setback_mm"])
        notes.append(f"gripper: measured width->command table and arc from {a.gripper_tables} "
                     f"(fingertip set-back {setback.min():.1f}-{setback.max():.1f} mm compensated)")
    if a.grip_close_below_mm > 0:  # holding an object: close fully so the Robotiq stops on contact and applies force
        holding = width * 1000 < a.grip_close_below_mm
        grip = np.where(holding, 255.0, grip)
        notes.append(f"gripper: fully closed (255) while the TRumi gap is below {a.grip_close_below_mm:.0f} mm "
                     f"({holding.mean() * 100:.0f}% of the episode)")

    # --- IK through the trajectory, starting from the best collision-free configuration
    pl = Planner(cell, controller_yaw_deg=ctrl_yaw, tcp_z_mm=a.tcp_z_mm)
    tcp_site_z = lambda i: 0.145 + ((a.tcp_z_mm or SIM_TCP_Z_MM) - setback[i] - SIM_TCP_Z_MM) / 1000
    pl.m.site_pos[pl.site][2] = tcp_site_z(0)  # IK target = where the real fingertips are at this opening
    starts = pl.solutions(p_w[0], R_w[0].as_matrix())
    if not starts:
        raise SystemExit("no collision-free robot configuration reaches the episode's first pose")
    lim = np.degrees(pl.m.jnt_range[:6])

    def follow(q_start):
        """IK through the whole episode starting from one arm configuration."""
        qs, e_pos, e_rot, hits = [q_start], [], [], []
        for i in range(len(t)):
            pl.m.site_pos[pl.site][2] = tcp_site_z(i)
            q, ep_, er_ = solve_ik(pl.m, pl.d, pl.site, p_w[i], R_w[i].as_matrix(), qs[-1])
            qs.append(q)
            e_pos.append(ep_)
            e_rot.append(er_)
            hits.append(pl.contacts(q))
        qs = np.array(qs[1:])
        n_lim = int(((np.degrees(qs) < lim[:, 0] + 5) | (np.degrees(qs) > lim[:, 1] - 5)).any(1).sum())
        return (sum(map(bool, hits)), n_lim, max(e_pos)), (qs, np.array(e_pos), np.array(e_rot), hits)

    # the first configuration found is not always the sensible one (e.g. elbow-down into the table): follow the episode
    # from each candidate start and keep the one with the fewest contact frames, then fewest frames near a joint limit
    # with --reference_q_deg (e.g. for a joint-space dataset, where every episode must use the same arm configuration
    # and the same joint winding), also start from the IK solution nearest that reference, and keep the candidate
    # closest to it among those with no more than ~0.5 s of extra contact frames
    candidates = list(starts[:a.max_starts])
    if a.reference_q_deg is not None:
        q_ref = np.radians(a.reference_q_deg)
        q_near, _, _ = solve_ik(pl.m, pl.d, pl.site, p_w[0], R_w[0].as_matrix(), q_ref)
        candidates.insert(0, q_near)
    tried = [follow(q0) for q0 in candidates]
    if a.reference_q_deg is not None:
        fewest = min(c[0][0] for c in tried)
        ok = [k for k in range(len(tried)) if tried[k][0][0] <= fewest + 30 and tried[k][0][2] < 0.002]
        best = min(ok or range(len(tried)), key=lambda k: np.linalg.norm(np.degrees(tried[k][1][0][0]) - a.reference_q_deg))
    else:
        best = min(range(len(tried)), key=lambda k: tried[k][0])
    qs, e_pos, e_rot, hits = tried[best][1]
    if len(tried) > 1:
        notes.append(f"arm configuration: tried {len(tried)} starting configurations, kept #{best + 1} "
                     f"(contact frames per candidate: {[c[0][0] for c in tried]})"
                     + (f"; start is {np.linalg.norm(np.degrees(qs[0]) - a.reference_q_deg):.0f} deg from the reference" if a.reference_q_deg is not None else ""))
    speed = np.degrees(np.abs(np.diff(qs, axis=0)) / np.diff(t)[:, None])
    slow = max(1.0, speed.max() / a.speed_cap_deg_s)
    over_limit_s = float(np.sum(np.diff(t)[speed.max(1) > 180]))
    near_lim = (np.degrees(qs) < lim[:, 0] + 5) | (np.degrees(qs) > lim[:, 1] - 5)
    collisions = {}
    for i, h in enumerate(hits):
        for name in h:
            collisions.setdefault(name, []).append(t[i])

    out = session / a.out_subdir
    out.mkdir(exist_ok=True)
    stem = f"ep{a.episode}_{a.arm}" + (f"_air{a.z_offset_mm:.0f}mm" if a.z_offset_mm else "")
    ok = e_pos.max() < 0.002 and e_rot.max() < 1.0 and not near_lim.any() and not collisions
    rep = [f"PRE-FLIGHT {session.name} episode {a.episode} ({a.arm} arm): {'PASS' if ok else 'NEEDS ATTENTION'}",
           f"calibration: {calib_src}", *notes,
           f"trajectory: {len(t)} steps, {t[-1]:.1f} s; glitch jumps smoothed: {n_jumps}; floor clamps: {n_floor}",
           f"IK: position error max {e_pos.max()*1000:.1f} mm, rotation error max {e_rot.max():.2f} deg",
           f"joint limits: {'OK' if not near_lim.any() else f'{near_lim.any(1).sum()} frames within 5 deg of a limit'}",
           f"joint speed: above the UR5e's 180 deg/s for {over_limit_s:.2f} s in total (max {speed.max():.0f} deg/s)",
           f"joint speed: max {speed.max():.0f} deg/s at recorded speed -> replay {slow:.1f}x slower "
           f"({t[-1]*slow:.0f} s) to stay under {a.speed_cap_deg_s:.0f} deg/s",
           "collisions: " + ("none" if not collisions else
                             "; ".join(f"{k} at {v[0]:.1f}-{v[-1]:.1f} s ({len(v)} frames)" for k, v in collisions.items())),
           f"TCP (fingertip) offset used: {a.tcp_z_mm if a.tcp_z_mm else 155.8} mm from the flange (set the same on the pendant)",
           f"path lifted by {a.z_offset_mm:.0f} mm (in-the-air rehearsal)" if a.z_offset_mm else "path at recorded height",
           f"TRumi correction (marker frame x y z): {a.correction_mm} mm",
           "box: NOT CHECKED (--no_box)" if a.no_box else "box: from the cell file"]
    (out / f"{stem}_report.txt").write_text("\n".join(rep) + "\n")

    traj = {"description": "joint-space trajectory for the real UR5e (UR controller base frame); times already slowed down",
            "session": str(session), "episode": a.episode, "arm": a.arm, "calibration": calib_src,
            "speed_cap_deg_s": a.speed_cap_deg_s, "slowdown": slow, "time_over_180_deg_s": over_limit_s,
            "smooth_s": a.smooth_s, "tcp_z_mm": a.tcp_z_mm or 155.8,
            "t_s": (t * slow).round(4).tolist(), "q_rad": qs.round(6).tolist(),
            # pose of the pendant TCP (closed fingertip ends): the real fingertips plus the arc set-back along the tool axis
            "tcp_pose_ctrl": [list(p) + list(r) for p, r in zip((p_ctrl + setback[:, None] / 1000 * R_ctrl.as_matrix()[:, :, 2]).round(5),
                                                                R_ctrl.as_rotvec().round(5))],
            "fingertip_target_ctrl": p_ctrl.round(5).tolist(), "fingertip_setback_mm": setback.round(2).tolist(),
            "fingertip_raw_ctrl": (raw_pos @ T_ctrl_marker[:3, :3].T + T_ctrl_marker[:3, 3]).round(5).tolist(),
            "t_recorded_s": t.round(4).tolist(), "gripper_width_m": width.round(5).tolist(),
            "gripper_tables": str(a.gripper_tables) if a.gripper_tables else None,
            "gripper_0open_255closed": grip.round(1).tolist(), "preflight_pass": bool(ok), "z_offset_mm": a.z_offset_mm,
            "correction_mm": a.correction_mm, "box_checked": not a.no_box}
    json.dump(traj, open(out / f"{stem}_robot_trajectory.json", "w"))

    if a.no_video:  # checks and trajectory only; no half-empty video
        print("\n".join(rep))
        print(f"saved {out/(stem + '_report.txt')}, {out/(stem + '_robot_trajectory.json')} (no video: --no_video)")
        return
    # --- video: twin replay (left) + GoPro (right)
    steps = list(range(0, len(t), 60 // OUT_FPS))
    cam_meta = ep["cameras"][g]
    gopro = read_video_frames(session / "demos" / cam_meta["video_path"],
                              [cam_meta["video_start_end"][0] + SLAM_STRIDE * s for s in steps])
    r = mujoco.Renderer(pl.m, 480, 640)
    r_close = mujoco.Renderer(pl.m, 200, 260)  # inset: close-up that follows the gripper
    vclose = mujoco.MjvCamera()
    vclose.distance, vclose.azimuth, vclose.elevation = 0.45, 110.0, -20.0
    vc = mujoco.MjvCamera()
    vc.lookat[:] = p_w.mean(0)
    vc.distance, vc.azimuth, vc.elevation = 1.6, 110.0, -30.0
    tmp = out / f"{stem}_tmp.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x480", "-r",
                           str(OUT_FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(tmp)], stdin=subprocess.PIPE)
    d = mujoco.MjData(pl.m)  # arm set kinematically; the gripper linkage is simulated so the fingers open/close
    substeps = max(1, round((1 / OUT_FPS) / pl.m.opt.timestep))
    for n, s in enumerate(steps):
        d.ctrl[:6] = qs[s]
        d.ctrl[6] = grip[s]
        for _ in range(substeps):
            d.qpos[:6] = qs[s]
            d.qvel[:6] = 0
            mujoco.mj_step(pl.m, d)
        r.update_scene(d, camera=vc)
        sc = r.scene
        for k in range(0, len(p_w), 6):
            if sc.ngeom >= sc.maxgeom:
                break
            col = [1, 0.1, 0.1, 0.9] if hits[k] else [1, 0.6, 0.1, 0.35 if k > s else 0.9]
            mujoco.mjv_initGeom(sc.geoms[sc.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, [0.004, 0, 0], p_w[k],
                                np.eye(3).flatten(), np.array(col, dtype=np.float32))
            sc.ngeom += 1
        twin = r.render().copy()
        vclose.lookat[:] = d.site_xpos[pl.site]
        r_close.update_scene(d, camera=vclose)
        twin[-204:-4, -264:-4] = r_close.render()
        twin[-206:-204, -266:-2] = twin[-4:-2, -266:-2] = 255  # white frame around the inset
        twin[-206:-2, -266:-264] = twin[-206:-2, -4:-2] = 255
        ff.stdin.write(np.hstack([twin, gopro[n]]).tobytes())
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
    ap.add_argument("--tcp_z_mm", type=float, default=173.8,
                    help="TCP offset from the flange: 173.8 mm = closed Robotiq fingertip ends, as set on this robot")
    ap.add_argument("--speed_cap_deg_s", type=float, default=45.0, help="joint speed cap for the hardware replay")
    ap.add_argument("--z_offset_mm", type=float, default=0.0, help="lift the whole path by this much (in-the-air rehearsal)")
    ap.add_argument("--correction_mm", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    help="add this to every TRumi position (marker frame x y z, mm), e.g. minus a touch test's average offset")
    ap.add_argument("--smooth_s", type=float, default=0.0, help="smooth the fingertip path over this window (s), e.g. 0.15")
    ap.add_argument("--reference_q_deg", type=float, nargs=6, default=None,
                    help="prefer the arm configuration closest to these joint angles (consistent joints across episodes)")
    ap.add_argument("--max_starts", type=int, default=4, help="arm configurations to try for the first pose")
    ap.add_argument("--out_subdir", default="preflight", help="output folder inside the session")
    ap.add_argument("--grip_close_below_mm", type=float, default=0.0,
                    help="command the gripper fully closed while the TRumi gap is below this (grip with force); 0 = off")
    ap.add_argument("--gripper_tables", type=pathlib.Path, default=None,
                    help="measured width->command table and fingertip arc (scripts/tools/gripper_sweep_table.py output)")
    ap.add_argument("--no_box", action="store_true", help="leave the box out of the collision check (position unknown)")
    ap.add_argument("--no_video", action="store_true", help="checks and trajectory only, no video (fast batch runs)")
    main(ap.parse_args())

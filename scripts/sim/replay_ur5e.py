"""Replay one TRumi arm on a simulated UR5e + Robotiq 2F-85 (MuJoCo), next to the GoPro view.

Usage (from ~/trumi):
    MUJOCO_GL=egl data/sim/.venv/bin/python scripts/sim/replay_ur5e.py \
        --session data/2026-10-01_conveyor_bimanual --episode 2 --arm right

Outputs in <session>/retarget/:
    ep<N>_<arm>_trajectory.csv  robot-independent: time, fingertip xyz (m) + quaternion, gripper width (m),
                                in the mapping-marker frame (z = height above the marker)
    ep<N>_<arm>_ur5e.csv        the same targets in the robot base frame + UR5e joint angles + IK errors
    ep<N>_<arm>_ur5e_sim.mp4    simulated UR5e (left) next to the GoPro view (right)

Notes:
    - Kinematic replay: the arm is set to the IK solution every frame (no dynamics); only the gripper
      linkage is simulated so the fingers open/close realistically.
    - The marker -> robot-base placement is a stand-in (there is no real calibration yet): the base is
      put behind the demonstrated motion, facing along the average pointing direction, at the
      distance (from --base_distances) that gives the smallest IK error.
"""

import pathlib
import pickle
import subprocess

import argparse

import av
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

REPO = pathlib.Path(__file__).resolve().parents[2]
SIM_DATA = REPO / "data" / "sim"  # venv, robot models and scans (not versioned; see scripts/sim/setup_sim.sh)
MENAGERIE = SIM_DATA / "mujoco_menagerie"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

SLAM_STRIDE = 2  # raw video frame = video_start + SLAM_STRIDE * step
OUT_FPS = 30  # plan steps are 60 Hz -> show every 2nd step (real time)
JUMP_M = 0.025  # a single 1/60 s step longer than this (1.5 m/s) is treated as a SLAM glitch
FLOOR_M = 0.005  # never command the fingertip below this height above the marker
UR5E_MAX_SPEED_DEG_S = 180.0  # UR5e joint speed limit (all joints)

# TRumi fingertip frame = GoPro camera frame (x right, y down, z forward); fingers open along x.
# Robotiq pinch frame: z = approach, fingers close along y.  So: z_r = z_c, y_r = x_c, x_r = -y_c.
TRUMI_TO_ROBOTIQ = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

# measured in this model (see notes): fingertip opening (m, relative to fully closed) vs. actuator ctrl
OPENING_M = np.array([0.0, 8.7, 20.3, 31.8, 43.0, 53.9, 64.4, 74.2, 83.4]) / 1000
OPENING_CTRL = np.array([255, 224, 192, 160, 128, 96, 64, 32, 0], dtype=float)

HOME_Q = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
ARMS = {"right": 0, "left": 1}  # gripper index in the dataset plan (0 = right, 1 = left)


def build_model():
    """UR5e (Menagerie scene) with a Robotiq 2F-85 attached at the wrist, plus a marker square on the floor."""
    spec = mujoco.MjSpec.from_file(str(MENAGERIE / "universal_robots_ur5e" / "scene.xml"))
    spec.option.impratio = 10  # settings recommended by the 2F-85 model
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    gripper = mujoco.MjSpec.from_file(str(MENAGERIE / "robotiq_2f85" / "2f85.xml"))
    spec.attach(gripper, site=spec.site("attachment_site"), prefix="g_")
    marker = spec.worldbody.add_body(name="marker", mocap=True)
    marker.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.08, 0.08, 0.001], rgba=[0.05, 0.05, 0.05, 1],
        contype=0, conaffinity=0,
    )
    model = spec.compile()
    return model, mujoco.MjData(model)


def clean_trajectory(t, pos, rot):
    """Replace single-step jumps (SLAM glitches) by interpolation and keep the fingertip above the floor."""
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    bad = np.zeros(len(pos), dtype=bool)
    bad[1:] |= step > JUMP_M
    good = ~bad
    pos = pos.copy()
    for k in range(3):
        pos[:, k] = np.interp(t, t[good], pos[good, k])
    rot = Slerp(t[good], rot[good])(t)
    n_floor = int((pos[:, 2] < FLOOR_M).sum())
    pos[:, 2] = np.maximum(pos[:, 2], FLOOR_M)
    return pos, rot, int(bad.sum()), n_floor


def placement(pos, rot, distance):
    """Stand-in marker->base transform: base behind the motion, facing the average pointing direction."""
    d = rot.apply([0, 0, 1])[:, :2].mean(0)
    d /= np.linalg.norm(d)
    centroid = pos[:, :2].mean(0)
    base_xy = centroid - distance * d
    yaw = np.arctan2(d[1], d[0])
    R_mb = R.from_euler("z", yaw)  # base orientation in the marker frame
    t_mb = np.array([base_xy[0], base_xy[1], 0.0])
    return R_mb, t_mb


def solve_ik(model, data, site_id, p_target, R_target, q0, iters=200, lam=0.05):
    """Damped least-squares IK on the 6 arm joints for the pinch site."""
    lo, hi = model.jnt_range[:6, 0], model.jnt_range[:6, 1]
    q = q0.copy()
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for _ in range(iters):
        data.qpos[:6] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        e_p = p_target - data.site_xpos[site_id]
        e_r = R.from_matrix(R_target @ data.site_xmat[site_id].reshape(3, 3).T).as_rotvec()
        if np.linalg.norm(e_p) < 1e-4 and np.linalg.norm(e_r) < 1e-3:
            break
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        J = np.vstack([jacp[:, :6], jacr[:, :6]])
        dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(6), np.concatenate([e_p, e_r]))
        q = np.clip(q + dq, lo, hi)
    return q, np.linalg.norm(e_p), np.degrees(np.linalg.norm(e_r))


def run_ik(model, data, site_id, p_base, R_base):
    """IK for the whole trajectory, warm-started from the previous frame; first frame tries several seeds."""
    seeds = [HOME_Q + np.array([dp, 0, 0, 0, 0, dw]) for dp in (-1.0, 0.0, 1.0) for dw in (-1.5, 0.0, 1.5)]
    best = min(
        (solve_ik(model, data, site_id, p_base[0], R_base[0].as_matrix(), s) for s in seeds),
        key=lambda r: (r[1] > 0.002, np.abs(r[0] - HOME_Q).sum()),
    )
    qs, ep, er = [best[0]], [best[1]], [best[2]]
    for i in range(1, len(p_base)):
        q, e1, e2 = solve_ik(model, data, site_id, p_base[i], R_base[i].as_matrix(), qs[-1])
        qs.append(q)
        ep.append(e1)
        er.append(e2)
    return np.array(qs), np.array(ep), np.array(er)


def read_video_frames(path, frame_idxs):
    wanted, out = set(frame_idxs), {}
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        for i, f in enumerate(c.decode(s)):
            if i in wanted:
                out[i] = f.to_ndarray(format="rgb24", width=640, height=480)
            if i >= max(wanted):
                break
    return [out[i] for i in frame_idxs]


def main(session, episode, arm, base_distances):
    session = session.resolve()
    ep = pickle.load(open(session / "dataset_plan.pkl", "rb"))[episode - 1]
    g = ARMS[arm]
    ts = np.asarray(ep["episode_timestamps"], dtype=float)
    t = ts - ts[0]
    pose = np.asarray(ep["grippers"][g]["tcp_pose"], dtype=float)
    width = np.asarray(ep["grippers"][g]["gripper_width"], dtype=float)
    pos, rot_trumi, n_jumps, n_floor = clean_trajectory(t, pose[:, :3], R.from_rotvec(pose[:, 3:]))
    rot_robotiq = rot_trumi * R.from_matrix(TRUMI_TO_ROBOTIQ)

    out_dir = session / "retarget"
    out_dir.mkdir(exist_ok=True)
    stem = f"ep{episode}_{arm}"
    q_trumi = rot_trumi.as_quat()
    np.savetxt(out_dir / f"{stem}_trajectory.csv", np.column_stack([t, pos, q_trumi, width]), delimiter=",",
               header="t_s,x_m,y_m,z_m,qx,qy,qz,qw,gripper_width_m  (TRumi fingertip, mapping-marker frame)",
               comments="", fmt="%.6f")

    model, data = build_model()
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "g_pinch")

    # choose the stand-in base placement that the UR5e can follow best
    results = []
    for dist in [float(x) for x in base_distances.split(",")]:
        R_mb, t_mb = placement(pos, rot_robotiq, dist)
        p_base = R_mb.inv().apply(pos - t_mb)
        R_base = R_mb.inv() * rot_robotiq
        qs, ep_err, er_err = run_ik(model, data, site_id, p_base, R_base)
        speed = np.degrees(np.abs(np.diff(qs, axis=0)) / np.diff(t)[:, None])
        results.append((ep_err.max(), dist, R_mb, t_mb, p_base, R_base, qs, ep_err, er_err, speed))
        print(f"base {dist:.2f} m behind the motion: IK position error max {ep_err.max()*1000:6.1f} mm, "
              f"rotation error max {er_err.max():5.2f} deg, fastest joint {speed.max():5.0f} deg/s")
    _, dist, R_mb, t_mb, p_base, R_base, qs, ep_err, er_err, speed = min(results, key=lambda r: r[0])

    ctrl = np.interp(np.clip(width, OPENING_M[0], OPENING_M[-1]), OPENING_M, OPENING_CTRL)
    np.savetxt(out_dir / f"{stem}_ur5e.csv",
               np.column_stack([t, p_base, R_base.as_quat(), qs, ctrl, ep_err * 1000, er_err]), delimiter=",",
               header="t_s,x_m,y_m,z_m,qx,qy,qz,qw,q1_rad,q2_rad,q3_rad,q4_rad,q5_rad,q6_rad,"
                      "gripper_ctrl_0open_255closed,ik_pos_err_mm,ik_rot_err_deg  (UR5e base frame)",
               comments="", fmt="%.6f")

    # --- render: simulated robot (left) + GoPro view (right)
    steps = list(range(0, len(t), 60 // OUT_FPS))
    cam = ep["cameras"][g]
    gopro = read_video_frames(session / "demos" / cam["video_path"],
                              [cam["video_start_end"][0] + SLAM_STRIDE * s for s in steps])
    renderer = mujoco.Renderer(model, 480, 640)
    vcam = mujoco.MjvCamera()
    vcam.lookat[:] = p_base.mean(0)
    vcam.distance, vcam.azimuth, vcam.elevation = 1.3, 150.0, -25.0
    mid = model.body("marker").mocapid[0]
    substeps = max(1, round((1 / OUT_FPS) / model.opt.timestep))

    tmp = out_dir / f"{stem}_ur5e_sim_tmp.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x480",
                           "-r", str(OUT_FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(tmp)],
                          stdin=subprocess.PIPE)
    mujoco.mj_resetData(model, data)
    data.mocap_pos[mid] = R_mb.inv().apply(-t_mb)  # marker origin in the base frame
    data.mocap_quat[mid] = np.roll(R_mb.inv().as_quat(), 1)  # scipy xyzw -> mujoco wxyz
    for n, s in enumerate(steps):
        data.ctrl[:6] = qs[s]
        data.ctrl[6] = ctrl[s]
        for _ in range(substeps):
            data.qpos[:6] = qs[s]
            data.qvel[:6] = 0
            mujoco.mj_step(model, data)
        renderer.update_scene(data, camera=vcam)
        sc = renderer.scene
        for k in range(0, len(p_base), 6):  # the recorded fingertip path, faint
            if sc.ngeom >= sc.maxgeom:
                break
            mujoco.mjv_initGeom(sc.geoms[sc.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, [0.004, 0, 0], p_base[k],
                                np.eye(3).flatten(), np.array([1, 0.2, 0.2, 0.35 if k > s else 0.9], dtype=np.float32))
            sc.ngeom += 1
        ff.stdin.write(np.hstack([renderer.render(), gopro[n]]).tobytes())
    ff.stdin.close()
    ff.wait()
    out = out_dir / f"{stem}_ur5e_sim.mp4"
    label = (f"drawtext=fontfile={FONT}:text='simulated UR5e + Robotiq 2F-85':x=10:y=10:fontsize=20:fontcolor=white:"
             f"box=1:boxcolor=black@0.5,drawtext=fontfile={FONT}:text='TRumi {arm} camera (episode {episode})':"
             f"x=650:y=10:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.5")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(tmp), "-vf", label, "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(out)], check=True)
    tmp.unlink()

    lim = np.degrees(model.jnt_range[:6])
    near_limit = (np.degrees(qs) < lim[:, 0] + 5) | (np.degrees(qs) > lim[:, 1] - 5)
    print("\n=== RETARGET REPORT ===")
    print(f"trajectory: {len(t)} steps ({t[-1]:.1f} s); glitch jumps smoothed: {n_jumps}; floor clamps: {n_floor}")
    print(f"chosen stand-in placement: base {dist:.2f} m behind the motion's centre")
    print(f"IK position error: median {np.median(ep_err)*1000:.1f} mm, max {ep_err.max()*1000:.1f} mm")
    print(f"IK rotation error: median {np.median(er_err):.2f} deg, max {er_err.max():.2f} deg")
    print(f"fastest joint: {speed.max():.0f} deg/s (UR5e limit {UR5E_MAX_SPEED_DEG_S:.0f}); "
          f"frames over limit: {(speed.max(1) > UR5E_MAX_SPEED_DEG_S).sum()}")
    print(f"frames within 5 deg of a joint limit: {near_limit.any(1).sum()}")
    print(f"gripper: TRumi width {width.min()*1000:.0f}-{width.max()*1000:.0f} mm -> 2F-85 ctrl {ctrl.min():.0f}-{ctrl.max():.0f}")
    print(f"saved: {out}\n       {out_dir / (stem + '_ur5e.csv')}\n       {out_dir / (stem + '_trajectory.csv')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Replay one TRumi arm on a simulated UR5e + Robotiq 2F-85.")
    ap.add_argument("--session", required=True, type=pathlib.Path, help="Processed session directory.")
    ap.add_argument("--episode", required=True, type=int, help="Episode number in the dataset plan (1-based).")
    ap.add_argument("--arm", choices=list(ARMS), default="right")
    ap.add_argument("--base_distances", default="0.35,0.45,0.55",
                    help="Candidate horizontal distances (m) from the base to the motion's centre.")
    a = ap.parse_args()
    main(a.session, a.episode, a.arm, a.base_distances)

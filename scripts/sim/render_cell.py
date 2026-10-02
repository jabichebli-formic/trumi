"""Render the digital twin of the cell (scan + marker + box + robot) from three viewpoints, optionally with the
grasp (green) and release (orange) fingertip positions of every episode of a session.

The robot is placed with the calibration (as in preflight.py); the marker pattern and the box come from
data/sim/cell.json. Collision boxes other than the box are hidden so the scan stays readable.

Usage (from ~/trumi):
    MUJOCO_GL=egl data/sim/.venv/bin/python scripts/sim/render_cell.py [--session data/<session> --correction_mm -12.3 0.3 -8.5]
Output: data/sim/plan/cell_views.png (or --out)
"""

import argparse
import json
import pathlib
import pickle
import subprocess
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from episode_report import events  # noqa: E402
from plan_cell import Planner  # noqa: E402
from playback_touches import FONT, calibrated_placement  # noqa: E402
from view_twin import REPO  # noqa: E402

ARM_UP_DEG = [13.9, -96.7, 14.0, -180.1, -93.2, 169.8]  # where the real robot is parked during recording


def session_points(session, correction_mm):
    """Grasp and release fingertip positions (marker frame, m) of every episode in the session's dataset plan."""
    grasps, releases = [], []
    for ep in pickle.load(open(session / "dataset_plan.pkl", "rb")):
        p = np.asarray(ep["grippers"][0]["tcp_pose"], float)[:, :3] + np.array(correction_mm) / 1000
        g, r = events(np.asarray(ep["grippers"][0]["gripper_width"], float))
        if g is not None:
            grasps.append(p[g])
        if r is not None:
            releases.append(p[r])
    return np.array(grasps), np.array(releases)


def main(a):
    cell = json.load(open(a.cell))
    T_cm = np.array(json.load(open(a.calibration))["T_base_marker"])
    yaw, _, _, Twm = calibrated_placement(cell, T_cm)
    cell["marker"]["centre"] = Twm[:3, 3].tolist()
    cell["marker"]["yaw_deg"] = float(np.degrees(np.arctan2(Twm[1, 0], Twm[0, 0])))
    pl = Planner(cell, controller_yaw_deg=yaw, tcp_z_mm=a.tcp_z_mm)
    for g in range(pl.m.ngeom):  # hide the collision boxes except the box and the marker
        nm = mujoco.mj_id2name(pl.m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if pl.m.geom_bodyid[g] == 0 and nm and not nm.startswith(("marker_paper", "marker_bit", "box_", "scan")):
            pl.m.geom_rgba[g][3] = 0.0
        if nm.startswith("box_"):
            pl.m.geom_rgba[g] = [0.72, 0.52, 0.32, 1.0]
    pl.d.qpos[:] = 0
    pl.d.qpos[:6] = np.radians(a.joints_deg)
    mujoco.mj_forward(pl.m, pl.d)

    pts = []
    if a.session:
        gr, rel = session_points(a.session, a.correction_mm)
        to_w = lambda P: (Twm[:3, :3] @ P.T).T + Twm[:3, 3] if len(P) else P
        far = lambda P: np.linalg.norm(P[:, :2], axis=1) > 0.8  # tracking glitches far from the cell
        pts = [(p, [0.1, 0.9, 0.2, 1]) for p in to_w(gr[~far(gr)])] + [(p, [1, 0.5, 0, 1]) for p in to_w(rel[~far(rel)])]
        print(f"{len(gr)} grasps (green), {len(rel)} releases (orange); left out as far-off: {far(gr).sum()} grasps, {far(rel).sum()} releases")

    box_c = np.array(cell["box"]["centre"]) if cell.get("box") else Twm[:3, 3]
    mid = (Twm[:3, 3] + box_c) / 2
    view_az = cell["marker"]["yaw_deg"] + 90  # looking along the marker's +y (toward the belt), like the photo
    views = [("like the photo: from the front of the lower table", mid + [0, 0, 0.1], 1.6, view_az, -38),
             ("top-down (belt at the top)", mid + [0, 0.2, 0], 2.2, view_az, -89.9),
             ("close-up: marker and box", mid, 0.8, view_az - 25, -50)]
    r = mujoco.Renderer(pl.m, 600, 800)
    ims = []
    for title, look, dist, az, el in views:
        cam = mujoco.MjvCamera()
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = look, dist, az, el
        r.update_scene(pl.d, camera=cam)
        sc = r.scene
        for p, col in pts:
            if sc.ngeom < sc.maxgeom:
                mujoco.mjv_initGeom(sc.geoms[sc.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, [0.008, 0, 0], p, np.eye(3).flatten(),
                                    np.array(col, dtype=np.float32))
                sc.ngeom += 1
        ims.append((title, r.render()))
    grid = np.hstack([im for _, im in ims])
    txt = ",".join(f"drawtext=fontfile={FONT}:text='{t}':x={10 + 800 * i}:y=10:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.6"
                   for i, (t, _) in enumerate(ims))
    if pts:
        txt += (f",drawtext=fontfile={FONT}:text='green = where the gripper closed (grasp)   orange = where it opened (release)':"
                "x=10:y=565:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.6")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{grid.shape[1]}x{grid.shape[0]}", "-i", "-",
                    "-vf", txt, "-frames:v", "1", str(a.out)], input=grid.tobytes(), check=True)
    print(f"saved {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Render the cell twin from three viewpoints.")
    ap.add_argument("--cell", type=pathlib.Path, default=REPO / "data" / "sim" / "cell.json")
    ap.add_argument("--calibration", type=pathlib.Path, default=REPO / "data" / "robot" / "marker_in_robot_base.json")
    ap.add_argument("--session", type=pathlib.Path, default=None, help="overlay grasp/release points of this session")
    ap.add_argument("--correction_mm", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    ap.add_argument("--joints_deg", type=float, nargs=6, default=ARM_UP_DEG)
    ap.add_argument("--tcp_z_mm", type=float, default=173.8)
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "data" / "sim" / "plan" / "cell_views.png")
    main(ap.parse_args())

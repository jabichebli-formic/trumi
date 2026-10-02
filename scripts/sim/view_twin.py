"""Digital twin: the phone scan of the cell + a UR5e with Robotiq 2F-85 placed where the real robot stands.

Usage (from ~/trumi, in a terminal on the desktop):
    data/sim/.venv/bin/python scripts/sim/view_twin.py
    # headless preview images instead of a window:
    MUJOCO_GL=egl data/sim/.venv/bin/python scripts/sim/view_twin.py --render twin.png

In the viewer: left-drag rotates, right-drag pans, scroll zooms. Open the "Control" panel (right side)
and drag the sliders to move each robot joint; the arm holds the pose you set.

Placement comes from the scan (data/sim/scans/New-conveyor2/): base centre fitted to the scanned
base cylinder, height from the mounting plate, starting pose fitted to the scanned arm. Expect ~1-2 cm
accuracy; this is for previewing and planning, not a replacement for the robot calibration.
"""

import argparse
import pathlib
import subprocess

import mujoco
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
SIM_DATA = REPO / "data" / "sim"  # venv, robot models and scans (not versioned; see scripts/sim/setup_sim.sh)
MENAGERIE = SIM_DATA / "mujoco_menagerie"
SCAN = SIM_DATA / "scans" / "New-conveyor2"
BASE_Z = 0.885  # top of the scanned mounting plate (m, scan coordinates)
CUT_RADIUS = 0.25  # scanned robot = everything within this distance of the base axis ...
CUT_ABOVE_Z = 0.895  # ... and above the mounting plate (plate and frame are kept)


def mesh_without_scanned_robot(bx, by):
    """Write (once) a copy of the scan mesh with the scanned robot's triangles removed."""
    src, dst = SCAN / "mesh.obj", SCAN / "mesh_norobot.obj"
    if dst.is_file():
        return dst
    lines = src.read_text().splitlines()
    verts = np.array([[float(x) for x in ln.split()[1:4]] for ln in lines if ln.startswith("v ")])
    # OBJ is in glTF axes (y up); the viewer rotates it so that up = +z: (x, y, z)_obj -> (x, -z, y)
    inside = (np.hypot(verts[:, 0] - bx, -verts[:, 2] - by) < CUT_RADIUS) & (verts[:, 1] > CUT_ABOVE_Z)
    kept, dropped = [], 0
    for ln in lines:
        if ln.startswith("f ") and any(inside[int(tok.split("/")[0]) - 1] for tok in ln.split()[1:]):
            dropped += 1
            continue
        kept.append(ln)
    dst.write_text("\n".join(kept) + "\n")
    print(f"removed the scanned robot: {dropped:,} triangles ({inside.sum():,} vertices) -> {dst.name}")
    return dst


def build_twin(show_scanned_robot=False):
    bx, by, _ = np.load(SCAN / "base_circle.npy")
    pose_file = SCAN / "scan_robot_pose.npy"
    q0 = np.load(pose_file)[3:] if pose_file.is_file() else np.array([0, -1.57, 1.57, -1.57, -1.57, 0])

    spec = mujoco.MjSpec.from_file(str(MENAGERIE / "universal_robots_ur5e" / "ur5e.xml"))
    spec.option.impratio = 10  # settings recommended by the 2F-85 model
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.body("base").pos = [bx, by, BASE_Z]
    gripper = mujoco.MjSpec.from_file(str(MENAGERIE / "robotiq_2f85" / "2f85.xml"))
    spec.attach(gripper, site=spec.site("attachment_site"), prefix="g_")

    spec.add_texture(name="scan_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(SCAN / "texture.png"))
    mat = spec.add_material(name="scan_mat", specular=0)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "scan_tex"
    scan_mesh = SCAN / "mesh.obj" if show_scanned_robot else mesh_without_scanned_robot(bx, by)
    spec.add_mesh(name="scan", file=str(scan_mesh), inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
    spec.worldbody.add_geom(name="scan_geom", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="scan", material="scan_mat",
                            quat=[0.7071068, 0.7071068, 0, 0], contype=0, conaffinity=0)  # glTF y-up -> z-up
    spec.worldbody.add_light(pos=[bx, by, 3.0], dir=[0, 0, -1], diffuse=[0.4, 0.4, 0.4])
    spec.visual.headlight.ambient = [0.8, 0.8, 0.8]
    spec.visual.headlight.diffuse = [0.2, 0.2, 0.2]
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 960

    model = spec.compile()
    data = mujoco.MjData(model)
    data.qpos[:6] = q0
    data.ctrl[:6] = q0
    mujoco.mj_forward(model, data)
    return model, data, np.array([bx, by, BASE_Z])


def render_preview(model, data, base, out):
    r = mujoco.Renderer(model, 480, 640)
    cam = mujoco.MjvCamera()
    tiles = []
    # 4 low close-ups at plate level (to judge the base height), then an overview and a top-down view
    for az, el, dist, look_z in [(0, -4, 0.55, base[2] + 0.06), (90, -4, 0.55, base[2] + 0.06),
                                 (180, -4, 0.55, base[2] + 0.06), (270, -4, 0.55, base[2] + 0.06),
                                 (135, -35, 2.0, 1.05), (90, -89, 1.6, 1.05)]:
        cam.lookat[:] = [base[0], base[1], look_z]
        cam.distance, cam.azimuth, cam.elevation = dist, az, el
        r.update_scene(data, camera=cam)
        tiles.append(r.render())
    grid = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1920x960", "-i", "-",
                    "-frames:v", "1", str(out)], input=grid.tobytes(), check=True)
    print(f"saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="View the scanned cell with the UR5e + 2F-85 in place.")
    ap.add_argument("--render", type=pathlib.Path, help="Save preview images here instead of opening a window.")
    ap.add_argument("--show_scanned_robot", action="store_true",
                    help="Keep the scanned robot in the scan (by default it is cut out so the sim robot is visible).")
    a = ap.parse_args()
    model, data, base = build_twin(a.show_scanned_robot)
    print(f"UR5e base placed at x={base[0]:.3f} y={base[1]:.3f} z={base[2]:.3f} m (scan coordinates)")
    if a.render:
        render_preview(model, data, base, a.render)
    else:
        import mujoco.viewer
        mujoco.viewer.launch(model, data)

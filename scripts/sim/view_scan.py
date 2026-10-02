"""View a Scaniverse (or any single-mesh, textured) .glb scan in MuJoCo.

Usage (from ~/trumi):
    # interactive viewer (run in a terminal on the desktop, needs a screen)
    data/sim/.venv/bin/python scripts/sim/view_scan.py --glb ~/Desktop/New-conveyor1.glb
    # or: save preview images without a screen
    MUJOCO_GL=egl data/sim/.venv/bin/python scripts/sim/view_scan.py \
        --glb ~/Desktop/New-conveyor1.glb --render preview.png

The .glb is converted once to OBJ + PNG in data/sim/scans/<name>/ (the original is not changed).
glTF is y-up, MuJoCo is z-up, so the scan is rotated +90 deg about x. Units stay in metres.
"""

import argparse
import json
import pathlib
import struct
import subprocess

import mujoco
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
SIM_DATA = REPO / "data" / "sim"  # venv, robot models and scans (not versioned; see scripts/sim/setup_sim.sh)
COMPONENT = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def read_glb(path):
    """Return the glTF JSON and the binary chunk of a .glb file."""
    b = path.read_bytes()
    magic, _, _ = struct.unpack("<4sII", b[:12])
    if magic != b"glTF":
        raise ValueError(f"{path} is not a .glb file")
    jlen, _ = struct.unpack("<II", b[12:20])
    gltf = json.loads(b[20:20 + jlen])
    off = 20 + jlen
    blen, _ = struct.unpack("<II", b[off:off + 8])
    return gltf, b[off + 8:off + 8 + blen]


def accessor(gltf, binary, idx):
    a = gltf["accessors"][idx]
    bv = gltf["bufferViews"][a["bufferView"]]
    dtype, n = COMPONENT[a["componentType"]], NCOMP[a["type"]]
    start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = bv.get("byteStride", 0)
    if stride and stride != n * np.dtype(dtype).itemsize:
        raw = np.frombuffer(binary, np.uint8, count=stride * a["count"], offset=start).reshape(a["count"], stride)
        return raw[:, : n * np.dtype(dtype).itemsize].copy().view(dtype).reshape(a["count"], n)
    return np.frombuffer(binary, dtype, count=a["count"] * n, offset=start).reshape(a["count"], n)


def convert(glb):
    """Convert a single-mesh textured .glb to OBJ + PNG (cached). Returns (obj_path, png_path)."""
    out = SIM_DATA / "scans" / glb.stem.replace(" ", "_").replace("(", "").replace(")", "")
    obj, png = out / "mesh.obj", out / "texture.png"
    if obj.is_file() and png.is_file():
        return obj, png
    out.mkdir(parents=True, exist_ok=True)
    gltf, binary = read_glb(glb)
    prim = gltf["meshes"][0]["primitives"][0]
    v = accessor(gltf, binary, prim["attributes"]["POSITION"])
    uv = accessor(gltf, binary, prim["attributes"]["TEXCOORD_0"])
    f = accessor(gltf, binary, prim["indices"]).reshape(-1, 3).astype(np.int64) + 1
    with open(obj, "w") as fh:
        fh.write("".join(f"v {x:.5f} {y:.5f} {z:.5f}\n" for x, y, z in v))
        fh.write("".join(f"vt {u:.5f} {1 - w:.5f}\n" for u, w in uv))  # glTF uv origin is top-left
        fh.write("".join(f"f {a}/{a} {b}/{b} {c}/{c}\n" for a, b, c in f))
    tex_idx = gltf["materials"][prim["material"]]["pbrMetallicRoughness"]["baseColorTexture"]["index"]
    img = gltf["images"][gltf["textures"][tex_idx]["source"]]
    bv = gltf["bufferViews"][img["bufferView"]]
    jpg = out / "texture.jpg"
    jpg.write_bytes(binary[bv.get("byteOffset", 0): bv.get("byteOffset", 0) + bv["byteLength"]])
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(jpg), str(png)], check=True)
    jpg.unlink()
    print(f"converted {glb.name}: {len(v):,} vertices, {len(f):,} triangles -> {out}")
    return obj, png


def scene_xml(obj, png):
    return f"""
<mujoco model="scan">
  <compiler angle="radian"/>
  <visual><headlight ambient="0.6 0.6 0.6" diffuse="0.3 0.3 0.3"/><global offwidth="1280" offheight="960"/></visual>
  <asset>
    <texture name="scan_tex" type="2d" file="{png}"/>
    <material name="scan_mat" texture="scan_tex" specular="0" shininess="0"/>
    <mesh name="scan" file="{obj}" inertia="shell"/>
  </asset>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" diffuse="0.5 0.5 0.5"/>
    <geom type="mesh" mesh="scan" material="scan_mat" quat="0.7071068 0.7071068 0 0" contype="0" conaffinity="0"/>
  </worldbody>
</mujoco>"""


def render_views(model, out_png):
    """Save a 2x2 grid of views (top-down + three angles) to out_png."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, 480, 640)
    cam = mujoco.MjvCamera()
    # centre of the scan in world coordinates (after the y-up -> z-up rotation)
    c = data.geom_xpos[0] + data.geom_xmat[0].reshape(3, 3) @ model.geom_aabb[0][:3]
    size = np.linalg.norm(model.geom_aabb[0][3:])
    tiles = []
    for az, el in [(90, -89), (45, -35), (135, -35), (225, -35)]:
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = c, 1.4 * size, az, el
        renderer.update_scene(data, camera=cam)
        tiles.append(renderer.render())
    grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x960", "-i", "-",
                    "-frames:v", "1", str(out_png)], input=grid.tobytes(), check=True)
    print(f"saved {out_png} (top-down, then 3 angled views)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="View a textured .glb scan in MuJoCo.")
    ap.add_argument("--glb", required=True, type=pathlib.Path)
    ap.add_argument("--render", type=pathlib.Path, help="Save preview images here instead of opening a window.")
    a = ap.parse_args()
    obj, png = convert(a.glb.expanduser().resolve())
    model = mujoco.MjModel.from_xml_string(scene_xml(obj, png))
    if a.render:
        render_views(model, a.render)
    else:
        import mujoco.viewer
        mujoco.viewer.launch(model)

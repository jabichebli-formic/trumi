"""Per-episode summary of a processed session: where the gripper closed (grasp) and opened again (release).

Positions are the TRumi fingertip in the mapping-marker frame (cm): origin = marker centre, +x = marker right edge,
+y = marker top edge, +z = up from the marker surface. With --calibration they are also given in the UR controller's
base frame (mm, as shown on the pendant with Feature = Base).

Usage (from ~/trumi):
    uv run python scripts/tools/episode_report.py --session data/2026-10-02_real_conveyor_setup
    uv run python scripts/tools/episode_report.py --session <s> --offset_mm -12.3 0.3 -8.5 --calibration data/robot/marker_in_robot_base.json
--offset_mm: correction added to every TRumi position (marker frame, mm), e.g. the average offset a touch test found,
negated. Episodes whose video name contains "touch" are skipped.
"CHECK" flags episodes whose path jumps faster than 3 m/s or strays > 0.9 m from the marker: SLAM tracking went
wrong even though the pipeline kept the episode (exclude it with a check_result.txt containing "false").
"""

import argparse
import json
import pathlib
import pickle

import numpy as np


def events(w):
    """Indices where the gripper closes (falls below half-way between open and closed) and opens again; None if never."""
    hi, lo = np.percentile(w, 95), w.min()
    if hi - lo < 0.02:  # never opened/closed by more than 2 cm
        return None, None
    mid = (hi + lo) / 2
    closed = np.flatnonzero(w < mid)
    if not len(closed):
        return None, None
    g = closed[0]
    opened = np.flatnonzero(w[g:] > mid)
    return g, (g + opened[0] if len(opened) else None)


def path_checks(t, p, max_speed=3.0):
    """(frames moving faster than max_speed m/s, farthest distance from the marker in the plane (m)) for a fingertip path.
    Large values mean SLAM tracking went wrong even though the pipeline kept the episode."""
    v = np.linalg.norm(np.diff(p, axis=0), axis=1) / np.diff(t)
    return int((v > max_speed).sum()), float(np.linalg.norm(p[:, :2], axis=1).max())


def main(a):
    plan = pickle.load(open(a.session / "dataset_plan.pkl", "rb"))
    off = np.array(a.offset_mm) / 1000
    T = np.array(json.load(open(a.calibration))["T_base_marker"]) if a.calibration else None
    rows = []
    print(f"{'episode':32s} {'len':>5s}  {'grasp x y z (cm)':>22s}  {'release x y z (cm)':>22s}" + ("   grasp in robot base (mm)" if T is not None else ""))
    for i, ep in enumerate(plan):
        name = "_".join(pathlib.Path(ep["cameras"][0]["video_path"]).parent.name.split("_")[4:])  # drop demo_<serial>_<date>_<time>_
        if "touch" in name:
            continue
        t = np.asarray(ep["episode_timestamps"], float)
        p = np.asarray(ep["grippers"][0]["tcp_pose"], float)[:, :3] + off
        g, r = events(np.asarray(ep["grippers"][0]["gripper_width"], float))
        fmt = lambda k: " ".join(f"{v*100:6.1f}" for v in p[k]) if k is not None else f"{'none':>20s}"
        n_jumps, far = path_checks(t, p)
        line = f"ep{i + 1:<3d}{name[:28]:28s} {t[-1] - t[0]:4.1f}s  {fmt(g):>22s}  {fmt(r):>22s}"
        if T is not None and g is not None:
            line += "   " + " ".join(f"{v*1000:7.1f}" for v in (T @ np.r_[p[g], 1])[:3])
        if n_jumps or far > 0.9:
            line += f"   CHECK: {n_jumps} jumps > 3 m/s, farthest {far:.2f} m from the marker"
        print(line)
        rows.append((p[g] if g is not None else None, p[r] if r is not None else None))
    for k, label in ((0, "grasp"), (1, "release")):
        pts = np.array([r[k] for r in rows if r[k] is not None])
        if len(pts) > 1:
            print(f"{label:8s} mean {np.round(pts.mean(0) * 100, 1)} cm, spread (std) {np.round(pts.std(0) * 100, 1)} cm over {len(pts)} episodes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Grasp/release positions per episode.")
    ap.add_argument("--session", type=pathlib.Path, required=True)
    ap.add_argument("--offset_mm", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    ap.add_argument("--calibration", type=pathlib.Path, default=None)
    main(ap.parse_args())

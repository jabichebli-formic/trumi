"""Compare the robot's wrist-camera view of its gripper with the TRumi's, using gripper-calibration videos
(gripper opening and closing, camera still) from both.

Both videos are run through the pipeline's ArUco detector. The two finger tags (IDs 0 and 1) give, per frame,
their 3D position relative to the camera; the x-distance between them is what the pipeline uses as the gripper
width. Frames are matched by that width, so robot and TRumi are compared at the same opening:
  - tag position differences (mm) = how far the robot's fingers sit from where the TRumi's were in the image
  - tag depth vs opening = the Robotiq's arc (the TRumi's fingers move on linear rails, constant depth)
  - pipeline check: the pipeline only accepts finger tags 64-80 mm in front of the camera
Images: robot vs TRumi at open / half / closed, as a 50/50 blend, an edge overlay (green = TRumi,
magenta = robot), and with the UMI-style policy mask and the finger mask applied.

Usage (from ~/trumi):
    uv run python scripts/tools/compare_gripper_views.py --robot_video <robot.MP4> --trumi_video <trumi_calibration.MP4>
Outputs in --out (default data/robot/view_compare/): report.txt, tags_vs_opening.png, compare_<open|half|closed>.jpg
"""

import argparse
import json
import pathlib

import av
import cv2
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from trumi.utils.cv_util import (  # noqa: E402
    convert_fisheye_intrinsics_resolution, detect_localize_aruco_tags, draw_predefined_mask, parse_aruco_config,
    parse_fisheye_intrinsics)

REPO = pathlib.Path(__file__).resolve().parents[2]
CALIB = REPO / "example" / "calibration"
Z_OK = (0.064, 0.080)  # depth window the pipeline accepts for finger tags (cv_util.get_gripper_width)


def detect(video, stride):
    """Per sampled frame: time, frame (BGR), and {tag id: (tvec, pixel centre)} for the finger tags 0 and 1."""
    cfg = parse_aruco_config(yaml.safe_load((CALIB / "aruco_config.yaml").read_text()))
    intr = parse_fisheye_intrinsics(json.loads((CALIB / "gopro13_intrinsics_2_7k.json").read_text()))
    rows = []
    with av.open(str(video)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        fi = convert_fisheye_intrinsics_resolution(opencv_intr_dict=intr, target_resolution=np.array([s.width, s.height]))
        for i, fr in enumerate(c.decode(s)):
            if i % stride:
                continue
            img = fr.to_ndarray(format="bgr24")
            tags = detect_localize_aruco_tags(draw_predefined_mask(img.copy(), color=(0, 0, 0), mirror=True, gripper=False, finger=False),
                                              cfg["aruco_dict"], cfg["marker_size_map"], fi, refine_subpix=True)
            rows.append((float(fr.pts * s.time_base), img, {k: (v["tvec"], v["corners"].mean(0)) for k, v in tags.items() if k in (0, 1)}))
    return rows


def table(rows):
    """Frames where both finger tags were seen: width (tag x-distance), tvec of tag 0 and 1, pixel centres."""
    out = [(t, d[1][0][0] - d[0][0][0], d[0][0], d[1][0], d[0][1], d[1][1], k) for k, (t, _, d) in enumerate(rows) if 0 in d and 1 in d]
    return out


def masked(img, **kw):
    return draw_predefined_mask(img.copy(), color=(0, 0, 0), **kw)


def label(img, text):
    img = img.copy()
    cv2.putText(img, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 9)
    cv2.putText(img, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 3)
    return img


def main(a):
    a.out.mkdir(parents=True, exist_ok=True)
    R_rows, T_rows = detect(a.robot_video, a.stride), detect(a.trumi_video, a.stride)
    R, T = table(R_rows), table(T_rows)
    if not R or not T:
        raise SystemExit(f"both finger tags seen together in {len(R)} robot frames and {len(T)} TRumi frames: cannot compare")
    rw, tw = np.array([r[1] for r in R]), np.array([r[1] for r in T])
    lines = [f"robot video: {a.robot_video} ({len(R_rows)} frames sampled, both tags in {len(R)})",
             f"TRumi video: {a.trumi_video} ({len(T_rows)} frames sampled, both tags in {len(T)})",
             f"tag x-distance (= the pipeline's gripper width signal): robot {rw.min()*1000:.1f}-{rw.max()*1000:.1f} mm, "
             f"TRumi {tw.min()*1000:.1f}-{tw.max()*1000:.1f} mm", ""]

    # pipeline acceptance of the robot's tags (depth window)
    for name, rows in (("robot", R_rows), ("TRumi", T_rows)):
        for tid in (0, 1):
            z = np.array([d[tid][0][2] for _, _, d in rows if tid in d])
            if len(z):
                ok = ((z > Z_OK[0]) & (z < Z_OK[1])).mean()
                lines.append(f"{name} tag {tid}: seen in {len(z)}/{len(rows)} frames, depth {z.min()*1000:.1f}-{z.max()*1000:.1f} mm, "
                             f"{ok*100:.0f}% inside the pipeline's {Z_OK[0]*1000:.0f}-{Z_OK[1]*1000:.0f} mm window")
            else:
                lines.append(f"{name} tag {tid}: never seen")
    lines.append("")

    # matched-opening comparison: robot frames vs the TRumi frames at the same tag x-distance
    lines.append("tag position difference, robot minus TRumi at the same opening (camera axes: x right, y down, z forward), mm:")
    lines.append(f"{'opening':>9s} | {'tag 0 (left)  dx    dy    dz':>30s} | {'tag 1 (right)  dx    dy    dz':>31s} | pixel shift tag0 / tag1")
    picks = {}
    lo, hi = max(rw.min(), tw.min()), min(rw.max(), tw.max())
    for nm, w in (("closed", lo), ("quarter", lo + 0.25 * (hi - lo)), ("half", (lo + hi) / 2), ("3/4", lo + 0.75 * (hi - lo)), ("open", hi)):
        ri = int(np.argmin(np.abs(rw - w)))
        ti = int(np.argmin(np.abs(tw - rw[ri])))
        r, t = R[ri], T[ti]
        d0, d1 = (r[2] - t[2]) * 1000, (r[3] - t[3]) * 1000
        p0, p1 = np.linalg.norm(r[4] - t[4]), np.linalg.norm(r[5] - t[5])
        lines.append(f"{rw[ri]*1000:6.1f} mm | {d0[0]:+6.1f} {d0[1]:+6.1f} {d0[2]:+6.1f}{'':>11s} | {d1[0]:+6.1f} {d1[1]:+6.1f} {d1[2]:+6.1f}{'':>12s} | "
                     f"{p0:5.0f} px / {p1:5.0f} px")
        picks[nm] = (R_rows[r[6]][1], T_rows[t[6]][1], rw[ri])
    lines.append("(image is 2704 x 2028 px; dz > 0 = robot tag further from the camera)")
    lines.append("")

    # arc: how much the robot's tags move toward/away from the camera between closed and open
    for name, tab in (("robot", R), ("TRumi", T)):
        w = np.array([r[1] for r in tab])
        zc = np.array([(r[2][2] + r[3][2]) / 2 for r in tab])
        yc = np.array([(r[2][1] + r[3][1]) / 2 for r in tab])
        closed, opened = w < np.percentile(w, 10), w > np.percentile(w, 90)
        lines.append(f"{name}: from closed to open, the tags move {(zc[opened].mean()-zc[closed].mean())*1000:+.1f} mm in depth "
                     f"and {(yc[opened].mean()-yc[closed].mean())*1000:+.1f} mm up/down (y)")
    report = "\n".join(lines)
    (a.out / "report.txt").write_text(report + "\n")
    print(report)

    # plot: tag depth and height vs opening
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, tab, col in (("robot", R, "m"), ("TRumi", T, "g")):
        w = np.array([r[1] for r in tab]) * 1000
        for tid, mk in ((0, "o"), (1, "^")):
            ax[0].scatter(w, [r[2 + tid][2] * 1000 for r in tab], s=6, c=col, marker=mk, label=f"{name} tag {tid}")
            ax[1].scatter(w, [r[2 + tid][1] * 1000 for r in tab], s=6, c=col, marker=mk, label=f"{name} tag {tid}")
    ax[0].axhspan(Z_OK[0] * 1000, Z_OK[1] * 1000, color="0.9", zorder=0, label="pipeline accepts")
    ax[0].set(xlabel="opening signal: tag x-distance (mm)", ylabel="tag depth in front of camera (mm)", title="depth (the Robotiq arc shows here)")
    ax[1].set(xlabel="opening signal: tag x-distance (mm)", ylabel="tag height below image centre line (mm)", title="up/down position")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(a.out / "tags_vs_opening.png", dpi=110)

    # images: robot | TRumi | blend | edges, then the same robot/TRumi pair with the two masks
    for nm in ("open", "half", "closed"):
        rimg, timg, w = picks[nm]
        blend = cv2.addWeighted(rimg, 0.5, timg, 0.5, 0)
        edges = (0.35 * cv2.cvtColor(cv2.cvtColor(timg, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)).astype(np.uint8)
        edges[cv2.Canny(cv2.GaussianBlur(timg, (5, 5), 0), 60, 140) > 0] = (0, 255, 0)
        edges[cv2.Canny(cv2.GaussianBlur(rimg, (5, 5), 0), 60, 140) > 0] = (255, 0, 255)
        top = np.hstack([label(rimg, "robot"), label(timg, "TRumi"), label(blend, "50/50 blend"), label(edges, "edges: green TRumi, magenta robot")])
        umi = dict(mirror=False, gripper=True, finger=False)
        fin = dict(mirror=False, gripper=True, finger=True)
        bot = np.hstack([label(masked(rimg, **umi), "robot, UMI mask"), label(masked(timg, **umi), "TRumi, UMI mask"),
                         label(masked(rimg, **fin), "robot, finger mask"), label(masked(timg, **fin), "TRumi, finger mask")])
        sheet = np.vstack([top, bot])
        sheet = cv2.resize(sheet, (sheet.shape[1] // 3, sheet.shape[0] // 3), interpolation=cv2.INTER_AREA)
        cv2.putText(sheet, f"{nm}: tag x-distance {w*1000:.0f} mm", (10, sheet.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imwrite(str(a.out / f"compare_{nm}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"\nsaved {a.out}/report.txt, tags_vs_opening.png, compare_open/half/closed.jpg")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compare robot vs TRumi wrist-camera views of the gripper.")
    ap.add_argument("--robot_video", type=pathlib.Path, required=True)
    ap.add_argument("--trumi_video", type=pathlib.Path, required=True)
    ap.add_argument("--stride", type=int, default=3, help="use every Nth frame")
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "data" / "robot" / "view_compare")
    main(ap.parse_args())

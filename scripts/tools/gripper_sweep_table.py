"""Turn a gripper sweep (scripts/robot/gripper_sweep.py log + the wrist-GoPro video recorded during it) into the two
tables the robot needs to reproduce TRumi gripper motion:

  1. width -> Robotiq command: for a TRumi dataset gripper width w (m, 0 = closed), the Robotiq position (0-255)
     whose finger-tag distance on the robot equals the TRumi's at w. The pipeline's width is the TRumi tag x-distance
     minus its fully-closed value (min_width in the TRumi session's gripper_range.json).
  2. arc: how far the Robotiq fingertips sit back toward the flange at each position, relative to fully closed
     (the TRumi's fingers run on linear rails, the Robotiq's on a curved linkage). Measured as the change of the
     finger tags' depth in front of the camera; the fingers keep their orientation, so every finger point moves alike.

Holds in the video are found as stretches where the tag distance is steady, then matched to the logged steps by a
single clock offset between the camera and this computer.

Usage (from ~/trumi):
    uv run python scripts/tools/gripper_sweep_table.py --video <sweep.MP4> --sweep_json data/robot/gripper_sweep_<time>.json
Output: <sweep_json stem>_tables.json and _tables.png next to the sweep log.
"""

import argparse
import json
import pathlib
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from compare_gripper_views import detect  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_RANGE = next((REPO / "data" / "2026-10-02_real_conveyor_setup" / "demos").glob("gripper_calibration_*/gripper_range.json"), None)


def steady_segments(t, s, win=0.5, tol=0.0004, min_len=1.0):
    """(start, end) times of stretches where the signal stays within +-tol around its running median for >= min_len s."""
    steady = np.zeros(len(t), bool)
    for i in range(len(t)):
        m = (t > t[i] - win / 2) & (t < t[i] + win / 2)
        steady[i] = m.sum() >= 3 and np.ptp(s[m]) < 2 * tol
    segs, start = [], None
    for i, st in enumerate(np.append(steady, False)):
        if st and start is None:
            start = i
        elif not st and start is not None:
            if t[i - 1] - t[start] >= min_len:
                segs.append((t[start], t[i - 1]))
            start = None
    return segs


def main(a):
    log = json.load(open(a.sweep_json))
    steps = log["steps"]
    rows = detect(a.video, a.stride, keep_images=False)
    T = np.array([r[0] for r in rows if 0 in r[2] and 1 in r[2]])
    D = np.array([r[2][1][0][0] - r[2][0][0][0] for r in rows if 0 in r[2] and 1 in r[2]])
    Z = np.array([(r[2][0][0][2] + r[2][1][0][2]) / 2 for r in rows if 0 in r[2] and 1 in r[2]])
    Y = np.array([(r[2][0][0][1] + r[2][1][0][1]) / 2 for r in rows if 0 in r[2] and 1 in r[2]])
    print(f"video: {len(rows)} frames sampled, both tags in {len(T)}, {T[-1] - T[0]:.1f} s")
    segs = steady_segments(T, D)
    print(f"steady stretches found in the video: {len(segs)}")

    # hold windows from the log (computer clock), then the camera-clock offset that puts them on steady stretches
    t0 = steps[0]["t_cmd"]
    holds = [(s["t_end"] - (log["hold_s"] if i else 3.0) - t0, s["t_end"] - t0) for i, s in enumerate(steps)]
    mid = np.array([(h0 + h1) / 2 for h0, h1 in holds])
    # pick the offset where a higher command always means a smaller tag distance (rank correlation -1), then the one
    # that puts the most holds on steady stretches; steady stretches alone are ambiguous (repeated open/closed holds)
    cmds = np.array([s["cmd"] for s in steps], float)
    rank = lambda v: np.argsort(np.argsort(v)).astype(float)
    best = None
    for off in np.arange(-20, 20, 0.02):
        med = np.array([np.median(D[(T >= h0 + off + 0.3) & (T <= h1 + off - 0.3)]) if ((T >= h0 + off + 0.3) & (T <= h1 + off - 0.3)).sum() >= 5
                        else np.nan for h0, h1 in holds])
        ok = ~np.isnan(med)
        if ok.sum() < 8:
            continue
        corr = np.corrcoef(rank(cmds[ok]), rank(med[ok]))[0, 1]
        inside = sum(any(s0 + 0.2 <= m + off <= s1 - 0.2 for s0, s1 in segs) for m in mid[ok])
        score = (round(-corr, 2), inside)
        if best is None or score > best[0]:
            best = (score, off, int(ok.sum()))
    (neg_corr, inside), off, n_in = best
    print(f"clock offset {off:+.2f} s: {n_in} of {len(steps)} logged holds fall inside the video, {inside} on steady stretches, "
          f"command vs tag distance rank correlation {-neg_corr:+.2f} (expect -1)")

    table = []
    for s, (h0, h1) in zip(steps, holds):
        m = (T >= h0 + off + 0.3) & (T <= h1 + off - 0.3)
        if m.sum() < 5:
            print(f"  step {s['step']:2d} (cmd {s['cmd']:3d}): not in the video, skipped")
            continue
        table.append({"step": s["step"], "cmd": s["cmd"], "pos": s["pos"], "tag_distance_mm": float(np.median(D[m]) * 1000),
                      "tag_depth_mm": float(np.median(Z[m]) * 1000), "tag_y_mm": float(np.median(Y[m]) * 1000),
                      "spread_mm": float(np.ptp(D[m]) * 1000), "n": int(m.sum())})
    for r in table:
        print(f"  step {r['step']:2d}: command {r['cmd']:3d} (reported {r['pos']:3d}) -> tag distance {r['tag_distance_mm']:6.1f} mm "
              f"(spread {r['spread_mm']:.1f}), tag depth {r['tag_depth_mm']:6.1f} mm, n={r['n']}")

    # average the closing and opening passes per reported position
    pos = sorted({r["pos"] for r in table})
    dist = np.array([np.mean([r["tag_distance_mm"] for r in table if r["pos"] == p]) for p in pos])
    depth = np.array([np.mean([r["tag_depth_mm"] for r in table if r["pos"] == p]) for p in pos])
    hyst = [abs(np.subtract(*[r["tag_distance_mm"] for r in table if r["pos"] == p])) for p in pos if sum(r["pos"] == p for r in table) == 2]
    print(f"closing vs opening pass at the same position: tag distance differs by up to {max(hyst) if hyst else float('nan'):.1f} mm")
    pos = np.array(pos, float)

    rng = json.load(open(a.trumi_range))
    tmin, tmax = rng["min_width"] * 1000, rng["max_width"] * 1000
    widths = np.linspace(0, tmax - tmin, 18)  # TRumi dataset width (mm)
    order = np.argsort(dist)
    cmd = np.interp(widths + tmin, dist[order], pos[order])  # clips to the robot's open/closed ends
    arc = depth[np.argmax(pos)] - depth  # mm the fingertips sit back toward the flange vs fully closed
    print(f"\nTRumi tag distance {tmin:.1f}-{tmax:.1f} mm; robot {dist.min():.1f} (pos {pos[np.argmin(dist)]:.0f}) - {dist.max():.1f} (pos {pos[np.argmax(dist)]:.0f}) mm")
    print("TRumi width (mm) -> Robotiq command:", ", ".join(f"{w:.0f}->{c:.0f}" for w, c in zip(widths[::3], cmd[::3])))
    print("Robotiq position -> fingertips set back from the closed position (mm):", ", ".join(f"{p:.0f}:{r:.1f}" for p, r in zip(pos, arc)))
    out = {"source": {"video": str(a.video), "sweep_json": str(a.sweep_json), "trumi_gripper_range": str(a.trumi_range),
                      "clock_offset_s": off},
           "width_to_cmd": {"trumi_width_m": (widths / 1000).round(5).tolist(), "robotiq_cmd": cmd.round(1).tolist(),
                            "note": "TRumi dataset gripper width (0 = closed) -> Robotiq position 0-255; clipped to the robot's range"},
           "arc": {"robotiq_pos": pos.tolist(), "fingertip_setback_mm": arc.round(2).tolist(),
                   "note": "fingertips this much closer to the flange than when fully closed; add it along the tool axis to keep them in place"},
           "steps": table}
    stem = pathlib.Path(a.sweep_json).with_suffix("")
    json.dump(out, open(f"{stem}_tables.json", "w"), indent=1)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(T, D * 1000, ".", ms=2, c="0.6")
    for r in table:
        h0, h1 = holds[r["step"]]
        ax[0].hlines(r["tag_distance_mm"], h0 + off, h1 + off, colors="m", lw=3)
    ax[0].set(xlabel="video time (s)", ylabel="tag distance (mm)", title="video signal and matched holds (magenta)")
    ax[1].plot(widths, cmd, "o-")
    ax[1].set(xlabel="TRumi dataset width (mm, 0 = closed)", ylabel="Robotiq command (0 open, 255 closed)", title="width -> command")
    ax[2].plot(pos, arc, "o-")
    ax[2].set(xlabel="Robotiq position", ylabel="fingertip set-back vs closed (mm)", title="arc (fingertips toward the flange)")
    fig.tight_layout()
    fig.savefig(f"{stem}_tables.png", dpi=110)
    print(f"\nsaved {stem}_tables.json and {stem}_tables.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Width->command and arc tables from a gripper sweep video + log.")
    ap.add_argument("--video", type=pathlib.Path, required=True)
    ap.add_argument("--sweep_json", type=pathlib.Path, required=True)
    ap.add_argument("--trumi_range", type=pathlib.Path, default=DEFAULT_RANGE, help="the TRumi session's gripper_range.json")
    ap.add_argument("--stride", type=int, default=2)
    main(ap.parse_args())

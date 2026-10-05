"""Review what retargeting did to each episode (the `retarget` stage of scripts/postprocess_session.py).

Per episode (<retarget folder>/review/ep<N>.png):
  1. fingertip x, y, z in the robot base frame: raw TRumi path (after the touch-test correction) vs the final path
     the robot follows (jump clean-up + smoothing)
  2. how far the final path is from the raw one, with grasp and release marked
  3. joint speeds of the final joint trajectory, against the UR5e's 180 deg/s
  4. gripper: TRumi width -> Robotiq command, and the arc set-back the IK compensates
Plus overview.png: per-episode largest/typical path change, time above 180 deg/s, peak joint speed.

Usage (from ~/trumi):
    uv run python scripts/tools/review_retarget.py --session data/<session> --retarget data/<session>/retarget_<robot>
"""

import argparse
import json
import pathlib
import pickle

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

JOINTS = ["base", "shoulder", "elbow", "wrist 1", "wrist 2", "wrist 3"]


def events(g):
    closed = np.flatnonzero(g > 128)
    if not len(closed):
        return None, None
    gi = closed[0]
    opened = np.flatnonzero(g[gi:] < 128)
    return gi, (gi + opened[0] if len(opened) else None)


def main(a):
    plan = pickle.load(open(a.session / "dataset_plan.pkl", "rb"))
    out = a.retarget / "review"
    out.mkdir(exist_ok=True)
    summary = []
    for k in range(1, len(plan) + 1):
        tr = json.load(open(a.retarget / f"ep{k}_right_robot_trajectory.json"))
        if "fingertip_raw_ctrl" not in tr:
            raise SystemExit(f"{a.retarget} was made before raw paths were stored: rerun the retarget stage")
        name = "_".join(pathlib.Path(plan[k - 1]["cameras"][0]["video_path"]).parent.name.split("_")[4:])
        t = np.array(tr["t_recorded_s"])
        raw, fin = np.array(tr["fingertip_raw_ctrl"]) * 1000, np.array(tr["fingertip_target_ctrl"]) * 1000
        q, g = np.array(tr["q_rad"]), np.array(tr["gripper_0open_255closed"])
        dev = np.linalg.norm(fin - raw, axis=1)
        sp = np.degrees(np.abs(np.diff(q, axis=0)) / np.diff(t)[:, None])
        gi, ri = events(g)
        summary.append({"ep": k, "video": name, "dev_max": dev.max(), "dev_med": np.median(dev), "over": tr["time_over_180_deg_s"],
                        "peak": sp.max(), "dev_grasp": dev[gi] if gi is not None else np.nan, "dev_release": dev[ri] if ri is not None else np.nan})

        fig, ax = plt.subplots(2, 2, figsize=(14, 8))
        for i, (c, lab) in enumerate(zip("rgb", "xyz")):
            ax[0, 0].plot(t, raw[:, i], c=c, lw=0.8, alpha=0.35)
            ax[0, 0].plot(t, fin[:, i], c=c, lw=1.6, label=f"{lab} (final)")
        ax[0, 0].set(title="fingertip in the robot base frame: faint = raw TRumi, solid = what the robot follows", xlabel="s", ylabel="mm")
        ax[0, 0].legend(fontsize=8)
        ax[0, 1].plot(t, dev, c="k")
        for idx, lab, col in ((gi, "grasp", "g"), (ri, "release", "m")):
            if idx is not None:
                ax[0, 1].axvline(t[idx], c=col, ls="--", label=f"{lab}: {dev[idx]:.1f} mm")
        ax[0, 1].set(title=f"change from raw (clean-up + smoothing): median {np.median(dev):.1f} mm, max {dev.max():.1f} mm", xlabel="s", ylabel="mm")
        ax[0, 1].legend(fontsize=8)
        for j in range(6):
            ax[1, 0].plot(t[1:], sp[:, j], lw=1, label=JOINTS[j])
        ax[1, 0].axhline(180, c="r", ls="--", label="UR5e limit 180 deg/s")
        ax[1, 0].set(title=f"joint speeds at recorded speed: {tr['time_over_180_deg_s']:.2f} s above 180 deg/s", xlabel="s", ylabel="deg/s")
        ax[1, 0].legend(fontsize=7, ncol=4)
        ax[1, 1].plot(t, np.array(tr["gripper_width_m"]) * 1000, c="tab:blue", label="TRumi width (mm)")
        ax[1, 1].plot(t, np.array(tr["fingertip_setback_mm"]), c="tab:orange", label="arc set-back compensated (mm)")
        ax2 = ax[1, 1].twinx()
        ax2.plot(t, g, c="k", lw=1, label="Robotiq command (0 open, 255 closed)")
        ax2.set_ylim(-5, 260)
        ax[1, 1].set(title="gripper", xlabel="s", ylabel="mm")
        ax[1, 1].legend(fontsize=8, loc="upper left")
        ax2.legend(fontsize=8, loc="upper right")
        fig.suptitle(f"episode {k} ({name})  |  smoothing {tr.get('smooth_s', 0)} s  |  pre-flight {'PASS' if tr['preflight_pass'] else 'check report'}")
        fig.tight_layout()
        fig.savefig(out / f"ep{k}.png", dpi=90)
        plt.close(fig)

    s = summary
    fig, ax = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    x = [r["ep"] for r in s]
    ax[0].bar(x, [r["dev_max"] for r in s], color="0.7", label="largest change (at tracking spikes)")
    ax[0].bar(x, [r["dev_med"] for r in s], color="k", label="typical (median) change")
    ax[0].scatter(x, [r["dev_grasp"] for r in s], c="g", zorder=3, label="change at the grasp")
    ax[0].set(ylabel="mm", title="path change from raw TRumi")
    ax[0].legend(fontsize=8)
    ax[1].bar(x, [r["over"] for r in s], color=["r" if r["over"] > a.max_over_s else "0.5" for r in s])
    ax[1].axhline(a.max_over_s, c="r", ls="--")
    ax[1].set(ylabel="s", title=f"time above 180 deg/s (red: more than {a.max_over_s} s)")
    ax[2].bar(x, [r["peak"] for r in s], color="0.5")
    ax[2].axhline(180, c="r", ls="--")
    ax[2].set(ylabel="deg/s", xlabel="episode", title="peak joint speed")
    fig.tight_layout()
    fig.savefig(out / "overview.png", dpi=90)
    print(f"{len(s)} episodes | typical path change {np.median([r['dev_med'] for r in s]):.1f} mm | change at grasp: median "
          f"{np.nanmedian([r['dev_grasp'] for r in s]):.1f}, max {np.nanmax([r['dev_grasp'] for r in s]):.1f} mm | at release: max "
          f"{np.nanmax([r['dev_release'] for r in s]):.1f} mm")
    print("above 180 deg/s for more than", a.max_over_s, "s:", [f"ep{r['ep']} {r['video']} ({r['over']:.2f} s, peak {r['peak']:.0f})" for r in s if r["over"] > a.max_over_s] or "none")
    print("largest path changes:", [f"ep{r['ep']} {r['dev_max']:.0f} mm" for r in sorted(s, key=lambda r: -r["dev_max"])[:5]])
    print(f"saved {out}/ep<N>.png and overview.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Plots of what retargeting changed, per episode.")
    ap.add_argument("--session", type=pathlib.Path, required=True)
    ap.add_argument("--retarget", type=pathlib.Path, required=True)
    ap.add_argument("--max_over_s", type=float, default=0.1)
    main(ap.parse_args())

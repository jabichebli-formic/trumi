"""Offline test of how smoothly consecutive policy chunks join, with and without real-time chunking (RTC).

Steps through recorded dataset episodes as if they were live: every --replan_s a new chunk is predicted from the
frame and state at that moment; the robot would still be executing the old chunk during inference (--delay steps).
Reports the jump in joint position and velocity where the new chunk takes over. No robot or camera needed.

Usage (from ~/trumi, LeRobot 0.6 environment):
    ~/YAM/yam-lerobot/.venv/bin/python scripts/tools/rtc_seam_test.py --checkpoint <.../pretrained_model> \
        --dataset data/lerobot/trumi_conveyor_pick_v1 --rtc_horizon 25
"""

import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "robot"))
from run_policy import FPS, Policy  # noqa: E402


def run(pol, ds, episodes, replan, delay, use_rtc):
    jumps, vjumps, times = [], [], []
    for e in episodes:
        m = ds.meta.episodes[e]
        start, length = int(m["dataset_from_index"]), int(m["length"])
        cur = None  # (start frame, absolute chunk, raw chunk)
        for k in range(0, length - 1, replan):
            s = ds[start + k]
            img = (s["observation.images.wrist"].numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
            prev = None
            if use_rtc and cur is not None:
                prev = cur[2][k - cur[0]:]
            t0 = time.time()
            ch, raw = pol.chunk(img, s["observation.state"].numpy(), prev_raw=prev, delay=delay, return_raw=True)
            times.append(time.time() - t0)
            if cur is not None:
                i_old = k - cur[0] + delay  # the new chunk takes over after the inference delay
                if i_old + 1 < len(cur[1]):
                    jumps.append(np.degrees(np.abs(ch[delay, :6] - cur[1][i_old, :6])).max())
                    vjumps.append(np.degrees(np.abs((ch[delay + 1, :6] - ch[delay, :6]) - (cur[1][i_old + 1, :6] - cur[1][i_old, :6]))).max() * FPS)
            cur = (k, ch, raw)
    return np.array(jumps), np.array(vjumps), np.array(times)


def main(a):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(a.repo_id, root=a.dataset)
    pol = Policy(a.checkpoint, a.device, rtc_horizon=a.rtc_horizon, seed=0)
    replan = int(round(a.replan_s * FPS))
    for use_rtc in (False, True):
        j, v, t = run(pol, ds, a.episodes, replan, a.delay, use_rtc)
        print(f"RTC {'on ' if use_rtc else 'off'} (horizon {a.rtc_horizon}, delay {a.delay} steps, new chunk every {replan} steps): "
              f"position jump at the seam median {np.median(j):.1f} deg, 90% {np.percentile(j, 90):.1f}, max {j.max():.1f} | "
              f"velocity jump median {np.median(v):.0f} deg/s, max {v.max():.0f} | inference median {np.median(t[1:]) * 1000:.0f} ms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Chunk seam jumps with and without RTC, on recorded episodes.")
    ap.add_argument("--checkpoint", type=pathlib.Path, required=True)
    ap.add_argument("--dataset", type=pathlib.Path, required=True)
    ap.add_argument("--repo_id", default="formic/trumi_conveyor_pick_v1")
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 10, 20])
    ap.add_argument("--replan_s", type=float, default=0.5)
    ap.add_argument("--delay", type=int, default=6, help="actions executed during one inference (~0.2 s at 30 fps)")
    ap.add_argument("--rtc_horizon", type=int, default=25)
    ap.add_argument("--device", default="cuda")
    main(ap.parse_args())

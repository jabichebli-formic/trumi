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
    jumps, vjumps, times, track = [], [], [], []
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
            # what would actually be executed from this chunk (until the next one takes over) vs what the human did
            n = min(replan + delay, length - k)
            truth = np.stack([ds[start + k + j]["action"].numpy() for j in range(delay, n)]) if n > delay else None
            if truth is not None:
                track.append(np.degrees(np.abs(ch[delay:n, :6] - truth[:, :6])).max(axis=1).mean())
            cur = (k, ch, raw)
    return np.array(jumps), np.array(vjumps), np.array(times), np.array(track)


def main(a):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(a.repo_id, root=a.dataset)
    replan = int(round(a.replan_s * FPS))
    for h in a.rtc_horizon:
        pol = Policy(a.checkpoint, a.device, rtc_horizon=h, seed=0)
        j, v, t, tr = run(pol, ds, a.episodes, replan, a.delay, h > 0)
        print(f"RTC {'horizon ' + str(h) if h else 'off       '} (delay {a.delay}, new chunk every {replan} steps): seam jump median {np.median(j):.2f} deg, "
              f"max {j.max():.1f} | velocity jump max {v.max():.0f} deg/s | executed actions vs human: {np.median(tr):.2f} deg median, "
              f"90% {np.percentile(tr, 90):.2f} | inference {np.median(t[1:]) * 1000:.0f} ms")
        del pol
        import torch

        torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Chunk seam jumps with and without RTC, on recorded episodes.")
    ap.add_argument("--checkpoint", type=pathlib.Path, required=True)
    ap.add_argument("--dataset", type=pathlib.Path, required=True)
    ap.add_argument("--repo_id", default="formic/trumi_conveyor_pick_v1")
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 10, 20])
    ap.add_argument("--replan_s", type=float, default=0.5)
    ap.add_argument("--delay", type=int, default=6, help="actions executed during one inference (~0.2 s at 30 fps)")
    ap.add_argument("--rtc_horizon", type=int, nargs="+", default=[0, 10, 25], help="0 = RTC off")
    ap.add_argument("--device", default="cuda")
    main(ap.parse_args())

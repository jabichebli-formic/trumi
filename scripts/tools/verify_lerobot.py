"""Verify an exported TRumi LeRobot dataset frame by frame, as the trainer will read it (LeRobot 0.6 environment).

Checks every frame of every episode: the image decodes and is not black (outside the policy mask) and not frozen
(identical to the previous frame), state/action are finite and in range, timestamps step by 1/fps, episode lengths
match the metadata. Then, for a few random frames per episode, it decodes the original GoPro frame, applies the same
mask and resize, and compares it with the dataset frame (PSNR): this proves each dataset frame is the right moment
of the right source video. Finally writes <root>_thumbnails.png (one frame per episode) next to the dataset folder
(not inside it, so it is not uploaded with the dataset).

Usage (from ~/trumi):
    ~/YAM/yam-lerobot/.venv/bin/python scripts/tools/verify_lerobot.py --root data/lerobot/<name>
"""

import argparse
import json
import pathlib
import pickle

import av
import cv2
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

SLAM_STRIDE = 2


def main(a):
    info = json.load(open(a.root / "trumi_export_info.json"))
    sources = [l.strip().split(",") for l in open(a.root / "trumi_sources.csv")][1:]
    ds = LeRobotDataset(a.repo_id, root=a.root)
    fps, n = ds.meta.fps, len(ds)
    mask = cv2.resize(cv2.imread(info["mask"], cv2.IMREAD_GRAYSCALE), tuple(info["image_size"]), interpolation=cv2.INTER_NEAREST) > 0
    print(f"{a.root.name}: {ds.meta.total_episodes} episodes, {n} frames, {fps} fps")

    loader = torch.utils.data.DataLoader(ds, batch_size=64, num_workers=a.workers, shuffle=False)
    per_ep = {}
    prev, prev_ep, problems = None, None, []
    for b in loader:
        imgs = b["observation.images.wrist"].numpy()  # (B, C, H, W) in [0, 1]
        for i in range(len(imgs)):
            ep, fi = int(b["episode_index"][i]), int(b["frame_index"][i])
            img = imgs[i].transpose(1, 2, 0)
            r = per_ep.setdefault(ep, {"frames": 0, "dark": 0, "frozen": 0, "bad_values": 0, "bad_time": 0, "brightness": []})
            r["frames"] += 1
            br = float(img[~mask].mean())
            r["brightness"].append(br)
            r["dark"] += br < 0.05
            if prev is not None and prev_ep == ep and np.abs(img - prev).mean() < 1e-4:
                r["frozen"] += 1
            prev, prev_ep = img, ep
            s, act = b["observation.state"][i].numpy(), b["action"][i].numpy()
            ok = np.isfinite(s).all() and np.isfinite(act).all() and (np.abs(s[:6]) < 2 * np.pi + 0.1).all() and -0.01 <= s[6] <= 1.01
            r["bad_values"] += not ok
            r["bad_time"] += abs(float(b["timestamp"][i]) - fi / fps) > 1e-3
    for ep, r in sorted(per_ep.items()):
        expected = int(ds.meta.episodes[ep]["length"])
        flags = [f"{k} {r[k]}" for k in ("dark", "frozen", "bad_values", "bad_time") if r[k]]
        if r["frames"] != expected:
            flags.append(f"{r['frames']} frames read vs {expected} in metadata")
        if flags:
            problems.append(f"episode {ep} ({sources[ep][2]}): " + ", ".join(flags))
    print(f"frames checked: {sum(r['frames'] for r in per_ep.values())} | brightness outside the mask: "
          f"min {min(min(r['brightness']) for r in per_ep.values()):.2f}, median {np.median(np.concatenate([r['brightness'] for r in per_ep.values()])):.2f}")
    print("problems:", "none" if not problems else "\n  " + "\n  ".join(problems))

    # spot-check against the original GoPro frames
    session = pathlib.Path(info["session"])
    plan = pickle.load(open(session / "dataset_plan.pkl", "rb"))
    step = round(60 / fps)
    rng = np.random.default_rng(0)
    psnr, thumbs = [], []
    for ep in range(ds.meta.total_episodes):
        pe = plan[int(sources[ep][1]) - 1]
        cam = pe["cameras"][0]
        start = int(ds.meta.episodes[ep]["dataset_from_index"])
        length = int(ds.meta.episodes[ep]["length"])
        ks = sorted(rng.choice(length, size=min(a.spot_checks, length), replace=False).tolist())
        want = {cam["video_start_end"][0] + SLAM_STRIDE * step * k: k for k in ks}
        with av.open(str(session / "demos" / cam["video_path"])) as c:
            for i, fr in enumerate(c.decode(video=0)):
                if i in want:
                    orig = fr.to_ndarray(format="rgb24")
                    orig[cv2.imread(info["mask"], cv2.IMREAD_GRAYSCALE) > 0] = 0
                    orig = cv2.resize(orig, tuple(info["image_size"]), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
                    got = ds[start + want[i]]["observation.images.wrist"].numpy().transpose(1, 2, 0)
                    psnr.append(10 * np.log10(1 / max(np.mean((orig - got) ** 2), 1e-10)))
                if i >= max(want):
                    break
        thumbs.append((ds[start + int(length * 0.4)]["observation.images.wrist"].numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
    psnr = np.array(psnr)
    print(f"source check ({len(psnr)} random frames vs the original GoPro videos): PSNR min {psnr.min():.1f} dB, median {np.median(psnr):.1f} dB "
          f"(> 30 dB = same frame; a wrong frame or video gives ~10-20 dB)")
    cols = 8
    h, w = 120, 160
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
    for k, im in enumerate(thumbs):
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
        cv2.putText(im, f"{k}: {sources[k][2].split('.')[0]}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        sheet[(k // cols) * h:(k // cols + 1) * h, (k % cols) * w:(k % cols + 1) * w] = im
    thumbs_path = a.root.parent / f"{a.root.name}_thumbnails.png"
    cv2.imwrite(str(thumbs_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f"saved {thumbs_path}")
    print("RESULT:", "OK" if not problems and psnr.min() > 30 else "CHECK THE PROBLEMS ABOVE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Frame-by-frame check of an exported TRumi LeRobot dataset.")
    ap.add_argument("--root", type=pathlib.Path, required=True)
    ap.add_argument("--repo_id", default="formic/trumi_conveyor_pick_v1")
    ap.add_argument("--spot_checks", type=int, default=3, help="random frames per episode compared with the GoPro video")
    ap.add_argument("--workers", type=int, default=8)
    main(ap.parse_args())

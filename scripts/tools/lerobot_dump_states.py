"""Dump an exported LeRobot dataset's per-episode state/action arrays and video locations, so the digital twin
(scripts/sim/replay_dataset.py, sim environment without LeRobot) can replay exactly what the policy trains on.

Usage (from ~/trumi, LeRobot 0.6 environment):
    ~/YAM/yam-lerobot/.venv/bin/python scripts/tools/lerobot_dump_states.py --root data/lerobot/<name>
Output: data/lerobot/<name>_review/ (episodes.json + ep<E>_state.npy / ep<E>_action.npy)
"""

import argparse
import csv
import json
import pathlib

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

KEY = "observation.images.wrist"


def main(a):
    ds = LeRobotDataset(a.repo_id, root=a.root)
    out = a.root.parent / f"{a.root.name}_review"
    out.mkdir(exist_ok=True)
    sources = {int(r["episode_index"]): r["source_video"] for r in csv.DictReader(open(a.root / "trumi_sources.csv"))}
    cols = ds.hf_dataset.with_format("numpy")
    ep_idx = np.asarray(cols["episode_index"])
    state, action = np.asarray(cols["observation.state"]), np.asarray(cols["action"])
    episodes = []
    for e in range(ds.meta.total_episodes):
        m = ds.meta.episodes[e]
        sel = np.flatnonzero(ep_idx == e)
        np.save(out / f"ep{e}_state.npy", state[sel])
        np.save(out / f"ep{e}_action.npy", action[sel])
        video = a.root / ds.meta.video_path.format(video_key=KEY, chunk_index=m[f"videos/{KEY}/chunk_index"],
                                                  file_index=m[f"videos/{KEY}/file_index"])
        episodes.append({"episode": e, "source_video": sources.get(e, ""), "frames": int(len(sel)), "fps": ds.meta.fps,
                         "video": str(video), "from_timestamp": float(m[f"videos/{KEY}/from_timestamp"]),
                         "to_timestamp": float(m[f"videos/{KEY}/to_timestamp"])})
    json.dump({"root": str(a.root), "names": ds.meta.features["observation.state"]["names"], "episodes": episodes},
              open(out / "episodes.json", "w"), indent=1)
    print(f"dumped {len(episodes)} episodes to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Dump state/action and video locations of a LeRobot dataset for twin replay.")
    ap.add_argument("--root", type=pathlib.Path, required=True)
    ap.add_argument("--repo_id", default="formic/trumi_conveyor_pick_v1")
    main(ap.parse_args())

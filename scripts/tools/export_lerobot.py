"""Export a processed TRumi session as a LeRobot v3.0 dataset in the target robot's joint space (e.g. for pi0.5).

Per frame (default 30 fps = every 2nd 60 Hz SLAM step):
  observation.images.wrist  the TRumi GoPro frame with the policy mask applied (fingers + gripper body blacked out, the
                            same mask must be applied on the robot), resized to --width x --height
  observation.state         robot joints (rad, 6) + gripper (Robotiq command / 255: 0 open, 1 closed) at this frame
  action                    the same at the next frame (absolute targets)
  task                      --task
The joints come from scripts/sim/preflight.py robot trajectories (IK for the real robot, calibrated to the marker,
fingertip arc compensated); run it for every episode first with --gripper_tables and --reference_q_deg.

Runs with LeRobot 0.6 (e.g. the YAM environment; not the TRumi or sim environments):
    ~/YAM/yam-lerobot/.venv/bin/python scripts/tools/export_lerobot.py --session data/<session> \
        --trajectories data/<session>/preflight_printed_tips --out data/lerobot/<name>
"""

import argparse
import csv
import json
import pathlib
import pickle
import shutil

import av
import cv2
import numpy as np
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO = pathlib.Path(__file__).resolve().parents[2]
SLAM_STRIDE = 2  # SLAM steps are every 2nd video frame (120 fps video -> 60 Hz)
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]


def frames_at(video, indices):
    """Decode the given frame indices (sorted) from a video, as RGB arrays."""
    want, out = set(indices), {}
    with av.open(str(video)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        for i, fr in enumerate(c.decode(s)):
            if i in want:
                out[i] = fr.to_ndarray(format="rgb24")
            if i >= indices[-1]:
                break
    return [out[i] for i in indices]


def main(a):
    plan = pickle.load(open(a.session / "dataset_plan.pkl", "rb"))
    mask = cv2.imread(str(a.mask), cv2.IMREAD_GRAYSCALE) > 0
    step = round(60 / a.fps)
    if a.out.exists():
        if not a.overwrite:
            raise SystemExit(f"{a.out} exists (use --overwrite)")
        shutil.rmtree(a.out)
    names = [f"{j}.pos" for j in JOINTS] + ["gripper.pos"]
    features = {
        "observation.images.wrist": {"dtype": "video", "shape": (a.height, a.width, 3), "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (7,), "names": names},
        "action": {"dtype": "float32", "shape": (7,), "names": names},
    }
    ds = LeRobotDataset.create(repo_id=a.repo_id, fps=a.fps, features=features, root=a.out, robot_type=a.robot_type,
                               use_videos=True, rgb_encoder=RGBEncoderConfig(vcodec="h264", pix_fmt="yuv420p", g=2, crf=23))
    sources = []
    episodes = range(len(plan)) if a.limit is None else range(min(a.limit, len(plan)))
    for e in episodes:
        ep = plan[e]
        tr = json.load(open(a.trajectories / f"ep{e + 1}_right_robot_trajectory.json"))
        q = np.asarray(tr["q_rad"], np.float32)
        g = np.asarray(tr["gripper_0open_255closed"], np.float32) / 255.0
        n = len(q)
        if n != len(ep["episode_timestamps"]):
            raise SystemExit(f"episode {e + 1}: trajectory has {n} steps, dataset plan {len(ep['episode_timestamps'])}")
        steps = list(range(0, n, step))
        cam = ep["cameras"][0]
        video = a.session / "demos" / cam["video_path"]
        imgs = frames_at(video, [cam["video_start_end"][0] + SLAM_STRIDE * s for s in steps])
        state = np.concatenate([q, g[:, None]], 1)
        for k, (s, img) in enumerate(zip(steps, imgs)):
            img = img.copy()
            img[mask] = 0
            img = cv2.resize(img, (a.width, a.height), interpolation=cv2.INTER_AREA)
            nxt = steps[k + 1] if k + 1 < len(steps) else s
            ds.add_frame({"observation.images.wrist": img, "observation.state": state[s], "action": state[nxt], "task": a.task})
        ds.save_episode()
        src = pathlib.Path(cam["video_path"]).parent.name
        sources.append({"episode_index": len(sources), "plan_episode": e + 1, "source_video": "_".join(src.split("_")[4:]),
                        "frames": len(steps), "preflight_pass": tr.get("preflight_pass")})
        print(f"episode {len(sources) - 1}: {sources[-1]['source_video']}, {len(steps)} frames")
    ds.finalize()
    with open(a.out / "trumi_sources.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sources[0]))
        w.writeheader()
        w.writerows(sources)
    json.dump({"session": str(a.session), "trajectories": str(a.trajectories), "mask": str(a.mask), "fps": a.fps,
               "image_size": [a.width, a.height], "task": a.task,
               "state_action": "UR joints in rad (6) + gripper = Robotiq command / 255 (0 open, 1 closed); action = next frame",
               "notes": "apply the same mask (resized from 2704x2028) to the robot's wrist GoPro before the policy"},
              open(a.out / "trumi_export_info.json", "w"), indent=1)
    print(f"saved {len(sources)} episodes to {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Export a TRumi session as a LeRobot v3.0 dataset in robot joint space.")
    ap.add_argument("--session", type=pathlib.Path, required=True)
    ap.add_argument("--trajectories", type=pathlib.Path, required=True, help="folder with preflight ep<N>_right_robot_trajectory.json")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--repo_id", default="formic/trumi_conveyor_pick_v1")
    ap.add_argument("--task", default="pick the cup from the conveyor and place it in the box")
    ap.add_argument("--robot_type", default="ur5e_robotiq_2f85")
    ap.add_argument("--mask", type=pathlib.Path, default=REPO / "data" / "robot" / "policy_mask_2704x2028.png")
    ap.add_argument("--fps", type=int, default=30, choices=[15, 20, 30, 60])
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--limit", type=int, default=None, help="only the first N episodes (for a test)")
    ap.add_argument("--overwrite", action="store_true")
    main(ap.parse_args())

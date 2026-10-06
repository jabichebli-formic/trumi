"""Does the policy use the camera? Swap images and states between recorded frames and see what changes the plan.

For pairs of frames (i, j) from different episodes, all taken shortly before the grasp:
  image change : plan(image_j, state_i) vs plan(image_i, state_i)
  state change : plan(image_i, state_j) vs plan(image_i, state_i)
measured as where the fingertip would be at the end of the 1.7 s chunk (UR kinematics + the robot calibration, cm).
A policy that steers by what it sees moves its target when the image changes; one that replays an average motion
barely does. Noise is fixed (same seed) so the differences come from the inputs only.

Usage (from ~/trumi, LeRobot 0.6 environment):
    ~/YAM/yam-lerobot/.venv/bin/python scripts/tools/vision_sensitivity.py --checkpoint <.../pretrained_model> \
        --dataset data/lerobot/trumi_conveyor_pick_v1
"""

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "robot"))
from run_policy import DH_A, DH_ALPHA, DH_D, REPO, Policy  # noqa: E402


def tip_cm(q, Ti, tcp=0.2572):
    M = np.eye(4)
    for i in range(6):
        ct, st, ca, sa = np.cos(q[i]), np.sin(q[i]), np.cos(DH_ALPHA[i]), np.sin(DH_ALPHA[i])
        M = M @ np.array([[ct, -st * ca, st * sa, DH_A[i] * ct], [st, ct * ca, -ct * sa, DH_A[i] * st], [0, sa, ca, DH_D[i]], [0, 0, 0, 1]])
    return (Ti @ np.r_[M[:3, 3] + M[:3, 2] * tcp, 1])[:3] * 100


def main(a):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(a.repo_id, root=a.dataset)
    Ti = np.linalg.inv(np.array(json.load(open(a.calibration))["T_base_marker"]))
    frames = []  # (image, state, true fingertip at the grasp) about a.lead_s before each episode's grasp
    for e in range(0, ds.meta.total_episodes, max(1, ds.meta.total_episodes // a.n)):
        m = ds.meta.episodes[e]
        start, length = int(m["dataset_from_index"]), int(m["length"])
        g = np.array([ds[start + k]["observation.state"][6].item() for k in range(length)])
        closed = np.flatnonzero(g > 0.5)
        if not len(closed):
            continue
        k = max(0, closed[0] - int(a.lead_s * 30))
        s = ds[start + k]
        img = (s["observation.images.wrist"].numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
        frames.append((img, s["observation.state"].numpy(), tip_cm(ds[start + closed[0]]["observation.state"].numpy()[:6], Ti)))
    pol = Policy(a.checkpoint, a.device, seed=0)
    end = lambda img, st: tip_cm(pol.chunk(img, st)[-1, :6], Ti)
    base = [end(f[0], f[1]) for f in frames]
    d_img, d_state, err_true = [], [], []
    n = len(frames)
    for i in range(n):
        j = (i + n // 2) % n  # a frame from a different episode
        d_img.append(np.linalg.norm(end(frames[j][0], frames[i][1]) - base[i]))
        d_state.append(np.linalg.norm(end(frames[i][0], frames[j][1]) - base[i]))
        err_true.append(np.linalg.norm(base[i] - frames[i][2]))
    spread = np.linalg.norm(np.array([f[2] for f in frames]) - np.mean([f[2] for f in frames], 0), axis=1)
    print(f"{a.checkpoint.parent.parent.parent.name}/{a.checkpoint.parent.name}: {n} frames, {a.lead_s} s before the grasp")
    print(f"   real grasp spots differ from their average by {np.median(spread):.1f} cm (median)")
    print(f"   planned target (end of the 1.7 s chunk) vs the real grasp: {np.median(err_true):.1f} cm (median)")
    print(f"   target moves when ONLY THE IMAGE is swapped: {np.median(d_img):.1f} cm median | when ONLY THE STATE is swapped: {np.median(d_state):.1f} cm")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Image vs state sensitivity of a policy's plan.")
    ap.add_argument("--checkpoint", type=pathlib.Path, required=True)
    ap.add_argument("--dataset", type=pathlib.Path, required=True)
    ap.add_argument("--repo_id", default="formic/trumi_conveyor_pick_v1")
    ap.add_argument("--n", type=int, default=19, help="number of episodes to sample")
    ap.add_argument("--lead_s", type=float, default=1.0)
    ap.add_argument("--calibration", type=pathlib.Path, default=REPO / "data" / "robot" / "marker_in_robot_base.json")
    ap.add_argument("--device", default="cuda")
    main(ap.parse_args())

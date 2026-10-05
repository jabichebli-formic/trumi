"""Step the Robotiq gripper from open to closed and back while the wrist GoPro records, logging what the gripper
reports at every step. Only the gripper moves; the arm is not commanded.

Each step: command a position, wait until the fingers stop, hold still (so the video shows a clear plateau), log the
commanded and actual position. Matching the video's plateaus to these steps (in order) gives the TRumi-width ->
Robotiq-command table and the fingertip arc for the fingers that are mounted.

Usage (from ~/trumi):
    data/sim/.venv/bin/python scripts/robot/gripper_sweep.py --robot_ip 192.168.10.205            # status + plan only
    data/sim/.venv/bin/python scripts/robot/gripper_sweep.py --robot_ip 192.168.10.205 --execute  # moves the gripper
Needs the gripper activated from the pendant. Output: data/robot/gripper_sweep_<time>.json
"""

import argparse
import datetime
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from replay_on_robot import RobotiqSocket  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]


def wait_until_stopped(g, timeout=6.0):
    """OBJ: 0 = moving, 1/2 = stopped on an object, 3 = reached the requested position."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        obj = g.get("OBJ")
        if obj != 0:
            return obj
        time.sleep(0.05)
    return None


def main(a):
    targets = np.round(np.linspace(0, 255, a.steps)).astype(int).tolist()
    plan = [0] + targets[1:] + targets[::-1][1:]  # open, close step by step, open step by step
    print(f"plan: {len(plan)} positions (0 = open, 255 = closed): {plan}")
    print(f"speed {a.speed}/255, force {a.force}/255, hold {a.hold_s} s per step -> about {len(plan) * (a.hold_s + 0.6) + a.start_hold_s:.0f} s")
    g = RobotiqSocket(a.robot_ip, speed=a.speed, force=a.force)
    status = {v: g.get(v) for v in ("ACT", "STA", "GTO", "FLT", "POS", "OBJ")}
    print("gripper status:", status)
    if status["FLT"] != 0:
        raise SystemExit("gripper reports a fault; clear it on the pendant first")
    if not a.execute:
        print("status only (add --execute to move the gripper)")
        return
    log = []
    for i, p in enumerate(plan):
        t_cmd = time.time()
        g.set(p)
        obj = wait_until_stopped(g)
        time.sleep(a.start_hold_s if i == 0 else a.hold_s)
        row = {"step": i, "cmd": p, "pos": g.get("POS"), "obj": obj, "flt": g.get("FLT"), "t_cmd": t_cmd, "t_end": time.time()}
        log.append(row)
        print(f"  step {i:2d}: command {p:3d} -> gripper reports {row['pos']:3d} (OBJ {obj}, FLT {row['flt']})")
        if row["flt"]:
            print("  gripper fault: stopping")
            break
    g.set(0)  # leave it open
    g.close()
    out = REPO / "data" / "robot" / f"gripper_sweep_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
    json.dump({"robot_ip": a.robot_ip, "speed": a.speed, "force": a.force, "hold_s": a.hold_s, "steps": log}, open(out, "w"), indent=1)
    print(f"saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Step the gripper open->closed->open with holds, logging its position.")
    ap.add_argument("--robot_ip", required=True)
    ap.add_argument("--steps", type=int, default=10, help="positions from open to closed")
    ap.add_argument("--hold_s", type=float, default=2.0)
    ap.add_argument("--start_hold_s", type=float, default=3.0)
    ap.add_argument("--speed", type=int, default=64, help="0-255 (slow = 64)")
    ap.add_argument("--force", type=int, default=30, help="0-255 (gentle = 30)")
    ap.add_argument("--execute", action="store_true", help="actually move the gripper")
    main(ap.parse_args())

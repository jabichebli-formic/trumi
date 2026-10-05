"""Read-only check that the robot calibration and the pendant TCP still hold: shows, live, where the robot's TCP
(fingertip) is in the marker's frame. Touch the marker centre with the closed fingertips in freedrive: the reading
should be ~(0, 0, 0) mm within ~10 mm (corners: +-80 mm in x and y). A large offset means the robot base or the
marker moved since the calibration, or the pendant TCP does not match the fingertips.

Usage (from ~/trumi):
    data/sim/.venv/bin/python scripts/robot/check_marker_touch.py --robot_ip 192.168.10.204
Press Enter to record a reading, q + Enter to stop.
"""

import argparse
import json
import pathlib
import select
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]


def main(a):
    import rtde_receive

    T = np.array(json.load(open(a.calibration))["T_base_marker"])
    Tinv = np.linalg.inv(T)
    rr = rtde_receive.RTDEReceiveInterface(a.robot_ip)
    print("touch the marker centre with the closed fingertips; Enter = record, q + Enter = stop")
    readings = []
    while True:
        p = Tinv @ np.r_[rr.getActualTCPPose()[:3], 1]
        sys.stdout.write(f"\r  fingertip in the marker frame: x {p[0]*1000:+7.1f}  y {p[1]*1000:+7.1f}  z {p[2]*1000:+7.1f} mm   ")
        sys.stdout.flush()
        if select.select([sys.stdin], [], [], 0.2)[0]:
            line = sys.stdin.readline().strip().lower()
            if line == "q":
                break
            readings.append(p[:3] * 1000)
            print(f"\n  recorded: {np.round(p[:3] * 1000, 1).tolist()} mm (distance from the centre {np.linalg.norm(p[:3]) * 1000:.1f} mm)")
    rr.disconnect()
    if readings:
        r = np.array(readings)
        print(f"\n{len(r)} readings; mean {np.round(r.mean(0), 1).tolist()} mm")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Live fingertip position in the marker frame (read-only).")
    ap.add_argument("--robot_ip", required=True)
    ap.add_argument("--calibration", type=pathlib.Path, default=REPO / "data" / "robot" / "marker_in_robot_base.json")
    main(ap.parse_args())

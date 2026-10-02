"""Measure where the mapping marker is in the UR5e's base frame by touching 5 marker points with the gripper.

The 5 points are on the marker's black square (16 cm), in the marker's own axes (see assets/marker13_axes.png):
    centre (0, 0)   TL (-8, +8) cm   TR (+8, +8) cm   BR (+8, -8) cm   BL (-8, -8) cm
For each point, hand-guide the robot (freedrive) until the closed fingertips touch the point, then press Enter.

Modes:
    --robot_ip IP   read the TCP position and joint angles from the robot over RTDE. Read-only: this script never
                    sends motion commands. Needs the ur_rtde package (pip install ur_rtde).
    --manual        type the TCP X Y Z (mm) shown on the teach pendant (Move tab, Feature = Base) for each point.
    --self_test     check the maths on simulated touches with a known answer (no robot needed).

Before starting: on the pendant, set the TCP to the gripper fingertips (Installation > TCP) and note its offset.

Output (default data/robot/marker_in_robot_base.json): the 4x4 transform of the marker in the UR controller's
base frame (metres) plus quality checks: per-point fit error (expect ~1-2 mm), the 4 measured side lengths
(expect 160 mm), and the marker's tilt relative to the robot base.

Usage (from ~/trumi):
    data/sim/.venv/bin/python scripts/robot/calibrate_marker.py --self_test
    data/sim/.venv/bin/python scripts/robot/calibrate_marker.py --robot_ip 192.168.1.10
"""

import argparse
import datetime
import json
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
MARKER_SIZE = 0.16  # black square side (m)
H = MARKER_SIZE / 2
POINTS = {  # marker-frame coordinates (m); corner order matches the pipeline (src/trumi/utils/cv_util.py)
    "centre": (0.0, 0.0, 0.0),
    "TL (top-left)": (-H, H, 0.0),
    "TR (top-right)": (H, H, 0.0),
    "BR (bottom-right)": (H, -H, 0.0),
    "BL (bottom-left)": (-H, -H, 0.0),
}
FIT_WARN_MM = 3.0


def fit_rigid(A, B):
    """Least-squares rotation R and translation t with R @ A_i + t ~= B_i (Kabsch). A, B: (N, 3)."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    Rm = Vt.T @ D @ U.T
    return Rm, cb - Rm @ ca


def evaluate(touches):
    """Fit the marker pose from a list of (point name, xyz_base in m) touches and compute quality checks."""
    names = [n for n, _ in touches]
    A = np.array([POINTS[n] for n in names])
    B = np.array([p for _, p in touches])
    Rm, t = fit_rigid(A, B)
    resid = np.linalg.norm((A @ Rm.T + t) - B, axis=1) * 1000
    mean_of = {n: B[[i for i, m in enumerate(names) if m == n]].mean(0) for n in POINTS}
    corners = [mean_of[n] for n in ("TL (top-left)", "TR (top-right)", "BR (bottom-right)", "BL (bottom-left)")]
    sides = [float(np.linalg.norm(corners[i] - corners[(i + 1) % 4]) * 1000) for i in range(4)]
    per_point = {n: round(float(np.sqrt(np.mean(resid[[i for i, m in enumerate(names) if m == n]] ** 2))), 2) for n in POINTS}
    tilt = np.degrees(np.arccos(np.clip(abs(Rm[:, 2] @ [0, 0, 1]), -1, 1)))
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = Rm, t
    return {
        "T_base_marker": T.tolist(),
        "fit_error_mm": per_point,
        "n_touches": len(touches),
        "fit_rms_mm": float(np.sqrt(np.mean(resid ** 2))),
        "side_lengths_mm": [round(s, 1) for s in sides],
        "marker_tilt_deg": round(float(tilt), 2),
        "marker_yaw_in_base_deg": round(float(np.degrees(np.arctan2(Rm[1, 0], Rm[0, 0]))), 2),
    }


def report(res):
    ok = res["fit_rms_mm"] < FIT_WARN_MM and all(abs(s - MARKER_SIZE * 1000) < 4 for s in res["side_lengths_mm"])
    print(f"\nper-point fit error (mm, {res['n_touches']} touches):", res["fit_error_mm"])
    print(f"RMS fit error: {res['fit_rms_mm']:.2f} mm   (expect ~1-2 mm)")
    print(f"measured side lengths: {res['side_lengths_mm']} mm   (expect {MARKER_SIZE*1000:.0f})")
    print(f"marker tilt vs robot base: {res['marker_tilt_deg']:.2f} deg | marker +x direction in robot base: {res['marker_yaw_in_base_deg']:.1f} deg")
    t = np.array(res["T_base_marker"])[:3, 3]
    print(f"marker centre in robot base: x={t[0]*1000:.1f} y={t[1]*1000:.1f} z={t[2]*1000:.1f} mm")
    print("RESULT:", "GOOD" if ok else "CHECK: a touch may be off, a corner mislabelled, or the TCP is not at the fingertip -> redo")
    return ok


def collect_manual(rounds):
    touches = []
    print("For each point, touch it with the closed fingertips and type the TCP position from the pendant (mm), e.g. 412.3 -88.1 55.0")
    for r in range(rounds):
        print(f" round {r + 1} of {rounds} (lift the gripper off between touches)")
        for n in POINTS:
            while True:
                try:
                    x, y, z = (float(v) for v in input(f"  {n:18s} X Y Z (mm): ").replace(",", " ").split())
                    touches.append((n, np.array([x, y, z]) / 1000))
                    break
                except ValueError:
                    print("    please type three numbers")
    return touches, []


def collect_robot(ip, rounds):
    import rtde_receive  # from the ur_rtde package (read-only receive interface)

    rr = rtde_receive.RTDEReceiveInterface(ip)
    print(f"connected to {ip}. Use freedrive (pendant button) to touch each point with the closed fingertips, then press Enter.")
    touches, joints = [], []
    for r in range(rounds):
        print(f" round {r + 1} of {rounds} (lift the gripper off between touches)")
        for n in POINTS:
            input(f"  touch {n:18s} then press Enter ")
            samples = [rr.getActualTCPPose()[:3] for _ in range(25)]  # average 25 readings (robot held still)
            touches.append((n, np.mean(samples, 0)))
            joints.append((n, list(rr.getActualQ())))
            print(f"    TCP = {np.round(touches[-1][1] * 1000, 1)} mm (spread {np.ptp(samples, 0).max()*1000:.2f} mm)")
    rr.disconnect()
    return touches, joints


def self_test():
    rng = np.random.default_rng(0)
    from scipy.spatial.transform import Rotation as Rot

    T = Rot.from_euler("zyx", [rng.uniform(-180, 180), 0.6, -0.4], degrees=True).as_matrix()
    t = np.array([0.45, -0.12, -0.11])
    for rounds in (1, 2, 3):
        errs = []
        for trial in range(200):
            touches = [(n, T @ np.array(p) + t + rng.normal(0, 0.001, 3)) for _ in range(rounds) for n, p in POINTS.items()]
            Te = np.array(evaluate(touches)["T_base_marker"])
            yaw_err = np.degrees(np.arctan2(*(Te[:3, :3] @ T.T)[[1, 0], 0]))
            errs.append((np.linalg.norm(Te[:3, 3] - t) * 1000, abs(yaw_err)))
        e = np.array(errs)
        print(f"self-test, 1 mm touch noise, {rounds} round(s): position error {np.median(e[:,0]):.2f} mm, yaw error {np.median(e[:,1]):.2f} deg"
              f" -> ~{np.radians(np.median(e[:,1])) * 900:.1f} mm at 0.9 m reach (median of 200 trials)")
    touches = [(n, T @ np.array(p) + t + rng.normal(0, 0.001, 3)) for _ in range(2) for n, p in POINTS.items()]
    good = report(evaluate(touches))
    swapped = [({"TL (top-left)": "TR (top-right)", "TR (top-right)": "TL (top-left)"}.get(n, n), p) for n, p in touches]
    print("\nself-test with two corners swapped (should be flagged):")
    flagged = not report(evaluate(swapped))
    print("\nSELF-TEST", "PASSED" if good and flagged else "FAILED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Measure the marker pose in the UR5e base frame by touching 5 points.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--robot_ip")
    g.add_argument("--manual", action="store_true")
    g.add_argument("--self_test", action="store_true")
    ap.add_argument("--rounds", type=int, default=2, help="touch every point this many times (more = more accurate)")
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "data" / "robot" / "marker_in_robot_base.json")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        raise SystemExit
    touches, joints = collect_robot(a.robot_ip, a.rounds) if a.robot_ip else collect_manual(a.rounds)
    res = evaluate(touches)
    report(res)
    res.update({"frame": "UR controller base frame (as reported by the robot/pendant), metres",
                "measured_tcp_m": [[n, np.round(p, 5).tolist()] for n, p in touches],
                "joints_rad": [[n, q] for n, q in joints],
                "marker_size_m": MARKER_SIZE, "date": datetime.datetime.now().isoformat(timespec="seconds")})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"saved {a.out}")

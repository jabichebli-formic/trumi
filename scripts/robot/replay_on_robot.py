"""Replay a pre-flighted TRumi trajectory on the real UR5e (joint space), slowly and with confirmations.

Input: <session>/preflight/ep<N>_<arm>[_air..mm]_robot_trajectory.json from scripts/sim/preflight.py. The joint
angles in it were checked in the digital twin (reach, limits, collisions) and reproduce the planned fingertip poses
with UR's own kinematics. They are already slowed down to the pre-flight speed cap; --extra_slowdown slows further.

Safety:
  * Dry run by default: nothing moves without --execute. A dry run exercises the whole loop on a simulated robot.
  * Refuses trajectories that did not pass pre-flight (unless --force).
  * Asks you to type "yes" before moving to the start pose (slow moveJ) and again before running the trajectory
    (--yes skips the prompts when the operator has confirmed some other way, e.g. in chat right before the run).
  * The move to the start pose is a straight line in joint space; it is checked for collisions in the digital twin
    (tables, belt, rails, box, self) and refused if anything is hit. --move_to_start_only stops after that move.
  * Ctrl+C or a protective stop stops the robot (servoStop / stopScript). Keep the e-stop in reach regardless.
  * First run on hardware: use an "in the air" trajectory (preflight.py --z_offset_mm 50) with --no_gripper.

After the run it logs the robot's actual joints and fingertip (TCP) and compares them with the plan; if the pendant's
TCP matches the trajectory's tcp_z_mm, the fingertip should follow the plan within ~1-2 mm.

Needs the ur_rtde package (pip install ur_rtde) for --execute. Gripper: Robotiq URCap socket on port 63352
(not yet tested on this robot; use --no_gripper if it misbehaves).

Usage (from ~/trumi):
    data/sim/.venv/bin/python scripts/robot/replay_on_robot.py --trajectory <file.json>                    # dry run
    data/sim/.venv/bin/python scripts/robot/replay_on_robot.py --trajectory <file.json> --robot_ip 192.168.1.10 \
        --execute --no_gripper --extra_slowdown 2
"""

import argparse
import json
import pathlib
import socket
import time

import numpy as np

SERVO_HZ = 125
MOVE_TO_START_SPEED = 0.3  # rad/s
MOVE_TO_START_ACC = 0.5  # rad/s^2
MAX_JOINT_SPEED_DEG_S = 90  # hard cap for --execute (UR5e max is 180 deg/s)
MAX_JOINT_SPEED_RECORDED_DEG_S = 150  # cap with --recorded_speed (the demo's own timing)


class RobotiqSocket:
    """Minimal client for the Robotiq URCap gripper socket (port 63352)."""

    def __init__(self, ip, speed=255, force=100):
        self.s = socket.create_connection((ip, 63352), timeout=2.0)
        # activating from here would make the gripper open and close fully; do it from the pendant instead
        if self.get("ACT") != 1 or self.get("STA") != 3:
            raise RuntimeError("gripper is not activated: activate it from the pendant (Robotiq toolbar) first")
        for cmd in (f"SET SPE {int(speed)}", f"SET FOR {int(force)}", "SET GTO 1"):
            self._send(cmd)

    def _send(self, cmd):
        self.s.sendall((cmd + "\n").encode())
        return self.s.recv(1024).decode().strip()

    def get(self, var):
        """Read one gripper register (e.g. POS, OBJ, FLT); replies look like 'POS 3'."""
        return int(self._send(f"GET {var}").split()[-1])

    def set(self, pos_0_255):
        self._send(f"SET POS {int(np.clip(pos_0_255, 0, 255))}")

    def close(self):
        self.s.close()


class FakeRobot:
    """Stand-in for a dry run: follows commands perfectly, no network."""

    def __init__(self, q0):
        self.q = np.array(q0, float)

    def moveJ(self, q, speed, acc):
        print(f"   [dry run] moveJ to start at {speed} rad/s")
        self.q = np.array(q, float)

    def initPeriod(self):
        return time.time()  # keeps real time like the robot does, so a dry run takes as long as the real run

    def servoJ(self, q, v, a, dt, lookahead, gain):
        self.q = np.array(q, float)

    def waitPeriod(self, t0):
        time.sleep(max(0.0, t0 + 1.0 / SERVO_HZ - time.time()))

    def servoStop(self):
        pass

    def stopScript(self):
        pass

    def getActualQ(self):
        return self.q.tolist()

    def getActualTCPPose(self):
        return [float("nan")] * 6

    def isProtectiveStopped(self):
        return False


def confirm(msg, yes=False):
    if yes:
        print(f"{msg} -> confirmed by the operator beforehand (--yes)")
        return True
    return input(f"{msg} Type 'yes' to continue: ").strip().lower() == "yes"


def start_move_collisions(tr, q_from, q_to, n=60):
    """Collisions in the digital twin along the straight joint-space move q_from -> q_to (what moveJ does)."""
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "sim"))
    from plan_cell import Planner
    from playback_touches import calibrated_placement

    repo = pathlib.Path(__file__).resolve().parents[2]
    cell = json.load(open(repo / "data" / "sim" / "cell.json"))
    calib = json.load(open(repo / "data" / "robot" / "marker_in_robot_base.json"))
    yaw, _, _, Twm = calibrated_placement(cell, np.array(calib["T_base_marker"]))
    cell["marker"]["centre"] = Twm[:3, 3].tolist()
    cell["marker"]["yaw_deg"] = float(np.degrees(np.arctan2(Twm[1, 0], Twm[0, 0])))
    pl = Planner(cell, controller_yaw_deg=yaw, tcp_z_mm=tr["tcp_z_mm"])
    hits = set()
    for s in np.linspace(0, 1, n):
        hits.update(pl.contacts(q_from + s * (q_to - q_from)))
    return sorted(hits)


def main(a):
    tr = json.load(open(a.trajectory))
    if not tr.get("preflight_pass") and not a.force:
        raise SystemExit("this trajectory did not pass pre-flight (see its _report.txt); use --force only if you know why")
    if a.recorded_speed:  # the demo's own timing (what the policy is trained on)
        t = np.array(tr["t_s"]) / tr["slowdown"] * a.extra_slowdown
    else:
        t = np.array(tr["t_s"]) * a.extra_slowdown
    Q = np.array(tr["q_rad"])
    grip = np.array(tr["gripper_0open_255closed"])
    qd = np.degrees(np.abs(np.diff(Q, axis=0)) / np.diff(t)[:, None])
    print(f"trajectory: {a.trajectory.name} | {len(t)} steps, {t[-1]:.1f} s | max joint speed {qd.max():.0f} deg/s"
          f" | path lifted {tr.get('z_offset_mm', 0):.0f} mm | TCP z {tr['tcp_z_mm']} mm (pendant TCP must match)")
    print(f"start joints (deg): {np.degrees(Q[0]).round(1).tolist()}")
    if a.execute and a.extra_slowdown < 1.0:
        raise SystemExit("--extra_slowdown below 1 would speed the robot up; not allowed with --execute")
    cap = MAX_JOINT_SPEED_RECORDED_DEG_S if a.recorded_speed else MAX_JOINT_SPEED_DEG_S
    if a.execute and qd.max() > cap and not a.force:
        raise SystemExit(f"max joint speed {qd.max():.0f} deg/s exceeds {cap} deg/s; increase --extra_slowdown")

    if a.execute:
        if not a.robot_ip:
            raise SystemExit("--execute needs --robot_ip")
        import rtde_control
        import rtde_receive

        rr = rtde_receive.RTDEReceiveInterface(a.robot_ip)
        rc = rtde_control.RTDEControlInterface(a.robot_ip, frequency=SERVO_HZ)  # default would be the robot's 500 Hz
        state = rr
    else:
        print("DRY RUN (no --execute): simulating the robot; nothing will move")
        rc = state = FakeRobot(Q[0] + 0.1)
    gripper = None
    if a.execute and not a.no_gripper:
        gripper = RobotiqSocket(a.robot_ip, speed=a.gripper_speed, force=a.gripper_force)

    q_now = np.array(state.getActualQ())
    move = np.degrees(np.abs(Q[0] - q_now)).max()
    print(f"current joints (deg): {np.degrees(q_now).round(1).tolist()} | largest joint move to start: {move:.0f} deg")
    if move > 1.0:
        hits = start_move_collisions(tr, q_now, Q[0])
        print("twin check of the move to the start pose: " + ("no collisions" if not hits else f"COLLISIONS with {', '.join(hits)}"))
        if hits and a.execute:
            raise SystemExit("refusing the move to the start pose; bring the robot closer to the start pose by hand (freedrive) first")
    log = {"t": [], "q_cmd": [], "q_act": [], "tcp_act": []}
    try:
        if move > 1.0:
            if a.execute and not confirm("Move SLOWLY to the start pose (clear the workspace, e-stop in hand)?", a.yes):
                raise SystemExit("cancelled")
            rc.moveJ(Q[0].tolist(), MOVE_TO_START_SPEED, MOVE_TO_START_ACC)
        if a.move_to_start_only:
            print("at the start pose (--move_to_start_only): stopping here")
            return
        if gripper:
            gripper.set(grip[0])
        if a.execute and not confirm(f"Run the trajectory ({t[-1]:.0f} s)?", a.yes):
            raise SystemExit("cancelled")
        dt = 1.0 / SERVO_HZ
        last_grip, last_grip_t = grip[0], 0.0
        t0 = time.time()
        k = -1
        while True:
            k += 1
            cyc = rc.initPeriod()
            tk = time.time() - t0  # position on the path from the real clock, never from counting commands
            if tk > t[-1]:
                break
            q = np.array([np.interp(tk, t, Q[:, j]) for j in range(6)])
            rc.servoJ(q.tolist(), 0.0, 0.0, dt, a.lookahead, a.gain)
            g = float(np.interp(tk, t, grip))
            if gripper and abs(g - last_grip) > 10 and tk - last_grip_t > 0.1:
                gripper.set(g)
                last_grip, last_grip_t = g, tk
            if k % 5 == 0:
                log["t"].append(tk)
                log["q_cmd"].append(q.tolist())
                log["q_act"].append(state.getActualQ())
                log["tcp_act"].append(state.getActualTCPPose())
            if state.isProtectiveStopped():
                print("PROTECTIVE STOP detected - stopping")
                break
            rc.waitPeriod(cyc)
        took = time.time() - t0
        print(f"done in {took:.1f} s (planned {t[-1]:.1f} s)" + ("" if abs(took - t[-1]) < 0.1 * t[-1] + 0.5 else "  WARNING: timing does not match the plan"))
    except KeyboardInterrupt:
        print("\ninterrupted - stopping the robot")
    finally:
        rc.servoStop()
        rc.stopScript()
        if gripper:
            gripper.close()

    if log["t"]:
        qc, qa = np.array(log["q_cmd"]), np.array(log["q_act"])
        print(f"joint tracking (commanded vs actual): max {np.degrees(np.abs(qc - qa)).max():.2f} deg")
        tcp = np.array(log["tcp_act"])
        if np.isfinite(tcp).all():
            plan = np.array(tr["tcp_pose_ctrl"])
            plan_i = np.array([[np.interp(tk, t, plan[:, j]) for j in range(3)] for tk in log["t"]])
            dev = np.linalg.norm(tcp[:, :3] - plan_i, axis=1) * 1000
            # servoJ follows with a constant delay (lookahead + network): find it, then measure the real path error
            lt = np.array(log["t"])

            def lagged(lag):
                pl = np.array([[np.interp(tk - lag, t, plan[:, j]) for j in range(3)] for tk in lt])
                return np.linalg.norm(tcp[:, :3] - pl, axis=1) * 1000

            lags = np.arange(0.0, 0.3, 0.005)
            lag = lags[int(np.argmin([np.median(lagged(l)) for l in lags]))]
            d = lagged(lag)
            print(f"fingertip vs plan: robot follows {lag * 1000:.0f} ms behind the commands; path error with that delay "
                  f"removed: median {np.median(d):.1f} mm, max {d.max():.1f} mm (without removing it: median {np.median(dev):.1f}, "
                  f"max {dev.max():.1f} mm). A large path error means the pendant TCP differs from tcp_z_mm.")
        out = a.trajectory.with_name(a.trajectory.stem.replace("_robot_trajectory", "") + ("_robot_log.json" if a.execute else "_dryrun_log.json"))
        json.dump(log, open(out, "w"))
        print(f"saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Replay a pre-flighted trajectory on the real UR5e (joint space).")
    ap.add_argument("--trajectory", required=True, type=pathlib.Path)
    ap.add_argument("--robot_ip")
    ap.add_argument("--execute", action="store_true", help="actually move the robot (otherwise: dry run)")
    ap.add_argument("--no_gripper", action="store_true", help="don't command the gripper")
    ap.add_argument("--extra_slowdown", type=float, default=1.0, help=">1 = slower than the pre-flight trajectory")
    ap.add_argument("--lookahead", type=float, default=0.1, help="servoJ lookahead time (s)")
    ap.add_argument("--gain", type=float, default=300, help="servoJ gain")
    ap.add_argument("--force", action="store_true", help="allow trajectories that did not pass pre-flight")
    ap.add_argument("--recorded_speed", action="store_true", help="replay at the demo's own timing (cap 150 deg/s)")
    ap.add_argument("--gripper_speed", type=int, default=255, help="Robotiq speed 0-255")
    ap.add_argument("--gripper_force", type=int, default=50, help="Robotiq force 0-255 (gentle on a plastic cup)")
    ap.add_argument("--yes", action="store_true", help="skip the typed confirmations (operator confirmed beforehand)")
    ap.add_argument("--move_to_start_only", action="store_true", help="only move to the start pose, then stop")
    main(ap.parse_args())

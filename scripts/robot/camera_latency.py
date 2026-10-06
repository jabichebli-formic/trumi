"""Measure the wrist camera's real latency (light -> frame on this PC) using the gripper as a clock.

The Robotiq gripper opens and closes a few times; its own reported position (polled at ~50 Hz) says when the
fingers really moved. The same motion is seen in the live GoPro preview as the distance between the two finger ArUco
tags (raw frames, not masked). Latency = time the tags move in the video - time the fingers really moved, at the
half-way point of every open/close. Only the gripper moves; the arm is not commanded.

Usage (from ~/trumi, LeRobot environment): ~/YAM/yam-lerobot/.venv/bin/python scripts/robot/camera_latency.py --robot_ip 192.168.10.204
"""

import argparse
import pathlib
import socket
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from run_policy import GoProPreview, find_gopro_ip  # noqa: E402


def main(a):
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    cam = GoProPreview(a.gopro_ip or find_gopro_ip())
    video, grip, stop = [], [], threading.Event()

    def watch_video():  # finger-tag distance (pixels) per new frame, stamped when the frame was decoded here
        last = 0
        while not stop.is_set():
            if cam.n == last:
                time.sleep(0.002)
                continue
            last, img, t = cam.n, cam.frame, cam.t_frame
            corners, ids, _ = det.detectMarkers(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
            if ids is not None:
                c = {int(i): cc[0].mean(0) for cc, i in zip(corners, ids.ravel()) if i in (0, 1)}
                if 0 in c and 1 in c:
                    video.append((t, float(np.linalg.norm(c[0] - c[1]))))

    def watch_gripper():  # the gripper's own position report (separate read-only connection)
        with socket.create_connection((a.robot_ip, 63352), timeout=3) as sk:
            while not stop.is_set():
                sk.sendall(b"GET POS\n")
                v = sk.recv(64)
                grip.append((time.time(), int(v.decode().split()[-1])))
                time.sleep(0.02)

    threads = [threading.Thread(target=watch_video, daemon=True), threading.Thread(target=watch_gripper, daemon=True)]
    for th in threads:
        th.start()
    from replay_on_robot import RobotiqSocket

    g = RobotiqSocket(a.robot_ip, speed=a.speed, force=a.force)
    try:
        time.sleep(1.0)
        for _ in range(a.cycles):
            for target in (a.closed, 0):
                g.set(target)
                time.sleep(a.hold_s)
    finally:
        g.set(0)
        g.close()
        time.sleep(0.5)
        stop.set()
        cam.close()
    V, G = np.array(video), np.array(grip, float)
    if len(V) < 20 or len(G) < 20:
        raise SystemExit(f"not enough data: {len(V)} frames with both finger tags, {len(G)} gripper readings")

    def crossings(t, x):  # times the signal passes half-way between its low and high levels, with direction
        lo, hi = np.percentile(x, 5), np.percentile(x, 95)
        mid = (lo + hi) / 2
        out = []
        for i in range(1, len(x)):
            if (x[i - 1] - mid) * (x[i] - mid) < 0:
                f = (mid - x[i - 1]) / (x[i] - x[i - 1])
                out.append((t[i - 1] + f * (t[i] - t[i - 1]), np.sign(x[i] - x[i - 1])))
        return out

    gc = crossings(G[:, 0], -G[:, 1])  # closing = POS up = tag distance down: flip so both rise when opening
    vc = crossings(V[:, 0], V[:, 1])
    lat = []
    for tg, sg in gc:
        cand = [tv - tg for tv, sv in vc if sv == sg and -0.2 < tv - tg < 2.0]
        if cand:
            lat.append(min(cand, key=abs))
    lat = np.array(lat) * 1000
    fps = (len(V) - 1) / (V[-1, 0] - V[0, 0])
    print(f"{len(gc)} gripper moves, {len(lat)} matched in the video ({fps:.0f} fps with both tags visible)")
    print(f"camera latency (finger motion seen on this PC minus real finger motion): median {np.median(lat):.0f} ms, "
          f"range {lat.min():.0f}-{lat.max():.0f} ms  (frame spacing {1000 / fps:.0f} ms adds up to that much uncertainty)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Wrist camera latency using the gripper as a clock (gripper moves).")
    ap.add_argument("--robot_ip", required=True)
    ap.add_argument("--gopro_ip")
    ap.add_argument("--cycles", type=int, default=4)
    ap.add_argument("--hold_s", type=float, default=1.5)
    ap.add_argument("--closed", type=int, default=200, help="close to this position (not fully: fingers stay apart)")
    ap.add_argument("--speed", type=int, default=255)
    ap.add_argument("--force", type=int, default=30)
    main(ap.parse_args())

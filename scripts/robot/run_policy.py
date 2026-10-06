"""Run a trained pi0.5 policy (LeRobot 0.6) on the UR5e + Robotiq 2F-85 with the wrist GoPro.

Pipeline (the same as the training data, see docs/POST_PROCESSING.md):
  GoPro HERO13 over USB-C -> Open GoPro preview stream (UDP, HEVC 1920x1440 4:3, 30 fps)
  -> policy mask (fingers + gripper body, same file as the dataset export) -> 640x480 RGB
  + state = UR joints (rad, 6) + gripper (last command / 255)
  -> pi0.5 -> chunk of 50 actions (30 fps, absolute joint targets + gripper)
  -> background servo thread at 125 Hz: follows the newest chunk by wall clock, joint-speed limited, gripper via the
     Robotiq URCap socket.
A new chunk is predicted every --replan_s; the actions that are already in the past when it arrives (inference time)
are skipped, and the joint-speed limiter makes the switch from the old chunk smooth.

Modes:
  --dataset ROOT                  offline check, no robot or camera: feeds stored dataset frames and states, compares
                                  the predicted chunks with the recorded actions (run where the GPU is).
  --camera_test                   only the GoPro stream: frame rate, frame age, saves preprocessed frames.
  --robot_ip IP                   SHADOW mode (default): live camera + robot state, policy runs and logs what it would
                                  do, the robot does NOT move.
  --robot_ip IP --execute         moves the robot. Check list: e-stop in hand, nobody within reach, Remote mode,
                                  pendant TCP = --tcp_z_mm, gripper activated.
Safety (execute): joint-speed limit (--max_joint_speed_deg_s, default 60), a chunk is refused if its first target is
more than --max_jump_deg from the robot or any target puts the fingertip below --min_height_mm above the marker plane
(robot calibration) or outside the UR joint limits; --max_seconds; Ctrl+C stops (servoStop).

Usage (from ~/trumi, LeRobot 0.6 environment, e.g. ~/YAM/yam-lerobot/.venv/bin/python):
  python scripts/robot/run_policy.py --checkpoint <.../pretrained_model> --dataset data/lerobot/trumi_conveyor_pick_v1
  python scripts/robot/run_policy.py --camera_test
  python scripts/robot/run_policy.py --checkpoint <...> --robot_ip 192.168.10.204               # shadow
  python scripts/robot/run_policy.py --checkpoint <...> --robot_ip 192.168.10.204 --execute     # moves
Logs: data/robot/policy_runs/<time>/ (log.json, frames/).
"""

import argparse
import datetime
import json
import pathlib
import sys
import threading
import time
import urllib.request

import cv2
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
FPS = 30  # dataset / action rate
SERVO_HZ = 125
IMG_W, IMG_H = 640, 480
TASK = "pick the cup from the conveyor and place it in the box"
# UR5e DH (nominal) for the fingertip height check
DH_D = [0.1625, 0, 0, 0.1333, 0.0997, 0.0996]
DH_A = [0, -0.425, -0.3922, 0, 0, 0]
DH_ALPHA = [np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]


# ---------------------------------------------------------------- camera
class GoProPreview:
    """Latest frame of the HERO13 preview stream over USB-C (Open GoPro HTTP on port 8080, UDP HEVC on port 8554)."""

    def __init__(self, ip, port=8554):
        import av

        self.ip, self.port, self.av = ip, port, av
        self.frame, self.t_frame, self.n, self.stop_flag, self.error = None, 0.0, 0, False, None
        self._http("/gopro/camera/control/wired_usb?p=1")
        self._http("/gopro/camera/stream/stop")
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()  # listen first: the HEVC parameter sets are only sent when the stream starts
        time.sleep(1.0)
        self._http(f"/gopro/camera/stream/start?port={port}")
        t0 = time.time()
        while self.frame is None and self.error is None and time.time() - t0 < 15:  # first frame takes ~2 s
            time.sleep(0.05)
        if self.frame is None:
            raise RuntimeError(f"GoPro preview stream did not start: {self.error or 'no frame within 15 s'}")

    def _http(self, path):
        with urllib.request.urlopen(f"http://{self.ip}:8080{path}", timeout=5) as r:
            return r.read()

    def _run(self):
        url = f"udp://@0.0.0.0:{self.port}?overrun_nonfatal=1&fifo_size=50000000"
        try:
            opts = {"fflags": "nobuffer", "flags": "low_delay", "probesize": "500000", "analyzeduration": "500000"}
            with self.av.open(url, options=opts, timeout=30) as c:
                s = c.streams.video[0]
                # slice threads only: frame threading (AUTO/FRAME, 16 threads here) holds back 15 frames = 0.5 s at
                # 30 fps; slice decoding holds back none and decodes 2.7K at ~110 fps (scratch test, 2026-10-06)
                s.thread_type = "SLICE"
                for fr in c.decode(s):
                    if self.stop_flag:
                        break
                    self.frame, self.t_frame = fr.to_ndarray(format="rgb24"), time.time()
                    self.n += 1
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"

    def latest(self, max_age_s=0.3):
        if self.frame is None or time.time() - self.t_frame > max_age_s:
            why = self.error or ("no frame received yet" if self.frame is None else f"last frame {time.time() - self.t_frame:.2f} s old")
            raise RuntimeError(f"no fresh camera frame: {why}")
        return self.frame, self.t_frame

    def close(self):
        self.stop_flag = True
        try:
            self._http("/gopro/camera/stream/stop")
        except Exception:
            pass


def find_gopro_ip():
    """The camera is at 172.2X.1YZ.51 on the USB network; this PC gets an address in the same /24."""
    import subprocess

    out = subprocess.run(["ip", "-4", "-o", "addr"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        addr = line.split()[3].split("/")[0]
        if addr.startswith("172.2") and line.split()[1].startswith("enx"):
            return ".".join(addr.split(".")[:3] + ["51"])
    raise SystemExit("no GoPro USB network found: plug the GoPro in with USB-C (Preferences > Connections > USB: GoPro Connect)")


class Preprocess:
    """Frame (any size, 4:3) -> masked 640x480 RGB, as in scripts/tools/export_lerobot.py."""

    def __init__(self, mask_path):
        self.mask_full = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        self.masks = {}

    def __call__(self, img):
        h, w = img.shape[:2]
        if abs(w / h - 4 / 3) > 0.01:
            raise ValueError(f"camera frame is {w}x{h}, expected 4:3 (GoPro in 4:3 Wide video mode)")
        if (w, h) not in self.masks:
            self.masks[(w, h)] = cv2.resize(self.mask_full, (w, h), interpolation=cv2.INTER_NEAREST) > 0
        img = img.copy()
        img[self.masks[(w, h)]] = 0
        return cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- policy
# settings written by newer LeRobot versions (e.g. 0.6.2 on the training server) that LeRobot 0.6.0 does not know;
# they may only be dropped when they have these values (the features they control are switched off)
NEWER_LEROBOT_OFF = {"use_visual_memory": False, "use_proprioceptive_memory": False, "rtc_training_max_delay": 0,
                     "memory_frames": None, "memory_stride": None, "memory_temporal_attention_every": None}


def compatible_checkpoint(checkpoint, cfg_class):
    """The checkpoint folder itself, or (if its config has settings this LeRobot does not know, all switched off) a
    folder next to it in the scratch area with links to the same files and a config without those settings."""
    import dataclasses
    import tempfile

    cfg = json.load(open(checkpoint / "config.json"))
    known = {f.name for f in dataclasses.fields(cfg_class)} | {"type"}
    extra = {k: v for k, v in cfg.items() if k not in known}
    if not extra:
        return checkpoint
    bad = {k: v for k, v in extra.items() if k not in NEWER_LEROBOT_OFF or (NEWER_LEROBOT_OFF[k] is not None and v != NEWER_LEROBOT_OFF[k])}
    if bad:
        raise SystemExit(f"checkpoint uses settings this LeRobot version does not support: {bad}")
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="trumi_ckpt_"))
    for f in checkpoint.iterdir():
        if f.name != "config.json":
            (tmp / f.name).symlink_to(f.resolve())
    json.dump({k: v for k, v in cfg.items() if k not in extra}, open(tmp / "config.json", "w"), indent=1)
    print(f"checkpoint config: dropped settings unknown to this LeRobot, all switched off: {extra}")
    return tmp


class Policy:
    def __init__(self, checkpoint, device, compile_model=False, seed=None, rtc_horizon=0):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.policies.utils import prepare_observation_for_inference

        self.seed = seed
        checkpoint = compatible_checkpoint(pathlib.Path(checkpoint), PI05Config)
        self.torch, self.prepare, self.device = torch, prepare_observation_for_inference, torch.device(device)
        t0 = time.time()
        cfg = PreTrainedConfig.from_pretrained(str(checkpoint))
        cfg.compile_model = compile_model  # training used torch.compile: the first prediction would take ~6 min
        cfg.device = device
        self.rtc = rtc_horizon > 0
        if self.rtc:  # real-time chunking (inference-time guidance, no retraining): new chunks continue the old one
            from lerobot.policies.rtc.configuration_rtc import RTCConfig

            cfg.rtc_config = RTCConfig(execution_horizon=rtc_horizon)
        self.policy = PI05Policy.from_pretrained(str(checkpoint), config=cfg).to(self.device).eval()
        over = {"device": device}
        pre_over = {"device_processor": over}
        if (checkpoint / "tokenizer").is_dir():  # older LeRobot resolves the saved "tokenizer" relative to the working directory
            pre_over["tokenizer_processor"] = {"tokenizer_name": str((checkpoint / "tokenizer").resolve())}
        self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=str(checkpoint),
                                                       preprocessor_overrides=pre_over,
                                                       postprocessor_overrides={"device_processor": over})
        print(f"policy loaded from {checkpoint} in {time.time() - t0:.0f} s on {device}")

    def chunk(self, img_rgb, state, task=TASK, prev_raw=None, delay=0, return_raw=False):
        """img: 480x640x3 uint8 RGB (already masked), state: (7,) -> (50, 7) absolute actions.
        RTC: prev_raw = the not-yet-executed rest of the current chunk as the model output it (normalised), delay =
        actions that will be executed while this prediction runs; the new chunk is steered to continue prev_raw."""
        torch = self.torch
        obs = {"observation.images.wrist": img_rgb, "observation.state": np.asarray(state, np.float32)}
        if self.seed is not None:  # the action chunk starts from random noise: fix it to compare runs
            torch.manual_seed(self.seed)
        kwargs = {}
        if self.rtc and prev_raw is not None and len(prev_raw):
            kwargs = {"prev_chunk_left_over": torch.as_tensor(prev_raw, device=self.device), "inference_delay": int(delay)}
        with torch.no_grad():  # not inference_mode: RTC guidance needs autograd inside the sampler
            batch = self.pre(self.prepare(obs, self.device, task, "ur5e_robotiq_2f85"))
            raw = self.policy.predict_action_chunk(batch, **kwargs)  # (1, T, 7), normalised
            a = torch.stack([self.post(raw[:, i, :]) for i in range(raw.shape[1])], dim=1)
        out = a[0].float().cpu().numpy()
        return (out, raw[0].float().cpu().numpy()) if return_raw else out


# ---------------------------------------------------------------- safety helpers
def fingertip_height_mm(q, T_marker_inv, tcp_z):
    """Height of the TCP (closed fingertip ends) above the marker plane, from UR kinematics."""
    T = np.eye(4)
    for i in range(6):
        ct, st, ca, sa = np.cos(q[i]), np.sin(q[i]), np.cos(DH_ALPHA[i]), np.sin(DH_ALPHA[i])
        T = T @ np.array([[ct, -st * ca, st * sa, DH_A[i] * ct], [st, ct * ca, -ct * sa, DH_A[i] * st], [0, sa, ca, DH_D[i]], [0, 0, 0, 1]])
    tip = T[:3, 3] + T[:3, 2] * tcp_z
    return float((T_marker_inv @ np.r_[tip, 1])[2] * 1000)


def check_chunk(chunk, q_now, a, T_marker_inv, i_now=0):
    """Reasons to refuse a chunk (empty list = fine). i_now: index of the action due right now."""
    problems = []
    i_now = int(np.clip(i_now, 0, len(chunk) - 1))
    jump = np.degrees(np.abs(chunk[i_now, :6] - q_now)).max()
    if jump > a.max_jump_deg:
        problems.append(f"target due now is {jump:.0f} deg from the robot (> {a.max_jump_deg})")
    if i_now >= len(chunk) - 5:
        problems.append(f"chunk already over when it arrived (action {i_now} of {len(chunk)} due): latency too large")
    if np.abs(chunk[:, :6]).max() > 2 * np.pi - 0.05:
        problems.append("a target is at a joint limit (+-360 deg)")
    low = min(fingertip_height_mm(q, T_marker_inv, a.tcp_z_mm / 1000) for q in chunk[:, :6])
    if low < a.min_height_mm:
        problems.append(f"fingertip would go to {low:.0f} mm above the marker plane (< {a.min_height_mm})")
    return problems


# ---------------------------------------------------------------- execution
class Servo:
    """Background 125 Hz loop: follows the newest chunk by wall clock with a joint-speed limit; sends gripper commands."""

    def __init__(self, rc, rr, gripper, q0, g0, max_speed_deg_s, blend_s=0.0, gripper_delay_s=0.0):
        self.rc, self.rr, self.gripper = rc, rr, gripper
        self.gripper_delay_s = gripper_delay_s
        self.g_hist = [(time.time(), float(g0))]  # (time, gripper command 0-255) to look up the state at a past time
        self.blend_s, self.old, self.t_switch = blend_s, None, 0.0
        self.q_cmd, self.g_sent = np.array(q0, float), float(g0)
        self.vmax = np.radians(max_speed_deg_s)
        self.traj = None  # (t_start, chunk) with action i due at t_start + i / FPS
        self.lock, self.stop_flag, self.error = threading.Lock(), False, None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def set_chunk(self, t_start, chunk):
        with self.lock:
            self.old, self.t_switch = self.traj, time.time()  # crossfade from the previous chunk (blend_s)
            self.traj = (t_start, chunk)

    @staticmethod
    def _at(traj, now):
        t0, ch = traj
        x = (now - t0) * FPS
        if x >= len(ch) - 1:
            return ch[-1]  # chunk ran out before a new one: hold the last target
        i = max(int(x), 0)
        f = min(max(x - i, 0.0), 1.0)
        return ch[i] * (1 - f) + ch[i + 1] * f

    def _target(self, now):
        with self.lock:
            traj, old, ts = self.traj, self.old, self.t_switch
        if traj is None:
            return None
        new = self._at(traj, now)
        if old is None or self.blend_s <= 0 or now - ts >= self.blend_s:
            return new
        w = (now - ts) / self.blend_s
        w = w * w * (3 - 2 * w)  # smoothstep: no velocity jump at either end of the crossfade
        return self._at(old, now) * (1 - w) + new * w

    def _run(self):
        dt = 1.0 / SERVO_HZ
        last_grip_t = 0.0
        try:
            while not self.stop_flag:
                cyc = self.rc.initPeriod()
                now = time.time()
                tgt = self._target(now)
                if tgt is not None:
                    step = np.clip(tgt[:6] - self.q_cmd, -self.vmax * dt, self.vmax * dt)  # joint-speed limit
                    self.q_cmd = self.q_cmd + step
                    g_tgt = self._target(now - self.gripper_delay_s) if self.gripper_delay_s > 0 else tgt
                    g = float(np.clip(g_tgt[6], 0, 1)) * 255  # gripper can lag the arm on purpose (--gripper_delay_s)
                    if self.gripper and abs(g - self.g_sent) > 10 and now - last_grip_t > 0.1:
                        self.gripper.set(g)
                        self.g_sent, last_grip_t = g, now
                        self.g_hist.append((now, g))
                self.rc.servoJ(self.q_cmd.tolist(), 0.0, 0.0, dt, 0.1, 300)
                if self.rr.isProtectiveStopped():
                    self.error = "protective stop"
                    break
                self.rc.waitPeriod(cyc)
        except Exception as e:  # surface to the main loop
            self.error = f"{type(e).__name__}: {e}"


def move_home(a):
    """Slow, twin-checked joint move to the in-distribution start pose (data/robot/policy_home_*.json)."""
    import rtde_control
    import rtde_receive
    from replay_on_robot import start_move_collisions

    home = np.array(json.load(open(a.home_pose))["q_rad"])
    rr = rtde_receive.RTDEReceiveInterface(a.robot_ip)
    q = np.array(rr.getActualQ())
    move = np.degrees(home - q)
    print(f"home {np.round(np.degrees(home), 1).tolist()} deg | move per joint {np.round(move, 0).tolist()} deg")
    if np.abs(move).max() < 0.5:
        print("already at home")
        return
    hits = start_move_collisions({"tcp_z_mm": a.tcp_z_mm}, q, home)
    print("twin check of the move: " + ("no collisions" if not hits else f"COLLISIONS with {', '.join(hits)}"))
    if hits and not a.ignore_twin:
        raise SystemExit("refusing the move home; bring the robot closer by hand (freedrive) first, or --ignore_twin "
                         "if the operator has checked the path on the real cell (the twin's box/belt are approximate)")
    if hits:
        print("WARNING: twin reports contacts, moving anyway because the operator checked the real path (--ignore_twin)")
    if not a.yes and input("Move SLOWLY to the home pose (e-stop in hand, workspace clear)? Type 'yes': ").strip() != "yes":
        raise SystemExit("cancelled")
    rc = rtde_control.RTDEControlInterface(a.robot_ip)
    try:
        rc.moveJ(home.tolist(), a.home_speed, 0.5)
    finally:
        rc.stopScript()
    q = np.array(rr.getActualQ())
    rr.disconnect()
    print(f"at home: largest joint error {np.degrees(np.abs(q - home)).max():.2f} deg")


# ---------------------------------------------------------------- modes
def run_dataset(a):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    pol = Policy(a.checkpoint, a.device, a.compile, a.seed)
    ds = LeRobotDataset(a.repo_id, root=a.dataset)
    eps = a.episodes or list(range(0, ds.meta.total_episodes, max(1, ds.meta.total_episodes // 5)))
    rows, chunks = [], {}
    for e in eps:
        m = ds.meta.episodes[e]
        start, length = int(m["dataset_from_index"]), int(m["length"])
        for k in range(0, length - 1, a.dataset_stride):
            s = ds[start + k]
            img = (s["observation.images.wrist"].numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
            t0 = time.time()
            ch = pol.chunk(img, s["observation.state"].numpy())
            chunks[f"ep{e}_f{k}"] = ch
            dt_ms = (time.time() - t0) * 1000
            n = min(len(ch), length - k)
            truth = np.stack([ds[start + k + j]["action"].numpy() for j in range(n)])
            err = np.degrees(np.abs(ch[:n, :6] - truth[:, :6]))
            rows.append({"episode": e, "frame": k, "infer_ms": dt_ms, "j_err_1": err[0].max(), "j_err_15": err[:15].max(axis=1).mean(),
                         "j_err_all": err.max(axis=1).mean(), "g_err": float(np.abs(ch[:n, 6] - truth[:, 6]).mean())})
            print(f"  episode {e} frame {k:3d}: inference {dt_ms:5.0f} ms | joint error vs recorded: next action {rows[-1]['j_err_1']:.1f} deg, "
                  f"first 0.5 s {rows[-1]['j_err_15']:.1f} deg, whole chunk {rows[-1]['j_err_all']:.1f} deg | gripper {rows[-1]['g_err']:.3f}")
    r = {k: np.array([x[k] for x in rows]) for k in rows[0]}
    if a.save_chunks:
        np.savez(a.save_chunks, **chunks)
        print(f"saved {len(chunks)} predicted chunks to {a.save_chunks}")
    if pol.device.type == "cuda":
        import torch

        print(f"GPU memory: peak allocated {torch.cuda.max_memory_allocated() / 1e9:.1f} GB, "
              f"peak reserved {torch.cuda.max_memory_reserved() / 1e9:.1f} GB")
    print(f"\n{len(rows)} chunks: inference median {np.median(r['infer_ms'][1:]):.0f} ms (first {r['infer_ms'][0]:.0f} ms) | "
          f"joint error median: next action {np.median(r['j_err_1']):.1f} deg, first 0.5 s {np.median(r['j_err_15']):.1f} deg, "
          f"1.7 s chunk {np.median(r['j_err_all']):.1f} deg | gripper {np.median(r['g_err']):.3f}")


def run_camera_test(a, out):
    cam = GoProPreview(a.gopro_ip or find_gopro_ip())
    prep = Preprocess(a.mask)
    try:
        t_end, ages, n0 = time.time() + a.max_seconds, [], None
        time.sleep(2)
        n0, t0 = cam.n, time.time()
        while time.time() < t_end:
            img, t_img = cam.latest(max_age_s=2)
            ages.append(time.time() - t_img)
            cv2.imwrite(str(out / "frames" / f"{len(ages):04d}.jpg"), cv2.cvtColor(prep(img), cv2.COLOR_RGB2BGR))
            time.sleep(0.5)
        fps = (cam.n - n0) / (time.time() - t0)
        print(f"camera: {img.shape[1]}x{img.shape[0]}, {fps:.1f} fps decoded, frame age when read: median {np.median(ages) * 1000:.0f} ms "
              f"(decode only; not the full glass-to-PC latency) | preprocessed frames in {out / 'frames'}")
    finally:
        cam.close()


def run_robot(a, out):
    import rtde_receive

    calib = json.load(open(a.calibration))
    T_marker_inv = np.linalg.inv(np.array(calib["T_base_marker"]))
    rr = rtde_receive.RTDEReceiveInterface(a.robot_ip)
    from replay_on_robot import RobotiqSocket

    gripper = None
    if a.execute and not a.no_gripper:
        gripper = RobotiqSocket(a.robot_ip, speed=a.gripper_speed, force=a.gripper_force)
        g_state = gripper.get("POS") / 255.0
    else:  # shadow mode only reads the gripper: opening the control connection sends commands
        import socket

        with socket.create_connection((a.robot_ip, 63352), timeout=3) as sk:
            sk.sendall(b"GET POS\n")
            g_state = int(sk.recv(64).decode().split()[-1]) / 255.0
    cam = GoProPreview(a.gopro_ip or find_gopro_ip())
    prep = Preprocess(a.mask)
    pol = Policy(a.checkpoint, a.device, a.compile, rtc_horizon=a.rtc_horizon)
    servo, rc, log = None, None, []
    hist, stop_hist = [], threading.Event()  # (time, joints) at ~100 Hz: the state at the moment a frame was taken

    def record_state():
        while not stop_hist.is_set():
            hist.append((time.time(), np.array(rr.getActualQ())))
            if len(hist) > 1000:
                del hist[:200]
            time.sleep(0.01)

    def state_at(t):
        ts = np.array([h[0] for h in hist])
        k = int(np.clip(np.searchsorted(ts, t), 1, len(ts) - 1))
        f = float(np.clip((t - ts[k - 1]) / max(ts[k] - ts[k - 1], 1e-6), 0, 1))
        return hist[k - 1][1] * (1 - f) + hist[k][1] * f

    threading.Thread(target=record_state, daemon=True).start()
    try:
        time.sleep(2)
        if a.execute:
            import rtde_control

            if not a.yes and input("Run the policy on the robot (e-stop in hand, workspace clear)? Type 'yes': ").strip() != "yes":
                raise SystemExit("cancelled")
            rc = rtde_control.RTDEControlInterface(a.robot_ip, frequency=SERVO_HZ)
            servo = Servo(rc, rr, gripper, rr.getActualQ(), g_state * 255, a.max_joint_speed_deg_s, a.blend_s, a.gripper_delay_s)
            servo.thread.start()
        t_end = time.time() + a.max_seconds if a.max_seconds > 0 else float("inf")
        print(("EXECUTING" if a.execute else "SHADOW MODE (robot does not move)")
              + (f" for up to {a.max_seconds:.0f} s" if a.max_seconds > 0 else " until Ctrl+C") + "; Ctrl+C to stop")
        cur = None  # (t0, absolute chunk, raw chunk): action i of the current chunk is due at t0 + i / FPS
        latencies = []
        while time.time() < t_end:
            t_loop = time.time()
            img, t_img = cam.latest()
            t_obs = t_img - a.camera_latency_s  # when the frame was really taken (measured with camera_latency.py)
            q = np.array(rr.getActualQ())  # now (for the safety check)
            q_obs = state_at(t_obs)  # the state at the moment the frame was taken: image and state match, as in training
            if servo:
                g_state = [g for t, g in servo.g_hist if t <= t_obs][-1:] or [servo.g_hist[0][1]]
                g_state = g_state[0] / 255.0
            obs_img = prep(img)
            prev_raw, delay, t0_new = None, 0, t_obs
            if cur is not None and pol.rtc and not a.sync_steps:
                idx = int(round((t_obs - cur[0]) * FPS))  # index of the current chunk being executed at the observation
                if idx < len(cur[2]) - 1:
                    prev_raw = cur[2][max(idx, 0):]
                    delay = int(np.ceil(max(latencies[-5:]) * FPS)) if latencies else 0
                    t0_new = cur[0] + max(idx, 0) / FPS  # keep the new chunk on the old chunk's timeline
            ch, raw = pol.chunk(obs_img, np.r_[q_obs, g_state], prev_raw=prev_raw, delay=delay, return_raw=True)
            t_ready = time.time()
            latencies.append(t_ready - t_obs)
            q = np.array(rr.getActualQ())
            i_now = 0 if a.sync_steps else (t_ready - t0_new) * FPS
            problems = check_chunk(ch, q, a, T_marker_inv, i_now)
            entry = {"t": t_img, "t_obs": t_obs, "q_obs": q_obs.tolist(), "i_now": float(i_now), "t0": t0_new, "rtc_delay": delay, "infer_ms": (t_ready - t_loop) * 1000, "frame_age_ms": (t_loop - t_img) * 1000, "q": q.tolist(),
                     "g": g_state, "chunk": np.round(ch, 5).tolist(), "problems": problems}
            log.append(entry)
            with open(out / "steps.jsonl", "a") as fh:  # saved as we go: a killed run keeps its log
                fh.write(json.dumps(entry) + "\n")
            cv2.imwrite(str(out / "frames" / f"{len(log):04d}.jpg"), cv2.cvtColor(obs_img, cv2.COLOR_RGB2BGR))
            move = np.degrees(np.abs(ch[min(14, len(ch) - 1), :6] - q)).max()
            print(f"  {len(log):3d}: inference {entry['infer_ms']:4.0f} ms, frame age {entry['frame_age_ms']:3.0f} ms | in 0.5 s: largest joint move "
                  f"{move:5.1f} deg, gripper {ch[min(14, len(ch) - 1), 6]:.2f}" + (f" | REFUSED: {'; '.join(problems)}" if problems else ""))
            if problems:
                if a.execute:
                    print("stopping: unsafe chunk")
                    break
            elif servo:
                if servo.error:
                    print(f"stopping: {servo.error}")
                    break
                if a.sync_steps:
                    # synchronous: the arm stood still during inference, so start this chunk now from action 0, run
                    # the first sync_steps actions as trained (no stitching), then wait for the arm before looking again
                    servo.set_chunk(time.time(), ch[: a.sync_steps + 1])
                else:
                    servo.set_chunk(t0_new, ch)  # past actions are skipped by wall clock
            if not problems:
                cur = (t0_new, ch, raw)
            if a.sync_steps:
                time.sleep(a.sync_steps / FPS)
                if servo:  # let the speed-limited arm reach the last target (at most 1 s)
                    t_wait = time.time()
                    while time.time() - t_wait < 1.0 and np.degrees(np.abs(np.array(rr.getActualQ()) - ch[a.sync_steps, :6])).max() > 1.0:
                        time.sleep(0.02)
                    time.sleep(a.camera_latency_s)  # the next frame must show the arm already stopped
            else:
                time.sleep(max(0.0, a.replan_s - (time.time() - t_loop)))
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        stop_hist.set()
        if servo:
            servo.stop_flag = True
            servo.thread.join(timeout=1)
        if rc:
            rc.servoStop()
            rc.stopScript()
        cam.close()
        json.dump({"args": {k: str(v) for k, v in vars(a).items()}, "steps": log}, open(out / "log.json", "w"))
        print(f"saved {out / 'log.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run a pi0.5 policy on the UR5e with the wrist GoPro (shadow mode by default).")
    ap.add_argument("--checkpoint", type=pathlib.Path, help=".../checkpoints/<step>/pretrained_model")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model (faster per step, ~6 min warm-up)")
    ap.add_argument("--dataset", type=pathlib.Path, help="offline check against a LeRobot dataset")
    ap.add_argument("--repo_id", default="formic/trumi_conveyor_pick_v1")
    ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--dataset_stride", type=int, default=30, help="frames between checks in --dataset mode")
    ap.add_argument("--seed", type=int, default=None, help="fix the noise the action chunk starts from (comparisons)")
    ap.add_argument("--save_chunks", type=pathlib.Path, help="--dataset mode: save the predicted chunks (.npz)")
    ap.add_argument("--camera_test", action="store_true")
    ap.add_argument("--gopro_ip", help="default: found from the USB network (172.2X.1YZ.51)")
    ap.add_argument("--robot_ip")
    ap.add_argument("--execute", action="store_true", help="move the robot (otherwise shadow mode)")
    ap.add_argument("--home", action="store_true", help="first move slowly to the in-distribution home pose (robot moves)")
    ap.add_argument("--home_only", action="store_true", help="only move to the home pose, then stop (robot moves)")
    ap.add_argument("--home_pose", type=pathlib.Path, default=REPO / "data" / "robot" / "policy_home_robot1.json")
    ap.add_argument("--home_speed", type=float, default=0.3, help="rad/s")
    ap.add_argument("--ignore_twin", action="store_true", help="home move: operator checked the real path, ignore twin contacts")
    ap.add_argument("--yes", action="store_true", help="skip the typed confirmation (operator confirmed beforehand)")
    ap.add_argument("--no_gripper", action="store_true")
    ap.add_argument("--gripper_speed", type=int, default=255)
    ap.add_argument("--gripper_force", type=int, default=50)
    ap.add_argument("--max_seconds", type=float, default=30, help="0 = run until Ctrl+C")
    ap.add_argument("--replan_s", type=float, default=0.5, help="predict a new chunk this often")
    ap.add_argument("--camera_latency_s", type=float, default=0.07,
                    help="wrist camera latency (light -> frame on this PC), measured with scripts/robot/camera_latency.py: "
                         "0.073 s for the HERO13 USB preview with slice decoding (0.57 s before, 2026-10-06); 0 = ignore")
    ap.add_argument("--gripper_delay_s", type=float, default=0.0,
                    help="send gripper commands this much later than the arm's (e.g. 0.25): closes nearer the object")
    ap.add_argument("--blend_s", type=float, default=0.0,
                    help="continuous replanning: crossfade from the old to the new chunk over this time (e.g. 0.25) "
                         "instead of jumping; the new chunk is used fully after it")
    ap.add_argument("--sync_steps", type=int, default=0,
                    help="synchronous: predict, run the first N actions of the chunk (e.g. 25 = 0.83 s), predict again "
                         "(arm pauses ~0.2 s per chunk; no stitching, no RTC). 0 = continuous replanning")
    ap.add_argument("--rtc_horizon", type=int, default=0,
                    help="real-time chunking: steer each new chunk to continue the first N actions of the old one "
                         "(0 = off; e.g. 25 with --replan_s 0.5)")
    ap.add_argument("--max_joint_speed_deg_s", type=float, default=60)
    ap.add_argument("--max_jump_deg", type=float, default=20)
    ap.add_argument("--min_height_mm", type=float, default=0, help="fingertip never below this height above the marker plane")
    ap.add_argument("--tcp_z_mm", type=float, default=257.2)
    ap.add_argument("--calibration", type=pathlib.Path, default=REPO / "data" / "robot" / "marker_in_robot_base.json")
    ap.add_argument("--mask", type=pathlib.Path, default=REPO / "data" / "robot" / "policy_mask_2704x2028.png",
                    help="must match the dataset the checkpoint was trained on (fingers-visible: policy_mask_gripper_only_2704x2028.png)")
    a = ap.parse_args()
    a.sync_steps = min(max(a.sync_steps, 0), 49)  # chunks have 50 actions
    out = REPO / "data" / "robot" / "policy_runs" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if a.dataset:
        run_dataset(a)
    elif a.camera_test:
        (out / "frames").mkdir(parents=True)
        run_camera_test(a, out)
    elif a.robot_ip and a.home_only:
        move_home(a)
    elif a.robot_ip:
        if a.home:
            move_home(a)
        if not a.checkpoint:
            raise SystemExit("--checkpoint is required")
        (out / "frames").mkdir(parents=True)
        run_robot(a, out)
    else:
        ap.error("choose --dataset, --camera_test or --robot_ip")

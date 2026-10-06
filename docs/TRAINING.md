# Training pi0.5 on a TRumi dataset (GPU server)

How the conveyor-pick policies were trained (2026-10-05), so the next run is the same recipe. Dataset preparation is in
`docs/POST_PROCESSING.md`.

## Where
- Server `zerogrid2` (ssh alias): 2x RTX PRO 6000 Blackwell (96 GB), 40 cores, 251 GB RAM.
- Environment `~/ur5e-lerobot/.venv` (LeRobot 0.6.2). `lerobot/pi05_base` and PaliGemma are in the Hugging Face cache.
- Datasets copied to `~/datasets/<name>` (rsync, then md5 of every file compared on both machines).

## Recipe (`~/trumi_runs/train_<variant>.sh`)
```bash
cd ~/ur5e-lerobot
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8
.venv/bin/lerobot-train \
  --dataset.repo_id=formic/<name> --dataset.root=$HOME/datasets/<name> \
  --dataset.video_backend=pyav --dataset.image_transforms.enable=true \
  --policy.path=lerobot/pi05_base --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true --policy.compile_model=true --policy.push_to_hub=false \
  --rename_map="{\"observation.images.wrist\": \"observation.images.left_wrist_0_rgb\"}" \
  --batch_size=16 --steps=40000 --save_freq=10000 --log_freq=100 \
  --wandb.enable=true --wandb.project=trumi --wandb.disable_artifact=true \
  --output_dir=outputs/<run> --job_name=<run>
```
- 40k x 16 = 640k samples, as the team's earlier UR5e run and openpi's `pi05_droid_finetune` (20k x 32).
- Without gradient checkpointing batch 16 runs out of GPU memory (96 GB); with it ~70 GB, 1.28 s/step, ~14 h.
- The torchcodec / libavutil message at start is a warning only (video is decoded with pyav).
- One wrist camera goes into pi0.5's `left_wrist_0_rgb` slot (team convention); the other two slots are masked empty.
- Data loading waits ~0.001 s per 1.3 s step: two runs on the two GPUs use ~2-3 of 40 cores.
- Always run a 200-step smoke test first (`--steps=200 --save_checkpoint=false`).

## Overnight safety
- **Watchdog on the server** (`scripts/tools/train_watchdog.py`, copied to `~/trumi_runs/`, tmux `trumi-watchdog`):
  every 5 min; on a resource failure (GPU/system memory, disk, killed, CUDA fault, no progress for 30 min) it stops
  run v2 and restarts run v1 alone, resuming from its latest checkpoint (`--config_path=<run>/checkpoints/last/
  pretrained_model/train_config.json --resume=true`). Other errors are logged, not retried. Events:
  `~/trumi_runs/watchdog.log`, current state: `~/trumi_runs/status.txt`.
- **Checkpoint backup on the desktop** (`scripts/tools/checkpoint_backup.py`, tmux `trumi-backup`): every 10 min
  copies each finished checkpoint (24 GB: 8.8 GB model + 15 GB optimizer state) to `data/checkpoints/<run>/<step>/`,
  compares every file's md5, and only then deletes older checkpoints on the server, which keeps just the latest
  (needed to resume). A checkpoint still being written (newer than `last`, or changed in the last 5 min) is never
  touched. Log: `data/checkpoints/backup.log`.

## If something goes wrong
- **Resource failure** (out of memory, disk full, killed, hung): the watchdog stops v2 and resumes v1 by itself.
- **Resume any run by hand** from its latest checkpoint (keeps the same output folder and wandb run settings):
  ```bash
  ssh zerogrid2
  cd ~/ur5e-lerobot
  tmux new-session -d -s trumi-v1 "bash -c 'CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 .venv/bin/lerobot-train \
    --config_path=outputs/pi05_trumi_conveyor_v1_finger_mask_b16_40k/checkpoints/last/pretrained_model/train_config.json \
    --resume=true 2>&1 | tee -a ~/logs_pi05_trumi_v1_finger_mask.log; exec bash'"
  ```
  (v2: GPU 1, `pi05_trumi_conveyor_v2_fingers_visible_b16_40k`, `~/logs_pi05_trumi_v2_fingers_visible.log`.)
- **Failed before its first checkpoint** (step < 10k): start the script again (`~/trumi_runs/train_v1_finger_mask.sh`)
  after moving the old output folder aside (LeRobot refuses an existing output folder).
- **wandb**: project `trumi` (team `msaryan-formic`), logs only: `--wandb.disable_artifact=true` stops LeRobot from
  uploading every checkpoint's 8.8 GB model.
- **Desktop backup stopped** (e.g. the desktop restarted): `tmux new-session -d -s trumi-backup "python3
  scripts/tools/checkpoint_backup.py; exec bash"` from ~/trumi; it picks up where it left off.

## Checking on it
```bash
ssh zerogrid2 cat ~/trumi_runs/status.txt          # both runs: step, s/step, time left, disk
ssh zerogrid2 tail ~/trumi_runs/watchdog.log       # failures / restarts
tail ~/trumi/data/checkpoints/backup.log           # backups and pruning
ssh zerogrid2 -t tmux attach -t trumi-v1           # live training output (Ctrl-b d to detach)
```

## Running a trained policy on the robot (`scripts/robot/run_policy.py`)
- **Camera: one USB-C cable**, no capture card. The HERO13's USB networking ("GoPro Connect") gives an Open GoPro
  preview stream (UDP, HEVC 1920x1440 4:3, 30 fps). Finger-tag positions in the preview match the recorded 2.7K video
  within 5 px of 1920 (< 2 px at the policy's 640x480). The preview does not stream while the camera records.
- **Environment**: `~/YAM/yam-lerobot/.venv` (LeRobot 0.6.0, ur_rtde, CUDA). Checkpoints from the training server's
  LeRobot 0.6.2 load after dropping six newer config settings that are switched off in our runs (the script refuses if
  they are on) and pointing the tokenizer at the checkpoint folder. Seeded predictions agree with 0.6.2 within 0.47 deg.
- **This PC (RTX 5080, 16 GB)**: load 48 s (peak 18 GB system RAM), 9.9 GB GPU, ~154 ms per 50-action chunk without
  torch.compile (`--compile` gives the training speed after ~6 min of warm-up).
- **Offline check** (no robot): `--dataset data/lerobot/<name>` compares predicted chunks with the recorded actions
  (40k v1: 0.8-0.9 deg on training frames, i.e. the pipeline is wired right; it says nothing about generalisation).
- **Shadow mode** (default with `--robot_ip`): live camera + robot state, the policy predicts and logs, nothing moves
  (the gripper is only read). `--execute` moves the robot: joint-speed limit, refuses chunks that start far from the
  robot or reach below the marker plane, Ctrl+C stops.
- **Mask must match the checkpoint**: v1 = `policy_mask_2704x2028.png` (default), v2 (fingers visible) =
  `--mask data/robot/policy_mask_gripper_only_2704x2028.png`.
- **Smooth motion: use RTC** (`--rtc_horizon 25`). Without it every new chunk (each 0.5 s) is an independent sample and
  the target jumps where it takes over (robot 1, 90 deg/s: median 2.9 deg, max 16 deg, velocity up to 70 deg/s).
  Real-time chunking (LeRobot's inference-time guidance, no retraining) steers each new chunk to continue the rest of
  the old one, on the old chunk's timeline. Offline on recorded episodes (`scripts/tools/rtc_seam_test.py`): seam
  jumps 0.6 -> 0.1 deg median, 1.9 -> 0.2 deg max, velocity 47 -> 6 deg/s max, for +21 ms per prediction.
- **What worked on robot 1 (2026-10-06): continuous replanning** (no RTC, no sync), which was jerky. Smooth it with
  `--blend_s 0.25`: each new chunk is crossfaded in over 0.25 s (smoothstep) and then used fully, so what the policy
  predicts is unchanged. Replaying the logged run through the controller: velocity jumps 114 -> 16 deg/s (99th pct),
  peak acceleration 14,300 -> 2,000 deg/s^2. RTC (horizon 25) was smooth live but kept closing at one spot; sync
  (`--sync_steps`) also did not work for the user.
- **Camera latency 73 ms** (HERO13 USB preview, `scripts/robot/camera_latency.py`: the gripper opens/closes, its
  reported position vs the finger tags in the live video; 8 moves, 48-101 ms). It was 0.57 s until 2026-10-06: FFmpeg
  frame-threaded HEVC decoding (16 threads on this PC) holds back 15 frames = 0.5 s at 30 fps; `GoProPreview` now uses
  slice threading (no frames held back, 2.7K still decodes at ~110 fps). Check any new decoding code for this.
  `run_policy.py --camera_latency_s 0.07` (default) pairs each frame with the robot state and gripper command from when
  it was taken (state history at 100 Hz), places the chunk on that timeline (actions already in the past are skipped)
  and checks safety against the action due now. With 0.57 s that left ~23 of 50 actions in the past on arrival and a
  chunk was refused (24 deg from the robot). UMI's HDMI path (Media Mod + Cam Link 4K) is set to 0.17 s in their code,
  so a capture card would not be faster than the fixed USB preview.
- **Jump check per joint** (`--max_jump_deg`, default 57 44 23 49 24 56 deg, base .. wrist roll): a new chunk is
  refused if its target due now is further than this from the robot. Each limit is 1.2 x the 99th percentile of that
  joint's change over 0.5 s (one replan) in the human demos. The old flat 20 deg refused legitimate re-plans on robot 1
  (a 22 deg shoulder dive for a newly placed cup; a 38 deg wrist roll over the box at release); of 122 logged
  hand-overs on 2026-10-06 the per-joint limits refuse none. The speed cap and crossfade still bound how fast any jump
  is executed.
- **First working run (2026-10-06, 111103)**: 40k v1, `--blend_s 0.35 --camera_latency_s 0.07
  --max_joint_speed_deg_s 180`: 44.6 s, 5 grasps, 4 releases into the box before the old 20 deg check stopped it.
- **Floor: hold, don't stop** (`--min_height_mm`, default 0 = marker plane). The demos set cups down on the box floor
  (box on the marker's table, floor ~3 mm above the plane; demo fingertip down to -1.1 cm there, within calibration
  error), so the policy sometimes plans the end of a chunk slightly below the plane. Such targets are now held at the last
  target above the floor (gripper values kept, so it still releases) instead of stopping the run. Both floor stops on
  2026-10-06 (-4 and -8 mm, 1.2-1.6 s ahead in the chunk) pass this way; of 715 logged chunks none is refused.
- **Per-joint run (2026-10-06, 111930)**: same settings with the per-joint jump limits: 193 s, 18 grasps, 18 releases,
  ended by the old floor stop.

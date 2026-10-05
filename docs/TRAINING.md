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

## Checking on it
```bash
ssh zerogrid2 cat ~/trumi_runs/status.txt          # both runs: step, s/step, time left, disk
ssh zerogrid2 tail ~/trumi_runs/watchdog.log       # failures / restarts
tail ~/trumi/data/checkpoints/backup.log           # backups and pruning
ssh zerogrid2 -t tmux attach -t trumi-v1           # live training output (Ctrl-b d to detach)
```

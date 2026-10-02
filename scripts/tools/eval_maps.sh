#!/usr/bin/env bash
# Evaluate candidate maps against a session's demos, so you can pick the best mapping take.
# A good map passes BOTH tests:
#   1) the session's demos localize in it (pipeline step 03), and
#   2) the mapping-marker calibration is tight (the "Tag position std" that step 05 prints; ~1-2 cm is good).
#
# Usage (from ~/trumi):
#   bash scripts/tools/eval_maps.sh <session_dir> <candidate_dir> [<candidate_dir> ...]
#     session_dir   : a session already processed through step 01 (demos/*/raw_video.mp4 + imu_data.json)
#     candidate_dir : a folder made by test_mappings.sh with raw_video.mp4, imu_data.json and map_atlas.osa
# Example:
#   bash scripts/tools/eval_maps.sh data/2026-10-01_conveyor_bimanual data/_mapping_tests/conveyor/demos/left_mapping{1,2,3}
#
# Work happens in data/_map_eval/<session>/<candidate>/ ; the session itself is not modified.
set -u
if [ $# -lt 2 ]; then
  sed -n '2,15p' "$0"
  exit 1
fi
cd ~/trumi || exit 1
SESSION=${1%/}
shift
EVAL=data/_map_eval/$(basename "$SESSION")
CI=example/calibration/gopro13_intrinsics_2_7k.json
AC=example/calibration/aruco_config.yaml

# 1) set up one folder per candidate (demo videos + IMU copied fresh) and detect tags in its mapping video
for C in "$@"; do
  C=${C%/}
  name=$(basename "$C")
  E=$EVAL/$name/demos
  M=$E/mapping_$name
  mkdir -p "$M"
  for d in "$SESSION"/demos/demo_*/; do
    n=$(basename "$d")
    mkdir -p "$E/$n"
    cp --update=none "$d/raw_video.mp4" "$d/imu_data.json" "$E/$n/"
  done
  cp --update=none "$C/raw_video.mp4" "$C/imu_data.json" "$C/map_atlas.osa" "$M/"
  (
    [ -f "$M/tag_detection.pkl" ] || uv run python scripts/detect_aruco.py -i "$M/raw_video.mp4" -o "$M/tag_detection.pkl" \
      -ci $CI -ac $AC -n 4 -sfs 2 > "$EVAL/$name/detect_aruco.log" 2>&1
  ) &
done
wait

# 2) pipeline step 03 against each candidate map (one candidate at a time; step 03 itself runs videos in parallel)
for C in "$@"; do
  name=$(basename "${C%/}")
  E=$EVAL/$name/demos
  echo "step 03 with map from $name ..."
  uv run python scripts/scripts_slam_pipeline/03_batch_slam.py --input_dir "$E" \
    --map_path "$E/mapping_$name/map_atlas.osa" > "$EVAL/$name/step03.log" 2>&1
done

# 3) marker calibration exactly as step 05 does it, then summary
echo
echo "=== SUMMARY (good map: all demos tracked AND marker spread ~1-2 cm) ==="
for C in "$@"; do
  name=$(basename "${C%/}")
  E=$EVAL/$name/demos
  M=$E/mapping_$name
  spread="n/a (mapping video did not localize)"
  if [ -f "$M/camera_trajectory.csv" ]; then
    spread=$(uv run python scripts/calibrate_slam_tag.py --tag_detection "$M/tag_detection.pkl" \
      --csv_trajectory "$M/camera_trajectory.csv" --output "$EVAL/$name/tx_slam_tag.json" --keyframe_only 2>&1 \
      | grep -o '\[.*\]' | tail -1)
  fi
  tracked=$(uv run python - "$E" <<'EOF'
import glob, sys, pandas as pd
E = sys.argv[1]; demos = sorted(glob.glob(f"{E}/demo_*")); ok, worst = 0, 100.0
for d in demos:
    try:
        df = pd.read_csv(f"{d}/camera_trajectory.csv")
        pct = 100 * (~df["is_lost"].astype(str).str.lower().eq("true")).mean(); ok += 1; worst = min(worst, pct)
    except FileNotFoundError:
        worst = 0.0
print(f"demos localized {ok}/{len(demos)} (worst {worst:.0f}% tracked)")
EOF
)
  echo "$name: $tracked | marker spread (cm) $spread"
done

#!/usr/bin/env bash
# Test-build a SLAM map from each candidate mapping video, exactly like pipeline steps 01 + 02,
# so you can pick a take that works before running the full pipeline.
# Each video gets its own folder <out_dir>/demos/<parent>_<name>/. Originals are only copied, never changed.
#
# Usage (from ~/trumi):
#   bash scripts/tools/test_mappings.sh <out_dir> <mapping1.MP4> [<mapping2.MP4> ...]
# Example:
#   bash scripts/tools/test_mappings.sh data/_mapping_tests ~/Downloads/test2-conveyer/*/mapping*.MP4
#
# Use a take whose summary line says "MAP BUILT" and "tag_init_success=1" (map started from the marker).
set -u
if [ $# -lt 2 ]; then
  sed -n '2,11p' "$0"
  exit 1
fi
cd ~/trumi || exit 1
OUT=$1
shift

# 1) copy each mapping video into its own folder as raw_video.mp4
for v in "$@"; do
  name="$(basename "$(dirname "$v")")_$(basename "${v%.*}")"
  d="$OUT/demos/$name"
  mkdir -p "$d"
  cp --update=none "$v" "$d/raw_video.mp4"
done

# 2) pipeline step 01: extract IMU -> imu_data.json in each folder
uv run python scripts/scripts_slam_pipeline/01_extract_gopro_imu.py "$OUT"

# 3) pipeline step 02: build a map from each video (in parallel), log per folder
for d in "$OUT"/demos/*/; do
  (
    uv run python scripts/scripts_slam_pipeline/02_create_map.py \
      --input_dir "$d" --map_path "$d/map_atlas.osa" > "$d/create_map.log" 2>&1
    rc=$?   # capture before anything else overwrites $?
    echo "$(basename "$d"): step 02 exit code $rc"
  ) &
done
wait

# 4) summary
echo
echo "=== SUMMARY ==="
for d in "$OUT"/demos/*/; do
  n=$(basename "$d")
  if [ -f "$d/map_atlas.osa" ]; then map="MAP BUILT"; else map="NO MAP"; fi
  init=$(grep -m1 -o -E "tag_init_success=1|init_success=1" "$d/slam_stdout_mapping.txt" 2>/dev/null)
  echo "$n: $map | first map start: ${init:-none}"
done

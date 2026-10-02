#!/usr/bin/env bash
# Set up the simulation tools (scripts/sim/*.py) in data/sim/ (not versioned):
#   data/sim/.venv              Python venv with MuJoCo and helpers (separate from the TRumi pipeline venv)
#   data/sim/mujoco_menagerie   UR5e + Robotiq 2F-85 models from google-deepmind/mujoco_menagerie (pinned commit)
#   data/sim/scans              converted phone scans (created by view_scan.py / view_twin.py)
# Safe to re-run; existing pieces are kept.
#
# Usage (from ~/trumi):  bash scripts/sim/setup_sim.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
SIM=data/sim
MENAGERIE_COMMIT=4d038b3feae26ec82b46a4d586379114012a8ac7
mkdir -p "$SIM"

if [ ! -x "$SIM/.venv/bin/python" ]; then
  uv venv "$SIM/.venv" --python 3.12 -q
fi
VIRTUAL_ENV="$PWD/$SIM/.venv" uv pip install -q mujoco==3.14.0 numpy==2.5.3 scipy==1.18.1 av==18.1.0

if [ ! -d "$SIM/mujoco_menagerie" ]; then
  git clone -q --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git "$SIM/mujoco_menagerie"
  git -C "$SIM/mujoco_menagerie" sparse-checkout set universal_robots_ur5e robotiq_2f85
  git -C "$SIM/mujoco_menagerie" checkout -q "$MENAGERIE_COMMIT"
fi

"$SIM/.venv/bin/python" -c "import mujoco; print('MuJoCo', mujoco.__version__, 'ready')"
for m in universal_robots_ur5e robotiq_2f85; do
  [ -f "$SIM/mujoco_menagerie/$m/scene.xml" ] && echo "model ready: $m (menagerie $(git -C "$SIM/mujoco_menagerie" rev-parse --short HEAD))"
done

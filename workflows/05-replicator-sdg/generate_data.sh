#!/usr/bin/env bash
# Three Replicator runs, one per distractor set, 2000 frames each at 960x544 (Omniverse Basic Training 05, 19:28-21:43).
#
#   ISAAC_SIM_PATH=~/isaacsim OUT=$PWD/data bash generate_data.sh
#
# On Ubuntu one script runs all three; on Windows the tutorial needed one .bat per set because Isaac Sim did not
# relaunch after the first run (21:51-23:12).
#
# Pinned: Isaac Sim 5.0.0 workstation install (./python.sh, Python 3.11), Ubuntu 22.04, RTX GPU.
# not executed: needs GPU.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ISAAC_SIM_PATH=${ISAAC_SIM_PATH:-$HOME/isaacsim}
OUT=${OUT:-$HERE/data}
NUM_FRAMES=${NUM_FRAMES:-2000}
WIDTH=${WIDTH:-960}
HEIGHT=${HEIGHT:-544}

mkdir -p "$OUT/distractors_warehouse" "$OUT/distractors_additional" "$OUT/no_distractors"
cd "$ISAAC_SIM_PATH"

./python.sh "$HERE/replicator_pallet_jack.py" --height "$HEIGHT" --width "$WIDTH" --num_frames "$NUM_FRAMES" \
  --distractors warehouse --data_dir "$OUT/distractors_warehouse"
./python.sh "$HERE/replicator_pallet_jack.py" --height "$HEIGHT" --width "$WIDTH" --num_frames "$NUM_FRAMES" \
  --distractors additional --data_dir "$OUT/distractors_additional"
./python.sh "$HERE/replicator_pallet_jack.py" --height "$HEIGHT" --width "$WIDTH" --num_frames "$NUM_FRAMES" \
  --distractors none --data_dir "$OUT/no_distractors"

# Dataset check and the 14% split (CPU, no NVIDIA software):
python3 "$HERE/kitti_tools.py" stats --root "$OUT"
python3 "$HERE/kitti_tools.py" split --root "$OUT" --val 0.14 --out "$OUT/splits"

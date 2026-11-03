#!/usr/bin/env bash
# Evaluate a trained policy with Isaac Lab's rsl_rl play script (tutorial #12, 06:13-07:13; tutorial #14, 11:53-12:59).
#
#   bash play.sh g1                                   # built-in task: latest checkpoint is picked automatically
#   bash play.sh push  logs/rsl_rl/atlas_push_object_g1/<run>/model_1500.pt
#   bash play.sh anymal logs/rsl_rl/anymal_c_flat/<run>/model_<iter>.pt
#
# Gotcha (tutorial #14, 12:10-12:21): for a custom task, play without --checkpoint loads "but the robot is not moving".
# This script refuses to run the custom task without a checkpoint path.
# Side effect: play.py exports exported/policy.onnx and exported/policy.pt next to the checkpoint (see export_onnx.sh).
#
# Pinned: Isaac Sim 5.0.0, Isaac Lab v2.2.0 (verify), Python 3.11. not executed: needs GPU.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ISAACLAB_PATH=${ISAACLAB_PATH:-$HOME/IsaacLab}
export ISAACLAB_PATH
PLAY=scripts/reinforcement_learning/rsl_rl/play.py
MODE=${1:-push}
CKPT=${2:-}

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-env_isaaclab}"

case "$MODE" in
  g1)
    cd "$ISAACLAB_PATH"
    python "$PLAY" --task Isaac-Velocity-Rough-G1-v0 --num_envs 5 ${CKPT:+--checkpoint "$CKPT"}
    ;;
  anymal)
    cd "$ISAACLAB_PATH"
    python "$PLAY" --task Isaac-Velocity-Flat-Anymal-C-Play-v0 --num_envs 5 ${CKPT:+--checkpoint "$CKPT"}
    ;;
  push)
    if [ -z "$CKPT" ]; then
      echo "custom task: pass the checkpoint, e.g. bash play.sh push logs/rsl_rl/atlas_push_object_g1/<run>/model_1500.pt"
      exit 2
    fi
    python "$HERE/scripts/run_custom_task.py" play --task Atlas-PushObject-G1-Play-v0 --num_envs 5 --checkpoint "$CKPT"
    ;;
  *)
    echo "usage: bash play.sh {g1|anymal|push} [checkpoint.pt]"; exit 2
    ;;
esac

#!/usr/bin/env bash
# Train with Isaac Lab's rsl_rl script (tutorials #12 and #14).
#
#   bash train.sh spot          # tutorial #12: Spot on flat ground, GUI, small env count
#   bash train.sh g1            # tutorial #12: G1 on rough terrain, 10000 envs, headless, to 500 iterations
#   bash train.sh anymal        # ANYmal C flat: the policy that sim_to_sim.md moves to Newton
#   bash train.sh push-debug    # tutorial #14: custom task with 4 envs and the GUI
#   bash train.sh push          # custom task at scale, headless
#
# Pinned: Isaac Sim 5.0.0, Isaac Lab v2.2.0 (verify), Python 3.11 in conda env env_isaaclab, rsl_rl.
# Checkpoints: $ISAACLAB_PATH/logs/rsl_rl/<experiment_name>/<timestamp>/model_<iter>.pt, every 50 iterations.
# not executed: needs GPU.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ISAACLAB_PATH=${ISAACLAB_PATH:-$HOME/IsaacLab}
export ISAACLAB_PATH
TRAIN=scripts/reinforcement_learning/rsl_rl/train.py
MODE=${1:-push-debug}

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-env_isaaclab}"

case "$MODE" in
  spot)
    cd "$ISAACLAB_PATH"
    python "$TRAIN" --task Isaac-Velocity-Flat-Spot-v0 --num_envs 512
    ;;
  g1)
    cd "$ISAACLAB_PATH"
    python "$TRAIN" --task Isaac-Velocity-Rough-G1-v0 --num_envs 10000 --headless --max_iterations 500
    ;;
  anymal)
    cd "$ISAACLAB_PATH"
    python "$TRAIN" --task Isaac-Velocity-Flat-Anymal-C-v0 --num_envs 4096 --headless
    ;;
  push-debug)
    python "$HERE/scripts/run_custom_task.py" train --task Atlas-PushObject-G1-v0 --num_envs 4
    ;;
  push)
    python "$HERE/scripts/run_custom_task.py" train --task Atlas-PushObject-G1-v0 --num_envs 4096 --headless
    ;;
  *)
    echo "usage: bash train.sh {spot|g1|anymal|push-debug|push}"; exit 2
    ;;
esac

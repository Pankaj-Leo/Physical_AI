#!/usr/bin/env bash
# Export a trained rsl_rl policy to ONNX and check its I/O widths, ready for sim-to-sim in Newton (sim_to_sim.md).
#
#   bash export_onnx.sh anymal logs/rsl_rl/anymal_c_flat/<run>/model_<iter>.pt
#   bash export_onnx.sh push   logs/rsl_rl/atlas_push_object_g1/<run>/model_1500.pt
#
# How: Isaac Lab's play.py writes <run>/exported/policy.onnx and policy.pt before it starts the rollout loop
# (verify: Isaac Lab v2.2 play.py calls export_policy_as_onnx / export_policy_as_jit). Recording a short video makes
# play.py exit on its own instead of looping until the window closes. The tutorials stop at play.py; export is not
# shown in them (digest 02, section 2.9 lists "no policy export" as a gap).
#
# Pinned: Isaac Sim 5.0.0, Isaac Lab v2.2.0 (verify), Python 3.11. not executed: needs GPU.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ISAACLAB_PATH=${ISAACLAB_PATH:-$HOME/IsaacLab}
export ISAACLAB_PATH
MODE=${1:?usage: bash export_onnx.sh {anymal|push} <checkpoint.pt>}
CKPT=${2:?pass the checkpoint path}
OUT=${OUT:-$HERE/artifacts}

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-env_isaaclab}"
mkdir -p "$OUT"

case "$MODE" in
  anymal)
    (cd "$ISAACLAB_PATH" && python scripts/reinforcement_learning/rsl_rl/play.py \
      --task Isaac-Velocity-Flat-Anymal-C-Play-v0 --num_envs 1 --headless --video --video_length 50 \
      --checkpoint "$CKPT")
    OBS=48; ACT=12; NAME=anymal_c_flat_policy.onnx
    ;;
  push)
    python "$HERE/scripts/run_custom_task.py" play --task Atlas-PushObject-G1-Play-v0 --num_envs 1 --headless \
      --video --video_length 50 --checkpoint "$CKPT"
    OBS=${OBS:-28}; ACT=${ACT:-5}; NAME=push_object_g1_policy.onnx   # 3n+13 and n for n arm joints
    ;;
  *)
    echo "usage: bash export_onnx.sh {anymal|push} <checkpoint.pt>"; exit 2
    ;;
esac

case "$CKPT" in /*) RUN_DIR=$(dirname "$CKPT") ;; *) RUN_DIR=$(dirname "$ISAACLAB_PATH/$CKPT") ;; esac
cp "$RUN_DIR/exported/policy.onnx" "$OUT/$NAME"
python "$HERE/scripts/check_onnx_policy.py" "$OUT/$NAME" --obs "$OBS" --act "$ACT"
echo "wrote $OUT/$NAME"

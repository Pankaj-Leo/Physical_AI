#!/usr/bin/env bash
# Install Isaac Lab on top of the Isaac Sim pip package in a conda env (tutorial #3, 01:01-04:39).
#
# Pinned (the tutorials use Isaac Sim 5.0.0; the transcripts do not state an Isaac Lab tag, CUDA, or driver version):
#   Ubuntu 22.04 x86_64, RTX GPU, NVIDIA driver from the Isaac Sim 5.0.0 requirements page
#   Python 3.11 (conda env "env_isaaclab", the name tutorial #3 shows)
#   isaacsim[all,extscache]==5.0.0 from pypi.nvidia.com
#   Isaac Lab tag v2.2.0                        # verify: the Isaac Lab release paired with Isaac Sim 5.0
#   torch 2.7.0 + cu128                         # verify: the wheel the Isaac Lab v2.2 pip-install page lists
#
# not executed: needs GPU. Run on the GPU box from any directory.
set -euo pipefail

ISAACSIM_VERSION=${ISAACSIM_VERSION:-5.0.0}
ISAACLAB_REF=${ISAACLAB_REF:-v2.2.0}
ISAACLAB_PATH=${ISAACLAB_PATH:-$HOME/IsaacLab}
ENV_NAME=${ENV_NAME:-env_isaaclab}

# Step 1 (tutorial #3, 01:27-01:60): system update and Miniconda are assumed done; `conda` must be on PATH.
command -v conda >/dev/null || { echo "install Miniconda first"; exit 1; }
nvidia-smi >/dev/null || { echo "no NVIDIA driver visible"; exit 1; }

# Step 2 (02:03-02:22): create and activate the env.
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n "$ENV_NAME" python=3.11
conda activate "$ENV_NAME"

# Step 3 (02:22-02:32): upgrade pip.
pip install --upgrade pip

# Step 4 (02:32-02:44): PyTorch (CUDA build; slow).
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# Step 5 (02:44-03:16): the Isaac Sim pip package, "not the full Isaac Sim installation".
pip install "isaacsim[all,extscache]==${ISAACSIM_VERSION}" --extra-index-url https://pypi.nvidia.com
echo "Optional check: run 'isaacsim' once; if the GUI opens, close it (tutorial #3, 03:00-03:16)."

# Step 6 (03:24-04:01): clone Isaac Lab with the env active and install it into the same env.
if [ ! -d "$ISAACLAB_PATH" ]; then
  git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_PATH"
fi
cd "$ISAACLAB_PATH"
git checkout "$ISAACLAB_REF"
./isaaclab.sh --install

# Step 7 (04:01-04:39): verify with a quadruped example; the tutorial removes --headless to watch it train.
echo "Verify: python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Rough-Anymal-C-v0 --num_envs 64"

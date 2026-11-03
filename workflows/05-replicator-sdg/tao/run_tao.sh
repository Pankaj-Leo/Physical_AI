#!/usr/bin/env bash
# TAO fine-tune of DetectNet_v2 (ResNet-18) on the Replicator pallet-jack data (Omniverse Basic Training 06).
# The tutorial drives these steps from the NVIDIA repo's Jupyter notebook; this is the same sequence as a script.
#
#   export LOCAL_PROJECT_DIR=$PWD/..          # the folder that holds data/ (tutorial: "set up this variable", 10:15)
#   export NGC_API_KEY=...                    # NGC -> Setup -> Generate API key (18:24-19:06)
#   bash run_tao.sh
#
# Prerequisites (tutorial order, 13:13-22:45): Docker, NVIDIA Container Toolkit (nvidia-ctk runtime configure, restart
# Docker; nvidia-smi in a container shows the GPU), docker login nvcr.io as user $oauthtoken, pip install nvidia-tao.
# Pinned: Ubuntu 22.04, Python 3.10 conda env (the tutorial's TAO sandbox), TAO Toolkit 5.x launcher, one RTX GPU.
# not executed: needs GPU and an NGC account.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Tutorial gotcha (10:55-12:07, 23:22-25:01): the notebook lost this variable; re-check it before every step.
: "${LOCAL_PROJECT_DIR:?set LOCAL_PROJECT_DIR to the folder that holds data/}"
: "${NGC_API_KEY:?set NGC_API_KEY}"
WS=/workspace/tao-experiments
SETS=(distractors_warehouse distractors_additional no_distractors)

# Mount LOCAL_PROJECT_DIR into the TAO containers at /workspace/tao-experiments.
if [ ! -f "$HOME/.tao_mounts.json" ]; then
  cat > "$HOME/.tao_mounts.json" <<EOF
{"Mounts": [{"source": "$LOCAL_PROJECT_DIR", "destination": "$WS"}],
 "DockerOptions": {"user": "$(id -u):$(id -g)"}}
EOF
fi
mkdir -p "$LOCAL_PROJECT_DIR/specs" "$LOCAL_PROJECT_DIR/data/tfrecords" "$LOCAL_PROJECT_DIR/detectnet_v2"
cp "$HERE"/detectnet_v2_train_resnet18_kitti.txt "$HERE"/detectnet_v2_inference_kitti.txt "$LOCAL_PROJECT_DIR/specs/"

# Docker login uses the literal user name $oauthtoken.
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin

# Step 1: pretrained DetectNet_v2 ResNet-18 from NGC.
ngc registry model download-version nvidia/tao/pretrained_detectnet_v2:resnet18 \
  --dest "$LOCAL_PROJECT_DIR/detectnet_v2/pretrained_resnet18"

# Step 2: KITTI -> TFRecords, one run per distractor set (holds out 14% as partition 0).
for SET in "${SETS[@]}"; do
  sed "s/__SET__/$SET/g" "$HERE/detectnet_v2_tfrecords_kitti.txt" > "$LOCAL_PROJECT_DIR/specs/tfrecords_$SET.txt"
  mkdir -p "$LOCAL_PROJECT_DIR/data/tfrecords/$SET"
  tao model detectnet_v2 dataset_convert \
    -d "$WS/specs/tfrecords_$SET.txt" \
    -o "$WS/data/tfrecords/$SET/$SET"
done

# Step 3-4: train (the tutorial's run: about 1 h 45 min on one GPU, mAP above 81%, as reported in the video).
tao model detectnet_v2 train \
  -e "$WS/specs/detectnet_v2_train_resnet18_kitti.txt" \
  -r "$WS/detectnet_v2/experiment_dir_unpruned" \
  --gpus 1

# Step 5: evaluate on the held-out 14%.
tao model detectnet_v2 evaluate \
  -e "$WS/specs/detectnet_v2_train_resnet18_kitti.txt" \
  -m "$WS/detectnet_v2/experiment_dir_unpruned/weights/resnet18_detector.hdf5"  # verify: weights file name

# Step 6: inference on real photos placed in $LOCAL_PROJECT_DIR/real_images.
tao model detectnet_v2 inference \
  -e "$WS/specs/detectnet_v2_inference_kitti.txt" \
  -i "$WS/real_images" \
  -r "$WS/detectnet_v2/inference_real"

#!/usr/bin/env bash
# GPU numbers for this workflow. Written for a CUDA GPU; not yet run (no CUDA machine was available).
# Expect the 16,384-world rows to be limited by device memory; watch mem_mb in the CSV.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
rm -f results/gpu_scale.csv
$PY scale_benchmark.py --device cuda:0 --solver mujoco_warp --worlds 256 1024 4096 16384 --graph --frames 200 \
    --warmup 10 --out results/gpu_scale.csv
$PY scale_benchmark.py --device cuda:0 --solver xpbd --worlds 256 1024 4096 16384 --graph --frames 200 \
    --warmup 10 --out results/gpu_scale.csv --append
# Clips for workflow 07: 16 worlds at 1280x720 (Cosmos Transfer's 720p input). Large; stays in out/.
$PY render_clips.py --device cuda:0 --worlds 16 --width 1280 --height 720 --frames 121 --out out/clips_720p
cp out/clips_720p/contact_sheet.png results/gpu_contact_sheet.png
cp out/clips_720p/render_summary.json results/gpu_render_summary.json
$PY plot_scale.py results/gpu_scale.csv results/gpu_scale.png

#!/usr/bin/env bash
# Reproduce every CPU number in results/ (about 3 minutes on 4 cores once Warp kernels are cached).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
rm -f results/cpu_scale.csv
# Required run: XPBD, 1/2/4 worlds, plain kernel launches (no graph capture).
$PY scale_benchmark.py --device cpu --solver xpbd --worlds 1 2 4 --no-graph --frames 50 --out results/cpu_scale.csv
# Extra: the same with a captured Warp graph (Warp 1.17 replays graphs on CPU too), and more worlds.
$PY scale_benchmark.py --device cpu --solver xpbd --worlds 1 2 4 16 64 --graph --frames 50 \
    --out results/cpu_scale.csv --append
# Extra: MuJoCo-Warp (the solver the talk's Isaac Lab numbers use), plain launches.
$PY scale_benchmark.py --device cpu --solver mujoco_warp --worlds 1 2 4 --no-graph --frames 20 \
    --out results/cpu_scale.csv --append
$PY scale_benchmark.py --device cpu --solver mujoco_warp --worlds 1 4 16 --graph --frames 20 \
    --out results/cpu_scale.csv --append
# Clips: 3 worlds, 160x90, 121 frames at 30 fps. PNG/MP4 stay in out/ (git-ignored); summaries are copied.
$PY render_clips.py --device cpu --worlds 3 --width 160 --height 90 --frames 121 --out out/clips
cp out/clips/contact_sheet.png out/clips/render_summary.json out/clips/world_001.json results/
$PY plot_scale.py results/cpu_scale.csv results/cpu_scale.png

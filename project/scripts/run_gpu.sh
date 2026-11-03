#!/usr/bin/env bash
# GPU run of Phases 0 and 1 on one CUDA machine. Written for a GPU; not yet run (no CUDA machine was available).
#
# Usage: scripts/run_gpu.sh [PYTHON]
#   Needs an NVIDIA GPU (24-48 GB for the full MPM sweep), a recent driver, and newton[examples] installed.
#   Writes results/phase0_gpu/ and results/phase1/. The CPU baseline in results/phase0/ is left untouched.
set -euo pipefail

PY="${1:-python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$HERE/results/phase0_gpu/logs" "$HERE/results/phase1/logs"

"$PY" - <<'PYCHECK'
import newton, warp as wp
wp.init()
assert wp.get_cuda_device_count() > 0, "Warp found no CUDA device: check the driver and the warp-lang build"
d = wp.get_device("cuda:0")
print(f"newton {newton.__version__}, warp {wp.__version__}, {d.name} ({d.total_memory / 2**30:.0f} GB), mempool={d.is_mempool_enabled}")
PYCHECK

echo "== Phase 0 on cuda:0: 6 solvers x 400 frames, CUDA graphs where the solver allows"
"$PY" -m ssw.bakeoff --device cuda:0 --frames 400 --out "$HERE/results/phase0_gpu" \
    --determinism xpbd mujoco_warp --determinism-frames 100 \
    --usd xpbd --usd-frames 150 2>&1 | tee "$HERE/results/phase0_gpu/logs/bakeoff.log"

echo "== Upstream mpm_anymal on cuda:0 (100 frames, headless)"
"$PY" -m newton.examples mpm_anymal --device cuda:0 --viewer null --num-frames 100 --quiet \
    2>&1 | tee "$HERE/results/phase1/logs/mpm_anymal_gpu.log"

echo "== Phase 1 sweep on cuda:0: 4 materials x 3 seeds x 500 frames, full resolution"
"$PY" -m ssw.terrain sweep --device cuda:0 --materials rigid sand snow clay --seeds 0 1 2 --frames 500 \
    --out "$HERE/results/phase1" 2>&1 | tee "$HERE/results/phase1/logs/sweep.log"

echo "done: results/phase0_gpu and results/phase1"

#!/usr/bin/env bash
# Phase 0 CPU baseline: the solver bake-off pinned to the Warp CPU device, plus the CPU smoke tests.
# This reproduces the committed results/phase0 numbers. The GPU run is scripts/run_gpu.sh.
#
# Usage: scripts/run_phase0.sh [PYTHON]
#   PYTHON defaults to `python`; it must have newton[examples] installed.
# Expected wall time on a 4-core x86 CPU: ~3 min with a warm Warp kernel cache, ~6 min cold.
set -euo pipefail

PY="${1:-python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$HERE/results/phase0"
LOGS="$OUT/logs"
mkdir -p "$LOGS"
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"

"$PY" -c "import newton, warp; print('newton', newton.__version__, 'warp', warp.__version__)"

echo "== bake-off: 6 solvers x 400 frames, determinism (xpbd, mujoco_warp), USD export (xpbd)"
"$PY" -m ssw.bakeoff --device cpu --frames 400 --out "$OUT" \
    --determinism xpbd mujoco_warp --determinism-frames 100 \
    --usd xpbd --usd-frames 150 2>&1 | tee "$LOGS/bakeoff.log"

echo "== ANYmal C walking policy on CPU (upstream example, 50 frames)"
set +e
timeout 300 "$PY" -m newton.examples robot_anymal_c_walk --device cpu --viewer null --num-frames 50 --quiet \
    > "$LOGS/anymal_cpu.log" 2>&1
echo "robot_anymal_c_walk exit code: $?" | tee -a "$LOGS/anymal_cpu.log"

echo "== mpm_anymal on CPU (upstream example; expected to refuse CPU)"
timeout 120 "$PY" -m newton.examples mpm_anymal --device cpu --viewer null --num-frames 5 --quiet \
    > "$LOGS/mpm_anymal_cpu.log" 2>&1
echo "mpm_anymal exit code: $?" | tee -a "$LOGS/mpm_anymal_cpu.log"
set -e

echo "== Phase 1 CPU smoke: ANYmal rigid baseline + tiny two-way MPM beds"
SMOKE="$HERE/results/phase1_cpu_smoke"
"$PY" -m ssw.terrain run --device cpu --material rigid --frames 150 --out "$SMOKE"
"$PY" -m ssw.terrain sweep --device cpu --tiny --materials rigid sand snow clay --seeds 0 --frames 50 --out "$SMOKE"
"$PY" -m ssw.terrain run --device cpu --material sand --tiny --coupling one_way --frames 50 --out "$SMOKE"

echo "done: $OUT"

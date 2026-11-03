# Sand, Solvers, and World Models

A legged robot in [Newton](https://github.com/newton-physics/newton), from a solver bake-off to granular
terrain to Cosmos 3 as a critic. Built for an NVIDIA GPU (`--device auto` picks `cuda:0`, CUDA graphs on); so far
only the CPU path has been run. GPU commands: [`../RUN_ON_GPU.md`](../RUN_ON_GPU.md). This is the hands-on project planned in
`../planning/02-project-plan.md`; the thesis is "evaluate somewhere other than where you generated."

| Phase | What | Built for | Run so far |
|---|---|---|---|
| 0 | Solver bake-off on `quadruped.urdf`: settle height, energy, ms/step, determinism, USD export | Any CUDA GPU | CPU baseline **executed** (results below); GPU run not yet |
| 1 | ANYmal C walking policy on rigid ground vs MPM sand / snow / clay | 1x 24–48 GB CUDA GPU | Rigid baseline and tiny MPM smoke test **executed on CPU**; full sweep not yet run on GPU |
| 2 | Cosmos3-Nano Reasoner as judge; Cosmos 3 transfer with depth/seg controls | 24 GB GPU (Reasoner), 80 GB GPU (Generator) | Request building, reply parsing and scoring **tested offline**; no model calls made |

## Install

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -e ".[dev,cosmos]"      # pulls newton[examples]; see the pin note in pyproject.toml
```

Tested with Newton 1.7.0.dev0 installed editable from the local `newton-main` snapshot, Warp 1.17.0, MuJoCo 3.12.0,
mujoco-warp 3.12.0, Python 3.11. Newton releases monthly and renames things, so pin a release when 1.7 ships.

## Run

```bash
# GPU: Phase 0 + Phase 1 sweep on cuda:0 -> results/phase0_gpu, results/phase1 (not yet run)
scripts/run_gpu.sh

# Phase 0 CPU baseline (about 3 min with a warm Warp kernel cache) -> results/phase0
scripts/run_phase0.sh                       # or: python -m ssw.bakeoff --device cpu --out results/phase0
python -m ssw.bakeoff --replot --out results/phase0   # redraw plots from the CSVs

# Phase 1
python -m ssw.terrain materials                                   # material table and its sources
python -m ssw.terrain sweep --materials rigid sand snow clay --seeds 0 1 2 --frames 500 \
    --out results/phase1                                          # GPU (auto -> cuda:0): not yet run
python -m ssw.terrain run --device cpu --material rigid --frames 150        # CPU: ANYmal on rigid ground
python -m ssw.terrain run --device cpu --material sand --tiny --frames 50   # CPU: 462-particle smoke test

# Phase 2 (offline parts run anywhere)
python -m ssw.judge ask --manifest clips.jsonl --dry-run          # print the exact Reasoner requests
python -m ssw.judge ask --manifest clips.jsonl --base-url http://HOST:8000/v1 --out verdicts.jsonl   # needs a server
python -m ssw.judge score --verdicts verdicts.jsonl               # precision/recall vs simulator ground truth
python -m ssw.transfer --depth d.mp4 --seg s.mp4 --material sand --dry-run --out transfer_requests/

# Checks
pytest -q              # CPU tests; Newton smoke tests run a few frames, GPU tests skip without CUDA
pytest -q -m gpu       # on a GPU machine: CUDA-graph replay matches eager stepping
ruff check . && ruff format --check .
```

## Phase 0 results (CPU, 2026-10-03)

Quadruped dropped from 0.7 m, PD drives holding a crouch, 100 fps x 10 substeps (dt = 1 ms), 4 s, one world,
4-core Xeon at 2.1 GHz. Details, caveats and the determinism check are in [`docs/phase0-report.md`](docs/phase0-report.md).

| Solver path | Stable | Settle height [m] | Settle time [s] | Final KE [J] | Energy drift [J/s] | ms / 10 ms frame (median) |
|---|---|---|---|---|---|---|
| XPBD | yes | 0.4598 | 1.05 | 0 | +2e-5 | 14.1 |
| Featherstone | yes | 0.4492 | 0.65 | 0.48 | +0.076 | 17.8 |
| SemiImplicit | yes | 0.4324 | 1.85 | 1.68 | −0.15 | 8.8 |
| MuJoCo (mujoco_warp on Warp CPU) | yes | 0.4592 | 0.58 | 4e-11 | −2e-8 | 57.0 |
| MuJoCo (`use_mujoco_cpu=True`, MuJoCo C) | yes | 0.4592 | 0.58 | 7e-11 | −4e-7 | 5.7 |
| VBD (`rigid_compliant_alm=True`) | yes | 0.4584 | 1.02 | 9e-7 | −0.003 | 22.8 |

- All six paths run on CPU. XPBD, MuJoCo and VBD agree within 1.4 mm; Featherstone and SemiImplicit sink 1–3 cm
  more at Newton's default penalty stiffness (`ke = 2.5e3`).
- The two MuJoCo backends agree within 1 µm. CPU speed rankings say nothing about GPU throughput.
- XPBD trajectories are bit-identical run to run and across processes on CPU. `SolverMuJoCo(deterministic=RUN_TO_RUN)`
  fails to compile a mujoco_warp sensor kernel on CPU (see the report).
- `robot_anymal_c_walk` runs on CPU (yes). `mpm_anymal` refuses CPU (no), but `SolverImplicitMPM` runs on CPU at
  tiny scale through `ssw.terrain --tiny`.
- Upstream `mpm_anymal` is one-way coupled: the sand never pushes back on the robot. `ssw.terrain` adds two-way
  coupling (default) following `mpm_twoway_coupling`.

<p>
  <img src="results/phase0/base_height.png" width="49%" alt="Base height per solver">
  <img src="results/phase0/ms_per_frame.png" width="49%" alt="Warm ms per frame per solver">
</p>

## Layout

```
pyproject.toml            package sand-solvers-worldmodels, ruff + pytest config
scripts/run_phase0.sh     reproduces results/phase0 and results/phase1_cpu_smoke
src/ssw/metrics.py        pure NumPy: rigid-body energy (XYZW, Z-up), settling, timing, bit-identity,
                          judge precision/recall/confusion, depth si-RMSE, segmentation mIoU
src/ssw/bakeoff.py        Phase 0
src/ssw/terrain.py        Phase 1 (ANYmal C + ONNX policy + SolverMuJoCo + SolverImplicitMPM)
src/ssw/judge.py          Phase 2 Reasoner client, prompts, <think> parsing, scoring
src/ssw/transfer.py       Phase 2 transfer requests (vLLM-Omni form, Diffusers kwargs, Framework spec)
tests/                    unit tests plus a CPU smoke test (skipped if newton is missing)
results/phase0/           bakeoff.csv, summary.csv, run_metadata.json, plots, rollout.usda, logs/
results/phase1_cpu_smoke/ ANYmal rigid baseline and tiny MPM runs (plumbing checks, not results)
docs/phase0-report.md     one-page Phase 0 note
```

## What is tested and what is not

Functions that need CUDA or a model server say so in their docstrings ("Not executed: needs CUDA"). In particular:
the CUDA graph path in `TerrainScene`, the full-resolution sweep, `Judge.ask` against a live vLLM server,
`post_vllm_omni`, and the Diffusers kwargs have never been run. The Reasoner prompts follow the Cosmos cookbook,
and the plausibility prompt's setup sentence was adapted to this scene. The Phase 1 material mapping (mixed-MPM
Table 5 onto `model.mpm.*`) is a first guess documented in `ssw.terrain.MATERIALS`.

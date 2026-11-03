# 06 — Newton at GPU scale: parallel worlds and tiled cameras

> Reproduces the scaling workflow from the SIGGRAPH 2026 Newton talk: one robot replicated into thousands of
> parallel worlds, stepped as one captured graph, and rendered by a tiled camera into the color, depth and
> segmentation clips that the Cosmos workflow (07) consumes. The code targets a GPU; here it ran at tiny scale on
> CPU.

## Source evidence

There are no public Newton screenshots in this repo. The Newton slide deck is marked confidential and is never
embedded. The evidence is the public talk transcript and the open-source code.

| Evidence | What it shows |
|---|---|
| `NVIDIA Physical AI \| SIGGRAPH 2026/Newton Open-Source GPU Physics for Robot Simulation  Physical AI Day at SIGGRAPH 2026.txt` | Public talk (auto-captioned). Model/State/Control/Contacts/Solver separation, finalize uploads to the GPU, explicit stepping for "tens of thousands of environments", benchmark claims for steps/s, memory and tiled rendering. |
| `newton-main/newton/examples/basic/example_basic_urdf.py` | The quadruped, `ModelBuilder.replicate`, `wp.ScopedCapture` graph capture, `wp.capture_launch` replay. |
| `newton-main/newton/examples/robot/example_robot_anymal_c_walk.py` | Graph capture on CPU or mempool-enabled GPUs, a warm-up before capture, a state copy for odd substep counts. |
| `newton-main/newton/examples/sensors/example_sensor_tiled_camera.py` | `SensorTiledCamera` color, depth and shape-index outputs for 24 worlds, a semantic palette by shape. |
| `newton-main/asv/benchmarks/simulation/bench_quadruped_xpbd.py`, `bench_sensor_tiled_camera.py`, `bench_mujoco.py`, `bench_anymal.py` | Upstream benchmark code: the quadruped at 200 worlds (XPBD), tiled-camera scenes at 4096 worlds, skipped when no CUDA device is present. |

Transcript steps used (timestamps from the auto-captioned transcript, hedged as reported in the talk):
- 06:19-07:06: model (constants), state, control and contacts as explicit classes, separate from the solver.
- 09:21-09:50: build with helpers (USD, MJCF, URDF), then `finalize()` uploads everything to the GPU "in a very
  efficient tensorized format that the solvers can operate on directly".
- 09:56-10:16: an explicit loop gives "complete control over the stepping, the memory allocations", important "when
  you come to do RL and try and scale it up to tens of thousands of environments".
- 23:11-23:34: across Isaac Lab RL environments with the MuJoCo solver, "somewhere between 2 to 3x" steps per second
  compared with Isaac Lab 2 on PhysX.
- 23:44-24:12: "3 to 4x less memory"; "previously you could do 4,000 environments. Now you should be able to do
  ... 16,000 on the same GPU".
- 24:22-25:08: the tiled camera sensor, "a very high throughput raycast based sensor ... designed for tiled
  environments", designed for H200/B200-class hardware, reported at "2 to 3x faster tiled rendering" and "5 to 10x
  less memory" than Isaac Lab 2, from benchmarks by a partner (Flexion).

## What the workflow does

```
quadruped.urdf --add_urdf--> ModelBuilder (1 robot)
                                 | replicate(world_count = N)        all worlds overlap at the origin and never
                                 v                                   collide with each other (one world id each)
                      ModelBuilder (N worlds) --finalize(device)--> Model: one flat array per quantity, N x robots
                                 |
            ParallelSim: state_0/state_1, control, contacts, SolverXPBD or SolverMuJoCo (MuJoCo-Warp)
                                 | warm-up frame, then wp.ScopedCapture(one frame = 10 substeps)
                                 v
                 wp.capture_launch(graph) per frame  -->  env-steps/s, ms/frame, memory  -->  CSV
                                 |
            SensorTiledCamera.update(state, one camera per world): color, planar depth, shape index
                                 v
   per world: PNG sequences + H.264 mp4 (121 frames at 30 fps) + JSON ground truth + clips.jsonl manifest
```

Why GPU parallelism matters here: RL and synthetic data need many independent episodes. After `replicate` and
`finalize`, N robots are one set of arrays, and one kernel launch advances all of them. Per-frame cost then stays
nearly flat while N is small and the device is underused, so throughput grows almost linearly in N until the
device saturates. A captured graph removes the per-launch host overhead, which otherwise dominates at small N. The
CPU numbers below show both effects at small scale. The talk's 4,000-to-16,000-environment claim is the
memory-bound end of the same curve, on a GPU.

The clip renderer drops each world's robot with a different initial roll (0° to 50°), so some worlds land standing
and some fall. The sidecar JSON records the per-frame base pose (position and XYZW quaternion), the tilt, and a
fall flag using the thresholds of `ssw.terrain` (base below 0.25 m or tilt above 60°). `clips.jsonl` has the
fields `python -m ssw.judge ask --manifest` reads (`clip`, `material`, `seed`, `fell`, `forward_distance_m`) plus
the depth and segmentation videos that `python -m ssw.transfer --depth --seg` takes as controls.

## Code

| File | Purpose | Runs on | Status |
|---|---|---|---|
| `scale_benchmark.py` | Replicate, capture, time; sweep world counts; CSV + metadata | CPU (executed) or CUDA | CPU N = 1, 2, 4 without graph executed (required run); CPU graph runs to N = 64 executed; `gpu_sweep()` 256 to 16,384 worlds: not executed: needs GPU |
| `render_clips.py` | `SensorTiledCamera` color/depth/semantic for K worlds, PNG + mp4 + JSON sidecars + manifest | CPU (executed) or CUDA | 3 worlds at 160x90, 121 frames executed; `gpu_preset()` 16 worlds at 1280x720: not executed: needs GPU |
| `plot_scale.py` | `results/cpu_scale.png` from the CSV | CPU | executed here |
| `run_cpu.sh` | The exact commands behind every file in `results/` | CPU | its commands were executed one by one here |
| `tests/test_gpu_scale_cpu.py` | Replicate + step, graph replay equals plain stepping bit for bit, GPU paths marked, tiny clip render | CPU | executed here |

Reused from the project package: `ssw.bakeoff.INITIAL_JOINT_Q` and `solver_overrides` (robot set-up) and
`ssw.terrain.tilt_deg` (fall detection), with local fallbacks if `ssw` is not installed.

## Run it (GPU box)

Pinned versions: Newton 1.7.0.dev0 (the `newton-main/` snapshot), Warp 1.17.0, MuJoCo-Warp as pulled in by
`newton[examples]`, a CUDA 12 driver, a data-center GPU (the talk names H200/B200 for the tiled camera). Not
executed here.

```bash
source .venv/bin/activate
cd physical-ai-project/workflows/06-newton-gpu-scale
# Throughput and memory, 256 to 16,384 worlds, CUDA graph capture (mempool must be enabled, the Warp default)
python scale_benchmark.py --device cuda:0 --solver mujoco_warp --worlds 256 1024 4096 16384 --frames 200 \
    --out results/gpu_scale.csv
python scale_benchmark.py --device cuda:0 --solver xpbd --worlds 256 1024 4096 16384 --frames 200 \
    --out results/gpu_scale.csv --append
python plot_scale.py results/gpu_scale.csv results/gpu_scale.png
# Clips for workflow 07 at Cosmos Transfer's 720p, 16:9 (ssw.transfer DEFAULTS), 121 frames at 30 fps
python render_clips.py --device cuda:0 --worlds 16 --width 1280 --height 720 --out out/clips
# Upstream benchmarks for comparison (from newton-main/AGENTS.md and the bench docstrings)
cd ../../../newton-main && uvx --with virtualenv asv run --launch-method spawn main^!
uv run asv/benchmarks/simulation/bench_sensor_tiled_camera.py --bench TiledCameraQuadruped
```

## Source step → code map

| Source step (transcript or code) | Code location |
|---|---|
| Talk 09:21-09:50: add URDF/USD/MJCF, `finalize()` uploads tensorized arrays | `scale_benchmark.py: build_robot()`, `build_model()` |
| `example_basic_urdf.py`: `scene.replicate(quadruped, world_count)` | `scale_benchmark.py: build_model()` |
| Talk 09:56-10:16: explicit stepping loop with state swap | `scale_benchmark.py: ParallelSim.simulate()` |
| `example_basic_urdf.capture()`: `wp.ScopedCapture` around `simulate()`, `wp.capture_launch` | `ParallelSim.capture()`, `ParallelSim.step()` |
| `example_robot_anymal_c_walk.capture()`: graph on `device.is_cpu or device.is_mempool_enabled`, warm-up first, copy state on odd substeps | `ParallelSim.graph_wanted`, `capture()`, `simulate()` |
| Talk 23:11-24:12: steps/s and memory vs environment count | `run_one()` columns `env_steps_per_s`, `sim_array_mb`, `mem_mb`; `gpu_sweep()` |
| Talk 23:20-23:28: the comparison used the MuJoCo solver | `--solver mujoco_warp` (`SolverMuJoCo(use_mujoco_cpu=False)`) |
| Talk 24:22-24:44: tiled camera sensor for tiled environments | `render_clips.py: render()` (`SensorTiledCamera.update`) |
| `example_sensor_tiled_camera.py`: semantic colors by shape index | `render_clips.py: shape_classes()`, `CLASSES` |
| `bench_sensor_tiled_camera.py`: `_look_at_transform`, checkerboard material, 4096 worlds | `render_clips.py: look_at()`, `render()` |
| Workflow 07 input: `ssw.judge` manifest, `ssw.transfer` depth and seg controls | `render_clips.py: render()` writes `clips.jsonl`, `world_XXX_{depth,seg}.mp4` |

## Gotchas from the sources

- **Warm up before capturing.** `example_robot_anymal_c_walk._warmup_graph_capture` runs one step first so solver
  buffers that are allocated lazily exist before recording. `ParallelSim.capture()` does the same.
- **Odd substep counts break the state swap inside a graph.** The captured Python swap is baked in, so after an odd
  number of substeps the next replay would start from the wrong buffer. Upstream copies `state_1` into `state_0`
  on the last substep; so does `simulate()`. `render_clips.py` uses 32 substeps per 30 fps frame so no copy is
  needed.
- **Keep replicated worlds at the origin.** The `ModelBuilder.replicate` docstring recommends viewer world offsets
  over physical spacing, for numerical stability. Worlds never collide with each other, so overlapping is free.
  The tiled camera renders each world separately.
- **The tiled camera is built for data-center GPUs** (talk 24:44-24:52, H200s and B200s). It does run on the Warp
  CPU device, at about 2.6 Mpixel/s here; that is fine for tests and useless for training.
- **Found while building: without a graph, CPU time is launch overhead.** XPBD costs about 13-14 ms per frame for 1,
  2 or 4 worlds alike, so throughput scales with N for free. The same frame replayed as a CPU graph costs 0.84 ms
  for 1 world. MuJoCo-Warp launches far more kernels: 68 ms without a graph and 0.89 ms with one.
- **Found while building: on CPU, `ScopedCapture` records without executing.** After `capture()` the state has
  advanced by the warm-up frame only. Replays then match plain stepping bit for bit
  (`test_graph_replay_matches_plain_stepping`, XPBD; the same holds for MuJoCo-Warp in a manual check).
- **Found while building: two deprecations in Newton 1.4+.** `create_default_light()` and
  `assign_checkerboard_material()` still flip `enable_shadows` and `enable_textures` as a side effect, with a
  DeprecationWarning. Set `sensor.default_render_config` explicitly first.
- **The quadruped holds a PD pose; it does not walk.** `forward_distance_m` is about 0, so every clip is a
  "failure" for `ssw.judge`'s walk-forward question. The fall flag is the meaningful label for these clips.

## What we verified here

Executed in the cloud container (4 CPU cores, no GPU) on 2026-10-03 with Newton 1.7.0.dev0 and Warp 1.17.0
(`results/cpu_scale_metadata.json`). All numbers are CPU numbers and say nothing about GPU throughput.

`results/cpu_scale.csv` (100 Hz control, 10 substeps, 5 warm-up frames excluded):

| Solver | Graph | Worlds | ms per frame (all worlds) | env-steps/s |
|---|---|---|---|---|
| XPBD | no | 1 / 2 / 4 | 12.80 / 13.93 / 13.68 | 78 / 144 / 292 |
| XPBD | yes (CPU graph) | 1 / 2 / 4 / 16 / 64 | 0.84 / 1.21 / 1.44 / 4.25 / 16.64 | 1,190 / 1,648 / 2,774 / 3,765 / 3,846 |
| MuJoCo-Warp | no | 1 / 2 / 4 | 67.76 / 69.52 / 71.16 | 15 / 29 / 56 |
| MuJoCo-Warp | yes (CPU graph) | 1 / 4 / 16 | 0.89 / 2.68 / 12.24 | 1,128 / 1,493 / 1,307 |

- The required run (XPBD, N = 1, 2, 4, no graph) is in the first row. All rows stayed finite.
- Warp arrays held by model, states, control and contacts grow from 0.10 MB (1 world) to 1.06 MB (64 worlds). The
  process peak RSS of 366-496 MB is mostly the Python, Warp and Newton runtime. Neither is a GPU memory number.
- CPU throughput saturates near 3,800 env-steps/s by 16-64 worlds (XPBD) because one CPU thread is the device.
  This is the curve the GPU sweep is meant to push out; we have no GPU number.
- `render_clips.py`: 3 worlds at 160x90, 121 frames at 30 fps, color + planar depth (8-bit video, 16-bit mm PNG) +
  semantic. Median 16.8 ms per rendered frame for all three worlds and 4.5 ms per simulated frame (32 substeps,
  CPU graph). Ground truth: the 0° world stayed up; the 25° and 50° worlds fell (fall frames 14 and 8). mp4s were
  written with the system ffmpeg (libx264). `results/contact_sheet.png` shows color, depth and segmentation at
  frames 0, 30, 60 and 120. `results/render_summary.json` and one full sidecar `results/world_001.json` are
  committed; the PNG and mp4 files stay in `out/` (git-ignored).
- `pytest 06-newton-gpu-scale`: 4 tests pass, including the bitwise graph-replay check.

Not executed: any CUDA run, the 256 to 16,384 world sweep, 1280x720 rendering, the upstream asv benchmarks (they
skip without CUDA), ANYmal C (its asset is downloaded at run time; the bundled quadruped was used instead).

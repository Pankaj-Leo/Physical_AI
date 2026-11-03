# Phase 0: one quadruped, six solver paths, one CPU

Run on 2026-10-03, Newton 1.7.0.dev0 (editable from the local `newton-main` snapshot), Warp 1.17.0, MuJoCo 3.12.0,
mujoco-warp 3.12.0, Python 3.11.15, Linux x86-64, 4-core Intel Xeon at 2.1 GHz, 16 GB RAM, **no GPU** (Warp
`cpu` device). Raw data: `results/phase0/` (`bakeoff.csv`, `summary.csv`, `run_metadata.json`, `logs/`).
Reproduce with `scripts/run_phase0.sh`.

## Method

- Asset: `newton/examples/assets/quadruped.urdf` (13 bodies, 12 revolute joints plus a free base), loaded once per
  solver with the same builder code as Newton's `basic_urdf` example: geometry-derived inertia plus 0.01 kg m²
  rotational armature, PD drives (`target_ke` 2000, `target_kd` 1) holding a crouched pose, μ = 1.0.
- Scenario: drop the base from 0.7 m onto a ground plane and let the drives hold the pose for 4 s.
- Shared settings: 100 frames/s, 10 substeps (dt = 1 ms), 400 frames, one world. Newton's `CollisionPipeline`
  runs every substep for the four Newton-contact solvers; both MuJoCo paths use MuJoCo's own contacts.
  `newton.eval_fk` is called after `finalize()` and on the first state for every solver.
- Only VBD gets different parameters (the values the upstream example uses for VBD: shape ke 1e5, kd 1e4,
  drive kd 10), because its penalty contacts and ALM drives need them. This is recorded per row in `summary.csv`.
- Metrics (all in `ssw.metrics`, unit-tested): settle height = mean base z over the last 20% of frames; settle time
  = first time after which base z stays within 5 mm of that; energy = Σ ½mv² + ½ωᵀIω + mgz_com over all bodies
  (PE reference z = 0); drift = least-squares slope of energy over the last half of the run; wall-clock time per
  frame measured with `time.perf_counter` around the substep loop after `wp.synchronize_device`, first 5 frames
  dropped. "Stable" = finite, base z in [−0.5, 5] m, |E| < 10⁴ J.

## Results

| Solver path | Stable | Settle height [m] | Settle time [s] | Min base z [m] | Final KE [J] | Energy drift, last 2 s [J/s] | ms / frame (median, p90) | Real-time factor |
|---|---|---|---|---|---|---|---|---|
| `SolverXPBD` | yes | 0.4598 | 1.05 | 0.440 | 0 | +2e-5 | 14.1, 19.7 | 0.71 |
| `SolverFeatherstone` | yes | 0.4492 | 0.65 | 0.426 | 0.48 | +0.076 | 17.8, 22.8 | 0.56 |
| `SolverSemiImplicit` | yes | 0.4324 | 1.85 | 0.346 | 1.68 | −0.15 | 8.8, 14.2 | 1.14 |
| `SolverMuJoCo` (mujoco_warp kernels on Warp CPU) | yes | 0.4592 | 0.58 | 0.440 | 4e-11 | −2e-8 | 57.0, 68.0 | 0.18 |
| `SolverMuJoCo(use_mujoco_cpu=True)` (MuJoCo C) | yes | 0.4592 | 0.58 | 0.440 | 7e-11 | −4e-7 | 5.7, 6.7 | 1.75 |
| `SolverVBD(rigid_compliant_alm=True)` | yes | 0.4584 | 1.02 | 0.430 | 9e-7 | −0.003 | 22.8, 26.3 | 0.44 |

A second full run gave the same physics to every printed digit and median timings within 6% of these
(XPBD 14.6, Featherstone 18.0, SemiImplicit 8.9, mujoco_warp 55.0, MuJoCo C 5.8, VBD 24.2 ms). Kernel compilation is
excluded; on a cold Warp kernel cache it added up to about 11 s per solver (setup plus first frame) in our first trial.

![Base height per solver](../results/phase0/base_height.png)

![Warm step cost per solver](../results/phase0/ms_per_frame.png)

## Interpretation

- **All six paths run the same asset on CPU and stay stable.** That is the Phase 0 exit criterion.
- **Three groups by contact model.** XPBD, both MuJoCo paths and VBD settle within 1.4 mm of each other
  (0.458–0.460 m; XPBD matches the 0.46 m that the upstream `basic_urdf` test asserts). Featherstone and
  SemiImplicit use Newton's penalty contact with the default shape stiffness `ke = 2.5e3` (the URDF path does not
  set a stiffer one), so the robot sinks 1–3 cm further into the ground and keeps oscillating: SemiImplicit still
  carries 1.7 J of kinetic energy after 4 s and bottoms out at 0.35 m on the first impact. Raising `ke` for those two
  is a tuning step for later, not a solver defect.
- **The two MuJoCo backends agree.** MuJoCo C (float64) and mujoco_warp on the Warp CPU device (float32) produce
  base heights that differ by at most 1 µm over 4 s.
- **Energy.** For the settled, actively held robot, drift over the last 2 s is effectively zero for XPBD, both
  MuJoCo paths and VBD. Featherstone gains 0.08 J/s and SemiImplicit loses 0.15 J/s because they have not settled.
  About 64 J of the 180 J start energy is gone by the end: potential energy lost in the drop, dissipated by contact
  and drive damping.
- **Timing (CPU only).** The MuJoCo C library is the fastest path here (5.7 ms per 10 ms frame, 1.75× real time),
  because it is native code built for one world. mujoco_warp is the slowest (57 ms): Warp's CPU backend runs each kernel launch
  serially, and a mujoco_warp step is many small launches. This ranking says nothing about GPU throughput, where mujoco_warp and the Warp solvers are built to
  run thousands of worlds per launch.

## Determinism

- What exists in the source: `CollisionPipeline(deterministic=True)` (sorts contacts after the narrow phase,
  "results independent of GPU thread scheduling"; `contact_matching != "disabled"` implies it), and a
  `deterministic: wp.DeterministicMode | None` argument on `SolverXPBD`, `SolverFeatherstone`, `SolverVBD` and
  `SolverMuJoCo` (Warp 1.17 modes: `NOT_GUARANTEED`, `RUN_TO_RUN`, `GPU_TO_GPU`; `None` inherits
  `wp.config.deterministic`). `SolverSemiImplicit` has no such argument.
- XPBD, 100 frames, full `body_q`/`body_qd` history (17,069 float32 values): **bit-identical** across two runs in
  one process, with and without the deterministic flags, and across a separate process (same SHA-256
  `bafef573…8ebb8`). mujoco_warp with default flags: also bit-identical run to run.
- **Failure found:** `SolverMuJoCo(deterministic=wp.DeterministicMode.RUN_TO_RUN)` on the CPU device fails at kernel
  build: `WarpCodegenError` in `mujoco_warp/_src/sensor.py:2307` (`_sensor_tactile`): "Deterministic mode does not
  support mixing 'max' and 'add' reductions on array 'sensordata_out'". Worth reporting upstream; check on GPU.
- Caveat: the Warp CPU backend runs kernels serially, so CPU bit-identity is expected and does not prove GPU
  determinism. The GPU check belongs to Phase 1.

## Toolchain checks that decide the Phase 1 hardware plan

| Check | Result |
|---|---|
| `python -m newton.examples robot_anymal_c_walk --device cpu --viewer null --num-frames 50` | **Yes**, exit 0. Cold: 68 s wall (asset clone plus kernel compile). Warm: 200 frames in 14 s. Our own loop (`ssw.terrain run --material rigid`) walks 2.71 m in 3 s (0.99 m/s, no fall) at 25 ms per 20 ms frame (0.8× real time). The example's `--test` mode fails at 200 frames only because it expects 9 m of travel, which needs a longer run. |
| `python -m newton.examples mpm_anymal --device cpu ...` | **No.** Exits 1 immediately with "Error: This example requires a GPU device." (a guard in the example's `__main__`). |
| `SolverImplicitMPM` on CPU at tiny scale | **Yes.** Building the scene ourselves (`ssw.terrain --tiny`: 462 particles, 0.1 m voxels) runs sand, snow and clay at about 0.14 s per 20 ms frame (0.14× real time). The default bed (0.03 m voxels, 3 particles per cell) is about 486k particles; that is a GPU job. |
| USD export (`newton.viewer.ViewerUSD`) | **Yes.** 150 frames of the XPBD rollout, `results/phase0/rollout.usda`, 0.97 MB, Z-up, 100 time codes/s; opens with `pxr.Usd`. Not opened in `usdview` (no display here). |

**Finding for Phase 1: upstream `mpm_anymal` is one-way coupled.** It registers the robot bodies as zero-mass
colliders, so the sand never pushes back. On CPU the ANYmal base trajectory with the one-way sand bed is
bit-identical to rigid ground (0.7191 m in 1 s for both). `ssw.terrain` defaults to two-way coupling, which feeds
the MPM collider impulses back as body wrenches, following `mpm_twoway_coupling`. With it, the tiny CPU beds already
change the gait: 0.61 m (sand), 0.61 m (snow), 0.63 m (clay) versus 0.72 m on rigid ground in 1 s. At 0.1 m
resolution these are plumbing checks, not results.

**Hardware plan:** Phase 0 and the rigid-ground part of Phase 1 can run on CPU. The MPM sweep needs a CUDA GPU, as
planned.

## What could not be run here, and why

- Anything on CUDA: GPU timings, CUDA graph capture, the full-resolution MPM sweep, `SolverMuJoCo` determinism on
  GPU. No GPU in this environment.
- `--viewer gl` and `usdview`: no display.
- `ViewerFile` recordings: skipped; CSV traces and the USD file cover replay for this phase.
- Cosmos 3 (Phase 2): no model weights or server. `ssw.judge` and `ssw.transfer` are tested in dry-run mode only.

## Caveats

- CPU only, one world, small scene (13 bodies). CPU timings are not representative of GPU timings and should not be
  used to rank solvers for RL-scale workloads.
- Timings are wall-clock on a shared 4-core VM; repeat runs moved medians by up to 6%.
- Featherstone and SemiImplicit are compared at Newton's default contact stiffness. That is a fair "out of the box"
  comparison, not a tuned one.
- The energy proxy includes drive work and contact dissipation, so "drift" is only meaningful after settling.

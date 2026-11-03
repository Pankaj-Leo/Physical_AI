# Sim-to-sim: an Isaac Lab policy in Newton

Train in one simulator and test in another before any hardware. NVIDIA's humanoid talk calls the second physics engine a
"litmus test" for overfitting: sim-to-sim first, then sim-to-real (`context/02-applications-and-ecosystem.md` §1.6,
[Humanoid_Ro], auto-captioned, so treat the wording as reported in the talk). The tutorials stop at `play.py`. This page
covers the step after it.

## The pipeline

```
Isaac Lab (PhysX)                          export                          Newton (SolverMuJoCo)
train.py -> model_<iter>.pt  --play.py-->  exported/policy.onnx  -------->  example_robot_anymal_c_walk scene
48 obs / 12 actions / 50 Hz                (bash export_onnx.sh anymal)     scripts/newton_sim2sim.py --policy ...
```

1. Train `Isaac-Velocity-Flat-Anymal-C-v0`: `bash train.sh anymal` (GPU, not executed).
2. Export: `bash export_onnx.sh anymal logs/rsl_rl/anymal_c_flat/<run>/model_<iter>.pt`. Isaac Lab's `play.py` writes
   `exported/policy.onnx` next to the checkpoint (verify on v2.2). `scripts/check_onnx_policy.py` checks for 48 inputs and
   12 outputs.
3. Run it in Newton: `python scripts/newton_sim2sim.py --policy artifacts/anymal_c_flat_policy.onnx --frames 150`. This
   reuses Newton's `newton-main/newton/examples/robot/example_robot_anymal_c_walk.py` scene and swaps only the policy.
4. Compare with the baseline: the same script with no `--policy` runs the PhysX-trained policy that ships with the
   Newton asset (`anymal_walking_policy_physx.onnx`). The flagship package does the same thing on granular terrain:
   `physical-ai-project/project/src/ssw/terrain.py` (`python -m ssw.terrain run --material rigid --frames 150`).

## What must match between the two simulators

| Item | Isaac Lab side | Newton side | Where it is handled |
|---|---|---|---|
| Joint order | PhysX order, grouped by joint type: `LF_HAA, LH_HAA, RF_HAA, RH_HAA, LF_HFE, …` | URDF depth-first: `LF_HAA, LF_HFE, LF_KFE, RF_HAA, …` | `scripts/sim2sim_map.py: reorder_indices`; the example hard-codes the result as `lab_to_mujoco` / `mujoco_to_lab` |
| Quaternion order | (w, x, y, z) | (x, y, z, w) in `joint_q[3:7]` | `sim2sim_map.quat_rotate_inverse_xyzw`; `math_terms.quat_apply_inverse_wxyz` for the Isaac Lab side |
| Observation layout | base lin vel, base ang vel, projected gravity, command, joint pos − default, joint vel, last action (3+3+3+3+12+12+12 = 48) | same blocks, built by `_compute_obs_kernel` | `sim2sim_map.VELOCITY_OBS_LAYOUT`, `build_velocity_obs` |
| Velocities frame | body frame | Newton stores world-frame base velocity; the kernel rotates it into the body frame | `build_velocity_obs` |
| Action | `q_default + 0.5 · a`, joint position targets | same, with `joint_target_ke = 150`, `joint_target_kd = 5` | `sim2sim_map.action_to_targets` |
| Control rate | decimation 4 at 200 Hz physics = 50 Hz (verify in the ANYmal flat cfg) | `fps = 50`, 4 substeps | example constructor |
| Default pose | HFE ±0.4, KFE ∓0.8 (front/hind) | `initial_q` in the example | example constructor |
| Actuator model | Isaac Lab's `ANYMAL_C_CFG` uses an LSTM actuator network (verify) | ideal PD drive | not matched: this is a real sim-to-sim gap worth measuring |

Getting the first row wrong is the classic failure: every joint receives another joint's command and the robot
collapses on the first step. The tests derive both index arrays from joint names and assert they equal the arrays in
the Newton example, and that the URDF order matches the cached asset.

## Measured here (CPU)

`scripts/newton_sim2sim.py --frames 150` with the shipped PhysX policy, command (1.0, 0, 0) m/s, Newton 1.7.0.dev0,
Warp 1.17, Warp CPU device (`results/newton_sim2sim_shipped_policy_cpu.json`):

| Forward distance (3.0 s) | Mean forward speed | Lateral drift | Min base height | Fell |
|---|---|---|---|---|
| 2.714 m | 0.905 m/s | 0.039 m (to the left) | 0.507 m | no |

The forward distance equals the `ssw.terrain` rigid-ground baseline in
`physical-ai-project/project/results/phase1_cpu_smoke/runs.csv` (2.7144 m; that file logs the drift as −0.0393 m with
the opposite sign convention). A policy you export from Isaac Lab has no number yet: it needs the GPU training run.

## What to measure once you have your own policy

- Forward distance and speed tracking error at several commands (0.5, 1.0, 1.5 m/s; yaw rate ±0.5 rad/s).
- Drift and minimum base height (falls).
- The same rollouts in Isaac Lab `play.py` for the same command, to separate policy quality from simulator gap.
- Then change one thing at a time in Newton: contact stiffness (`builder.default_shape_cfg.ke`), friction (`mu`), PD
  gains, solver (`SolverMuJoCo` vs `SolverXPBD`), or terrain (`ssw.terrain --material sand`).

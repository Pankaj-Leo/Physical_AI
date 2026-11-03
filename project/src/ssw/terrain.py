"""Phase 1: ANYmal C walking policy on rigid ground vs. MPM granular terrain.

The scene mirrors Newton's ``robot_anymal_c_walk`` and ``mpm_anymal`` examples: ANYmal C
from the ``anybotics_anymal_c`` asset, the shipped PhysX-trained ONNX walking policy run with
Warp-NN, ``SolverMuJoCo`` for the robot (MuJoCo contacts against the ground plane under the
bed), and ``SolverImplicitMPM`` for a particle bed that touches only the shank colliders.
Unlike the upstream example, the bed material and density are parameters, so one script
covers rigid ground and the sand / snow / clay sweep.

Coupling. Upstream ``mpm_anymal`` registers the robot bodies as zero-mass (kinematic)
colliders, so the sand is pushed by the feet but never pushes back: on CPU we measured a
bit-identical base trajectory with and without the bed. The default here is ``two_way``,
which feeds the MPM collider impulses back to the robot as body wrenches, following the
``mpm_twoway_coupling`` example. ``--coupling one_way`` reproduces upstream.

Usage::

    # GPU (the target; written, not yet run: no CUDA machine was available)
    python -m ssw.terrain sweep --materials rigid sand snow clay --seeds 0 1 2 --frames 500 --out results/phase1

    # CPU smoke tests (executed): rigid-ground baseline, then a tiny MPM bed (462 particles, 0.1 m voxels)
    python -m ssw.terrain run --device cpu --material rigid --frames 150
    python -m ssw.terrain run --device cpu --material sand --tiny --frames 50

``--device auto`` (default) picks ``cuda:0`` when Warp sees a CUDA GPU and ``cpu`` otherwise.
On CUDA the robot substeps are captured once as a CUDA graph and replayed every frame.

Device notes (measured 2026-10-03, Newton 1.7.0.dev0, Warp 1.17, 4-core x86 CPU):

* ``python -m newton.examples mpm_anymal --device cpu`` exits immediately with
  "This example requires a GPU device." That guard lives in the example's ``__main__``.
* ``SolverImplicitMPM`` itself does run on the Warp CPU device: ``--tiny`` steps 462
  particles at about 0.14 s per 20 ms frame with two-way coupling. Full resolution (0.03 m
  voxels, 3 particles per cell, about 486k particles) is a GPU job, so non-tiny MPM on CPU
  is refused with a clear message.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from ssw import metrics
from ssw.device import AUTO, can_capture_graph, resolve_device

# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Material:
    """A bed material: particle density plus ``model.mpm.*`` per-particle overrides.

    ``mpm`` keys are the names registered by ``SolverImplicitMPM.register_custom_attributes``
    (``young_modulus``, ``friction``, ``yield_pressure``, ``tensile_yield_ratio``,
    ``yield_stress``, ``hardening``, ``dilatancy``, ``viscosity``, ...). Keys left out keep
    Newton's defaults (rigid-plastic: ``young_modulus=1e15``, ``friction=0.5``,
    ``yield_pressure=1e15``).
    """

    name: str
    density: float | None  # kg/m^3; None means no particles (rigid ground)
    mpm: dict = field(default_factory=dict)
    source: str = ""


# Mapping of the mixed-MPM paper's Table 5 (flow-rule parameters) onto Newton's attributes.
# The paper's compressive strength p_c maps to ``yield_pressure``, tensile ratio beta to
# ``tensile_yield_ratio``, friction mu to ``friction``, shear yield tau_c to
# ``yield_stress``, dilatancy theta to ``dilatancy``. Where the paper gives a range we take
# a value inside it and the sweep varies it. Snow's p_c range in the paper (1.5-10 MPa) is
# far stiffer than Newton's multi-material example (2e4 Pa); we start from the paper's
# lower bound and record the choice so that the Phase 1 report can revisit it.
MATERIALS: dict[str, Material] = {
    "rigid": Material("rigid", None, {}, "ground plane only, no particles"),
    "sand": Material(
        "sand",
        1600.0,
        {"friction": 0.68, "yield_pressure": 1.0e15, "tensile_yield_ratio": 0.0, "yield_stress": 0.0},
        "mixed-MPM Table 5: rho 1600, mu 0.68; cohesionless Drucker-Prager",
    ),
    "snow": Material(
        "snow",
        300.0,
        {
            "friction": 0.3,
            "yield_pressure": 1.5e6,
            "tensile_yield_ratio": 0.1,
            "hardening": 10.0,
            "dilatancy": 1.0,
        },
        "mixed-MPM Table 5: rho 250-400, p_c 1.5-10 MPa, beta 0.05-0.2; mu and hardening from "
        "Newton's mpm_multi_material snow (mu 0.1, hardening 10) raised to mu 0.3",
    ),
    "clay": Material(
        "clay",
        1800.0,
        {"friction": 0.0, "yield_pressure": 1.0e10, "yield_stress": 200.0, "tensile_yield_ratio": 1.0},
        "mixed-MPM Table 5: von Mises tau_c 200 Pa; mu 0 like Newton's mpm_multi_material mud; "
        "density 1800 is a typical wet-clay value, not from the paper",
    ),
}


@dataclass
class TerrainConfig:
    material: str = "rigid"
    device: str = AUTO
    frames: int = 150
    fps: int = 50
    substeps: int = 4
    voxel_size: float = 0.03
    particles_per_cell: float = 3.0
    bed_lo: tuple[float, float, float] = (-0.5, -0.5, 0.0)
    bed_hi: tuple[float, float, float] = (0.5, 2.5, 0.15)
    tolerance: float = 1.0e-6
    grid_type: str = "sparse"
    seed: int = 0
    command: tuple[float, float, float] = (1.0, 0.0, 0.0)
    fall_height: float = 0.25
    fall_tilt_deg: float = 60.0
    # "two_way": sand impulses are applied back to the robot as body forces (pattern of
    # Newton's mpm_twoway_coupling example). "one_way": robot bodies are kinematic colliders
    # for the sand and feel nothing (exactly what the upstream mpm_anymal example does).
    coupling: str = "two_way"
    overrides: dict = field(default_factory=dict)

    @property
    def spawn_height(self) -> float:
        """Base spawn height: upstream 0.62 m, raised by the bed depth so the feet start on the surface."""
        if MATERIALS[self.material].density is not None and self.coupling == "two_way":
            return 0.62 + float(self.bed_hi[2])
        return 0.62

    @classmethod
    def tiny(cls, **kw) -> TerrainConfig:
        """CPU smoke-test resolution: 0.1 m voxels, ~1 particle per cell, short bed."""
        base = {
            "voxel_size": 0.1,
            "particles_per_cell": 1.0,
            "bed_lo": (-0.5, -0.5, 0.0),
            "bed_hi": (0.5, 1.5, 0.1),
            "tolerance": 1.0e-5,
        }
        base.update(kw)
        return cls(**base)

    @property
    def frame_dt(self) -> float:
        return 1.0 / self.fps

    @property
    def sim_dt(self) -> float:
        return self.frame_dt / self.substeps


def estimate_particle_count(cfg: TerrainConfig) -> int:
    """Particle count ``add_particle_grid`` will create for ``cfg`` (0 for rigid ground)."""
    if MATERIALS[cfg.material].density is None:
        return 0
    lo, hi = np.asarray(cfg.bed_lo), np.asarray(cfg.bed_hi)
    res = np.ceil(cfg.particles_per_cell * (hi - lo) / cfg.voxel_size).astype(int)
    return int(np.prod(res + 1))


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

INITIAL_Q = {
    "RH_HAA": 0.0,
    "RH_HFE": -0.4,
    "RH_KFE": 0.8,
    "LH_HAA": 0.0,
    "LH_HFE": -0.4,
    "LH_KFE": 0.8,
    "RF_HAA": 0.0,
    "RF_HFE": 0.4,
    "RF_KFE": -0.8,
    "LF_HAA": 0.0,
    "LF_HFE": 0.4,
    "LF_KFE": -0.8,
}


class TerrainScene:
    """ANYmal C + ONNX policy on a ground plane, optionally with an MPM particle bed.

    Written for a CUDA device. Rigid mode and ``--tiny`` MPM mode were executed on CPU; the
    full-resolution bed and the CUDA-graph path in :meth:`_maybe_capture` have not been run yet.
    """

    def __init__(self, cfg: TerrainConfig):
        import newton
        import newton.utils
        import warp as wp
        from newton.examples.mpm.example_mpm_twoway_coupling import compute_body_forces, subtract_body_force
        from newton.examples.robot.example_robot_anymal_c_walk import (
            _build_joint_target_q_kernel,
            _compute_obs_kernel,
            lab_to_mujoco,
            mujoco_to_lab,
        )
        from newton.solvers import SolverImplicitMPM
        from warp_nn.runtime import OnnxRuntime

        self.wp, self.newton = wp, newton
        self._obs_kernel, self._target_kernel = _compute_obs_kernel, _build_joint_target_q_kernel
        self._force_kernel, self._subtract_kernel = compute_body_forces, subtract_body_force
        if cfg.coupling not in ("one_way", "two_way"):
            raise ValueError(f"coupling must be 'one_way' or 'two_way', got {cfg.coupling!r}")
        self.cfg = cfg
        self.material = MATERIALS[cfg.material]
        self.device = wp.get_device(resolve_device(cfg.device))
        self.sim_time = 0.0

        with wp.ScopedDevice(self.device):
            builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
            newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
            builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
                armature=0.06, limit_ke=1.0e3, limit_kd=1.0e1
            )
            builder.default_shape_cfg.ke = 5.0e4
            builder.default_shape_cfg.kd = 5.0e2
            builder.default_shape_cfg.kf = 1.0e3
            builder.default_shape_cfg.mu = 0.75

            asset_path = newton.utils.download_asset("anybotics_anymal_c")
            builder.add_urdf(
                str(asset_path / "urdf" / "anymal.urdf"),
                # Yaw 90 deg: the policy's forward (body +x) points along world +y.
                xform=wp.transform(
                    wp.vec3(0.0, 0.0, cfg.spawn_height),
                    wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5),
                ),
                floating=True,
                enable_self_collisions=False,
                collapse_fixed_joints=True,
                ignore_inertial_definitions=False,
            )
            for name, value in INITIAL_Q.items():
                idx = next(i for i, lbl in enumerate(builder.joint_label) if lbl.endswith(f"/{name}"))
                builder.joint_q[idx + 6] = value
            for i in range(builder.joint_dof_count):
                builder.joint_target_ke[i] = 150
                builder.joint_target_kd[i] = 5
            builder.add_ground_plane()

            self.has_particles = self.material.density is not None
            if self.has_particles:
                # Only the shanks (feet) collide with particles, as in the upstream example.
                for body in range(builder.body_count):
                    if "SHANK" not in builder.body_label[body]:
                        for shape in builder.body_shapes[body]:
                            builder.shape_flags[shape] &= ~newton.ShapeFlags.COLLIDE_PARTICLES
                SolverImplicitMPM.register_custom_attributes(builder)
                self._spawn_bed(builder)

            self.model = builder.finalize()
            self.solver = newton.solvers.SolverMuJoCo(self.model, ls_iterations=50, njmax=50)

            self.mpm_solver = None
            if self.has_particles:
                for key, value in {**self.material.mpm, **cfg.overrides}.items():
                    getattr(self.model.mpm, key).fill_(float(value))
                opts = SolverImplicitMPM.Config()
                opts.voxel_size = cfg.voxel_size
                opts.tolerance = cfg.tolerance
                opts.transfer_scheme = "pic"
                opts.grid_type = cfg.grid_type
                opts.grid_padding = 50 if cfg.grid_type == "fixed" else 0
                opts.max_active_cell_count = 1 << 15 if cfg.grid_type == "fixed" else -1
                opts.strain_basis = "P0"
                opts.max_iterations = 50
                opts.critical_fraction = 0.0
                opts.air_drag = 1.0
                opts.collider_velocity_mode = "backward"
                self.mpm_solver = SolverImplicitMPM(self.model, config=opts)

            self.state_0, self.state_1 = self.model.state(), self.model.state()
            self.control = self.model.control()
            newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
            self.two_way = self.mpm_solver is not None and cfg.coupling == "two_way"
            # Particles live in their own State so the MPM step never writes robot bodies.
            self.sand_state = self.state_0
            if self.mpm_solver is not None and not self.two_way:
                # Upstream mpm_anymal: zero effective mass = kinematic colliders, no feedback.
                self.mpm_solver.setup_collider(
                    body_mass=wp.zeros_like(self.model.body_mass), body_q=self.state_0.body_q
                )
            elif self.two_way:
                self.mpm_solver.setup_collider(model=self.model)
                self.sand_state = self.model.state()
                max_nodes = 1 << 20
                self._imp = wp.zeros(max_nodes, dtype=wp.vec3)
                self._imp_pos = wp.zeros(max_nodes, dtype=wp.vec3)
                self._imp_ids = wp.full(max_nodes, value=-1, dtype=int)
                self._collider_body = self.mpm_solver.collider_body_index
                self._sand_forces = wp.zeros_like(self.state_0.body_f)
                self._collect_impulses()

            policy_path = str(asset_path / "rl_policies" / "anymal_walking_policy_physx.onnx")
            self.policy = OnnxRuntime(policy_path, device=self.device)
            self._in = self.policy.input_names[0]
            self._out = self.policy.output_names[0]
            self._q0 = wp.clone(self.state_0.joint_q[7:])
            self._l2m = wp.array(np.asarray(lab_to_mujoco, dtype=np.int32), dtype=wp.int32)
            self._m2l = wp.array(np.asarray(mujoco_to_lab, dtype=np.int32), dtype=wp.int32)
            self._obs = wp.zeros((1, 48), dtype=wp.float32)
            self._prev_act = wp.zeros((1, 12), dtype=wp.float32)
            self._command = wp.vec3(*cfg.command)
            self.graph = None
            self._maybe_capture()

    def _spawn_bed(self, builder) -> None:
        wp = self.wp
        cfg = self.cfg
        lo, hi = np.asarray(cfg.bed_lo, dtype=float), np.asarray(cfg.bed_hi, dtype=float)
        res = np.ceil(cfg.particles_per_cell * (hi - lo) / cfg.voxel_size).astype(int)
        cell = (hi - lo) / res
        radius = float(np.max(cell) * 0.5)
        mass = float(np.prod(cell) * self.material.density)
        start = len(builder.particle_q)
        builder.add_particle_grid(
            pos=wp.vec3(*lo),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=int(res[0]) + 1,
            dim_y=int(res[1]) + 1,
            dim_z=int(res[2]) + 1,
            cell_x=float(cell[0]),
            cell_y=float(cell[1]),
            cell_z=float(cell[2]),
            mass=mass,
            jitter=0.0,
            radius_mean=radius,
        )
        # Seeded jitter (add_particle_grid's own jitter uses a fixed internal seed).
        rng = np.random.default_rng(cfg.seed)
        n = len(builder.particle_q) - start
        offsets = (rng.random((n, 3)) - 0.5) * 2.0 * radius
        for k in range(n):
            p = builder.particle_q[start + k]
            builder.particle_q[start + k] = wp.vec3(p[0] + offsets[k, 0], p[1] + offsets[k, 1], p[2] + offsets[k, 2])

    def _maybe_capture(self) -> None:
        """CUDA-graph capture of the robot substeps (CUDA only; not yet run on a GPU)."""
        wp = self.wp
        if not can_capture_graph(self.device):
            return
        self._simulate_robot()  # warm up lazily allocated solver buffers
        with wp.ScopedCapture(device=self.device) as capture:
            self._simulate_robot()
        self.graph = capture.graph

    def _policy_step(self) -> None:
        wp = self.wp
        wp.launch(
            self._obs_kernel,
            dim=1,
            inputs=[
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self._q0,
                self._l2m,
                wp.vec3(0.0, 0.0, -1.0),
                self._command,
                self._prev_act,
                self._obs,
            ],
            device=self.device,
        )
        act = self.policy({self._in: self._obs})[self._out]
        wp.launch(
            self._target_kernel,
            dim=19,
            inputs=[act, self._q0, self._m2l, 0.5, 7, self.control.joint_target_q],
            device=self.device,
        )
        wp.copy(self._prev_act, act)

    def _collect_impulses(self) -> None:
        imp, pos, ids = self.mpm_solver.collect_collider_impulses(self.sand_state)
        n = min(imp.shape[0], self._imp.shape[0])
        self._imp_ids.fill_(-1)
        self._imp[:n].assign(imp[:n])
        self._imp_pos[:n].assign(pos[:n])
        self._imp_ids[:n].assign(ids[:n])

    def _simulate_robot(self) -> None:
        wp = self.wp
        for _ in range(self.cfg.substeps):
            self.state_0.clear_forces()
            if self.two_way:
                # Sand reaction from the last MPM step, as a body wrench held over the frame.
                wp.launch(
                    self._force_kernel,
                    dim=self._imp_ids.shape[0],
                    inputs=[
                        self.cfg.frame_dt,
                        self._imp_ids,
                        self._imp,
                        self._imp_pos,
                        self._collider_body,
                        self.state_0.body_q,
                        self.model.body_com,
                        self.state_0.body_f,
                    ],
                    device=self.device,
                )
                self._sand_forces.assign(self.state_0.body_f)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.cfg.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _simulate_sand(self) -> None:
        wp = self.wp
        if self.two_way:
            # Hand MPM the robot velocity with last frame's sand force removed, so the solver
            # computes the full contact impulse (see mpm_twoway_coupling.subtract_body_force).
            wp.launch(
                self._subtract_kernel,
                dim=self.model.body_count,
                inputs=[
                    self.cfg.frame_dt,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self._sand_forces,
                    self.model.body_inv_inertia,
                    self.model.body_inv_mass,
                    self.sand_state.body_q,
                    self.sand_state.body_qd,
                ],
                device=self.device,
            )
            self.mpm_solver.step(self.sand_state, self.sand_state, contacts=None, control=None, dt=self.cfg.frame_dt)
            self._collect_impulses()
        else:
            # One implicit MPM step per frame on the frame dt, in place (as upstream).
            self.mpm_solver.step(self.state_0, self.state_0, contacts=None, control=None, dt=self.cfg.frame_dt)

    def step(self) -> None:
        with self.wp.ScopedDevice(self.device):
            self._policy_step()
            if self.graph is not None:
                self.wp.capture_launch(self.graph)
            else:
                self._simulate_robot()
            if self.mpm_solver is not None:
                self._simulate_sand()
        self.sim_time += self.cfg.frame_dt

    def base_pose(self) -> np.ndarray:
        """Base body transform ``(px, py, pz, qx, qy, qz, qw)``."""
        return self.state_0.body_q.numpy()[0].astype(np.float64)

    def particle_z_stats(self) -> tuple[float, float]:
        if not self.has_particles:
            return float("nan"), float("nan")
        z = self.sand_state.particle_q.numpy()[:, 2]
        return float(z.min()), float(z.max())


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def tilt_deg(q_xyzw: np.ndarray) -> float:
    """Angle between the body's +Z axis and world +Z [deg]."""
    r = metrics.quat_xyzw_to_matrix(np.asarray(q_xyzw))
    return float(np.degrees(np.arccos(np.clip(r[2, 2], -1.0, 1.0))))


def summarize_run(poses: np.ndarray, t: np.ndarray, cfg: TerrainConfig, step_ms: Sequence[float]) -> dict:
    """Distance, speed, fall flag and timing from per-frame base poses.

    Forward is world +y (the robot is spawned yawed by 90 degrees).
    """
    y = poses[:, 1]
    z = poses[:, 2]
    tilts = np.array([tilt_deg(q) for q in poses[:, 3:7]])
    fell = bool((z < cfg.fall_height).any() or (tilts > cfg.fall_tilt_deg).any())
    half = len(t) // 2
    speed = float((y[-1] - y[half]) / (t[-1] - t[half])) if t[-1] > t[half] else float("nan")
    ts = metrics.timing_stats(step_ms, skip=min(3, max(0, len(step_ms) - 1)))
    return {
        "material": cfg.material,
        "seed": cfg.seed,
        "frames": len(t) - 1,
        "sim_time_s": round(float(t[-1]), 3),
        "forward_distance_m": round(float(y[-1] - y[0]), 4),
        "lateral_drift_m": round(float(poses[-1, 0] - poses[0, 0]), 4),
        "forward_speed_2nd_half_mps": round(speed, 4),
        "mean_base_z_2nd_half_m": round(float(z[half:].mean()), 4),
        "min_base_z_m": round(float(z.min()), 4),
        "max_tilt_deg": round(float(tilts.max()), 2),
        "fell": fell,
        "ms_per_frame_median": round(ts.median_ms, 2),
        "realtime_factor": round(cfg.frame_dt * 1e3 / ts.median_ms, 3),
    }


def run(cfg: TerrainConfig, out_dir: Path | None = None, tiny: bool = False) -> dict:
    """Build the scene, run ``cfg.frames`` frames, return a summary (and write CSVs)."""
    import warp as wp

    cfg = replace(cfg, device=resolve_device(cfg.device))
    is_cpu = wp.get_device(cfg.device).is_cpu
    n_particles = estimate_particle_count(cfg)
    if is_cpu and n_particles > 20_000 and not tiny:
        raise SystemExit(
            f"Refusing to run {n_particles:,} MPM particles on CPU (voxel {cfg.voxel_size} m). "
            "Full-resolution terrain is a GPU job: pass --device cuda:0, or use --tiny for a CPU smoke test."
        )

    t0 = time.perf_counter()
    scene = TerrainScene(cfg)
    setup_s = time.perf_counter() - t0
    poses, t, step_ms, pz = [scene.base_pose()], [0.0], [], []
    for _ in range(cfg.frames):
        tic = time.perf_counter()
        scene.step()
        wp.synchronize_device(scene.device)
        step_ms.append((time.perf_counter() - tic) * 1e3)
        poses.append(scene.base_pose())
        t.append(scene.sim_time)
        pz.append(scene.particle_z_stats())
        if not np.all(np.isfinite(poses[-1])):
            break
    poses_np, t_np = np.asarray(poses), np.asarray(t)
    summary = summarize_run(poses_np, t_np, cfg, step_ms)
    summary.update(
        {
            "particles": int(scene.model.particle_count),
            "voxel_size_m": cfg.voxel_size,
            "device": str(scene.device),
            "setup_s": round(setup_s, 2),
            "tiny": tiny,
            "coupling": cfg.coupling if scene.has_particles else "n/a",
            "spawn_z_m": round(cfg.spawn_height, 3),
            "material_params": json.dumps({**scene.material.mpm, **cfg.overrides}, sort_keys=True),
        }
    )
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{cfg.material}_s{cfg.seed}" + (f"_{cfg.coupling}" if scene.has_particles else "")
        stem += "_tiny" if tiny else ""
        with (out_dir / f"{stem}_frames.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                ["frame", "t", "x", "y", "z", "qx", "qy", "qz", "qw", "step_ms", "particle_z_min", "particle_z_max"]
            )
            for i, (ti, p) in enumerate(zip(t_np, poses_np, strict=True)):
                ms = step_ms[i - 1] if i > 0 else math.nan
                pzmin, pzmax = pz[i - 1] if i > 0 else (math.nan, math.nan)
                w.writerow([i, f"{ti:.4f}", *(f"{v:.5f}" for v in p), f"{ms:.2f}", f"{pzmin:.4f}", f"{pzmax:.4f}"])
    return summary


def _write_summary(rows: Sequence[dict], path: Path) -> None:
    keys: list[str] = []
    for r in rows:
        keys.extend(k for k in r if k not in keys)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _append_rows(rows: Sequence[dict], path: Path) -> None:
    """Append rows to a CSV, writing the header only when the file is new."""
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--device", default=AUTO, help="auto (cuda:0 if present, else cpu), cpu, cuda:N")
        sp.add_argument("--frames", type=int, default=150, help="frames at 50 fps")
        sp.add_argument("--tiny", action="store_true", help="coarse CPU smoke-test resolution")
        sp.add_argument("--voxel-size", type=float, default=None)
        sp.add_argument("--particles-per-cell", type=float, default=None)
        sp.add_argument("--grid-type", choices=["sparse", "dense", "fixed"], default="sparse")
        sp.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a model.mpm.* value")
        sp.add_argument("--out", type=Path, default=None)
        sp.add_argument("--coupling", choices=["two_way", "one_way"], default="two_way")

    r = sub.add_parser("run", help="one material, one seed")
    common(r)
    r.add_argument("--material", choices=sorted(MATERIALS), default="rigid")
    r.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("sweep", help="materials x seeds (GPU job; written, not yet run on a GPU)")
    common(s)
    s.add_argument("--materials", nargs="+", choices=sorted(MATERIALS), default=["rigid", "sand", "snow", "clay"])
    s.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])

    s2 = sub.add_parser("materials", help="print the material table and exit")
    del s2
    args = p.parse_args(argv)

    if args.cmd == "materials":
        print(json.dumps({k: asdict(v) for k, v in MATERIALS.items()}, indent=2))
        return 0

    import warp as wp

    wp.config.log_level = wp.LOG_WARNING
    args.device = resolve_device(args.device)
    overrides = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        overrides[k] = float(v)

    def make_cfg(material: str, seed: int) -> TerrainConfig:
        kw = {"material": material, "seed": seed, "device": args.device, "frames": args.frames}
        kw["grid_type"] = args.grid_type
        kw["coupling"] = args.coupling
        kw["overrides"] = overrides
        if args.voxel_size is not None:
            kw["voxel_size"] = args.voxel_size
        if args.particles_per_cell is not None:
            kw["particles_per_cell"] = args.particles_per_cell
        return TerrainConfig.tiny(**kw) if args.tiny else TerrainConfig(**kw)

    if args.cmd == "run":
        cfgs = [make_cfg(args.material, args.seed)]
    else:
        cfgs = [make_cfg(m, sd) for m in args.materials for sd in args.seeds]

    rows = []
    for cfg in cfgs:
        print(f"[terrain] {cfg.material} seed={cfg.seed} particles~{estimate_particle_count(cfg):,}", flush=True)
        row = run(cfg, args.out, tiny=args.tiny)
        rows.append(row)
        print(f"[terrain] {json.dumps(row)}", flush=True)
    if args.out is not None:
        if args.cmd == "sweep":
            _write_summary(rows, args.out / ("sweep_tiny.csv" if args.tiny else "sweep.csv"))
        else:
            _append_rows(rows, args.out / "runs.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

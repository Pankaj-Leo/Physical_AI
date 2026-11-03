"""Throughput vs. number of parallel worlds for one quadruped, Newton + Warp.

One robot (``quadruped.urdf`` from ``newton-main``, set up exactly as ``example_basic_urdf``) is built once and
``ModelBuilder.replicate``-d to ``world_count`` worlds. All worlds share one set of arrays on the device; one
``solver.step`` advances them all. The step is captured as a Warp graph and replayed, the pattern the upstream
examples use (``example_basic_urdf.capture()``, ``example_robot_anymal_c_walk.capture()``).

Metrics per world count (one CSV row):
  env_steps_per_s     worlds x control frames per wall-clock second (one frame = 1/fps s, ``substeps`` solver steps)
  realtime_factor     simulated seconds per wall second, summed over worlds
  sim_array_mb        bytes of Warp arrays held by model, 2 states, control and contacts (device-agnostic)
  mem_kind / mem_mb   CUDA: mempool high-water mark; CPU: process peak RSS (cumulative across the sweep)

Kernel compilation and graph capture are excluded from timing (warm-up frames).

Usage::

    python scale_benchmark.py --out results/gpu_scale.csv                              # GPU target (not yet run)
    python scale_benchmark.py --device cpu --worlds 1 2 4 --out results/cpu_scale.csv  # CPU smoke test (executed)

``--device auto`` (default) uses ``cuda:0`` when present (256 to 16,384 worlds) and ``cpu`` otherwise (1, 2, 4).
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import platform
import resource
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

try:  # reuse the project package's Phase 0 settings when it is installed
    from ssw.bakeoff import INITIAL_JOINT_Q, solver_overrides
except ImportError:  # pragma: no cover - fallback keeps the folder standalone
    INITIAL_JOINT_Q = [0.2, 0.4, -0.6, -0.2, -0.4, 0.6, -0.2, 0.4, -0.6, 0.2, -0.4, 0.6]

    def solver_overrides(solver: str) -> dict:
        return {"target_ke": 2000.0, "target_kd": 1.0, "mu": 1.0}


GPU_WORLD_COUNTS = (256, 1024, 4096, 16384)
CPU_WORLD_COUNTS = (1, 2, 4)


def pick_device(spec: str = "auto") -> str:
    """``auto``: ``cuda:0`` when Warp sees a CUDA GPU, else ``cpu`` with a warning. CUDA without a GPU fails."""
    import warp as wp

    has_cuda = wp.get_cuda_device_count() > 0
    if spec == "auto":
        if not has_cuda:
            print("[warn] no CUDA device; falling back to cpu (smoke test, not GPU numbers)", file=sys.stderr)
        return "cuda:0" if has_cuda else "cpu"
    if spec.startswith("cuda") and not has_cuda:
        raise SystemExit(f"--device {spec} requested but Warp found no CUDA device")
    return spec


SOLVERS = ("xpbd", "mujoco_warp")


@dataclass
class BenchConfig:
    solver: str = "xpbd"
    fps: int = 100
    substeps: int = 10
    warmup_frames: int = 5
    frames: int = 50
    device: str = "auto"  # resolved in __post_init__: cuda:0 if present, else cpu
    graph: bool | None = None  # None: capture whenever Warp supports it on the device (CPU and CUDA in Warp 1.17)
    drop_height: float = 0.7

    def __post_init__(self) -> None:
        self.device = pick_device(self.device)

    @property
    def sim_dt(self) -> float:
        return 1.0 / self.fps / self.substeps


def build_robot(newton, wp, solver: str = "xpbd"):
    """Single quadruped builder, parameters as in ``example_basic_urdf`` (and ``ssw.bakeoff``)."""
    ov = solver_overrides(solver)
    b = newton.ModelBuilder()
    if solver == "mujoco_warp":
        newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.default_joint_cfg.armature = 0.01
    b.default_joint_cfg.target_ke = ov["target_ke"]
    b.default_joint_cfg.target_kd = ov["target_kd"]
    b.default_shape_cfg.mu = ov["mu"]
    b.add_urdf(
        newton.examples.get_asset("quadruped.urdf"),
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.7), wp.quat_identity()),
        floating=True,
        enable_self_collisions=False,
        ignore_inertial_definitions=True,
    )
    for body in range(b.body_count):
        inertia = np.asarray(b.body_inertia[body], dtype=np.float32).reshape(3, 3)
        b.body_inertia[body] = wp.mat33(inertia + np.eye(3, dtype=np.float32) * 0.01)
    b.joint_q[-12:] = INITIAL_JOINT_Q
    b.joint_target_q[-12:] = INITIAL_JOINT_Q
    return b


def build_model(world_count: int, cfg: BenchConfig):
    import newton
    import newton.examples
    import warp as wp

    robot = build_robot(newton, wp, cfg.solver)
    scene = newton.ModelBuilder()
    if cfg.solver == "mujoco_warp":
        newton.solvers.SolverMuJoCo.register_custom_attributes(scene)
    scene.replicate(robot, world_count)  # worlds overlap in space; they never collide with each other
    scene.add_ground_plane(cfg=robot.default_shape_cfg)
    model = scene.finalize(device=cfg.device)
    newton.eval_fk(model, model.joint_q, model.joint_qd, model)
    return model


class ParallelSim:
    """Model, solver, double-buffered state and an optional captured graph for one frame."""

    def __init__(self, model, cfg: BenchConfig):
        import newton
        import warp as wp

        self.wp, self.cfg, self.model = wp, cfg, model
        self.device = wp.get_device(cfg.device)
        if cfg.solver == "xpbd":
            self.solver = newton.solvers.SolverXPBD(model)
            self.pipeline = newton.CollisionPipeline(model)
            self.contacts = self.pipeline.contacts()
        elif cfg.solver == "mujoco_warp":
            self.solver = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=False)
            self.pipeline, self.contacts = None, None  # MuJoCo-Warp generates its own contacts
        else:
            raise ValueError(f"solver must be one of {SOLVERS}")
        self.state_0, self.state_1 = model.state(), model.state()
        self.control = model.control()
        newton.eval_fk(model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.graph = None

    def simulate(self) -> None:
        """One frame = ``substeps`` solver steps. Odd substep counts copy back so a graph replays correctly."""
        n = self.cfg.substeps
        for i in range(n):
            self.state_0.clear_forces()
            if self.pipeline is not None:
                self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.cfg.sim_dt)
            if self.graph_wanted and n % 2 == 1 and i == n - 1:
                self.state_0.assign(self.state_1)
            else:
                self.state_0, self.state_1 = self.state_1, self.state_0

    @property
    def graph_wanted(self) -> bool:
        if self.cfg.graph is not None:
            return self.cfg.graph
        return self.device.is_cpu or self.device.is_mempool_enabled

    def capture(self) -> None:
        """Warm up lazily allocated buffers, then record one frame (upstream pattern)."""
        self.simulate()
        with self.wp.ScopedCapture(device=self.device) as cap:
            self.simulate()
        self.graph = cap.graph

    def step(self) -> None:
        if self.graph is not None:
            self.wp.capture_launch(self.graph)
        else:
            self.simulate()


def _array_bytes(*objs) -> int:
    import warp as wp

    seen, total = set(), 0
    for obj in objs:
        if obj is None:
            continue
        for v in vars(obj).values():
            if isinstance(v, wp.array) and id(v) not in seen:
                seen.add(id(v))
                total += v.capacity
    return total


def device_memory_mb(device) -> tuple[str, float]:
    import warp as wp

    if device.is_cuda:
        if device.is_mempool_enabled:
            return "cuda_mempool_high", wp.get_mempool_used_mem_high(device) / 2**20
        return "cuda_used", (device.total_memory - device.free_memory) / 2**20
    return "cpu_peak_rss", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run_one(world_count: int, cfg: BenchConfig) -> dict:
    """Build, warm up (compile + capture), then time ``cfg.frames`` frames."""
    import warp as wp

    t0 = time.perf_counter()
    model = build_model(world_count, cfg)
    sim = ParallelSim(model, cfg)
    build_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(cfg.warmup_frames):
        sim.simulate()
    if sim.graph_wanted:
        sim.capture()
    wp.synchronize_device(sim.device)
    warmup_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(cfg.frames):
        sim.step()
    wp.synchronize_device(sim.device)
    wall = time.perf_counter() - t0

    body_q = sim.state_0.body_q.numpy()
    bodies_per_world = model.body_count // world_count
    base_z = body_q[::bodies_per_world, 2]
    mem_kind, mem_mb = device_memory_mb(sim.device)
    return {
        "device": str(sim.device),
        "solver": cfg.solver,
        "world_count": world_count,
        "graph": sim.graph is not None,
        "fps": cfg.fps,
        "substeps": cfg.substeps,
        "warmup_frames": cfg.warmup_frames,
        "frames": cfg.frames,
        "build_s": round(build_s, 3),
        "warmup_s": round(warmup_s, 3),
        "wall_s": round(wall, 4),
        "ms_per_frame": round(wall / cfg.frames * 1e3, 3),
        "env_steps_per_s": round(world_count * cfg.frames / wall, 1),
        "realtime_factor": round(world_count * cfg.frames / cfg.fps / wall, 3),
        "sim_array_mb": round(_array_bytes(model, sim.state_0, sim.state_1, sim.control, sim.contacts) / 2**20, 3),
        "mem_kind": mem_kind,
        "mem_mb": round(mem_mb, 1),
        "finite": bool(np.isfinite(body_q).all()),
        "base_z_mean_m": round(float(base_z.mean()), 4),
    }


def sweep(world_counts, cfg: BenchConfig, out_csv: Path | None = None, append: bool = False) -> list[dict]:
    """Run each world count; rewrite (or extend, with ``append``) the CSV after every row."""
    previous = []
    if append and out_csv is not None and out_csv.exists():
        with out_csv.open(newline="") as f:
            previous = list(csv.DictReader(f))
    rows = []
    for n in world_counts:
        row = run_one(n, cfg)
        rows.append(row)
        print(json.dumps(row), flush=True)
        if out_csv is not None:
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            with out_csv.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows([*previous, *rows])
    return rows


def gpu_sweep(out_csv: Path, device: str = "cuda:0", solver: str = "mujoco_warp") -> list[dict]:
    """The talk-scale sweep, 256 to 16,384 worlds with CUDA graph capture.

    Written for a CUDA GPU; not yet run on one. Expect the largest counts to hit device memory; watch ``mem_mb``.
    """
    cfg = BenchConfig(solver=solver, device=device, graph=True, warmup_frames=10, frames=200)
    return sweep(GPU_WORLD_COUNTS, cfg, out_csv)


def run_metadata(cfg: BenchConfig) -> dict:
    import newton
    import warp as wp

    return {
        "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "newton": newton.__version__,
        "warp": wp.__version__,
        "numpy": np.__version__,
        "last_invocation_config": asdict(cfg),
        "note": f"Measured on {cfg.device}. Per-row settings are columns of the CSV.",
        "cpu_count": __import__("os").cpu_count(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="auto", help="auto (cuda:0 if present, else cpu), cpu, cuda:N")
    ap.add_argument("--worlds", type=int, nargs="+", default=None)
    ap.add_argument("--solver", choices=SOLVERS, default="xpbd")
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--graph", dest="graph", action="store_true", default=None, help="force graph capture")
    g.add_argument("--no-graph", dest="graph", action="store_false", help="plain kernel launches")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--append", action="store_true", help="add rows to an existing --out CSV")
    args = ap.parse_args(argv)

    import warp as wp

    wp.config.log_level = wp.LOG_WARNING
    args.device = pick_device(args.device)
    is_cuda = wp.get_device(args.device).is_cuda
    counts = args.worlds or (GPU_WORLD_COUNTS if is_cuda else CPU_WORLD_COUNTS)
    cfg = BenchConfig(
        solver=args.solver, device=args.device, frames=args.frames, warmup_frames=args.warmup, graph=args.graph
    )
    rows = sweep(counts, cfg, args.out, append=args.append)
    if args.out is not None:
        meta = args.out.with_name(args.out.stem + "_metadata.json")
        meta.write_text(json.dumps(run_metadata(cfg), indent=2) + "\n")
    return 0 if all(r["finite"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())

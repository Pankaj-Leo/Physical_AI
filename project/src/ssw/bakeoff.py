"""Phase 0: solver bake-off on one quadruped URDF (GPU target, CPU fallback).

Loads ``newton/examples/assets/quadruped.urdf`` once per solver, drops it from 0.7 m onto a
ground plane with PD joint drives holding a crouched pose, and logs base height, mechanical
energy and wall-clock time per frame. All solvers share the same frame rate, substep count
and initial state; solver-specific parameter overrides (only VBD needs them) are recorded
in the summary.

Usage::

    python -m ssw.bakeoff --out results/phase0                 # all solvers, CSV + plots
    python -m ssw.bakeoff --solvers xpbd mujoco_c --frames 100  # subset
    python -m ssw.bakeoff --determinism xpbd --usd xpbd         # extras
    python -m ssw.bakeoff --device cpu --out results/phase0_cpu # force the CPU smoke test

``--device auto`` (default) runs on ``cuda:0`` when Warp sees a CUDA GPU and on ``cpu`` otherwise.
On CUDA the substep loop of the GPU-resident solvers is captured once as a CUDA graph and replayed
each frame, so the timing measures kernels, not Python launch overhead. ``mujoco_c`` steps the MuJoCo
C library on the host and is never captured.

The committed ``results/phase0`` numbers were measured on the Warp CPU device. The GPU path is
written but has not been run yet: no CUDA machine was available.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import datetime as _dt
import hashlib
import json
import os
import platform
import re
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ssw import metrics
from ssw.device import AUTO, can_capture_graph, describe, resolve_device

# Solver keys in run order. Each maps to (description, builder overrides, factory).
SOLVER_KEYS = (
    "xpbd",
    "featherstone",
    "semi_implicit",
    "mujoco_warp",
    "mujoco_c",
    "vbd",
)

INITIAL_JOINT_Q = [0.2, 0.4, -0.6, -0.2, -0.4, 0.6, -0.2, 0.4, -0.6, 0.2, -0.4, 0.6]


@dataclass
class BakeoffConfig:
    """Shared simulation settings. Identical for every solver."""

    fps: int = 100
    substeps: int = 10
    frames: int = 300
    drop_height: float = 0.7
    device: str = AUTO
    deterministic: bool = False
    # None: capture a CUDA graph whenever the solver and device allow it; False: always launch eagerly.
    graph: bool | None = None

    @property
    def frame_dt(self) -> float:
        return 1.0 / self.fps

    @property
    def sim_dt(self) -> float:
        return self.frame_dt / self.substeps


@dataclass
class RunResult:
    """Outcome of one solver run."""

    solver: str
    ran: bool
    reason: str = "ok"
    device: str = ""
    graph: bool = False
    stable: bool = False
    stability_reason: str = ""
    setup_s: float = float("nan")
    first_frame_s: float = float("nan")
    overrides: dict = field(default_factory=dict)
    t: list[float] = field(default_factory=list)
    base_z: list[float] = field(default_factory=list)
    kinetic: list[float] = field(default_factory=list)
    energy: list[float] = field(default_factory=list)
    step_ms: list[float] = field(default_factory=list)
    body_q_history: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Model and solver construction
# ---------------------------------------------------------------------------


def _import_newton():
    try:
        import newton
        import newton.examples
        import warp as wp
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(f"newton/warp not importable ({exc}). Install newton[examples] first.") from exc
    return wp, newton


def solver_overrides(solver: str) -> dict:
    """Builder parameter overrides per solver (mirrors ``example_basic_urdf``)."""
    if solver == "vbd":
        # VBD's penalty-based contacts and ALM drives need stiffer shapes and drives;
        # these are the values the upstream basic_urdf example uses for VBD.
        return {
            "target_ke": 2.0e3,
            "target_kd": 1.0e1,
            "shape_ke": 1.0e5,
            "shape_kd": 1.0e4,
            "mu": 1.0,
        }
    return {"target_ke": 2000.0, "target_kd": 1.0, "mu": 1.0}


def build_model(solver: str, cfg: BakeoffConfig):
    """Build and finalize the single-world quadruped scene for ``solver``."""
    wp, newton = _import_newton()
    ov = solver_overrides(solver)

    quadruped = newton.ModelBuilder()
    quadruped.default_joint_cfg.armature = 0.01
    quadruped.default_joint_cfg.target_ke = ov["target_ke"]
    quadruped.default_joint_cfg.target_kd = ov["target_kd"]
    quadruped.default_shape_cfg.mu = ov["mu"]
    if "shape_ke" in ov:
        quadruped.default_shape_cfg.ke = ov["shape_ke"]
        quadruped.default_shape_cfg.kd = ov["shape_kd"]

    quadruped.add_urdf(
        newton.examples.get_asset("quadruped.urdf"),
        xform=wp.transform(wp.vec3(0.0, 0.0, cfg.drop_height), wp.quat_identity()),
        floating=True,
        enable_self_collisions=False,
        ignore_inertial_definitions=True,
    )
    # Extra rotational armature on every body, as in the upstream example, for stability.
    for body in range(quadruped.body_count):
        inertia = np.asarray(quadruped.body_inertia[body], dtype=np.float32).reshape(3, 3)
        quadruped.body_inertia[body] = wp.mat33(inertia + np.eye(3, dtype=np.float32) * 0.01)

    quadruped.joint_q[-12:] = INITIAL_JOINT_Q
    quadruped.joint_target_q[-12:] = INITIAL_JOINT_Q

    scene = newton.ModelBuilder()
    scene.add_builder(quadruped)
    scene.add_ground_plane(cfg=quadruped.default_shape_cfg)
    if solver == "vbd":
        scene.color()

    model = scene.finalize(device=cfg.device)
    # Maximal-coordinate solvers and the collision pipeline need body_q from FK.
    newton.eval_fk(model, model.joint_q, model.joint_qd, model)
    return model


def make_solver(solver: str, model, cfg: BakeoffConfig):
    """Construct the solver. Returns ``(solver, uses_newton_contacts)``."""
    wp, newton = _import_newton()
    det = wp.DeterministicMode.RUN_TO_RUN if cfg.deterministic else None
    S = newton.solvers
    if solver == "xpbd":
        return S.SolverXPBD(model, deterministic=det), True
    if solver == "featherstone":
        return S.SolverFeatherstone(model, deterministic=det), True
    if solver == "semi_implicit":
        return S.SolverSemiImplicit(model), True
    if solver == "vbd":
        return S.SolverVBD(model, iterations=2, rigid_compliant_alm=True, deterministic=det), True
    if solver == "mujoco_warp":
        # mujoco_warp kernels compiled by Warp for the model's device; MuJoCo's own contacts.
        return S.SolverMuJoCo(model, use_mujoco_cpu=False, deterministic=det), False
    if solver == "mujoco_c":
        # The MuJoCo C library (mj_step) on the host, Newton only marshals state.
        return S.SolverMuJoCo(model, use_mujoco_cpu=True), False
    raise ValueError(f"unknown solver {solver!r}; choose from {SOLVER_KEYS}")


# Solvers whose step is a pure sequence of Warp launches and can be replayed as a CUDA graph.
# mujoco_c runs mj_step on the host; vbd toggles host-side history state every substep.
GRAPH_SOLVERS = frozenset({"xpbd", "featherstone", "semi_implicit", "mujoco_warp"})


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------


def _energy(model_arrays: dict, body_q: np.ndarray, body_qd: np.ndarray) -> metrics.EnergyBreakdown:
    return metrics.rigid_body_energy(
        body_q,
        body_qd,
        model_arrays["mass"],
        model_arrays["inertia"],
        model_arrays["com"],
    )


def run_solver(
    solver: str,
    cfg: BakeoffConfig,
    record_body_q: bool = False,
    frame_callback: Callable[[float, object, object], None] | None = None,
) -> RunResult:
    """Simulate ``cfg.frames`` frames with ``solver`` and collect per-frame metrics.

    Any exception during setup or stepping is caught and reported in ``RunResult.reason``.
    """
    wp, newton = _import_newton()
    res = RunResult(solver=solver, ran=False, overrides=solver_overrides(solver))
    try:
        cfg = dataclasses.replace(cfg, device=resolve_device(cfg.device))
        res.device = cfg.device
        with wp.ScopedDevice(cfg.device):
            device = wp.get_device(cfg.device)
            t0 = time.perf_counter()
            model = build_model(solver, cfg)
            sim, newton_contacts = make_solver(solver, model, cfg)
            state_0, state_1 = model.state(), model.state()
            control = model.control()
            newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
            pipeline = contacts = None
            if newton_contacts:
                pipeline = newton.CollisionPipeline(model, deterministic=cfg.deterministic)
                contacts = pipeline.contacts()
            res.setup_s = time.perf_counter() - t0

            arrays = {
                "mass": model.body_mass.numpy(),
                "inertia": model.body_inertia.numpy(),
                "com": model.body_com.numpy(),
            }
            history = []

            def record(t: float, ms: float):
                bq = state_0.body_q.numpy()
                bqd = state_0.body_qd.numpy()
                e = _energy(arrays, bq, bqd)
                res.t.append(t)
                res.base_z.append(float(bq[0, 2]))
                res.kinetic.append(e.kinetic)
                res.energy.append(e.total)
                res.step_ms.append(ms)
                if record_body_q:
                    history.append(np.concatenate([bq.ravel(), bqd.ravel()]))

            def simulate():
                nonlocal state_0, state_1
                for _ in range(cfg.substeps):
                    state_0.clear_forces()
                    if pipeline is not None:
                        pipeline.collide(state_0, contacts)
                    if solver == "vbd":
                        sim.set_rigid_history_update(True)
                    sim.step(state_0, state_1, control, contacts, cfg.sim_dt)
                    state_0, state_1 = state_1, state_0

            record(0.0, float("nan"))
            if frame_callback is not None:
                frame_callback(0.0, state_0, model)

            # An even substep count leaves state_0/state_1 where they started, which a replayed
            # graph requires (it always writes the same buffers).
            use_graph = (
                cfg.graph is not False
                and solver in GRAPH_SOLVERS
                and can_capture_graph(device)
                and cfg.substeps % 2 == 0
            )
            graph = None

            for frame in range(cfg.frames):
                if use_graph and graph is None and frame == 1:
                    # Frame 0 ran eagerly: kernels are compiled and lazy solver buffers allocated.
                    with wp.ScopedCapture(device=device) as capture:
                        simulate()
                    graph = capture.graph
                    res.graph = True
                tic = time.perf_counter()
                if graph is not None:
                    wp.capture_launch(graph)
                else:
                    simulate()
                wp.synchronize_device(device)
                ms = (time.perf_counter() - tic) * 1e3
                if frame == 0:
                    res.first_frame_s = ms / 1e3
                t = (frame + 1) * cfg.frame_dt
                record(t, ms)
                if frame_callback is not None:
                    frame_callback(t, state_0, model)
                if not np.isfinite(res.base_z[-1]) or abs(res.base_z[-1]) > 100.0:
                    break  # diverged; no point continuing
            res.ran = True
            if record_body_q:
                res.body_q_history = np.stack(history)
    except Exception as exc:
        res.reason = short_error(exc)
        traceback.print_exc()
        return res

    res.stable, res.stability_reason = metrics.is_stable(res.base_z, res.energy)
    if len(res.t) < cfg.frames + 1:
        res.stable = False
        res.stability_reason = f"diverged at t={res.t[-1]:.2f}s; " + res.stability_reason
    return res


def short_error(exc: BaseException, limit: int = 400) -> str:
    """One-line error: exception type, first line and last line of the message, site paths trimmed."""
    lines = [ln.strip() for ln in str(exc).splitlines() if ln.strip()]
    msg = lines[0] if lines else ""
    if len(lines) > 1:
        msg += " ... " + lines[-1].lstrip(";")
    msg = re.sub(r"/\S*/site-packages/", "<site-packages>/", msg)
    return f"{type(exc).__name__}: {msg}"[:limit]


def load_results(out: Path) -> tuple[list[RunResult], list[dict]]:
    """Rebuild :class:`RunResult` traces and summary rows from ``bakeoff.csv``/``summary.csv``."""
    with (out / "summary.csv").open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in list(r.items()):
            if v in ("True", "False"):
                r[k] = v == "True"
            else:
                with contextlib.suppress(ValueError):
                    r[k] = int(v) if v.lstrip("-").isdigit() else float(v)
    by_solver: dict[str, RunResult] = {}
    with (out / "bakeoff.csv").open() as f:
        for rec in csv.DictReader(f):
            rr = by_solver.setdefault(rec["solver"], RunResult(solver=rec["solver"], ran=True))
            rr.t.append(float(rec["t"]))
            rr.base_z.append(float(rec["base_z"]))
            rr.kinetic.append(float(rec["kinetic"]))
            rr.energy.append(float(rec["energy"]))
            rr.step_ms.append(float(rec["step_ms"]) if rec["step_ms"] else float("nan"))
    for r in rows:
        if r["solver"] in by_solver:
            by_solver[r["solver"]].stable = bool(r["stable"])
    return list(by_solver.values()), rows


def summarize(res: RunResult, cfg: BakeoffConfig, warmup_frames: int = 5) -> dict:
    """One summary row per solver."""
    row = {
        "solver": res.solver,
        "device": res.device,
        "cuda_graph": res.graph,
        "ran": res.ran,
        "stable": res.stable,
        "reason": res.reason if not res.ran else res.stability_reason,
        "fps": cfg.fps,
        "substeps": cfg.substeps,
        "sim_dt_s": cfg.sim_dt,
        "frames": len(res.t) - 1 if res.t else 0,
        "overrides": json.dumps(res.overrides, sort_keys=True),
    }
    if not res.ran or len(res.t) < warmup_frames + 3:
        return row
    ts = metrics.timing_stats(res.step_ms[1:], skip=warmup_frames)
    t = np.asarray(res.t)
    z = np.asarray(res.base_z)
    e = np.asarray(res.energy)
    row.update(
        {
            "setup_s": round(res.setup_s, 3),
            "first_frame_s": round(res.first_frame_s, 3),
            "ms_per_frame_median": round(ts.median_ms, 3),
            "ms_per_frame_p90": round(ts.p90_ms, 3),
            "ms_per_substep_median": round(ts.median_ms / cfg.substeps, 4),
            "realtime_factor": round(cfg.frame_dt * 1e3 / ts.median_ms, 3),
            "settle_height_m": round(metrics.settle_height(z), 4),
            "settle_time_s": round(metrics.settle_time(t, z), 3),
            "min_base_z_m": round(float(z.min()), 4),
            "final_kinetic_J": float(f"{res.kinetic[-1]:.3e}"),
            "energy_start_J": round(float(e[0]), 3),
            "energy_end_J": round(float(e[-1]), 3),
            "energy_drift_J_per_s": float(f"{metrics.energy_drift(t, e):.3e}"),
        }
    )
    return row


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_long_csv(results: Sequence[RunResult], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["solver", "frame", "t", "base_z", "kinetic", "energy", "step_ms"])
        for r in results:
            for i, (t, z, k, e, ms) in enumerate(zip(r.t, r.base_z, r.kinetic, r.energy, r.step_ms, strict=True)):
                w.writerow(
                    [
                        r.solver,
                        i,
                        f"{t:.4f}",
                        f"{z:.6f}",
                        f"{k:.6e}",
                        f"{e:.6f}",
                        "" if np.isnan(ms) else f"{ms:.3f}",
                    ]
                )


def write_summary_csv(rows: Sequence[dict], path: Path) -> None:
    keys: list[str] = []
    for r in rows:
        keys.extend(k for k in r if k not in keys)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# Okabe-Ito colorblind-safe palette.
PALETTE = {
    "xpbd": "#0072B2",
    "featherstone": "#E69F00",
    "semi_implicit": "#009E73",
    "mujoco_warp": "#CC79A7",
    "mujoco_c": "#D55E00",
    "vbd": "#56B4E9",
}
# Secondary encoding so identity never rests on color alone.
LINESTYLE = {
    "xpbd": "-",
    "featherstone": (0, (6, 2)),
    "semi_implicit": (0, (1, 1.5)),
    "mujoco_warp": (0, (3, 3)),
    "mujoco_c": "-",
    "vbd": (0, (6, 2, 1, 2)),
}
INK = "#222222"
MUTED = "#666666"
GRID = "#e6e6e6"


def _style_axes(ax) -> None:
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def plot_results(results: Sequence[RunResult], rows: Sequence[dict], out: Path) -> list[Path]:
    """Base height vs time (full + zoom on the settled phase) and a ms/frame bar chart."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 9.5, "text.color": INK})
    written = []
    ran = [r for r in results if r.ran and r.t]

    fig, (ax, az) = plt.subplots(1, 2, figsize=(10.0, 4.0), dpi=150)
    for r in ran:
        kw = {"color": PALETTE.get(r.solver, INK), "ls": LINESTYLE.get(r.solver, "-"), "lw": 1.6}
        ax.plot(r.t, r.base_z, label=r.solver + ("" if r.stable else " (unstable)"), **kw)
        az.plot(r.t, r.base_z, **kw)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("base height z [m]")
    ax.set_title("Drop from 0.7 m and settle", loc="left", fontsize=10)
    _style_axes(ax)
    ax.legend(frameon=False, ncol=1, fontsize=8.5, loc="upper right")

    # Zoom on the settled phase, with de-overlapped direct labels at the right edge.
    t_end = max(r.t[-1] for r in ran)
    t_zoom = 0.375 * t_end
    groups: dict[float, list[str]] = {}
    for r in ran:
        groups.setdefault(round(float(np.mean(r.base_z[-20:])), 4), []).append(r.solver)
    lo, hi = min(groups) - 0.01, max(groups) + 0.01
    az.set_xlim(t_zoom, t_end)
    az.set_ylim(lo, hi)
    min_gap = (hi - lo) * 0.07
    prev_y = None
    for key in sorted(groups, reverse=True):
        y = key if prev_y is None else min(key, prev_y - min_gap)
        prev_y = y
        az.annotate(
            " = ".join(sorted(groups[key])) + f"  {key:.3f} m",
            xy=(t_end, key),
            xytext=(t_end * 1.02, y),
            fontsize=8,
            color=INK,
            va="center",
            annotation_clip=False,
            arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.5},
        )
    az.set_xlabel("time [s]")
    az.set_title(f"Settled phase (t > {t_zoom:.1f} s), mean of last 0.2 s labeled", loc="left", fontsize=10)
    _style_axes(az)
    fig.suptitle("Quadruped base height per solver, CPU, dt = 1 ms", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.85, 1))
    p = out / "base_height.png"
    fig.savefig(p)
    plt.close(fig)
    written.append(p)

    timed = [r for r in rows if "ms_per_frame_median" in r]
    if timed:
        timed = sorted(timed, key=lambda r: r["ms_per_frame_median"])
        fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=150)
        names = [r["solver"] for r in timed]
        med = np.array([r["ms_per_frame_median"] for r in timed])
        p90 = np.array([r["ms_per_frame_p90"] for r in timed])
        y = np.arange(len(names))
        ax.barh(y, med, color=[PALETTE.get(n, "grey") for n in names], height=0.62, edgecolor="white", lw=2)
        ax.errorbar(med, y, xerr=[np.zeros(len(med)), p90 - med], fmt="none", ecolor=INK, lw=0.9, capsize=2)
        for yi, m, q in zip(y, med, p90, strict=True):
            ax.text(
                q + 0.6,
                yi,
                f"{m:.1f} ms",
                va="center",
                fontsize=8.5,
                color=INK,
                bbox={"fc": "white", "ec": "none", "pad": 0.5},
            )
        ax.set_yticks(y, names)
        rt = 1e3 / timed[0]["fps"]
        ax.axvline(rt, color=MUTED, ls="--", lw=0.9)
        ax.text(rt, -0.8, " real-time budget", color=MUTED, fontsize=8, va="center")
        ax.set_ylim(-1.1, len(names) - 0.4)
        ax.set_xlim(0, float(p90.max()) * 1.18)
        ax.set_xlabel(f"wall-clock ms per 10 ms frame ({timed[0]['substeps']} substeps); bar = median, whisker = p90")
        ax.set_title("Warm step cost per solver, 1 world, 4-core CPU (compile excluded)", loc="left", fontsize=10)
        _style_axes(ax)
        ax.grid(False, axis="y")
        fig.tight_layout()
        p = out / "ms_per_frame.png"
        fig.savefig(p)
        plt.close(fig)
        written.append(p)
    return written


def run_metadata(cfg: BakeoffConfig) -> dict:
    wp, newton = _import_newton()
    meta = {
        "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "newton": newton.__version__,
        "warp": wp.__version__,
        "device": describe(wp.get_device(resolve_device(cfg.device))),
        "config": asdict(cfg),
    }
    try:
        import mujoco

        meta["mujoco"] = mujoco.__version__
    except ImportError:
        meta["mujoco"] = None
    try:
        import mujoco_warp

        meta["mujoco_warp"] = getattr(mujoco_warp, "__version__", "installed")
    except ImportError:
        meta["mujoco_warp"] = None
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    meta["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        meta["cpu_model"] = platform.processor()
    try:
        with open("/proc/meminfo") as f:
            meta["mem_total"] = f.readline().split(":", 1)[1].strip()
    except OSError:
        pass
    return meta


def determinism_check(solver: str, cfg: BakeoffConfig, frames: int) -> dict:
    """Run ``solver`` twice in this process, with and without deterministic flags, and compare bitwise.

    The compared array is the full per-frame history of ``body_q`` and ``body_qd`` (float32).
    The SHA-256 of the first run's history is also returned so that separate processes
    (or machines) can be compared by re-running and diffing the hashes.
    """
    out = {"solver": solver, "frames": frames}
    for det in (False, True):
        c = BakeoffConfig(**{**asdict(cfg), "frames": frames, "deterministic": det})
        a = run_solver(solver, c, record_body_q=True)
        b = run_solver(solver, c, record_body_q=True)
        key = "deterministic_flags" if det else "default_flags"
        if not (a.ran and b.ran):
            out[key] = {"ran": False, "reason": a.reason if not a.ran else b.reason}
            continue
        out[key] = {
            "bit_identical": metrics.bitwise_identical(a.body_q_history, b.body_q_history),
            "max_abs_diff": metrics.max_abs_diff(a.body_q_history, b.body_q_history),
            "n_values_compared": int(a.body_q_history.size),
            "sha256": hashlib.sha256(np.ascontiguousarray(a.body_q_history).tobytes()).hexdigest(),
        }
    return out


def export_usd(solver: str, cfg: BakeoffConfig, path: Path, frames: int, max_bytes: int = 5_000_000) -> dict:
    """Record one rollout with ``newton.viewer.ViewerUSD``. Deletes the file if over ``max_bytes``."""
    _, newton = _import_newton()
    viewer = newton.viewer.ViewerUSD(output_path=str(path), fps=cfg.fps, up_axis="Z", num_frames=frames + 1)
    c = BakeoffConfig(**{**asdict(cfg), "frames": frames})

    def cb(t, state, model):
        if t == 0.0:
            viewer.set_model(model)
        viewer.begin_frame(t)
        viewer.log_state(state)
        viewer.end_frame()

    res = run_solver(solver, c, frame_callback=cb)
    viewer.close()
    info = {"solver": solver, "frames": frames, "ran": res.ran, "path": path.name}
    if path.exists():
        size = path.stat().st_size
        info["bytes"] = size
        if size > max_bytes:
            path.unlink()
            info["kept"] = False
            info["note"] = f"deleted: {size} bytes > {max_bytes}"
        else:
            info["kept"] = True
    return info


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--solvers", nargs="+", default=list(SOLVER_KEYS), choices=SOLVER_KEYS)
    p.add_argument("--frames", type=int, default=300, help="frames at 100 fps (default 300 = 3 s)")
    p.add_argument("--substeps", type=int, default=10)
    p.add_argument("--fps", type=int, default=100)
    p.add_argument("--device", default=AUTO, help="auto (cuda:0 if present, else cpu), cpu, cuda, cuda:N")
    p.add_argument("--no-graph", action="store_true", help="launch kernels eagerly even on CUDA")
    p.add_argument("--out", type=Path, default=Path("results/phase0"))
    p.add_argument(
        "--determinism",
        metavar="SOLVER",
        nargs="*",
        default=[],
        choices=SOLVER_KEYS,
        help="run the run-twice bit-identity check on these solvers",
    )
    p.add_argument("--determinism-frames", type=int, default=100)
    p.add_argument(
        "--usd",
        metavar="SOLVER",
        default=None,
        help="export one rollout of SOLVER to rollout.usda",
    )
    p.add_argument("--usd-frames", type=int, default=100)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--replot", action="store_true", help="only redraw plots from existing CSVs in --out")
    args = p.parse_args(argv)

    if args.replot:
        results, rows = load_results(args.out)
        for path in plot_results(results, rows, args.out):
            print(f"[bakeoff] wrote {path}")
        return 0

    wp, _ = _import_newton()
    wp.config.log_level = wp.LOG_WARNING
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = BakeoffConfig(
        fps=args.fps,
        substeps=args.substeps,
        frames=args.frames,
        device=resolve_device(args.device),
        graph=False if args.no_graph else None,
    )

    results, rows = [], []
    for s in args.solvers:
        print(
            f"[bakeoff] {s}: {cfg.frames} frames x {cfg.substeps} substeps, dt={cfg.sim_dt:.4g}s",
            flush=True,
        )
        r = run_solver(s, cfg)
        row = summarize(r, cfg)
        results.append(r)
        rows.append(row)
        print(f"[bakeoff] {s}: {json.dumps(row)}", flush=True)

    write_long_csv(results, args.out / "bakeoff.csv")
    write_summary_csv(rows, args.out / "summary.csv")
    meta = run_metadata(cfg)
    meta["solvers"] = {r.solver: {"ran": r.ran, "reason": r.reason} for r in results}

    if args.determinism:
        meta["determinism"] = []
        for solver in args.determinism:
            print(f"[bakeoff] determinism check on {solver}", flush=True)
            meta["determinism"].append(determinism_check(solver, cfg, args.determinism_frames))
            print(json.dumps(meta["determinism"][-1]), flush=True)
    if args.usd:
        print(f"[bakeoff] USD export of {args.usd}", flush=True)
        meta["usd_export"] = export_usd(args.usd, cfg, args.out / "rollout.usda", args.usd_frames)
        print(json.dumps(meta["usd_export"]), flush=True)
    if not args.no_plots:
        meta["plots"] = [str(x.name) for x in plot_results(results, rows, args.out)]

    (args.out / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(f"[bakeoff] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

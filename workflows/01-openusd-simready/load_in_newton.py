"""Load the SimReady ``asset.usda`` with Newton and simulate 1 s on CPU; compare with the source URDF.

Both paths use the same solver (XPBD), frame rate (100 Hz), substeps (10), drop height (0.7 m), joint armature
(0.01) and friction (1.0), so differences come from the asset conversion only. The URDF path keeps its inertial
tags (``ignore_inertial_definitions=False``) because the USD MassAPI was written from them.

Usage::

    python load_in_newton.py out/asset.usda --json results/newton_load.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
URDF = HERE.parents[2] / "newton-main" / "newton" / "examples" / "assets" / "quadruped.urdf"
STANDING_POSE_RAD = [0.2, 0.4, -0.6, -0.2, -0.4, 0.6, -0.2, 0.4, -0.6, 0.2, -0.4, 0.6]


def _builder(newton):
    b = newton.ModelBuilder()
    b.default_joint_cfg.armature = 0.01
    b.default_shape_cfg.mu = 1.0
    return b


def build_from_usd(asset: Path, drop_height: float = 0.7):
    import newton
    import warp as wp

    b = _builder(newton)
    info = b.add_usd(
        str(asset),
        xform=wp.transform(wp.vec3(0.0, 0.0, drop_height), wp.quat_identity()),
        enable_self_collisions=False,
    )
    return b, info


def build_from_urdf(drop_height: float = 0.7):
    import newton
    import warp as wp

    b = _builder(newton)
    b.default_joint_cfg.target_ke = 2000.0
    b.default_joint_cfg.target_kd = 1.0
    b.add_urdf(
        str(URDF),
        xform=wp.transform(wp.vec3(0.0, 0.0, drop_height), wp.quat_identity()),
        floating=True,
        enable_self_collisions=False,
        ignore_inertial_definitions=False,
    )
    b.joint_q[-12:] = STANDING_POSE_RAD
    b.joint_target_q[-12:] = STANDING_POSE_RAD
    return b


def named_q(labels, q_start, q) -> dict[str, float]:
    """Joint position by joint name. Importers order joints differently (Newton's USD import walks the tree
    depth-first: LF, LH, RF, RH; the URDF lists LF, RF, LH, RH), so compare by name, never by index."""
    out = {}
    for j, label in enumerate(labels):
        name = str(label).rsplit("/", 1)[-1]
        if name.endswith(("_HAA", "_HFE", "_KFE")):
            out[name] = round(float(q[int(q_start[j])]), 4)
    return dict(sorted(out.items()))


def describe(b) -> dict:
    """Counts and totals read from the builder before finalize."""
    return {
        "bodies": b.body_count,
        "joints": b.joint_count,
        "dofs": b.joint_dof_count,
        "shapes": b.shape_count,
        "total_mass_kg": round(float(sum(b.body_mass)), 4),
        "joint_order": [str(lb).rsplit("/", 1)[-1] for lb in b.joint_label],
        "joint_q_initial": named_q(b.joint_label, b.joint_q_start, b.joint_q),
        "joint_limit_pairs_rad": sorted(
            {
                (round(float(lo), 4), round(float(hi), 4))
                for lo, hi in zip(b.joint_limit_lower[-12:], b.joint_limit_upper[-12:], strict=True)
            }
        ),
        "target_ke": sorted({round(float(v), 3) for v in b.joint_target_ke[-12:]}),
    }


def simulate(b, seconds: float = 1.0, fps: int = 100, substeps: int = 10, device: str = "cpu") -> dict:
    """Ground plane + XPBD; returns base height trace and timings. Excludes kernel compilation from timing."""
    import newton
    import warp as wp

    scene = newton.ModelBuilder()
    scene.add_builder(b)
    scene.add_ground_plane()
    model = scene.finalize(device=device)
    newton.eval_fk(model, model.joint_q, model.joint_qd, model)
    solver = newton.solvers.SolverXPBD(model)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    dt = 1.0 / fps / substeps
    frames = round(seconds * fps)
    z, ms = [float(s0.body_q.numpy()[0, 2])], []
    for _ in range(frames):
        tic = time.perf_counter()
        for _ in range(substeps):
            s0.clear_forces()
            pipeline.collide(s0, contacts)
            solver.step(s0, s1, ctrl, contacts, dt)
            s0, s1 = s1, s0
        wp.synchronize_device(device)
        ms.append((time.perf_counter() - tic) * 1e3)
        z.append(float(s0.body_q.numpy()[0, 2]))
    q = s0.joint_q.numpy()
    zz = np.array(z)
    return {
        "frames": frames,
        "sim_time_s": seconds,
        "base_z_start_m": round(zz[0], 4),
        "base_z_end_m": round(zz[-1], 4),
        "base_z_min_m": round(float(zz.min()), 4),
        "finite": bool(np.isfinite(zz).all() and np.isfinite(q).all()),
        "joint_q_end": named_q(model.joint_label, model.joint_q_start.numpy(), q),
        "first_frame_ms_incl_compile": round(ms[0], 1),
        "median_ms_per_frame": round(float(np.median(ms[1:])) if len(ms) > 1 else ms[0], 2),
        "base_z_trace": [round(v, 4) for v in z[:: max(1, fps // 20)]],
    }


def run(asset: Path, seconds: float = 1.0, device: str = "cpu") -> dict:
    import newton
    import warp as wp

    wp.config.log_level = wp.LOG_WARNING
    b_usd, info = build_from_usd(asset)
    b_urdf = build_from_urdf()
    out = {
        "newton": newton.__version__,
        "warp": wp.__version__,
        "device": device,
        "solver": "SolverXPBD",
        "fps": 100,
        "substeps": 10,
        "asset": Path(asset).name,
        "usd_import": {
            "path_body_map_size": len(info.get("path_body_map", {})),
            "path_joint_map_size": len(info.get("path_joint_map", {})),
        },
        "usd": {"model": describe(b_usd)},
        "urdf": {"model": describe(b_urdf)},
    }
    out["usd"]["sim"] = simulate(b_usd, seconds, device=device)
    out["urdf"]["sim"] = simulate(b_urdf, seconds, device=device)
    dz = abs(out["usd"]["sim"]["base_z_end_m"] - out["urdf"]["sim"]["base_z_end_m"])
    qa, qb = out["usd"]["sim"]["joint_q_end"], out["urdf"]["sim"]["joint_q_end"]
    dq = max(abs(qa[k] - qb[k]) for k in qa) if qa.keys() == qb.keys() else float("nan")
    out["comparison"] = {
        "base_z_end_abs_diff_m": round(dz, 4),
        "joint_q_end_max_abs_diff_rad": round(dq, 4),
        "same_counts": out["usd"]["model"]["bodies"] == out["urdf"]["model"]["bodies"]
        and out["usd"]["model"]["dofs"] == out["urdf"]["model"]["dofs"],
        "mass_diff_kg": round(out["usd"]["model"]["total_mass_kg"] - out["urdf"]["model"]["total_mass_kg"], 4),
    }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("asset", type=Path)
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    res = run(args.asset, args.seconds, args.device)
    text = json.dumps(res, indent=2)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    ok = (
        res["usd"]["sim"]["finite"]
        and res["usd"]["sim"]["base_z_end_m"] > 0.1
        and not math.isnan(res["comparison"]["base_z_end_abs_diff_m"])
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""The real-time engineering-twin loop on CPU: trusted solver -> dataset -> AI surrogate -> design sweep -> verify.

Mirrors the "AI-Enhanced Simulation Workflows" slide of the SIGGRAPH 2026 real-time twin talk:
design -> simulation (``heat_solver``, Warp kernels) -> simulation dataset -> AI training (``surrogate``, numpy)
-> AI-physics model -> design optimization, with the winner checked against the solver.

Design space (fixed: 50 mm wide sink, 4 mm base, 1 mm fins, 100 mm long, 20 mm chip):
  n_fins 4..24 (channel spacing 1.13..15.3 mm), fin height 10..40 mm, heat load 2..20 W.

Usage::

    python run_twin.py                     # about 5 minutes on 4 CPU cores
    python run_twin.py --reuse-dataset     # retrain and re-sweep from results/dataset.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
N_RANGE = (4, 24)
H_RANGE = (10.0, 40.0)
Q_RANGE = (2.0, 20.0)
SWEEP_H = np.round(np.arange(10.0, 40.0 + 1e-9, 0.5), 2)
SWEEP_Q = (5.0, 10.0, 15.0, 20.0)
VERIFY_H, VERIFY_Q = 30.0, 10.0
DEVICE = "cuda:0"  # Warp device for the solver; main() resolves --device auto (cuda:0 if present, else cpu)


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


def _sibling(name: str):
    key = f"wf11_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def sample_designs(n: int, seed: int = 0) -> list[tuple[int, float, float]]:
    """Stratified (Latin-hypercube style) samples: fin count, fin height [mm], load [W]."""
    rng = np.random.default_rng(seed)
    u = (np.stack([rng.permutation(n) for _ in range(3)], axis=1) + rng.random((n, 3))) / n
    fins = np.floor(N_RANGE[0] + u[:, 0] * (N_RANGE[1] - N_RANGE[0] + 1)).astype(int)
    h = H_RANGE[0] + u[:, 1] * (H_RANGE[1] - H_RANGE[0])
    q = Q_RANGE[0] + u[:, 2] * (Q_RANGE[1] - Q_RANGE[0])
    return [(int(a), round(float(b), 3), round(float(c), 3)) for a, b, c in zip(fins, h, q, strict=True)]


def features(rows: list[dict]) -> np.ndarray:
    return np.array([[float(r["spacing_mm"]), float(r["fin_height_mm"]), float(r["power_w"])] for r in rows])


def build_dataset(n_samples: int, seed: int, out_csv: Path) -> list[dict]:
    hs = _sibling("heat_solver")
    rows = []
    for i, (n, h, q) in enumerate(sample_designs(n_samples, seed)):
        row = hs.solve_row(hs.Design(n_fins=n, fin_height_mm=h, power_w=q), device=DEVICE)
        rows.append(row)
        if i % 50 == 0:
            print(f"[data] {i}/{n_samples} n={n} H={h} Q={q} Tmax={row['t_max_rise_k']:.2f} K {row['wall_s']:.2f}s")
    write_csv(rows, out_csv)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def spacing_mm(n_fins, width=50.0, t=1.0):
    return (width - t) / (np.asarray(n_fins, dtype=float) - 1) - t


def train(rows: list[dict], seed: int = 0) -> dict:
    sur = _sibling("surrogate")
    x = features(rows)
    y = np.array([float(r["t_max_rise_k"]) for r in rows])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows))
    n_test = len(rows) // 5
    test, tr = idx[:n_test], idx[n_test:]
    t0 = time.perf_counter()
    mlp = sur.MLP(seed=seed).fit(x[tr], np.log(y[tr]))
    train_s = time.perf_counter() - t0
    quad = sur.QuadraticLS().fit(x[tr], np.log(y[tr]))
    metrics = {
        "split": {"train": len(tr), "test": len(test), "seed": seed},
        "mlp": {
            "architecture": "3-64-64-1 tanh, Adam, 6000 full-batch epochs, cosine LR from 3e-3",
            "train_s": round(train_s, 2),
            "final_loss_scaled": mlp.history[-1],
            "test": sur.error_metrics(y[test], np.exp(mlp.predict(x[test]))),
            "train": sur.error_metrics(y[tr], np.exp(mlp.predict(x[tr]))),
        },
        "quadratic_ls": {"test": sur.error_metrics(y[test], np.exp(quad.predict(x[test])))},
    }
    return {"mlp": mlp, "quad": quad, "metrics": metrics, "test_idx": test, "x": x, "y": y}


def timing(model, rows: list[dict]) -> dict:
    solver_s = np.array([float(r["wall_s"]) for r in rows])
    grid = np.array(
        [[s, h, q] for s in spacing_mm(np.arange(4, 25)) for h in SWEEP_H for q in SWEEP_Q],
        dtype=float,
    )
    model.predict(grid[:10])
    t0 = time.perf_counter()
    model.predict(grid)
    batch_s = time.perf_counter() - t0
    single = []
    for k in range(200):
        t0 = time.perf_counter()
        model.predict(grid[k : k + 1])
        single.append(time.perf_counter() - t0)
    per_query_batched = batch_s / len(grid)
    return {
        "solver_median_s": float(np.median(solver_s)),
        "solver_p90_s": float(np.percentile(solver_s, 90)),
        "surrogate_batch_designs": len(grid),
        "surrogate_batch_s": batch_s,
        "surrogate_per_query_batched_s": per_query_batched,
        "surrogate_single_query_median_s": float(np.median(single)),
        "speedup_batched": float(np.median(solver_s) / per_query_batched),
        "speedup_single_query": float(np.median(solver_s) / np.median(single)),
        "solver_time_for_sweep_est_s": float(np.median(solver_s) * len(grid)),
    }


def sweep(model) -> tuple[list[dict], dict]:
    """Surrogate over every (n, H, Q) on the grid; best fin count per (H, Q)."""
    fins = np.arange(N_RANGE[0], N_RANGE[1] + 1)
    s = spacing_mm(fins)
    optima, n_designs = [], 0
    curves = {}
    for q in SWEEP_Q:
        for h in SWEEP_H:
            x = np.stack([s, np.full_like(s, h), np.full_like(s, q)], axis=1)
            t = np.exp(model.predict(x))
            n_designs += len(x)
            k = int(np.argmin(t))
            optima.append(
                {
                    "power_w": q,
                    "fin_height_mm": h,
                    "best_n_fins": int(fins[k]),
                    "best_spacing_mm": round(float(s[k]), 3),
                    "pred_t_max_rise_k": round(float(t[k]), 3),
                }
            )
            if h == VERIFY_H:
                curves[q] = t
    return optima, {"designs_evaluated": n_designs, "curves_at_verify_h": curves, "fins": fins, "spacing": s}


def verify(model, optima: list[dict]) -> dict:
    """Solver at every fin count for (H = VERIFY_H, all loads) and at the surrogate's winner for a few more cases."""
    hs = _sibling("heat_solver")
    fins = np.arange(N_RANGE[0], N_RANGE[1] + 1)
    out = {"curves": {}, "winners": []}
    for q in SWEEP_Q:
        t_solver = np.array(
            [
                hs.solve(hs.Design(n_fins=int(n), fin_height_mm=VERIFY_H, power_w=q), device=DEVICE).t_max_rise_k
                for n in fins
            ]
        )
        out["curves"][q] = t_solver
    checks = [o for o in optima if o["fin_height_mm"] in (15.0, VERIFY_H, 40.0)]
    for o in checks:
        q, h = o["power_w"], o["fin_height_mm"]
        if h == VERIFY_H:
            curve = out["curves"][q]
            true_best = int(fins[int(np.argmin(curve))])
            t_at_pred = float(curve[o["best_n_fins"] - N_RANGE[0]])
            t_true_best = float(curve.min())
        else:
            res = hs.solve(hs.Design(n_fins=o["best_n_fins"], fin_height_mm=h, power_w=q), device=DEVICE)
            t_at_pred, true_best, t_true_best = res.t_max_rise_k, None, None
            neigh = []
            for n in (o["best_n_fins"] - 1, o["best_n_fins"] + 1):
                if N_RANGE[0] <= n <= N_RANGE[1]:
                    neigh.append(
                        (hs.solve(hs.Design(n_fins=n, fin_height_mm=h, power_w=q), device=DEVICE).t_max_rise_k, n)
                    )
            local_best = min([(t_at_pred, o["best_n_fins"]), *neigh])
            true_best, t_true_best = (local_best[1], local_best[0])
        out["winners"].append(
            {
                "power_w": q,
                "fin_height_mm": h,
                "surrogate_best_n": o["best_n_fins"],
                "surrogate_best_spacing_mm": o["best_spacing_mm"],
                "surrogate_pred_k": o["pred_t_max_rise_k"],
                "solver_at_surrogate_best_k": round(t_at_pred, 3),
                "prediction_error_k": round(o["pred_t_max_rise_k"] - t_at_pred, 3),
                "solver_best_n": true_best,
                "solver_best_k": round(t_true_best, 3),
                "regret_k": round(t_at_pred - t_true_best, 3),
                "solver_check": "full fin-count sweep" if h == VERIFY_H else "winner and +/-1 fin neighbours",
            }
        )
    return out


def plot(path: Path, tr: dict, sw: dict, ver: dict, best_design) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hs = _sibling("heat_solver")
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9), dpi=110)
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(SWEEP_Q)))
    s_fine = sw["spacing"]
    for c, q in zip(colors, SWEEP_Q, strict=True):
        ax[0].plot(s_fine, sw["curves_at_verify_h"][q], color=c, lw=1.6, label=f"surrogate, Q = {q:g} W")
        ax[0].plot(s_fine, ver["curves"][q], "o", color=c, ms=3.5, mfc="none", label=f"solver, Q = {q:g} W")
        k = int(np.argmin(sw["curves_at_verify_h"][q]))
        ax[0].plot(s_fine[k], sw["curves_at_verify_h"][q][k], "*", color="crimson", ms=11)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("channel spacing s [mm] (fin count 24 ... 4)")
    ax[0].set_ylabel("max temperature rise [K]")
    ax[0].set_title(f"Design sweep at fin height {VERIFY_H:g} mm (star: surrogate optimum)", fontsize=9)
    ax[0].legend(fontsize=6, ncol=2)
    ax[0].grid(alpha=0.3, which="both")

    test = tr["test_idx"]
    y = tr["y"][test]
    p = np.exp(tr["mlp"].predict(tr["x"][test]))
    ax[1].plot(y, p, "o", ms=3, alpha=0.7)
    lim = [min(y.min(), p.min()) * 0.95, max(y.max(), p.max()) * 1.05]
    ax[1].plot(lim, lim, "k--", lw=0.8)
    m = tr["metrics"]["mlp"]["test"]
    ax[1].set_title(f"MLP surrogate, held-out designs (n = {m['n']}): MAPE {m['mape_pct']:.2f} %", fontsize=9)
    ax[1].set_xlabel("solver T rise [K]")
    ax[1].set_ylabel("surrogate T rise [K]")
    ax[1].grid(alpha=0.3)

    field = hs.temperature_field(best_design, device=DEVICE)
    im = ax[2].imshow(field, origin="lower", cmap="inferno", extent=[0, best_design.width_mm, 0, field.shape[0] * 0.5])
    ax[2].set_title(
        f"Solver field, winner: {best_design.n_fins} fins, s = {best_design.spacing_mm:.2f} mm, "
        f"Q = {best_design.power_w:g} W",
        fontsize=9,
    )
    ax[2].set_xlabel("x [mm]")
    ax[2].set_ylabel("y [mm]")
    fig.colorbar(im, ax=ax[2], label="T rise [K]", shrink=0.85)
    fig.suptitle("Warp heat-sink solver (CPU) -> numpy MLP surrogate -> design sweep -> solver check", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def metadata(args) -> dict:
    import warp as wp

    return {
        "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "warp": wp.__version__,
        "numpy": np.__version__,
        "device": DEVICE,
        "n_samples": args.n_samples,
        "seed": args.seed,
        "note": "CPU-only container; no GPU. PhysicsNeMo script (train_physicsnemo.py) not executed: needs GPU.",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-samples", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=HERE / "results")
    ap.add_argument("--reuse-dataset", action="store_true")
    ap.add_argument("--device", default="auto", help="auto (cuda:0 if present, else cpu), cpu, cuda:N")
    args = ap.parse_args(argv)
    global DEVICE  # noqa: PLW0603 (one switch for every solver call in this script)

    import warp as wp

    DEVICE = pick_device(args.device)

    wp.config.log_level = wp.LOG_WARNING
    hs = _sibling("heat_solver")
    out = args.out
    ds = out / "dataset.csv"
    t0 = time.perf_counter()
    rows = read_csv(ds) if args.reuse_dataset and ds.exists() else build_dataset(args.n_samples, args.seed, ds)
    data_s = time.perf_counter() - t0
    tr = train(rows, args.seed)
    tm = timing(tr["mlp"], rows)
    t0 = time.perf_counter()
    optima, sw = sweep(tr["mlp"])
    sweep_s = time.perf_counter() - t0
    ver = verify(tr["mlp"], optima)
    write_csv(optima, out / "sweep_optima.csv")
    main_case = next(w for w in ver["winners"] if w["fin_height_mm"] == VERIFY_H and w["power_w"] == VERIFY_Q)
    best = hs.Design(n_fins=main_case["surrogate_best_n"], fin_height_mm=VERIFY_H, power_w=VERIFY_Q)
    plot(out / "twin.png", tr, sw, ver, best)
    report = {
        "dataset": {
            "designs": len(rows),
            "build_s": round(data_s, 1),
            "energy_balance_max_abs": max(abs(float(r["energy_balance_err"])) for r in rows),
        },
        "surrogate": tr["metrics"],
        "timing": tm,
        "sweep": {"designs_evaluated": sw["designs_evaluated"], "wall_s": sweep_s},
        "verification": ver["winners"],
        "main_case": main_case,
    }
    (out / "surrogate_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "run_metadata.json").write_text(json.dumps(metadata(args), indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("surrogate", "timing", "main_case")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

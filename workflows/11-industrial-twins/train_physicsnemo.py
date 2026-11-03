"""Train the same surrogate with NVIDIA PhysicsNeMo on a GPU.

not executed: needs GPU (and the ``nvidia-physicsnemo`` package with a CUDA build of PyTorch, neither installed in
the CPU container this repo was built in).

What it does: reads ``results/dataset.csv`` written by ``run_twin.py``, maps (spacing, fin height, load) to
log(T_max rise) with a PhysicsNeMo fully connected network of the same width as the numpy MLP, trains with Adam
on the GPU, and saves a checkpoint plus the held-out error, so the two surrogates can be compared row by row.

API uncertainty (checked against no installed package here, so verify before running):
  * ``physicsnemo.models.mlp.fully_connected.FullyConnected`` and its keyword names (``in_features``,
    ``layer_size``, ``out_features``, ``num_layers``, ``activation_fn``) follow the Modulus 24.x/25.x API that
    PhysicsNeMo inherited. If the import path moved, ``physicsnemo.models.mlp.FullyConnected`` is the next guess.
  * ``Module.save(path)`` writing a ``.mdlus`` checkpoint is assumed from the same lineage.
  * Everything else is plain PyTorch.

Usage (GPU box)::

    pip install nvidia-physicsnemo        # plus a CUDA build of torch
    python train_physicsnemo.py --data results/dataset.csv --epochs 6000 --out physicsnemo_run/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_xy(path: Path):
    """(x, y) as nested lists: x = [spacing_mm, fin_height_mm, power_w], y = t_max_rise_k."""
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    x = [[float(r["spacing_mm"]), float(r["fin_height_mm"]), float(r["power_w"])] for r in rows]
    y = [float(r["t_max_rise_k"]) for r in rows]
    return x, y


def train(data: Path, out: Path, epochs: int = 6000, lr: float = 3e-3, seed: int = 0) -> dict:
    """not executed: needs GPU. Same split rule as run_twin.train (seeded permutation, first 20 % held out)."""
    import numpy as np
    import torch
    from physicsnemo.models.mlp.fully_connected import FullyConnected  # API uncertain, see module docstring

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y = load_xy(data)
    x = np.asarray(x, dtype=np.float32)
    logy = np.log(np.asarray(y, dtype=np.float32))
    idx = np.random.default_rng(seed).permutation(len(x))
    n_test = len(x) // 5
    test, tr = idx[:n_test], idx[n_test:]
    lo, hi = x[tr].min(0), x[tr].max(0)
    mu, sd = logy[tr].mean(), logy[tr].std()

    def scale(a):
        return torch.tensor(2.0 * (a - lo) / (hi - lo) - 1.0, device=device)

    xt, yt = scale(x[tr]), torch.tensor((logy[tr] - mu) / sd, device=device).unsqueeze(1)
    model = FullyConnected(in_features=3, layer_size=64, out_features=1, num_layers=2, activation_fn="tanh").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.mean((model(xt) - yt) ** 2)
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        pred = torch.exp(model(scale(x[test])).squeeze(1) * sd + mu).cpu().numpy()
    true = np.asarray(y, dtype=np.float32)[test]
    err = pred - true
    report = {
        "framework": "physicsnemo",
        "device": str(device),
        "epochs": epochs,
        "test_n": len(test),
        "test_mae_k": float(np.abs(err).mean()),
        "test_max_abs_k": float(np.abs(err).max()),
        "test_mape_pct": float((np.abs(err) / true).mean() * 100.0),
    }
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out / "heat_sink_surrogate.mdlus"))  # PhysicsNeMo Module checkpoint (API uncertain)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=HERE / "results" / "dataset.csv")
    ap.add_argument("--epochs", type=int, default=6000)
    ap.add_argument("--out", type=Path, default=HERE / "out" / "physicsnemo")
    args = ap.parse_args(argv)
    try:
        import physicsnemo  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:
        print(f"PhysicsNeMo/PyTorch not installed ({exc}); this script needs a GPU box. Not executed.")
        return 2
    print(json.dumps(train(args.data, args.out, args.epochs), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

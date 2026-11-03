#!/usr/bin/env python3
"""Train a 3D Gaussian splat with the open-source gsplat library on a capture prepared by ``prepare_capture.py``.

gsplat is the library the NuRec slides name: "NuRec + gsplat" in the reference pipeline, gsplat as the core library
in the NuRec SDK overview, HiGS "available in gsplat", and in the talk NVIDIA says it pushes its libraries "back to
gsplat, back to the community". This script is the open path, not NuRec: no NCore, no Fixer, no USDZ export.

Two parts:

* ``load_dataset`` / ``plan``  - CPU, tested: reads ``transforms.json``, converts each camera to gsplat's convention
  (world-to-camera ``viewmats`` with OpenCV axes, pinhole ``Ks``), loads initial points from ``sparse/0/points3D.txt``.
* ``train``                    - not executed: needs GPU (CUDA, torch, ``pip install gsplat``). A minimal loop over
  ``gsplat.rasterization`` with ``gsplat.strategy.DefaultStrategy`` densification, L1 photometric loss, and the ego
  mask applied to the loss (the Mcity pipeline masks the hood out of the reconstruction).

The upstream alternative is gsplat's own trainer on the COLMAP export, as in the gsplat examples README, e.g.
``python examples/simple_trainer.py default --data_dir prepared/ --result_dir results/`` (check ``--help`` for your
gsplat version; its COLMAP reader may want binary files: ``colmap model_converter --input_path prepared/sparse/0
--output_path prepared/sparse/0 --output_type BIN``).

Usage::

    python train_gsplat.py --data prepared/ --dry-run      # CPU: load, check, print the plan
    python train_gsplat.py --data prepared/ --steps 7000   # GPU (not executed here)
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

GL_FROM_CV = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass(frozen=True)
class View:
    image: Path
    mask: Path | None
    K: np.ndarray  # 3x3 pinhole
    viewmat: np.ndarray  # 4x4 world-to-camera, OpenCV axes (gsplat convention)
    width: int
    height: int
    camera: str


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 7000
    sh_degree: int = 3
    sh_increase_every: int = 1000
    init_opacity: float = 0.1
    init_scale: float = 1.0  # times the mean nearest-neighbour distance
    lr_means: float = 1.6e-4  # times scene scale
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_sh0: float = 2.5e-3
    lr_shN: float = 2.5e-3 / 20
    random_init_points: int = 100_000
    seed: int = 0


def load_dataset(data: Path) -> tuple[list[View], np.ndarray | None]:
    """Views from ``transforms.json`` plus initial points from ``sparse/0/points3D.txt`` (None if empty)."""
    data = Path(data)
    tf = json.loads((data / "transforms.json").read_text())
    views = []
    for f in tf["frames"]:
        c2w_cv = np.asarray(f["transform_matrix"], float) @ GL_FROM_CV  # undo the OpenGL flip
        K = np.array([[f["fl_x"], 0.0, f["cx"]], [0.0, f["fl_y"], f["cy"]], [0.0, 0.0, 1.0]])
        mask = data / f["mask_path"] if "mask_path" in f else None
        views.append(
            View(data / f["file_path"], mask, K, np.linalg.inv(c2w_cv), int(f["w"]), int(f["h"]), f.get("camera", ""))
        )
    pts_file = data / "sparse" / "0" / "points3D.txt"
    pts = None
    if pts_file.is_file():
        rows = [ln.split() for ln in pts_file.read_text().splitlines() if ln and not ln.startswith("#")]
        if rows:
            pts = np.array(
                [[float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[6])] for r in rows]
            )
    return views, pts


def scene_scale(views: Sequence[View]) -> float:
    """Radius of the camera centres around their mean (as gsplat's trainer normalizes learning rates)."""
    centres = np.array([np.linalg.inv(v.viewmat)[:3, 3] for v in views])
    return float(np.linalg.norm(centres - centres.mean(axis=0), axis=1).max() * 1.1)


def plan(data: Path, cfg: TrainConfig) -> dict:
    views, pts = load_dataset(data)
    missing = [str(v.image) for v in views if not v.image.exists()][:5]
    cams = sorted({v.camera for v in views})
    return {
        "data": str(data),
        "views": len(views),
        "cameras": cams,
        "resolution": sorted({(v.width, v.height) for v in views}),
        "masked_views": sum(v.mask is not None for v in views),
        "init_points": 0 if pts is None else len(pts),
        "init": "points3D.txt" if pts is not None else f"random ({cfg.random_init_points} points in the camera bbox)",
        "scene_scale_m": round(scene_scale(views), 3),
        "missing_images": missing,
        "config": asdict(cfg),
        "status": "plan only (CPU). Training: not executed: needs GPU",
    }


def train(data: Path, out: Path, cfg: TrainConfig) -> Path:  # pragma: no cover - GPU path
    """Optimize Gaussians against the prepared views. Not executed: needs GPU (CUDA + torch + gsplat)."""
    try:
        import torch
        import torch.nn.functional as F
        from gsplat import rasterization
        from gsplat.strategy import DefaultStrategy
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("needs a CUDA machine with torch and gsplat (pip install gsplat)") from exc
    if not torch.cuda.is_available():
        raise SystemExit("not executed: needs GPU (torch.cuda.is_available() is False)")

    dev = torch.device("cuda")
    torch.manual_seed(cfg.seed)
    views, pts = load_dataset(data)
    scale = scene_scale(views)
    if pts is None:
        centres = np.array([np.linalg.inv(v.viewmat)[:3, 3] for v in views])
        lo, hi = centres.min(0) - scale, centres.max(0) + scale
        rng = np.random.default_rng(cfg.seed)
        xyz = rng.uniform(lo, hi, (cfg.random_init_points, 3))
        rgb = np.full((len(xyz), 3), 0.5)
    else:
        xyz, rgb = pts[:, :3], pts[:, 3:6] / 255.0
    means = torch.tensor(xyz, dtype=torch.float32, device=dev)
    n = len(means)
    d2 = torch.cdist(means[: min(n, 20000)], means[: min(n, 20000)]).topk(4, largest=False).values[:, 1:].mean()
    dist = float(d2.clamp_min(1e-4))
    sh_c0 = 0.28209479177387814  # zeroth SH basis constant
    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(means),
            "scales": torch.nn.Parameter(torch.full((n, 3), math.log(dist * cfg.init_scale), device=dev)),
            "quats": torch.nn.Parameter(torch.rand((n, 4), device=dev)),
            "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), cfg.init_opacity, device=dev))),
            "sh0": torch.nn.Parameter(((torch.tensor(rgb, dtype=torch.float32, device=dev) - 0.5) / sh_c0)[:, None]),
            "shN": torch.nn.Parameter(torch.zeros((n, (cfg.sh_degree + 1) ** 2 - 1, 3), device=dev)),
        }
    )
    lrs = {
        "means": cfg.lr_means * scale,
        "scales": cfg.lr_scales,
        "quats": cfg.lr_quats,
        "opacities": cfg.lr_opacities,
        "sh0": cfg.lr_sh0,
        "shN": cfg.lr_shN,
    }
    optimizers = {k: torch.optim.Adam([params[k]], lr=lr, eps=1e-15) for k, lr in lrs.items()}
    strategy = DefaultStrategy()
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=scale)

    def load(view: View):
        img = torch.tensor(np.asarray(Image.open(view.image).convert("RGB")), dtype=torch.float32, device=dev) / 255.0
        mask = None
        if view.mask is not None:
            mask = torch.tensor(np.asarray(Image.open(view.mask).convert("L")) > 127, device=dev)
        return img, mask

    for step in range(cfg.steps):
        v = views[step % len(views)]
        gt, mask = load(v)
        colors = torch.cat([params["sh0"], params["shN"]], 1)
        renders, _alphas, info = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=colors,
            viewmats=torch.tensor(v.viewmat, dtype=torch.float32, device=dev)[None],
            Ks=torch.tensor(v.K, dtype=torch.float32, device=dev)[None],
            width=v.width,
            height=v.height,
            sh_degree=min(step // cfg.sh_increase_every, cfg.sh_degree),
            packed=False,
        )
        strategy.step_pre_backward(params, optimizers, state, step, info)
        err = (renders[0] - gt).abs()
        loss = err[mask].mean() if mask is not None else F.l1_loss(renders[0], gt)
        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
        if step % 500 == 0:
            print(f"step {step} loss {loss.item():.4f} gaussians {len(params['means'])}")
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "gaussians.pt"
    torch.save({k: p.detach().cpu() for k, p in params.items()}, ckpt)
    return ckpt


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="folder written by prepare_capture.py")
    ap.add_argument("--out", type=Path, default=Path("gsplat_out"))
    ap.add_argument("--steps", type=int, default=TrainConfig.steps)
    ap.add_argument("--dry-run", action="store_true", help="load and plan on CPU; no torch or gsplat")
    args = ap.parse_args(argv)
    cfg = TrainConfig(steps=args.steps)
    if args.dry_run:
        print(json.dumps(plan(args.data, cfg), indent=2, default=str))
        return 0
    print(f"[train_gsplat] wrote {train(args.data, args.out, cfg)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

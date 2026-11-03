#!/usr/bin/env python3
"""Why reconstruction quality falls with distance from the recorded trajectory: a 3D visibility proxy (CPU).

A Gaussian reconstruction can only reproduce surfaces that some training camera saw, and it reproduces them best from
directions close to the ones it saw them from. This script measures both on a toy street: a ground plane, building
facades with a side street, and parked cars, all as boxes. Training views come from one drive in the right lane
(cameras 1.6 m above the road, one position per metre, all directions, like a six-camera roof rig). Novel views are the
same drive shifted left by an offset. For every surface point a novel camera sees, it asks:

* never seen          - no training camera had a clear line of sight (disocclusion: the kerb and facade behind a
                        parked car, deep into the side street). A reconstruction has nothing there: holes, floaters.
* seen, wide angle    - seen, but only from directions more than ``--angle-tol`` degrees away from the novel ray.
                        Appearance and geometry errors grow with that angle (blurry lane lines, ghosting).
* well supported      - seen from within ``--angle-tol`` degrees.

A 2D top-down version of this is degenerate: a dense line of 360 degree cameras samples every ray that crosses it, so
camera height has to be modelled. This is a geometric proxy, not a renderer: no Gaussians are trained, and the angle
tolerance is an illustrative threshold, not a NuRec parameter. It explains the shape of the trend, not NuRec's error.

Usage::

    python novel_view_envelope.py --out results/       # writes novel_view_envelope.csv/.json/.png
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

CAM_H = 1.6  # camera height above the road (m)
EGO_Y = -1.75  # right-lane centre; road from -5.5 to +5.5, parking lanes at +-6.5, facades from +-9
OFFSETS = (0.0, 0.5, 1.0, 2.0, 3.5, 5.0, 7.0)  # leftward shift of the novel trajectory (m)
MAX_RANGE = 60.0  # ignore hits farther than this (m)


def street_scene() -> np.ndarray:
    """Axis-aligned boxes (B, 6): xmin, ymin, zmin, xmax, ymax, zmax. The ground plane z = 0 is handled separately."""
    boxes = [
        (-20.0, 9.0, 0.0, 140.0, 10.0, 12.0),  # left facade
        (-20.0, -10.0, 0.0, 40.0, -9.0, 12.0),  # right facade, side street at x 40..52
        (52.0, -10.0, 0.0, 140.0, -9.0, 12.0),
        (39.0, -30.0, 0.0, 40.0, -9.0, 12.0),  # side-street walls and end wall
        (52.0, -30.0, 0.0, 53.0, -9.0, 12.0),
        (40.0, -31.0, 0.0, 52.0, -30.0, 12.0),
    ]
    for x in (10.0, 25.0, 33.0, 75.0, 90.0):  # parked cars, 1.5 m tall (below the 1.6 m cameras)
        boxes.append((x - 2.25, -7.45, 0.0, x + 2.25, -5.55, 1.5))
    for x in (15.0, 56.0, 70.0, 85.0):
        boxes.append((x - 2.25, 5.55, 0.0, x + 2.25, 7.45, 1.5))
    boxes.append((56.0, -7.8, 0.0, 64.0, -5.4, 3.2))  # parked box truck, right side
    boxes.append((42.5, 5.5, 0.0, 48.0, 7.5, 2.4))  # parked van, left side
    return np.array(boxes, float)


def _slab(o: np.ndarray, d: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Entry/exit ray parameters for rays (..., 3) against boxes (B, 6); broadcasting over leading dims."""
    lo, hi = boxes[..., :3], boxes[..., 3:]
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t1, t2 = (lo - o) * inv, (hi - o) * inv
    t1 = np.where(np.isnan(t1), -np.inf, t1)
    t2 = np.where(np.isnan(t2), np.inf, t2)
    tmin = np.minimum(t1, t2).max(axis=-1)
    tmax = np.maximum(t1, t2).min(axis=-1)
    return tmin, tmax


def cast(origins: np.ndarray, dirs: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First hit (boxes or ground) of each unit ray within MAX_RANGE. Returns points (K, 3) and hit mask (K,)."""
    tmin, tmax = _slab(origins[:, None], dirs[:, None], boxes[None])
    t_box = np.where((tmax >= tmin) & (tmin > 1e-6), tmin, np.inf).min(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_ground = np.where(dirs[:, 2] < -1e-9, -origins[:, 2] / dirs[:, 2], np.inf)
    t = np.minimum(t_box, t_ground)
    hit = t <= MAX_RANGE
    return origins + dirs * np.where(hit, t, 0.0)[:, None], hit


def visible_from(points: np.ndarray, cams: np.ndarray, boxes: np.ndarray, chunk: int = 64) -> np.ndarray:
    """(K, T) bool: clear line of sight from camera t to point k (no box strictly between them)."""
    out = np.zeros((len(points), len(cams)), bool)
    for s in range(0, len(points), chunk):
        P = points[s : s + chunk]
        o = cams[None, :, None, :]  # (1, T, 1, 3)
        d = P[:, None, None, :] - o  # (k, T, 1, 3); segment parameter 0..1
        tmin, tmax = _slab(o, d, boxes[None, None])
        blocked = (tmax >= tmin) & (tmin < 1.0 - 1e-4) & (tmax > 1e-6)
        out[s : s + chunk] = ~blocked.any(axis=2)
    return out


def ray_directions(az_step: float = 4.0, el_min: float = -40.0, el_max: float = 20.0, el_step: float = 3.0):
    az = np.radians(np.arange(0.0, 360.0, az_step))
    el = np.radians(np.arange(el_min, el_max + 1e-9, el_step))
    A, E = np.meshgrid(az, el, indexing="ij")
    return np.column_stack([(np.cos(E) * np.cos(A)).ravel(), (np.cos(E) * np.sin(A)).ravel(), np.sin(E).ravel()])


def support(novel: np.ndarray, angle_tol_deg: float = 10.0, boxes: np.ndarray | None = None) -> dict:
    """Support statistics for novel camera positions (N, 3) against the recorded drive."""
    boxes = street_scene() if boxes is None else boxes
    train = np.column_stack([np.arange(0.0, 101.0, 1.0), np.full(101, EGO_Y), np.full(101, CAM_H)])
    dirs1 = ray_directions()
    origins = np.repeat(novel, len(dirs1), axis=0)
    dirs = np.tile(dirs1, (len(novel), 1))
    pts, hit = cast(origins, dirs, boxes)
    pts, origins = pts[hit], origins[hit]
    vis = visible_from(pts, train, boxes)
    v_n = origins - pts
    v_t = train[None] - pts[:, None]
    cosang = np.einsum("kd,ktd->kt", v_n, v_t) / (
        np.linalg.norm(v_n, axis=1)[:, None] * np.linalg.norm(v_t, axis=2) + 1e-12
    )
    ang = np.where(vis, np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))), np.inf)
    min_ang = ang.min(axis=1)
    seen = np.isfinite(min_ang)
    good = min_ang <= angle_tol_deg
    ground = np.abs(pts[:, 2]) < 1e-6
    return {
        "surface_samples": len(pts),
        "well_supported": round(float(good.mean()), 4),
        "seen_wide_angle": round(float((seen & ~good).mean()), 4),
        "never_seen": round(float((~seen).mean()), 4),
        "median_min_angle_deg": round(float(np.median(min_ang[seen])), 2) if seen.any() else None,
        "ground_well_supported": round(float(good[ground].mean()), 4) if ground.any() else None,
        "min_training_views_per_point": int(vis.sum(axis=1).min()) if len(pts) else 0,
    }


def envelope(offsets: Sequence[float] = OFFSETS, angle_tol_deg: float = 10.0) -> list[dict]:
    """Lateral shifts of the drive (case "lateral"), plus cameras turned into the side street (case "turn")."""
    xs = np.arange(30.0, 71.0, 5.0)
    rows = []
    for off in offsets:
        novel = np.column_stack([xs, np.full_like(xs, EGO_Y + off), np.full_like(xs, CAM_H)])
        rows.append({"case": "lateral", "offset_m": off, **support(novel, angle_tol_deg)})
    ys = np.array([-12.0, -16.0, -20.0, -24.0])
    turn = np.column_stack([np.full_like(ys, 46.0), ys, np.full_like(ys, CAM_H)])
    rows.append({"case": "turn into side street", "offset_m": None, **support(turn, angle_tol_deg)})
    return rows


def plot(rows: list[dict], path: Path, angle_tol: float) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in rows if r["case"] == "lateral"]
    x = [r["offset_m"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    ax.stackplot(
        x,
        [r["well_supported"] for r in rows],
        [r["seen_wide_angle"] for r in rows],
        [r["never_seen"] for r in rows],
        labels=[f"seen within {angle_tol:g} deg", f"seen only from > {angle_tol:g} deg", "never seen"],
        colors=["#76b900", "#f2c14e", "#d1495b"],
    )
    ax.plot(
        x,
        [r["ground_well_supported"] for r in rows],
        "k-o",
        ms=3,
        lw=1.2,
        label=f"road surface only: seen within {angle_tol:g} deg",
    )
    for xv, label in ((3.5, "1 lane"), (7.0, "2 lanes")):
        ax.axvline(xv, color="k", lw=0.8, ls="--")
        ax.text(xv - 0.1, 0.03, label, ha="right", va="bottom", fontsize=8)
    ax.set_xlabel("lateral offset of the novel trajectory from the recorded drive (m)")
    ax.set_ylabel("fraction of visible surface samples")
    ax.set_title("Toy street (3D boxes, cameras at 1.6 m): support for shifted views")
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--angle-tol", type=float, default=10.0)
    args = ap.parse_args(argv)
    rows = envelope(angle_tol_deg=args.angle_tol)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "novel_view_envelope.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    meta = {
        "what": "3D visibility proxy for novel-view support vs lateral offset (toy street of boxes; no rendering)",
        "angle_tol_deg": args.angle_tol,
        "training_views": f"x = 0..100 m every 1 m, y = {EGO_Y} m, z = {CAM_H} m, all directions",
        "novel_views": "x = 30..70 m every 5 m, shifted left by offset_m; rays every 4 deg azimuth, 3 deg elevation",
        "rows": rows,
    }
    (args.out / "novel_view_envelope.json").write_text(json.dumps(meta, indent=2) + "\n")
    plot(rows, args.out / "novel_view_envelope.png", args.angle_tol)
    for r in rows:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

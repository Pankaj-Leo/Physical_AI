"""Plot env-steps/s and ms/frame against world count from a scale_benchmark CSV.

Usage::

    python plot_scale.py results/cpu_scale.csv results/cpu_scale.png
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> dict[tuple[str, str, str], list[dict]]:
    series = defaultdict(list)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            series[(row["device"], row["solver"], row["graph"])].append(row)
    return series


def plot(csv_path: Path, png_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = load(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=120)
    for (device, solver, graph), unsorted in sorted(series.items()):
        rows = sorted(unsorted, key=lambda r: int(r["world_count"]))
        n = [int(r["world_count"]) for r in rows]
        label = f"{solver}, {'graph' if graph == 'True' else 'no graph'} ({device})"
        axes[0].plot(n, [float(r["env_steps_per_s"]) for r in rows], marker="o", label=label)
        axes[1].plot(n, [float(r["ms_per_frame"]) for r in rows], marker="o", label=label)
    for ax, title in zip(axes, ("env-steps per second (higher is better)", "ms per frame, all worlds"), strict=True):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("parallel worlds")
        ax.set_title(title, fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=7)
    devices = ", ".join(sorted({device for device, _, _ in series}))
    fig.suptitle(f"Newton quadruped, 100 Hz x 10 substeps, on {devices}", fontsize=10)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path)
    plt.close(fig)
    return png_path


if __name__ == "__main__":
    plot(Path(sys.argv[1]), Path(sys.argv[2]))

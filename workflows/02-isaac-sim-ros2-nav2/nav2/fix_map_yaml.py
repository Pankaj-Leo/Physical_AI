"""Make a shipped occupancy-map YAML loadable by nav2_map_server (wizard step 5, "Hidden gotcha").

The wizard infographic lists the common issues with NVIDIA's carter_navigation map: map frame or origin mismatch,
hard-coded absolute paths, outdated map or resolution. Its fix: "convert map to clean format (validate, check
resolution & origin)" then "rewrite map and param paths". The talk adds that some downloads failed "because of some
relative names". This script does the path rewrite and the validation; it runs on CPU (tested here).

    python3 fix_map_yaml.py SRC.yaml --out-dir maps/ --container-dir /ws/nav2/maps
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

REQUIRED = ("image", "resolution", "origin", "occupied_thresh", "free_thresh", "negate")


def check_map(meta: dict) -> list[str]:
    """Problems that stop map_server from loading, or that misplace the map."""
    problems = [f"missing key '{k}'" for k in REQUIRED if k not in meta]
    if "resolution" in meta and not (0 < float(meta["resolution"]) <= 1.0):
        problems.append(f"resolution {meta['resolution']} m/cell is outside (0, 1]")
    if "origin" in meta and (not isinstance(meta["origin"], list) or len(meta["origin"]) != 3):
        problems.append("origin must be [x, y, yaw]")
    if {"occupied_thresh", "free_thresh"} <= meta.keys() and not (
        0 <= float(meta["free_thresh"]) < float(meta["occupied_thresh"]) <= 1
    ):
        problems.append("need 0 <= free_thresh < occupied_thresh <= 1")
    if "negate" in meta and meta["negate"] not in (0, 1, True, False):
        problems.append("negate must be 0 or 1")
    return problems


def fix_map_yaml(src: Path, out_dir: Path, container_dir: str) -> dict:
    """Copy the map image next to a rewritten YAML whose ``image`` is an absolute path valid inside the container."""
    meta = yaml.safe_load(src.read_text())
    problems = check_map(meta)
    if problems:
        raise ValueError(f"{src}: " + "; ".join(problems))
    image = Path(str(meta["image"]))
    image_src = image if image.is_absolute() else src.parent / image
    if not image_src.exists():
        # Shipped YAMLs sometimes point at a path from the author's machine; fall back to the file name.
        image_src = src.parent / image.name
    if not image_src.exists():
        raise FileNotFoundError(f"map image not found for {src}: tried {image} and {image.name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if image_src.resolve() != (out_dir / image_src.name).resolve():
        shutil.copy2(image_src, out_dir / image_src.name)
    fixed = dict(meta)
    fixed["image"] = f"{container_dir.rstrip('/')}/{image_src.name}"
    (out_dir / src.name).write_text(yaml.safe_dump(fixed, sort_keys=False))
    return fixed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "maps")
    ap.add_argument("--container-dir", default="/ws/nav2/maps")
    args = ap.parse_args(argv)
    fixed = fix_map_yaml(args.src, args.out_dir, args.container_dir)
    print(f"wrote {args.out_dir / args.src.name}: image={fixed['image']} resolution={fixed['resolution']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

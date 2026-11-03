"""Run the whole workflow on CPU and write the committed results.

    build_asset -> validate -> load_in_newton (1 s, XPBD, CPU) -> composition_demo

Generated layers go to ``out/`` (git-ignored). Small JSON reports go to ``results/``, and one layer
(``asset_physics.usda``) is copied to ``sample/`` so readers can see the UsdPhysics authoring without running.

Usage::

    python run_all.py            # about 1 minute on 4 CPU cores after Warp kernels are cached
"""

from __future__ import annotations

import datetime as _dt
import importlib.util
import json
import platform
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_sibling(name: str):
    """Import a module from this folder under a unique name (workflow folders reuse short names)."""
    spec = importlib.util.spec_from_file_location(f"wf01_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def metadata() -> dict:
    import newton
    import numpy
    import warp
    from pxr import Usd

    return {
        "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "usd": ".".join(map(str, Usd.GetVersion())),
        "newton": newton.__version__,
        "warp": warp.__version__,
        "numpy": numpy.__version__,
        "device": "cpu",
        "note": "CPU-only container; no GPU was used.",
    }


def main() -> int:
    out, results, sample = HERE / "out", HERE / "results", HERE / "sample"
    results.mkdir(exist_ok=True)
    sample.mkdir(exist_ok=True)
    build = load_sibling("build_asset").build(out)
    (results / "build_summary.json").write_text(json.dumps(build, indent=2) + "\n")

    validate = load_sibling("validate")
    rep = validate.report(validate.validate(out / "asset.usda"))
    (results / "validation_report.json").write_text(json.dumps(rep, indent=2) + "\n")

    newton_res = load_sibling("load_in_newton").run(out / "asset.usda", seconds=1.0)
    (results / "newton_load.json").write_text(json.dumps(newton_res, indent=2) + "\n")

    comp = load_sibling("composition_demo").run(out / "composition")
    (results / "composition_demo.json").write_text(json.dumps(comp, indent=2) + "\n")

    shutil.copy(out / "asset_physics.usda", sample / "asset_physics.usda")
    (results / "run_metadata.json").write_text(json.dumps(metadata(), indent=2) + "\n")
    ok = rep["passed"] and all(c["ok"] for c in comp.values()) and newton_res["usd"]["sim"]["finite"]
    print(f"validation passed: {rep['passed']}; composition cases ok: {all(c['ok'] for c in comp.values())}")
    z_usd, z_urdf = newton_res["usd"]["sim"]["base_z_end_m"], newton_res["urdf"]["sim"]["base_z_end_m"]
    print(f"newton base z after 1 s: USD {z_usd} m, URDF {z_urdf} m")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

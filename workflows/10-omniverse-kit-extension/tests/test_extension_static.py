"""Static checks on the Kit extension: manifests parse, module path resolves, Python compiles, imports are guarded."""

from __future__ import annotations

import ast
import importlib
import py_compile
import sys
from pathlib import Path

import tomllib

FOLDER = Path(__file__).resolve().parents[1]
EXT = FOLDER / "exts/physicalai.atlas.tools"


def _toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_extension_toml_has_required_keys():
    cfg = _toml(EXT / "config/extension.toml")
    pkg = cfg["package"]
    for key in ("version", "title", "description", "category", "keywords"):
        assert pkg.get(key), key
    assert pkg["version"].count(".") == 2
    for dep in ("omni.ui", "omni.usd", "omni.kit.commands"):
        assert dep in cfg["dependencies"]
    modules = cfg["python"]["module"]
    assert [m["name"] for m in modules] == ["physicalai.atlas.tools"]
    # readme/changelog the Extension Manager shows (Introduction 03) exist
    assert (EXT / pkg["readme"]).is_file() and (EXT / pkg["changelog"]).is_file()


def test_python_module_maps_to_package_folders():
    name = _toml(EXT / "config/extension.toml")["python"]["module"][0]["name"]
    pkg_dir = EXT.joinpath(*name.split("."))
    assert (pkg_dir / "__init__.py").is_file()
    assert (pkg_dir / "extension.py").is_file()


def test_all_extension_python_compiles():
    files = sorted(EXT.rglob("*.py"))
    assert len(files) >= 6
    for f in files:
        py_compile.compile(str(f), doraise=True)


def test_extension_class_implements_iext():
    tree = ast.parse((EXT / "physicalai/atlas/tools/extension.py").read_text(encoding="utf-8"))
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    cls = classes["AtlasToolsExtension"]
    assert any(ast.unparse(b) == "omni.ext.IExt" for b in cls.bases)
    methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    assert {"on_startup", "on_shutdown", "_on_load_robot", "_on_randomize_lighting", "_on_switch_variant"} <= methods
    src = ast.unparse(cls)
    for label in ("Load robot", "Randomize lighting", "Next variant", "Random variant"):
        assert f"'{label}'" in src


def test_package_imports_outside_kit():
    sys.path.insert(0, str(EXT))
    try:
        for mod in [m for m in sys.modules if m.startswith("physicalai")]:
            del sys.modules[mod]
        pkg = importlib.import_module("physicalai.atlas.tools")
        helpers = importlib.import_module("physicalai.atlas.tools.helpers")
    finally:
        sys.path.remove(str(EXT))
    assert pkg.AtlasToolsExtension is None  # omni is absent here, and the guard caught it
    assert helpers.DEFAULT_SEED == 2026


def test_kit_snippet_parses_and_adds_the_extension():
    kit = _toml(FOLDER / "kit/add_to_app.kit")
    deps = kit["dependencies"]
    assert "physicalai.atlas.tools" in deps
    assert "omni.physx.bundle" in deps  # Introduction 01: not in the template by default
    assert isinstance(kit["settings"]["app"]["exts"]["folders"]["++"], list)

"""CPU tests for the SimReady asset workflow: build, validate (positive and negative), Newton load, LIVERPS."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"wf01_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


build_asset = _load("build_asset")
validate = _load("validate")
composition_demo = _load("composition_demo")


@pytest.fixture(scope="module")
def asset_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("simready")
    build_asset.build(out)
    return out


def test_build_layers_and_instancing(asset_dir):
    for name in build_asset.ALL_LAYERS:
        assert (asset_dir / name).exists(), name
    summary = build_asset.build(asset_dir)
    assert summary["links"] == 13
    assert summary["joints"] == 12
    # 13 visuals but only 3 distinct parts: base shell, hip housing, leg segment.
    assert summary["instanced_visuals"] == 13
    assert summary["prototypes"] == 3


def test_validator_passes(asset_dir):
    checks = validate.validate(asset_dir / "asset.usda")
    failed = [c for c in checks if not c.ok]
    assert not failed, failed
    assert len(checks) == 13


def _copy(asset_dir: Path, tmp_path: Path) -> Path:
    dst = tmp_path / "asset"
    shutil.copytree(asset_dir, dst)
    return dst


def _failed(path: Path) -> set[str]:
    return {c.name for c in validate.validate(path / "asset.usda") if not c.ok}


def test_validator_catches_missing_payload(asset_dir, tmp_path):
    bad = _copy(asset_dir, tmp_path)
    (bad / "asset_sensors.usda").unlink()
    assert "payloads_resolve" in _failed(bad)


def test_validator_catches_unit_mismatch(asset_dir, tmp_path):
    from pxr import Sdf

    bad = _copy(asset_dir, tmp_path)
    layer = Sdf.Layer.FindOrOpen(str(bad / "asset_base.usda"))
    layer.pseudoRoot.SetInfo("metersPerUnit", 0.01)  # a centimeter layer referenced into a meter asset
    layer.Save()
    assert "layer_units_agree" in _failed(bad)


def test_validator_catches_second_articulation_root(asset_dir, tmp_path):
    from pxr import Usd, UsdPhysics

    bad = _copy(asset_dir, tmp_path)
    stage = Usd.Stage.Open(str(bad / "asset_physics.usda"))
    # The Basic Training 04 mistake: the importer put the root on the chassis link instead of the robot prim.
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath("/quadruped/base"))
    stage.GetRootLayer().Save()
    assert "single_articulation" in _failed(bad)


def test_validator_catches_missing_limits(asset_dir, tmp_path):
    from pxr import Usd

    bad = _copy(asset_dir, tmp_path)
    stage = Usd.Stage.Open(str(bad / "asset_physics.usda"))
    stage.GetPrimAtPath("/quadruped/joints/LF_KFE").GetAttribute("physics:upperLimit").Clear()
    stage.GetRootLayer().Save()
    assert "joint_limits" in _failed(bad)


def test_composition_liverps():
    res = composition_demo.run()
    assert all(r["ok"] for r in res.values()), res
    steps = res["reference_vs_variant_vs_local"]["steps"]
    assert steps["reference only"]["value"] == "red"
    assert steps["variant black selected"]["value"] == "black"
    assert steps["local pink authored, variant still black"]["value"] == "pink"


def test_newton_loads_asset_and_steps(asset_dir):
    pytest.importorskip("newton")
    load = _load("load_in_newton")
    b, _info = load.build_from_usd(asset_dir / "asset.usda")
    assert b.body_count == 13
    assert b.joint_dof_count == 18  # 6 free-base DOFs + 12 revolute
    sim = load.simulate(b, seconds=0.2)
    assert sim["finite"]
    assert 0.3 < sim["base_z_end_m"] < 0.75

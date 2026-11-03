"""CPU tests for the USD side of the extension, on real USD stages built with ``pxr`` (usd-core).

A small robot asset with two variant sets is written to a temp .usda file, then referenced into a stage exactly as the
"Load robot" button does; the variant and lighting buttons then act on the composed stage.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

pxr = pytest.importorskip("pxr")
from pxr import Usd, UsdGeom, UsdLux  # noqa: E402

FOLDER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FOLDER / "exts/physicalai.atlas.tools"))
from physicalai.atlas.tools import helpers, usd_ops  # noqa: E402

sys.path.pop(0)


@pytest.fixture
def robot_asset(tmp_path: Path) -> Path:
    """robot.usda: /Robot (defaultPrim) with variant sets 'material' {red, blue, steel} and 'config' {arm_5dof,
    arm_7dof}; each material variant authors a different displayColor."""
    path = tmp_path / "robot.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Robot").GetPrim()
    stage.SetDefaultPrim(root)
    body = UsdGeom.Cube.Define(stage, "/Robot/Body")
    vsets = root.GetVariantSets()
    material = vsets.AddVariantSet("material")
    colors = {"red": (1, 0, 0), "blue": (0, 0, 1), "steel": (0.6, 0.6, 0.65)}
    for name, rgb in colors.items():
        material.AddVariant(name)
        material.SetVariantSelection(name)
        with material.GetVariantEditContext():
            body.CreateDisplayColorAttr().Set([rgb])
    material.SetVariantSelection("red")
    config = vsets.AddVariantSet("config")
    for name in ("arm_5dof", "arm_7dof"):
        config.AddVariant(name)
    config.SetVariantSelection("arm_7dof")
    stage.GetRootLayer().Save()
    return path


def test_load_robot_references_the_asset_at_a_unique_path(robot_asset):
    stage = Usd.Stage.CreateInMemory()
    a = usd_ops.add_robot_reference(stage, str(robot_asset))
    b = usd_ops.add_robot_reference(stage, str(robot_asset))
    assert str(a.GetPath()) == "/World/robot"
    assert str(b.GetPath()) == "/World/robot_01"
    assert stage.GetPrimAtPath("/World/robot/Body").IsValid()  # composed through the reference
    assert a.HasAuthoredReferences()
    with pytest.raises(ValueError):
        usd_ops.add_robot_reference(stage, "")


def test_variant_sets_survive_the_reference_and_switch(robot_asset):
    stage = Usd.Stage.CreateInMemory()
    robot = usd_ops.add_robot_reference(stage, str(robot_asset))
    assert usd_ops.list_variant_sets(robot) == {
        "material": ["blue", "red", "steel"],
        "config": ["arm_5dof", "arm_7dof"],
    }
    body = stage.GetPrimAtPath(f"{robot.GetPath()}/Body")
    color = UsdGeom.Gprim(body).GetDisplayColorAttr()
    assert tuple(color.Get()[0]) == (1, 0, 0)

    # empty set name -> the look-like set "material"; authored order on this asset is alphabetical: blue, red, steel
    assert usd_ops.switch_variant(robot, "", "next") == ("material", "steel")
    assert tuple(round(c, 2) for c in color.Get()[0]) == (0.6, 0.6, 0.65)
    assert usd_ops.switch_variant(robot, "", "next") == ("material", "blue")
    assert tuple(color.Get()[0]) == (0, 0, 1)
    assert usd_ops.switch_variant(robot, "config", "next") == ("config", "arm_5dof")

    rng = random.Random(1)
    set_name, new = usd_ops.switch_variant(robot, "material", "random", rng)
    assert set_name == "material" and new != "blue"
    assert usd_ops.get_variant(robot, "material") == new
    assert usd_ops.switch_variant(robot, "Material", "next") is None  # case-sensitive set names
    with pytest.raises(ValueError):
        usd_ops.switch_variant(robot, "", "sideways")


def test_switch_variant_on_prim_without_sets_returns_none():
    stage = Usd.Stage.CreateInMemory()
    prim = UsdGeom.Xform.Define(stage, "/World/Plain").GetPrim()
    assert usd_ops.switch_variant(prim) is None


def test_randomize_lighting_creates_then_reuses_three_lights():
    stage = Usd.Stage.CreateInMemory()
    first = usd_ops.randomize_lighting(stage, random.Random(3))
    lights = [stage.GetPrimAtPath(f"/World/AtlasLights/Light_{i:02d}") for i in range(3)]
    assert all(p.IsValid() and p.IsA(UsdLux.SphereLight) for p in lights)
    for prim, p in zip(lights, first, strict=True):
        light = UsdLux.SphereLight(prim)
        assert light.GetIntensityAttr().Get() == pytest.approx(p["intensity"])
        assert light.GetEnableColorTemperatureAttr().Get() is True
        assert light.GetColorTemperatureAttr().Get() == pytest.approx(p["temperature"])
        translate = UsdGeom.Xformable(prim).GetOrderedXformOps()[0].Get()
        assert tuple(translate) == pytest.approx(p["position"])
    second = usd_ops.randomize_lighting(stage, random.Random(4))
    assert second != first
    children = stage.GetPrimAtPath("/World/AtlasLights").GetChildren()
    assert len(children) == 3  # reused, not duplicated
    assert len(UsdGeom.Xformable(lights[0]).GetOrderedXformOps()) == 1  # one translate op, overwritten


def test_selected_or_default():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/A")
    UsdGeom.Xform.Define(stage, "/World/B")
    assert str(usd_ops.selected_or_default(stage, ["/World/B"], "/World/A").GetPath()) == "/World/B"
    assert str(usd_ops.selected_or_default(stage, ["/World/Gone"], "/World/A").GetPath()) == "/World/A"
    assert usd_ops.selected_or_default(stage, [], "") is None
    assert helpers.DEFAULT_SEED == 2026

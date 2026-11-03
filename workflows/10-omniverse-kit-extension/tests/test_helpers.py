"""CPU tests for the pure helpers behind the extension's buttons."""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import pytest

FOLDER = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "atlas_helpers", FOLDER / "exts/physicalai.atlas.tools/physicalai/atlas/tools/helpers.py"
)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def test_default_robot_usd_reads_env():
    assert h.default_robot_usd({}) == ""
    assert h.default_robot_usd({"ATLAS_ROBOT_USD": "/assets/robot.usd"}) == "/assets/robot.usd"


@pytest.mark.parametrize(
    ("path", "name"),
    [
        ("/assets/franka_panda.usd", "franka_panda"),
        ("omniverse://host/Robots/Nova Carter.usda", "Nova_Carter"),
        ("https://x/y/robot.usdz?version=2", "robot"),
        ("C:\\assets\\ur10e.usd", "ur10e"),
        ("/assets/3d-robot.usd", "_3d_robot"),
        ("/", "Robot"),
    ],
)
def test_prim_name_from_asset(path, name):
    assert h.prim_name_from_asset(path) == name


def test_unique_prim_path():
    assert h.unique_prim_path([], "/World/Robot") == "/World/Robot"
    assert h.unique_prim_path(["/World/Robot"], "/World/Robot") == "/World/Robot_01"
    assert h.unique_prim_path(["/World/Robot", "/World/Robot_01"], "/World/Robot") == "/World/Robot_02"


def test_choose_variant_set_priority_and_preference():
    sets = {"config": ["arm_5dof", "arm_7dof"], "Looks": ["red", "blue"], "empty": []}
    assert h.choose_variant_set(sets) == "Looks"  # look-like set wins, case-insensitive
    assert h.choose_variant_set(sets, "config") == "config"
    assert h.choose_variant_set(sets, "missing") is None
    assert h.choose_variant_set(sets, "empty") is None  # no variants to switch to
    assert h.choose_variant_set({"b": ["x"], "a": ["y"]}) == "a"
    assert h.choose_variant_set({}) is None


def test_next_variant_wraps_and_handles_unset():
    names = ["black", "blue", "white"]
    assert h.next_variant(names, None) == "black"
    assert h.next_variant(names, "black") == "blue"
    assert h.next_variant(names, "white") == "black"
    assert h.next_variant(names, "Black") == "black"  # case-sensitive: unknown name restarts the cycle
    with pytest.raises(ValueError):
        h.next_variant([], None)


def test_random_variant_always_changes_when_possible():
    rng = random.Random(0)
    names = ["red", "blue", "steel"]
    picks = {h.random_variant(names, "red", rng) for _ in range(50)}
    assert picks == {"blue", "steel"}
    assert h.random_variant(["only"], "only", rng) == "only"


def test_sample_light_params_ranges_and_determinism():
    a = h.sample_light_params(random.Random(7), 4)
    b = h.sample_light_params(random.Random(7), 4)
    assert a == b and len(a) == 4
    for p in a:
        assert h.INTENSITY_RANGE[0] <= p["intensity"] <= h.INTENSITY_RANGE[1]
        assert h.TEMPERATURE_RANGE_K[0] <= p["temperature"] <= h.TEMPERATURE_RANGE_K[1]
        assert all(lo <= x <= hi for x, lo, hi in zip(p["position"], h.POSITION_MIN, h.POSITION_MAX, strict=True))
    with pytest.raises(ValueError):
        h.sample_light_params(random.Random(0), 0)


def test_parse_variant_event_from_browser_payload():
    assert h.VARIANT_EVENT == "atlas_switch_variant"
    assert h.parse_variant_event({}) == ("next", "")
    assert h.parse_variant_event({"mode": "random", "variant_set": " material "}) == ("random", "material")
    with pytest.raises(ValueError, match="mode"):
        h.parse_variant_event({"mode": "Next"})  # case-sensitive, like the tutorial's event names
    with pytest.raises(ValueError, match="variant_set"):
        h.parse_variant_event({"variant_set": 3})

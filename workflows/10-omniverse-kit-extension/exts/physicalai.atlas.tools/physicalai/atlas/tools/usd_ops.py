"""USD operations behind the Atlas Tools buttons, written against ``pxr`` only.

Inside Kit, ``extension.py`` passes ``omni.usd.get_context().get_stage()``; in this repo's tests the same functions run
on an in-memory ``Usd.Stage`` with the ``usd-core`` build of ``pxr`` (the usdview tutorial's Python environment).
Executed here (tests/test_usd_ops.py).
"""

from __future__ import annotations

import random

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

from . import helpers

LIGHTS_ROOT = "/World/AtlasLights"


def add_robot_reference(stage: Usd.Stage, asset_path: str, parent: str = "/World", name: str | None = None) -> Usd.Prim:
    """Reference a robot USD under ``parent`` at a fresh path and return the new prim.

    A reference (not a sublayer) keeps the asset's own layer stack intact and lets several robots share one file;
    variant sets authored in the asset stay selectable on the referencing prim.
    """
    if not asset_path:
        raise ValueError("empty asset path: set the Robot USD field or $ATLAS_ROBOT_USD")
    if not stage.GetPrimAtPath(parent):
        UsdGeom.Xform.Define(stage, parent)
    base = f"{parent.rstrip('/')}/{name or helpers.prim_name_from_asset(asset_path)}"
    existing = {str(p.GetPath()) for p in stage.Traverse()}
    path = helpers.unique_prim_path(existing, base)
    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    prim.GetReferences().AddReference(asset_path)
    return prim


def ensure_lights(stage: Usd.Stage, count: int = 3, root: str = LIGHTS_ROOT) -> list[Usd.Prim]:
    """Sphere lights ``root/Light_00`` .. ``Light_{count-1}``, created on first use and reused afterwards."""
    if not stage.GetPrimAtPath(root):
        UsdGeom.Xform.Define(stage, root)
    lights = []
    for i in range(count):
        path = f"{root}/Light_{i:02d}"
        prim = stage.GetPrimAtPath(path)
        if not prim:
            light = UsdLux.SphereLight.Define(stage, path)
            light.CreateRadiusAttr(0.25)
            prim = light.GetPrim()
        lights.append(prim)
    return lights


def apply_light_params(prim: Usd.Prim, params: dict) -> None:
    """Set intensity, colour temperature and translation on one light."""
    light = UsdLux.SphereLight(prim)
    light.CreateIntensityAttr().Set(float(params["intensity"]))
    light.CreateEnableColorTemperatureAttr().Set(True)
    light.CreateColorTemperatureAttr().Set(float(params["temperature"]))
    xform = UsdGeom.Xformable(prim)
    ops = [op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    op = ops[0] if ops else xform.AddTranslateOp()
    op.Set(Gf.Vec3d(*params["position"]))


def randomize_lighting(stage: Usd.Stage, rng: random.Random, count: int = 3) -> list[dict]:
    """The "Randomize lighting" button: sample ``count`` settings and apply them; returns what was applied."""
    params = helpers.sample_light_params(rng, count)
    for prim, p in zip(ensure_lights(stage, count), params, strict=True):
        apply_light_params(prim, p)
    return params


def list_variant_sets(prim: Usd.Prim) -> dict[str, list[str]]:
    """{variant set name: variant names} as composed on ``prim`` (includes sets that come through references)."""
    vsets = prim.GetVariantSets()
    return {name: list(vsets.GetVariantSet(name).GetVariantNames()) for name in vsets.GetNames()}


def get_variant(prim: Usd.Prim, set_name: str) -> str:
    return prim.GetVariantSets().GetVariantSet(set_name).GetVariantSelection()


def switch_variant(prim: Usd.Prim, preferred_set: str = "", mode: str = "next", rng: random.Random | None = None):
    """The "Next variant" / "Random variant" buttons. Returns (set name, new variant) or None if nothing to switch.

    Variant names are case-sensitive (Basic Training 08: "These names are case sensitive").
    """
    sets = list_variant_sets(prim)
    set_name = helpers.choose_variant_set(sets, preferred_set)
    if set_name is None:
        return None
    current = get_variant(prim, set_name) or None
    if mode == "next":
        new = helpers.next_variant(sets[set_name], current)
    elif mode == "random":
        new = helpers.random_variant(sets[set_name], current, rng or random.Random(helpers.DEFAULT_SEED))
    else:
        raise ValueError(f"mode must be 'next' or 'random', got {mode!r}")
    prim.GetVariantSets().GetVariantSet(set_name).SetVariantSelection(new)
    return set_name, new


def selected_or_default(stage: Usd.Stage, selected_paths: list[str], default: str) -> Usd.Prim | None:
    """First valid selected prim, else the prim at ``default`` (the last loaded robot), else None."""
    for p in [*selected_paths, default]:
        if p and Sdf.Path.IsValidPathString(p):
            prim = stage.GetPrimAtPath(p)
            if prim:
                return prim
    return None

"""LIVERPS in five small stages, reproducing the exercises from the OpenUSD tutorial transcripts.

LIVERPS is USD's strength order for opinions at one prim: Local, Inherits, VariantSets, (R)elocates,
References, Payloads, Specializes. Each case below builds an in-memory stage, resolves one attribute, and
records which layer/arc supplied the winning opinion.

1. reference_vs_variant_vs_local   red cube referenced; variant "black" selected; local "pink" authored.
                                   Module 3 (25:02-30:49): local beats variant beats reference.
2. sublayer_order                  two sublayers set the same value; the first listed wins.
                                   NCP 1.1 and Module 4 (30:57-31:11): "the first one will be the strongest layer".
3. local_vs_inherit                class opinion vs local opinion. NCP 1.2 (28:58): local beats inherits.
4. reference_vs_payload            same attribute from a reference and a payload; the reference wins.
                                   Module 3 (19:41-19:56): "references will be stronger than payloads".
5. payload_unloaded                the payload's opinion disappears when payloads are not loaded.
                                   Module 3 (21:43-22:21) and Basic Training 01 (31:41): payloads load on demand.

Usage::

    python composition_demo.py                  # prints a table
    python composition_demo.py --write out/composition   # also exports each case as .usda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom

RED, BLACK, WHITE, PINK = (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1.0, 0.4, 0.7)
GREEN, BLUE = (0.0, 0.6, 0.0), (0.0, 0.0, 1.0)
NAMES = {RED: "red", BLACK: "black", WHITE: "white", PINK: "pink", GREEN: "green", BLUE: "blue"}


def _color(prim: Usd.Prim) -> str:
    v = UsdGeom.Gprim(prim).GetDisplayColorAttr().Get()
    if not v:
        return "none"
    key = tuple(round(float(c), 3) for c in v[0])
    return NAMES.get(key, str(key))


def _set_color(prim: Usd.Prim, rgb) -> None:
    UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*rgb)])


def _source(prim: Usd.Prim) -> str:
    """Layer and path of the strongest opinion for displayColor (first entry of the property stack)."""
    stack = prim.GetAttribute("primvars:displayColor").GetPropertyStack(Usd.TimeCode.Default())
    if not stack:
        return "fallback"
    spec = stack[0]
    layer = spec.layer.identifier
    name = Path(layer).name if not spec.layer.anonymous else spec.layer.GetDisplayName() or "anon"
    return f"{name}:{spec.path}"


def _cube_layer(rgb, tag: str) -> Sdf.Layer:
    """An asset layer with a single Cube as defaultPrim."""
    layer = Sdf.Layer.CreateAnonymous(f"{tag}.usda")
    stage = Usd.Stage.Open(layer)
    cube = UsdGeom.Cube.Define(stage, "/Cube").GetPrim()
    _set_color(cube, rgb)
    stage.SetDefaultPrim(cube)
    return layer


def case_reference_variant_local() -> dict:
    asset = _cube_layer(RED, "red_cube")
    stage = Usd.Stage.CreateInMemory("shot.usda")
    prim = UsdGeom.Cube.Define(stage, "/World/cube_reference").GetPrim()
    prim.GetReferences().AddReference(asset.identifier)
    steps = {"reference only": (_color(prim), _source(prim))}

    vset = prim.GetVariantSets().AddVariantSet("paint")
    for name, rgb in (("black", BLACK), ("white", WHITE)):
        vset.AddVariant(name)
        vset.SetVariantSelection(name)
        with vset.GetVariantEditContext():
            _set_color(prim, rgb)
    vset.SetVariantSelection("black")
    steps["variant black selected"] = (_color(prim), _source(prim))
    vset.SetVariantSelection("white")
    steps["variant switched to white"] = (_color(prim), _source(prim))
    vset.SetVariantSelection("black")
    _set_color(prim, PINK)
    steps["local pink authored, variant still black"] = (_color(prim), _source(prim))
    return {"steps": steps, "final": _color(prim), "stage": stage}


def case_sublayer_order() -> dict:
    green = Sdf.Layer.CreateAnonymous("green_layer.usda")
    blue = Sdf.Layer.CreateAnonymous("blue_layer.usda")
    for layer, rgb in ((green, GREEN), (blue, BLUE)):
        s = Usd.Stage.Open(layer)
        _set_color(UsdGeom.Cube.Define(s, "/World/box").GetPrim(), rgb)
    stage = Usd.Stage.CreateInMemory("root.usda")
    stage.GetRootLayer().subLayerPaths = [green.identifier, blue.identifier]
    prim = stage.GetPrimAtPath("/World/box")
    first = (_color(prim), _source(prim))
    stage.GetRootLayer().subLayerPaths = [blue.identifier, green.identifier]
    swapped = (_color(stage.GetPrimAtPath("/World/box")), _source(stage.GetPrimAtPath("/World/box")))
    steps = {"subLayers [green, blue]": first, "subLayers [blue, green]": swapped}
    return {"steps": steps, "final": swapped[0], "stage": stage}


def case_local_vs_inherit() -> dict:
    stage = Usd.Stage.CreateInMemory("inherit.usda")
    cls = stage.CreateClassPrim("/_robot_paint")
    cls.CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray).Set([Gf.Vec3f(*GREEN)])
    prim = UsdGeom.Cube.Define(stage, "/World/box").GetPrim()
    prim.GetInherits().AddInherit("/_robot_paint")
    steps = {"inherits class green": (_color(prim), _source(prim))}
    _set_color(prim, PINK)
    steps["local pink authored"] = (_color(prim), _source(prim))
    return {"steps": steps, "final": _color(prim), "stage": stage}


# Anonymous layers live only while referenced from Python; an unloaded payload is not held by the stage,
# so keep every layer we create alive for the life of the process.
_KEEP: list[Sdf.Layer] = []


def _ref_and_payload_stage(load: Usd.Stage.InitialLoadSet) -> tuple[Usd.Stage, Usd.Prim]:
    ref = _cube_layer(RED, "ref_red")
    pay = _cube_layer(BLUE, "payload_blue")
    pay_stage = Usd.Stage.Open(pay)
    pay_stage.GetPrimAtPath("/Cube").CreateAttribute("note", Sdf.ValueTypeNames.String).Set("from payload")
    root = Sdf.Layer.CreateAnonymous("root.usda")
    s = Usd.Stage.Open(root)
    p = UsdGeom.Cube.Define(s, "/World/box").GetPrim()
    p.GetReferences().AddReference(ref.identifier)
    p.GetPayloads().AddPayload(pay.identifier)
    stage = Usd.Stage.Open(root, load)
    _KEEP.extend([ref, pay, root])
    return stage, stage.GetPrimAtPath("/World/box")


def case_reference_vs_payload() -> dict:
    stage, prim = _ref_and_payload_stage(Usd.Stage.LoadAll)
    note = prim.GetAttribute("note").Get()
    steps = {"reference red + payload blue": (_color(prim), _source(prim)), "payload-only attr": (str(note), "")}
    return {"steps": steps, "final": _color(prim), "stage": stage}


def case_payload_unloaded() -> dict:
    stage, prim = _ref_and_payload_stage(Usd.Stage.LoadNone)
    has_note = prim.GetAttribute("note").IsValid() and prim.GetAttribute("note").HasAuthoredValue()
    steps = {"LoadNone: payload-only attr present?": (str(has_note), ""), "color": (_color(prim), _source(prim))}
    stage.Load("/World/box")
    steps["after stage.Load(): payload-only attr"] = (str(prim.GetAttribute("note").Get()), "")
    return {"steps": steps, "final": str(has_note), "stage": stage}


CASES = {
    "reference_vs_variant_vs_local": (case_reference_variant_local, "pink"),
    "sublayer_order": (case_sublayer_order, "blue"),
    "local_vs_inherit": (case_local_vs_inherit, "pink"),
    "reference_vs_payload": (case_reference_vs_payload, "red"),
    "payload_unloaded": (case_payload_unloaded, "False"),
}


def run(write: Path | None = None) -> dict:
    """Run every case; ``expected`` is what LIVERPS predicts, ``ok`` whether pxr agreed."""
    out = {}
    for name, (fn, expected) in CASES.items():
        res = fn()
        if write is not None:
            write.mkdir(parents=True, exist_ok=True)
            res["stage"].GetRootLayer().Export(str(write / f"{name}.usda"))
        out[name] = {
            "steps": {k: {"value": v[0], "strongest_opinion": v[1]} for k, v in res["steps"].items()},
            "final": res["final"],
            "expected": expected,
            "ok": res["final"] == expected,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", type=Path, default=None, help="export each case's root layer as .usda here")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    res = run(args.write)
    for name, r in res.items():
        print(f"{name}: final={r['final']} expected={r['expected']} {'OK' if r['ok'] else 'MISMATCH'}")
        for step, v in r["steps"].items():
            print(f"    {step:45s} -> {v['value']:8s} {v['strongest_opinion']}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(res, indent=2) + "\n")
    return 0 if all(r["ok"] for r in res.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

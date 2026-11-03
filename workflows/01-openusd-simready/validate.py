"""SimReady-style validator for a robot asset entry point (``asset.usda``).

A "report card" in the sense of the SIGGRAPH 2026 SimReady talk: each check is a named pass/fail with a detail
string. Runs on any USD file, needs only ``usd-core`` (``pxr``).

Checks:
  stage_opens            the composed stage opens with all payloads loaded and has no composition errors
  default_prim           defaultPrim is set and valid
  up_axis                stage upAxis is the expected axis (Z for Newton and Isaac Sim robots)
  meters_per_unit        stage metersPerUnit (and kilogramsPerUnit) are the expected values
  layer_units_agree      every used layer that authors upAxis/metersPerUnit agrees with the stage (USD does not
                         convert units across references; a mismatched layer silently scales or rotates)
  payloads_resolve       every payload arc points at a layer that opens and at a prim that exists there
  payloads_effective     unloading payloads removes the physics (proves the features really are payloads)
  single_articulation    exactly one ArticulationRootAPI prim, and it is an ancestor of every rigid body
  joint_limits           every revolute/prismatic joint has finite lower < upper
  joint_bodies           every joint's body0/body1 targets exist and are rigid bodies
  bodies_have_mass       every rigid body has MassAPI with mass > 0 (or density > 0)
  bodies_have_collider   every rigid body owns at least one CollisionAPI prim (not one of a nested body)
  colliders_not_visible  colliders are not rendered (purpose guide or invisible), visuals carry no CollisionAPI

Usage::

    python validate.py out/asset.usda                 # prints the report, exit 1 on any failure
    python validate.py out/asset.usda --json report.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from pxr import Sdf, Usd, UsdGeom, UsdPhysics


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def _rigid_bodies(stage: Usd.Stage) -> list[Usd.Prim]:
    return [
        p
        for p in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
        if p.HasAPI(UsdPhysics.RigidBodyAPI)
    ]


def _owner_body(prim: Usd.Prim) -> Usd.Prim | None:
    p = prim
    while p and p.IsValid() and not p.IsPseudoRoot():
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            return p
        p = p.GetParent()
    return None


def _payload_items(prim: Usd.Prim) -> list[tuple[Sdf.Layer, Sdf.Payload]]:
    items = []
    for spec in prim.GetPrimStack():
        plist = spec.payloadList
        for p in (*plist.explicitItems, *plist.prependedItems, *plist.appendedItems, *plist.addedItems):
            items.append((spec.layer, p))
    return items


def check_payloads(stage: Usd.Stage) -> Check:
    seen, bad = 0, []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        for layer, p in _payload_items(prim):
            seen += 1
            if not p.assetPath:
                continue  # internal payload
            path = layer.ComputeAbsolutePath(p.assetPath)
            target = Sdf.Layer.FindOrOpen(path)
            if target is None:
                bad.append(f"{prim.GetPath()} -> {p.assetPath} (layer does not open)")
                continue
            prim_path = p.primPath if not p.primPath.isEmpty else Sdf.Path(f"/{target.defaultPrim}")
            if not target.defaultPrim and p.primPath.isEmpty:
                bad.append(f"{prim.GetPath()} -> {p.assetPath} (no primPath and no defaultPrim)")
            elif target.GetPrimAtPath(prim_path) is None:
                bad.append(f"{prim.GetPath()} -> {p.assetPath}{prim_path} (prim missing)")
    if seen == 0:
        return Check("payloads_resolve", False, "no payload arcs found (features are expected as payloads)")
    return Check("payloads_resolve", not bad, f"{seen} payload arcs; " + ("all resolve" if not bad else "; ".join(bad)))


def validate(
    path: str | Path, up_axis: str = "Z", meters_per_unit: float = 1.0, kg_per_unit: float = 1.0
) -> list[Check]:
    checks: list[Check] = []
    path = str(path)
    try:
        stage = Usd.Stage.Open(path, Usd.Stage.LoadAll)
    except Exception as exc:  # pxr raises Tf.ErrorException
        return [Check("stage_opens", False, f"{type(exc).__name__}: {exc}")]
    errors = [str(e) for e in stage.GetCompositionErrors()]
    n_prims = sum(1 for _ in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()))
    checks.append(Check("stage_opens", not errors, f"{n_prims} prims composed; composition errors: {errors or 'none'}"))

    dp = stage.GetDefaultPrim()
    checks.append(Check("default_prim", bool(dp and dp.IsValid()), f"defaultPrim = {dp.GetPath() if dp else None}"))

    axis = UsdGeom.GetStageUpAxis(stage)
    checks.append(Check("up_axis", axis == up_axis, f"upAxis = {axis} (expected {up_axis})"))
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    kpu = UsdPhysics.GetStageKilogramsPerUnit(stage)
    ok_units = math.isclose(mpu, meters_per_unit) and math.isclose(kpu, kg_per_unit)
    checks.append(Check("meters_per_unit", ok_units, f"metersPerUnit = {mpu}, kilogramsPerUnit = {kpu}"))

    mismatched = []
    for layer in stage.GetUsedLayers():
        if layer.anonymous:
            continue
        info = layer.pseudoRoot
        if info.HasInfo("upAxis") and info.GetInfo("upAxis") != axis:
            mismatched.append(f"{Path(layer.realPath).name}: upAxis {info.GetInfo('upAxis')}")
        if info.HasInfo("metersPerUnit") and not math.isclose(info.GetInfo("metersPerUnit"), mpu):
            mismatched.append(f"{Path(layer.realPath).name}: metersPerUnit {info.GetInfo('metersPerUnit')}")
    n_layers = sum(1 for la in stage.GetUsedLayers() if not la.anonymous)
    checks.append(
        Check("layer_units_agree", not mismatched, f"{n_layers} layers; " + ("; ".join(mismatched) or "all agree"))
    )

    checks.append(check_payloads(stage))
    unloaded = Usd.Stage.Open(path, Usd.Stage.LoadNone)
    n_loaded, n_unloaded = len(_rigid_bodies(stage)), len(_rigid_bodies(unloaded))
    checks.append(
        Check(
            "payloads_effective",
            n_loaded > 0 and n_unloaded == 0,
            f"rigid bodies with payloads loaded: {n_loaded}, unloaded: {n_unloaded}",
        )
    )

    bodies = _rigid_bodies(stage)
    roots = [
        p
        for p in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
        if p.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    ok_root = len(roots) == 1 and all(b.GetPath().HasPrefix(roots[0].GetPath()) for b in bodies)
    checks.append(
        Check(
            "single_articulation",
            ok_root,
            f"articulation roots: {[str(r.GetPath()) for r in roots]}; rigid bodies: {len(bodies)}",
        )
    )

    joints = [p for p in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()) if p.IsA(UsdPhysics.Joint)]
    bad_limits, bad_bodies = [], []
    body_paths = {b.GetPath() for b in bodies}
    for jp in joints:
        if jp.IsA(UsdPhysics.RevoluteJoint) or jp.IsA(UsdPhysics.PrismaticJoint):
            lo = jp.GetAttribute("physics:lowerLimit").Get()
            hi = jp.GetAttribute("physics:upperLimit").Get()
            if lo is None or hi is None or not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
                bad_limits.append(f"{jp.GetName()} ({lo}, {hi})")
        j = UsdPhysics.Joint(jp)
        for rel in (j.GetBody0Rel(), j.GetBody1Rel()):
            targets = rel.GetTargets()
            if targets and targets[0] not in body_paths:
                bad_bodies.append(f"{jp.GetName()}.{rel.GetName()} -> {targets[0]}")
    checks.append(
        Check("joint_limits", bool(joints) and not bad_limits, f"{len(joints)} joints; bad: {bad_limits or 'none'}")
    )
    checks.append(Check("joint_bodies", not bad_bodies, f"bad targets: {bad_bodies or 'none'}"))

    no_mass, no_collider = [], []
    colliders = [
        p
        for p in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
        if p.HasAPI(UsdPhysics.CollisionAPI)
    ]
    for b in bodies:
        m = UsdPhysics.MassAPI(b)
        mass = m.GetMassAttr().Get() if b.HasAPI(UsdPhysics.MassAPI) else None
        dens = m.GetDensityAttr().Get() if b.HasAPI(UsdPhysics.MassAPI) else None
        if not ((mass or 0.0) > 0.0 or (dens or 0.0) > 0.0):
            no_mass.append(b.GetName())
        if not any(_owner_body(c) == b for c in colliders):
            no_collider.append(b.GetName())
    checks.append(Check("bodies_have_mass", bool(bodies) and not no_mass, f"missing mass: {no_mass or 'none'}"))
    checks.append(
        Check("bodies_have_collider", bool(bodies) and not no_collider, f"no collider: {no_collider or 'none'}")
    )

    visible_colliders, colliding_visuals = [], []
    for c in colliders:
        img = UsdGeom.Imageable(c)
        if (
            img
            and img.ComputePurpose() not in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy)
            and img.ComputeVisibility() != UsdGeom.Tokens.invisible
        ):
            visible_colliders.append(str(c.GetPath()))
    for p in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if (
            p.IsA(UsdGeom.Mesh)
            and UsdGeom.Imageable(p).ComputePurpose() == UsdGeom.Tokens.render
            and p.HasAPI(UsdPhysics.CollisionAPI)
        ):
            colliding_visuals.append(str(p.GetPath()))
    checks.append(
        Check(
            "colliders_not_visible",
            not visible_colliders and not colliding_visuals,
            f"{len(colliders)} colliders; rendered colliders: {visible_colliders or 'none'}; "
            f"visual meshes with CollisionAPI: {colliding_visuals or 'none'}",
        )
    )
    return checks


def report(checks: list[Check]) -> dict:
    return {"passed": all(c.ok for c in checks), "checks": [asdict(c) for c in checks]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("asset", type=Path)
    ap.add_argument("--json", type=Path, default=None, help="write the report here")
    ap.add_argument("--up-axis", default="Z")
    ap.add_argument("--meters-per-unit", type=float, default=1.0)
    args = ap.parse_args(argv)
    checks = validate(args.asset, up_axis=args.up_axis, meters_per_unit=args.meters_per_unit)
    for c in checks:
        print(f"[{'PASS' if c.ok else 'FAIL'}] {c.name:24s} {c.detail}")
    rep = report(checks)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rep, indent=2) + "\n")
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

"""Build a layered SimReady robot asset from a URDF, following the "SimReady Frontier: USD Robot Asset
Structure" slide (Omniverse/OpenUSD deck, see README).

Layer graph written to ``--out`` (arrows are composition arcs, as on the slide)::

    asset.usda  (interface, entry point)
      |-- reference --> asset_sim_optimized.usda  (instanceable visuals)
      |                   |-- reference --> asset_sim_base.usda  (visual / collider split, material bindings)
      |                   |                   |-- sublayer --> asset_base.usda  (as imported)
      |                   |                   |      '-- reference --> parts.usda --sublayer--> materials.usda
      |                   |                   '-- sublayer --> materials.usda
      |                   '-- sublayer --> materials.usda
      '-- payload --> asset_physics.usda, asset_sensors.usda, asset_ros.usda, asset_control.usda

Every layer: ``upAxis = "Z"``, ``metersPerUnit = 1``, ``kilogramsPerUnit = 1``, ``defaultPrim = "quadruped"``
(``parts.usda`` uses ``/Parts``). Files are written as ``.usda`` (text) so diffs are readable; the slide's
``.usd`` names are kept otherwise.

Usage::

    python build_asset.py --out out/            # from newton-main/newton/examples/assets/quadruped.urdf
    python build_asset.py --urdf my.urdf --out out/
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_URDF = REPO_ROOT / "newton-main" / "newton" / "examples" / "assets" / "quadruped.urdf"

ROOT = "quadruped"
FEATURE_LAYERS = ("asset_physics.usda", "asset_sensors.usda", "asset_ros.usda", "asset_control.usda")
ALL_LAYERS = (
    "materials.usda",
    "parts.usda",
    "asset_base.usda",
    "asset_sim_base.usda",
    "asset_sim_optimized.usda",
    *FEATURE_LAYERS,
    "asset.usda",
)

# The URDF gives effort and velocity limits but no position limits. SimReady assets need them (the validator
# checks), so the "gap analysis" step of the SimReady playbook fills them here. These are our engineering
# choice for this toy quadruped, not datasheet values.
JOINT_LIMITS_DEG = {"HAA": (-45.0, 45.0), "HFE": (-90.0, 90.0), "KFE": (-90.0, 90.0)}
# PD gains from newton-main/newton/examples/basic/example_basic_urdf.py (target_ke=2000, target_kd=1 per rad).
DRIVE_KP_PER_RAD = 2000.0
DRIVE_KD_PER_RAD = 1.0
# Standing pose from the same example (radians), in URDF joint order LF, RF, LH, RH x (HAA, HFE, KFE).
STANDING_POSE_RAD = [0.2, 0.4, -0.6, -0.2, -0.4, 0.6, -0.2, 0.4, -0.6, 0.2, -0.4, 0.6]
FOOT_FRICTION = 1.0  # example_basic_urdf sets default_shape_cfg.mu = 1.0

LOOKS = {
    "body_paint": {"diffuseColor": (0.20, 0.32, 0.45), "metallic": 0.1, "roughness": 0.5},
    "joint_housing": {"diffuseColor": (0.85, 0.45, 0.10), "metallic": 0.0, "roughness": 0.6},
    "leg_metal": {"diffuseColor": (0.55, 0.57, 0.60), "metallic": 0.8, "roughness": 0.35},
}


# ---------------------------------------------------------------------------
# URDF parsing and kinematics (numpy, column vectors)
# ---------------------------------------------------------------------------


def rpy_to_matrix(rpy: tuple[float, float, float]) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def make_tf(xyz=(0.0, 0.0, 0.0), rot: np.ndarray | None = None) -> np.ndarray:
    tf = np.eye(4)
    tf[:3, :3] = np.eye(3) if rot is None else rot
    tf[:3, 3] = xyz
    return tf


def matrix_to_quat_wxyz(rot: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix to unit quaternion (w, x, y, z), w >= 0. USD's Gf quaternions are real-first."""
    m = np.asarray(rot, dtype=float)
    tr = np.trace(m)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return tuple(float(v) for v in q)


def rotation_x_to(axis: np.ndarray) -> np.ndarray:
    """Rotation that maps +X onto ``axis`` (USD revolute joints rotate about a principal axis)."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    x = np.array([1.0, 0.0, 0.0])
    v = np.cross(x, a)
    c = float(np.dot(x, a))
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([-1.0, -1.0, 1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


@dataclass
class Geom:
    kind: str  # cylinder | box | sphere
    dims: tuple[float, ...]  # cylinder (radius, length); box (x, y, z); sphere (radius,)
    tf: np.ndarray  # link -> geometry

    @property
    def part_name(self) -> str:
        mm = [round(d * 1000) for d in self.dims]
        if self.kind == "cylinder":
            return f"cylinder_r{mm[0]:03d}_l{mm[1]:03d}"
        if self.kind == "box":
            return f"box_{mm[0]:03d}x{mm[1]:03d}x{mm[2]:03d}"
        return f"sphere_r{mm[0]:03d}"


@dataclass
class Link:
    name: str
    geoms: list[Geom] = field(default_factory=list)
    mass: float = 0.0
    com: tuple[float, float, float] = (0.0, 0.0, 0.0)
    inertia: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    inertial_rot: np.ndarray = field(default_factory=lambda: np.eye(3))

    @property
    def role(self) -> str:
        if self.name == "base":
            return "base"
        return self.name.rsplit("_", 1)[-1]  # HAA, THIGH, SHANK


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    tf: np.ndarray  # parent link -> joint frame (= child link frame at q = 0)
    axis: np.ndarray
    effort: float | None
    velocity: float | None
    lower: float | None
    upper: float | None


@dataclass
class Robot:
    name: str
    links: dict[str, Link]
    joints: list[Joint]

    @property
    def root(self) -> str:
        children = {j.child for j in self.joints}
        roots = [n for n in self.links if n not in children]
        if len(roots) != 1:
            raise ValueError(f"expected one root link, found {roots}")
        return roots[0]

    def world_tfs(self) -> dict[str, np.ndarray]:
        """Link frames in the asset frame at q = 0 (root at the origin)."""
        out = {self.root: np.eye(4)}
        pending = list(self.joints)
        while pending:
            progressed = False
            for j in list(pending):
                if j.parent in out:
                    out[j.child] = out[j.parent] @ j.tf
                    pending.remove(j)
                    progressed = True
            if not progressed:
                raise ValueError("URDF joint tree is disconnected")
        return out


def _origin(el) -> np.ndarray:
    if el is None:
        return np.eye(4)
    xyz = [float(v) for v in el.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in el.get("rpy", "0 0 0").split()]
    return make_tf(xyz, rpy_to_matrix(rpy))


def parse_urdf(path: Path) -> Robot:
    """Parse links (collision geometry, inertial) and joints from a URDF. Visual elements fall back to collision."""
    root = ET.parse(path).getroot()
    links: dict[str, Link] = {}
    for le in root.findall("link"):
        link = Link(le.get("name"))
        shapes = le.findall("collision") or le.findall("visual")
        for ce in shapes:
            g = ce.find("geometry")
            tf = _origin(ce.find("origin"))
            if (c := g.find("cylinder")) is not None:
                link.geoms.append(Geom("cylinder", (float(c.get("radius")), float(c.get("length"))), tf))
            elif (b := g.find("box")) is not None:
                link.geoms.append(Geom("box", tuple(float(v) for v in b.get("size").split()), tf))
            elif (s := g.find("sphere")) is not None:
                link.geoms.append(Geom("sphere", (float(s.get("radius")),), tf))
            else:
                raise ValueError(f"{link.name}: only cylinder/box/sphere geometry is supported")
        ie = le.find("inertial")
        if ie is not None:
            link.mass = float(ie.find("mass").get("value"))
            itf = _origin(ie.find("origin"))
            link.com = tuple(float(v) for v in itf[:3, 3])
            link.inertial_rot = itf[:3, :3]
            a = {k: float(ie.find("inertia").get(k, 0.0)) for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")}
            link.inertia = np.array(
                [[a["ixx"], a["ixy"], a["ixz"]], [a["ixy"], a["iyy"], a["iyz"]], [a["ixz"], a["iyz"], a["izz"]]]
            )
        links[link.name] = link
    joints = []
    for je in root.findall("joint"):
        lim = je.find("limit")

        def _f(key, lim=lim):
            return float(lim.get(key)) if lim is not None and lim.get(key) is not None else None

        axis_el = je.find("axis")
        axis = np.array([float(v) for v in (axis_el.get("xyz") if axis_el is not None else "1 0 0").split()])
        joints.append(
            Joint(
                name=je.get("name"),
                type=je.get("type"),
                parent=je.find("parent").get("link"),
                child=je.find("child").get("link"),
                tf=_origin(je.find("origin")),
                axis=axis,
                effort=_f("effort"),
                velocity=_f("velocity"),
                lower=_f("lower"),
                upper=_f("upper"),
            )
        )
    return Robot(root.get("name"), links, joints)


def principal_inertia(link: Link) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Diagonalize the inertia tensor: (diagonalInertia, principalAxes quaternion wxyz) for UsdPhysics.MassAPI."""
    evals, evecs = np.linalg.eigh(link.inertia)
    if np.linalg.det(evecs) < 0:
        evecs[:, 2] *= -1.0
    rot = link.inertial_rot @ evecs
    return tuple(float(v) for v in evals), matrix_to_quat_wxyz(rot)


def joint_limits_deg(j: Joint) -> tuple[float, float, str]:
    """Position limits in degrees and where they came from."""
    if j.lower is not None and j.upper is not None and j.lower < j.upper:
        return math.degrees(j.lower), math.degrees(j.upper), "urdf"
    role = j.name.rsplit("_", 1)[-1]
    lo, hi = JOINT_LIMITS_DEG.get(role, (-180.0, 180.0))
    return lo, hi, "gap-fill (JOINT_LIMITS_DEG)"


# ---------------------------------------------------------------------------
# USD helpers
# ---------------------------------------------------------------------------


def new_layer(path: Path, default_prim: str = ROOT) -> Usd.Stage:
    """Create a layer with the stage metadata the validator checks."""
    if path.exists():
        path.unlink()
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    stage.GetRootLayer().documentation = f"Generated by build_asset.py ({path.name})"
    stage.GetRootLayer().defaultPrim = default_prim
    return stage


def save_feature(stage: Usd.Stage) -> None:
    """Feature layers only add opinions: keep the root an ``over`` (Define() on a child promotes it to ``def``)."""
    stage.GetRootLayer().GetPrimAtPath(f"/{ROOT}").specifier = Sdf.SpecifierOver
    stage.GetRootLayer().Save()


def set_pose(prim: Usd.Prim, tf: np.ndarray) -> None:
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in tf[:3, 3]]))
    w, x, y, z = matrix_to_quat_wxyz(tf[:3, :3])
    xf.AddOrientOp().Set(Gf.Quatf(w, x, y, z))


def cylinder_mesh(mesh: UsdGeom.Mesh, radius: float, length: float, segments: int = 24) -> None:
    """Triangulated cylinder along +Z, centered at the origin (stand-in for an imported CAD mesh)."""
    h = length / 2.0
    ang = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    ring = np.stack([radius * np.cos(ang), radius * np.sin(ang)], axis=1)
    pts = [(*p, -h) for p in ring] + [(*p, h) for p in ring] + [(0.0, 0.0, -h), (0.0, 0.0, h)]
    counts, idx = [], []
    bc, tc = 2 * segments, 2 * segments + 1
    for i in range(segments):
        j = (i + 1) % segments
        counts.append(4)
        idx += [i, j, segments + j, segments + i]
        counts.append(3)
        idx += [bc, j, i]
        counts.append(3)
        idx += [tc, segments + i, segments + j]
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in pts]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(-radius, -radius, -h), Gf.Vec3f(radius, radius, h)]))


def box_mesh(mesh: UsdGeom.Mesh, size: tuple[float, float, float]) -> None:
    hx, hy, hz = (s / 2.0 for s in size)
    pts = [(sx * hx, sy * hy, sz * hz) for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)]
    faces = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]]
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in pts]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4] * 6))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([i for f in faces for i in f]))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(-hx, -hy, -hz), Gf.Vec3f(hx, hy, hz)]))


def collider_prim(stage: Usd.Stage, path: str, geom: Geom) -> Usd.Prim:
    """Analytic collider (UsdGeom Cylinder/Cube/Sphere), the cheapest proxy a solver can use."""
    if geom.kind == "cylinder":
        g = UsdGeom.Cylinder.Define(stage, path)
        g.CreateRadiusAttr(geom.dims[0])
        g.CreateHeightAttr(geom.dims[1])
        g.CreateAxisAttr(UsdGeom.Tokens.z)
    elif geom.kind == "sphere":
        g = UsdGeom.Sphere.Define(stage, path)
        g.CreateRadiusAttr(geom.dims[0])
    else:
        g = UsdGeom.Cube.Define(stage, path)
        g.CreateSizeAttr(1.0)
        UsdGeom.Xformable(g).AddScaleOp()
    prim = g.GetPrim()
    UsdGeom.Imageable(prim).CreatePurposeAttr(UsdGeom.Tokens.guide)
    set_pose(prim, geom.tf)
    if geom.kind == "box":
        UsdGeom.Xformable(prim).AddScaleOp().Set(Gf.Vec3f(*map(float, geom.dims)))
    return prim


def look_for(link: Link) -> str:
    return {"base": "body_paint", "HAA": "joint_housing"}.get(link.role, "leg_metal")


# ---------------------------------------------------------------------------
# Layer writers
# ---------------------------------------------------------------------------


def write_materials(out: Path) -> None:
    stage = new_layer(out / "materials.usda")
    UsdGeom.Xform.Define(stage, f"/{ROOT}")
    UsdGeom.Scope.Define(stage, f"/{ROOT}/Looks")
    for name, p in LOOKS.items():
        mat = UsdShade.Material.Define(stage, f"/{ROOT}/Looks/{name}")
        sh = UsdShade.Shader.Define(stage, f"/{ROOT}/Looks/{name}/PreviewSurface")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*p["diffuseColor"]))
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(p["metallic"])
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(p["roughness"])
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    stage.GetRootLayer().Save()


def write_parts(out: Path, robot: Robot) -> dict[str, Geom]:
    """One part per unique geometry; links that share a shape reference the same part."""
    stage = new_layer(out / "parts.usda", default_prim="Parts")
    stage.GetRootLayer().subLayerPaths.append("./materials.usda")
    UsdGeom.Scope.Define(stage, "/Parts")
    parts: dict[str, Geom] = {}
    for link in robot.links.values():
        for g in link.geoms:
            parts.setdefault(g.part_name, g)
    for name, g in parts.items():
        UsdGeom.Xform.Define(stage, f"/Parts/{name}")
        mesh = UsdGeom.Mesh.Define(stage, f"/Parts/{name}/mesh")
        if g.kind == "cylinder":
            cylinder_mesh(mesh, *g.dims)
        elif g.kind == "box":
            box_mesh(mesh, g.dims)
        else:  # sphere as a coarse box-free stand-in: a short cylinder of equal radius
            cylinder_mesh(mesh, g.dims[0], 2 * g.dims[0])
    stage.GetRootLayer().Save()
    return parts


def write_base(out: Path, robot: Robot, world: dict[str, np.ndarray]) -> None:
    """As imported: one Xform per URDF link, flat under the root, each with a visuals group of part references."""
    stage = new_layer(out / "asset_base.usda")
    root = UsdGeom.Xform.Define(stage, f"/{ROOT}").GetPrim()
    root.SetCustomDataByKey("source", "newton-main/newton/examples/assets/quadruped.urdf")
    for link in robot.links.values():
        lp = UsdGeom.Xform.Define(stage, f"/{ROOT}/{link.name}").GetPrim()
        set_pose(lp, world[link.name])
        UsdGeom.Xform.Define(stage, f"/{ROOT}/{link.name}/visuals")
        for i, g in enumerate(link.geoms):
            vp = UsdGeom.Xform.Define(stage, f"/{ROOT}/{link.name}/visuals/geom_{i}").GetPrim()
            vp.GetReferences().AddReference("./parts.usda", Sdf.Path(f"/Parts/{g.part_name}"))
            set_pose(vp, g.tf)
    stage.GetRootLayer().Save()


def write_sim_base(out: Path, robot: Robot) -> None:
    """Reorganize for simulation: separate collider tree (analytic proxies, guide purpose), render purpose on
    visuals, material bindings. Sublayers asset_base and materials, as on the slide."""
    stage = new_layer(out / "asset_sim_base.usda")
    stage.GetRootLayer().subLayerPaths.append("./asset_base.usda")
    stage.GetRootLayer().subLayerPaths.append("./materials.usda")
    for link in robot.links.values():
        base = f"/{ROOT}/{link.name}"
        vis = stage.OverridePrim(f"{base}/visuals")
        UsdGeom.Imageable(vis).CreatePurposeAttr(UsdGeom.Tokens.render)
        UsdShade.MaterialBindingAPI.Apply(vis).Bind(
            UsdShade.Material(stage.GetPrimAtPath(f"/{ROOT}/Looks/{look_for(link)}"))
        )
        UsdGeom.Scope.Define(stage, f"{base}/collisions")
        for i, g in enumerate(link.geoms):
            collider_prim(stage, f"{base}/collisions/collider_{i}", g)
    stage.GetRootLayer().Save()


def write_sim_optimized(out: Path, robot: Robot) -> None:
    """Reference asset_sim_base and make every visual an instanceable reference, so links sharing a part share
    one prototype. (The slide also lists "merging meshes"; our URDF has one shape per link, nothing to merge.)"""
    stage = new_layer(out / "asset_sim_optimized.usda")
    stage.GetRootLayer().subLayerPaths.append("./materials.usda")
    prim = UsdGeom.Xform.Define(stage, f"/{ROOT}").GetPrim()
    prim.GetReferences().AddReference("./asset_sim_base.usda")
    for link in robot.links.values():
        for i in range(len(link.geoms)):
            stage.OverridePrim(f"/{ROOT}/{link.name}/visuals/geom_{i}").SetInstanceable(True)
    stage.GetRootLayer().Save()


def write_physics(out: Path, robot: Robot) -> list[dict]:
    """UsdPhysics feature layer: articulation root, rigid bodies with mass, colliders, joints with limits and
    drives, a physics material. Authored as overs so it only adds opinions."""
    stage = new_layer(out / "asset_physics.usda")
    root = stage.OverridePrim(f"/{ROOT}")
    UsdPhysics.ArticulationRootAPI.Apply(root)
    UsdGeom.Scope.Define(stage, f"/{ROOT}/PhysicsMaterials")
    pmat = UsdShade.Material.Define(stage, f"/{ROOT}/PhysicsMaterials/contact")
    mapi = UsdPhysics.MaterialAPI.Apply(pmat.GetPrim())
    mapi.CreateStaticFrictionAttr(FOOT_FRICTION)
    mapi.CreateDynamicFrictionAttr(FOOT_FRICTION)
    mapi.CreateRestitutionAttr(0.0)
    for link in robot.links.values():
        lp = stage.OverridePrim(f"/{ROOT}/{link.name}")
        UsdPhysics.RigidBodyAPI.Apply(lp)
        mass = UsdPhysics.MassAPI.Apply(lp)
        mass.CreateMassAttr(link.mass)
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*link.com))
        diag, (w, x, y, z) = principal_inertia(link)
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*diag))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(w, x, y, z))
        for i in range(len(link.geoms)):
            cp = stage.OverridePrim(f"/{ROOT}/{link.name}/collisions/collider_{i}")
            UsdPhysics.CollisionAPI.Apply(cp)
            UsdShade.MaterialBindingAPI.Apply(cp).Bind(pmat, materialPurpose="physics")
    UsdGeom.Scope.Define(stage, f"/{ROOT}/joints")
    rows = []
    for j in robot.joints:
        if j.type not in ("revolute", "continuous"):
            raise ValueError(f"{j.name}: joint type {j.type} not handled in this demo")
        jp = UsdPhysics.RevoluteJoint.Define(stage, f"/{ROOT}/joints/{j.name}")
        r_axis = rotation_x_to(j.axis)
        jp.CreateAxisAttr("X")
        jp.CreateBody0Rel().SetTargets([Sdf.Path(f"/{ROOT}/{j.parent}")])
        jp.CreateBody1Rel().SetTargets([Sdf.Path(f"/{ROOT}/{j.child}")])
        jp.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in j.tf[:3, 3]]))
        jp.CreateLocalRot0Attr(Gf.Quatf(*matrix_to_quat_wxyz(j.tf[:3, :3] @ r_axis)))
        jp.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        jp.CreateLocalRot1Attr(Gf.Quatf(*matrix_to_quat_wxyz(r_axis)))
        lo, hi, src = joint_limits_deg(j)
        jp.CreateLowerLimitAttr(lo)  # degrees for revolute joints (UsdPhysics convention)
        jp.CreateUpperLimitAttr(hi)
        drive = UsdPhysics.DriveAPI.Apply(jp.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        # UsdPhysics angular drive gains are per degree; Newton converts back to per radian on import.
        drive.CreateStiffnessAttr(DRIVE_KP_PER_RAD * math.pi / 180.0)
        drive.CreateDampingAttr(DRIVE_KD_PER_RAD * math.pi / 180.0)
        if j.effort is not None:
            drive.CreateMaxForceAttr(j.effort)
        rows.append({"joint": j.name, "lower_deg": lo, "upper_deg": hi, "limit_source": src, "effort": j.effort})
    save_feature(stage)
    return rows


def _forward_camera_rot() -> np.ndarray:
    """USD cameras look down -Z with +Y up. Columns: camera X, Y, Z axes in the base frame (look along +X, Z up)."""
    return np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def write_sensors(out: Path, robot: Robot) -> None:
    """Sensor feature layer: a front RGB-D camera (UsdGeom.Camera) and an IMU frame on the base link.
    The IMU has no standard OpenUSD schema; its parameters use our own ``ssw:sensor:*`` attributes."""
    stage = new_layer(out / "asset_sensors.usda")
    stage.OverridePrim(f"/{ROOT}")
    base = robot.links[robot.root]
    stage.OverridePrim(f"/{ROOT}/{base.name}")
    front = max(g.dims[1] / 2 for g in base.geoms) if base.geoms else 0.3
    cam = UsdGeom.Camera.Define(stage, f"/{ROOT}/{base.name}/front_camera")
    set_pose(cam.GetPrim(), make_tf((front + 0.02, 0.0, 0.05), _forward_camera_rot()))
    cam.CreateFocalLengthAttr(18.0)
    cam.CreateHorizontalApertureAttr(20.955)
    cam.CreateVerticalApertureAttr(15.716)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 50.0))
    cam.GetPrim().CreateAttribute("ssw:sensor:type", Sdf.ValueTypeNames.Token, custom=True).Set("rgbd")
    cam.GetPrim().CreateAttribute("ssw:sensor:rateHz", Sdf.ValueTypeNames.Float, custom=True).Set(30.0)
    imu = UsdGeom.Xform.Define(stage, f"/{ROOT}/{base.name}/imu").GetPrim()
    set_pose(imu, make_tf((0.0, 0.0, 0.0)))
    imu.CreateAttribute("ssw:sensor:type", Sdf.ValueTypeNames.Token, custom=True).Set("imu")
    imu.CreateAttribute("ssw:sensor:rateHz", Sdf.ValueTypeNames.Float, custom=True).Set(200.0)
    save_feature(stage)


def write_ros(out: Path, robot: Robot) -> None:
    """ROS interface feature layer: topic and frame names as ``ssw:ros:*`` metadata. Isaac Sim wires ROS 2 through
    OmniGraph nodes; this layer only records the contract so a bridge (or a test) can read it."""
    stage = new_layer(out / "asset_ros.usda")
    root = stage.OverridePrim(f"/{ROOT}")
    for key, val in {
        "namespace": f"/{ROOT}",
        "jointStateTopic": "joint_states",
        "jointCommandTopic": "joint_command",
        "odomFrame": "odom",
        "baseFrame": robot.root,
    }.items():
        root.CreateAttribute(f"ssw:ros:{key}", Sdf.ValueTypeNames.String, custom=True).Set(val)
    for link in robot.links.values():
        lp = stage.OverridePrim(f"/{ROOT}/{link.name}")
        lp.CreateAttribute("ssw:ros:frameId", Sdf.ValueTypeNames.String, custom=True).Set(link.name)
    cam = stage.OverridePrim(f"/{ROOT}/{robot.root}/front_camera")
    cam.CreateAttribute("ssw:ros:imageTopic", Sdf.ValueTypeNames.String, custom=True).Set("front_camera/rgb")
    cam.CreateAttribute("ssw:ros:depthTopic", Sdf.ValueTypeNames.String, custom=True).Set("front_camera/depth")
    imu = stage.OverridePrim(f"/{ROOT}/{robot.root}/imu")
    imu.CreateAttribute("ssw:ros:imuTopic", Sdf.ValueTypeNames.String, custom=True).Set("imu")
    save_feature(stage)


def write_control(out: Path, robot: Robot) -> None:
    """Control feature layer: drive targets for the standing pose, the initial joint state Newton reads
    (``newton:angular:position``, degrees), and the policy I/O contract as ``ssw:control:*``."""
    stage = new_layer(out / "asset_control.usda")
    root = stage.OverridePrim(f"/{ROOT}")
    names = [j.name for j in robot.joints]
    pose = dict(zip(names, STANDING_POSE_RAD, strict=False))
    for j in robot.joints:
        jp = stage.OverridePrim(f"/{ROOT}/joints/{j.name}")
        deg = math.degrees(pose.get(j.name, 0.0))
        jp.CreateAttribute("drive:angular:physics:targetPosition", Sdf.ValueTypeNames.Float).Set(deg)
        jp.CreateAttribute("newton:angular:position", Sdf.ValueTypeNames.Float, custom=True).Set(deg)
    root.CreateAttribute("ssw:control:jointOrder", Sdf.ValueTypeNames.StringArray, custom=True).Set(names)
    root.CreateAttribute("ssw:control:defaultJointPos", Sdf.ValueTypeNames.FloatArray, custom=True).Set(
        [pose.get(n, 0.0) for n in names]
    )
    root.CreateAttribute("ssw:control:actionScale", Sdf.ValueTypeNames.Float, custom=True).Set(0.5)
    root.CreateAttribute("ssw:control:rateHz", Sdf.ValueTypeNames.Float, custom=True).Set(50.0)
    save_feature(stage)


def write_interface(out: Path, robot: Robot) -> None:
    """Interface layer: the one file users open. References the optimized asset, payloads the features."""
    stage = new_layer(out / "asset.usda")
    prim = UsdGeom.Xform.Define(stage, f"/{ROOT}").GetPrim()
    prim.GetReferences().AddReference("./asset_sim_optimized.usda")
    for f in FEATURE_LAYERS:
        prim.GetPayloads().AddPayload(f"./{f}")  # earlier payloads are stronger: physics first
    model = Usd.ModelAPI(prim)
    model.SetKind(Kind.Tokens.component)
    model.SetAssetName(robot.name)
    model.SetAssetVersion("0.1.0")
    model.SetAssetIdentifier(Sdf.AssetPath("./asset.usda"))
    stage.GetRootLayer().Save()


def build(out: Path, urdf: Path = DEFAULT_URDF) -> dict:
    """Write all layers to ``out`` and return a summary."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    robot = parse_urdf(Path(urdf))
    world = robot.world_tfs()
    write_materials(out)
    parts = write_parts(out, robot)
    write_base(out, robot, world)
    write_sim_base(out, robot)
    write_sim_optimized(out, robot)
    joints = write_physics(out, robot)
    write_sensors(out, robot)
    write_ros(out, robot)
    write_control(out, robot)
    write_interface(out, robot)
    stage = Usd.Stage.Open(str(out / "asset.usda"))
    visuals = [p for p in stage.Traverse() if p.IsInstance()]
    return {
        "urdf": str(Path(urdf).name),
        "links": len(robot.links),
        "joints": len(robot.joints),
        "unique_parts": sorted(parts),
        "instanced_visuals": len(visuals),
        "prototypes": len(stage.GetPrototypes()),
        "joint_limits": joints,
        "layers": {name: (out / name).stat().st_size for name in ALL_LAYERS},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    ap.add_argument("--out", type=Path, default=HERE / "out")
    args = ap.parse_args(argv)
    summary = build(args.out, args.urdf)
    print(json.dumps({k: v for k, v in summary.items() if k != "joint_limits"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

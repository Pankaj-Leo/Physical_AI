"""Replicator synthetic data for a pallet-jack detector (Omniverse Basic Training 05).

Re-creates the standalone script the tutorial runs from NVIDIA's synthetic-data-generation training workflow repo
(``standalone_palletjack_sdg.py``; the tutorial reviews it at 24:59-30:30). Anatomy, in the tutorial's order:

1. argparse: ``--height 544 --width 960`` ("this is for the resolution", 25:26), ``--num_frames 2000``,
   ``--distractors warehouse|additional|none``, ``--data_dir``.
2. ``SimulationApp`` launch, then open the warehouse stage.
3. Assets from NVIDIA's S3 asset store: three pallet-jack variants, distractor props, textures.
4. Semantic label ``palletjack`` on the pallet jacks only ("we are only interested in what is a pallet jack", 28:22).
5. Camera "always looking at the center of the stage", position randomized in bounds (28:53-29:05).
6. Pallet-jack colour randomizer, pose randomizer with scale fixed (29:10-29:36).
7. Floor and wall texture randomizer (29:48-30:01).
8. ``KittiWriter`` from ``rep.WriterRegistry`` attached to the render product (30:01-30:15), then run.

Added here: a lighting randomizer. The tutorial names "lighting and point of views of the camera" as one of the three
domain-randomization techniques (11:53-11:59), but the script it walks through does not show one.

    ./python.sh replicator_pallet_jack.py --height 544 --width 960 --num_frames 2000 \
        --distractors warehouse --data_dir $OUT/distractors_warehouse

Pinned: Isaac Sim 5.0.0 (``./python.sh`` from the workstation install, Python 3.11), Replicator as bundled.
The tutorial ran Isaac Sim 4.5 docs-era code; module paths below moved in 4.5/5.0 and are marked ``# verify:``.
not executed: needs GPU (Isaac Sim, RTX). The pure helpers (``parse_args``, ``run_plan``, ``strip_semantics_except``)
are tested on CPU.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

CLASS_NAME = "palletjack"
DISTRACTOR_SETS = {
    "warehouse": "distractors_warehouse",
    "additional": "distractors_additional",
    "none": "no_distractors",
}

# Asset paths. ENV_URL and the props are relative to the Isaac Sim assets root; the pallet jacks are absolute URLs.
ENV_URL = "/Isaac/Environments/Simple_Warehouse/warehouse.usd"  # verify: path in the Isaac Sim 5.0 asset pack
_S3 = "http://omniverse-content-production.s3-us-west-2.amazonaws.com"
PALLETJACKS = [  # verify: the three pallet-truck variants the reference script references
    f"{_S3}/Assets/DigitalTwin/Assets/Warehouse/Equipment/Pallet_Trucks/Scale_A/PalletTruckScale_A01_PR_NVD_01.usd",
    f"{_S3}/Assets/DigitalTwin/Assets/Warehouse/Equipment/Pallet_Trucks/Heavy_Duty_A/HeavyDutyPalletTruck_A01_PR_NVD_01.usd",
    f"{_S3}/Assets/DigitalTwin/Assets/Warehouse/Equipment/Pallet_Trucks/Low_Profile_A/LowProfilePalletTruck_A01_PR_NVD_01.usd",
]
WAREHOUSE_DISTRACTORS = [  # verify: prop names under Simple_Warehouse/Props
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_A_01.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_01.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_PaletteA_01.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_RackPile_01.usd",
    "/Isaac/Environments/Simple_Warehouse/Props/SM_BucketPlastic_B.usd",
]
ADDITIONAL_DISTRACTORS = [  # verify: any extra props; the point is objects that look nothing like a warehouse
    "/Isaac/Props/YCB/Axis_Aligned/003_cracker_box.usd",
    "/Isaac/Props/YCB/Axis_Aligned/006_mustard_bottle.usd",
    "/Isaac/Props/KLT_Bin/small_KLT.usd",
    "/Isaac/Props/Blocks/basic_block.usd",
]
TEXTURES = [  # verify: any set of image textures works; pass --textures-file to use your own list
    "/Isaac/Materials/Textures/Patterns/nv_asphalt_yellow_weathered.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_tile_hexagonal_green_white.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_rubber_woven_charcoal.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_granite_tile.jpg",
    "/Isaac/Materials/Textures/Patterns/nv_wooden_wall.jpg",
]

# Randomization ranges (metres, degrees), warehouse frame.
CAMERA_POS_MIN, CAMERA_POS_MAX = (-9.2, -11.8, 0.4), (7.2, 15.8, 4.0)
OBJECT_POS_MIN, OBJECT_POS_MAX = (-6.0, -6.0, 0.0), (6.0, 12.0, 0.0)
LIGHT_POS_MIN, LIGHT_POS_MAX = (-8.0, -10.0, 4.0), (6.0, 14.0, 7.0)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Replicator pallet-jack SDG with a KITTI writer")
    ap.add_argument("--height", type=int, default=544, help="render height in pixels")
    ap.add_argument("--width", type=int, default=960, help="render width in pixels")
    ap.add_argument("--num_frames", type=int, default=2000)
    ap.add_argument("--distractors", choices=sorted(DISTRACTOR_SETS), default="warehouse")
    ap.add_argument("--data_dir", required=True, help="output folder for this distractor set")
    ap.add_argument("--num_lights", type=int, default=3)
    ap.add_argument("--textures-file", default=None, help="one texture path or URL per line")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args(argv)


@dataclass(frozen=True)
class Run:
    distractors: str
    data_dir: str
    num_frames: int


def run_plan(out_root: str, num_frames: int = 2000, sets: Iterable[str] = ("warehouse", "additional", "none")) -> list:
    """The three runs ``generate_data.sh`` performs: one Isaac Sim launch per distractor set (19:28-21:43)."""
    return [Run(s, f"{out_root.rstrip('/')}/{DISTRACTOR_SETS[s]}", num_frames) for s in sets]


def distractor_paths(kind: str) -> list[str]:
    return {"warehouse": WAREHOUSE_DISTRACTORS, "additional": ADDITIONAL_DISTRACTORS, "none": []}[kind]


def semantic_labels(prim) -> list[tuple[str, str]]:
    """(type, label) pairs on a prim, from the legacy ``SemanticsAPI`` attributes and the newer ``semantics:labels:*``.

    Legacy: ``semantic:<instance>:params:semanticType`` / ``semantic:<instance>:params:semanticData``.
    New (UsdSemantics LabelsAPI, Isaac Sim 5.0): ``semantics:labels:<type>`` token arrays.  # verify: 5.0 naming
    """
    out: list[tuple[str, str]] = []
    attrs = {a.GetName(): a for a in prim.GetAttributes()}
    for name, attr in attrs.items():
        if name.startswith("semantic:") and name.endswith(":params:semanticType"):
            data = attrs.get(name.replace(":semanticType", ":semanticData"))
            if data is not None:
                out.append((str(attr.Get()), str(data.Get())))
        elif name.startswith("semantics:labels:"):
            out += [(name.split(":", 2)[2], str(v)) for v in (attr.Get() or [])]
    return out


def strip_semantics_except(stage, keep: Iterable[str] = (CLASS_NAME,)) -> int:
    """Remove every semantic label whose value is not in ``keep``; returns how many prims were changed.

    The warehouse USD ships with labels on racks, floors and props. Left in place, the KITTI writer would emit boxes for
    them too. The reference script removes all but the pallet-jack class before writing.
    """
    keep = set(keep)
    changed = 0
    for prim in stage.Traverse():
        labels = semantic_labels(prim)
        if not labels or all(v in keep for _, v in labels):
            continue
        for attr in list(prim.GetAttributes()):
            name = attr.GetName()
            if name.startswith("semantic:") and name.endswith(":params:semanticData") and str(attr.Get()) not in keep:
                prim.RemoveProperty(name)
                prim.RemoveProperty(name.replace(":semanticData", ":semanticType"))
            elif name.startswith("semantics:labels:"):
                kept = [v for v in (attr.Get() or []) if str(v) in keep]
                if kept:
                    attr.Set(kept)
                else:
                    prim.RemoveProperty(name)
        changed += 1
    return changed


def _read_textures(path: str | None, assets_root: str) -> list[str]:
    if path:
        with open(path, encoding="utf-8") as fh:
            return [ln.strip() for ln in fh if ln.strip()]
    return [assets_root + t for t in TEXTURES]


def main(argv: Sequence[str] | None = None) -> int:  # not executed: needs GPU
    args = parse_args(argv)

    from isaacsim import SimulationApp  # verify: Isaac Sim 4.5+ entry point (was omni.isaac.kit)

    app = SimulationApp({"headless": args.headless, "renderer": "RayTracedLighting"})

    import omni.replicator.core as rep
    import omni.usd

    try:
        from isaacsim.core.utils.stage import open_stage  # verify: Isaac Sim 4.5+ namespace
        from isaacsim.storage.native import get_assets_root_path  # verify: Isaac Sim 4.5+ namespace
    except ImportError:  # Isaac Sim <= 4.2 names, as in the tutorial's documentation links
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        from omni.isaac.core.utils.stage import open_stage

    rep.set_global_seed(args.seed)
    assets_root = get_assets_root_path()
    if assets_root is None:
        print("could not reach the Isaac Sim assets root (Nucleus or S3)")
        app.close()
        return 1

    # 2. stage
    open_stage(assets_root + ENV_URL)
    for _ in range(10):
        app.update()  # let the stage finish loading
    stage = omni.usd.get_context().get_stage()
    removed = strip_semantics_except(stage, keep=())
    print(f"removed stray semantic labels from {removed} prims")

    # 3-4. pallet jacks (labelled) and distractors (unlabelled)
    jacks = [rep.create.from_usd(url, semantics=[("class", CLASS_NAME)], count=1) for url in PALLETJACKS]
    jack_group = rep.create.group(jacks)
    distractors = [rep.create.from_usd(assets_root + p, count=1) for p in distractor_paths(args.distractors)]
    distractor_group = rep.create.group(distractors) if distractors else None
    textures = _read_textures(args.textures_file, assets_root)

    # lighting rig, randomized per frame below
    lights = rep.create.light(light_type="Sphere", count=args.num_lights)

    # 5. camera and render product
    camera = rep.create.camera(clipping_range=(0.1, 1000000.0))
    render_product = rep.create.render_product(camera, (args.width, args.height))

    with rep.trigger.on_frame(num_frames=args.num_frames):
        with camera:
            rep.modify.pose(position=rep.distribution.uniform(CAMERA_POS_MIN, CAMERA_POS_MAX), look_at=(0, 0, 0))
        # 6. pallet-jack colour and pose; scale stays fixed
        with rep.get.prims(path_pattern="SteerAxles"):  # verify: the coloured body prim in the pallet-truck USDs
            rep.randomizer.color(colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1)))
        with jack_group:
            rep.modify.pose(
                position=rep.distribution.uniform(OBJECT_POS_MIN, OBJECT_POS_MAX),
                rotation=rep.distribution.uniform((0, 0, 0), (0, 0, 360)),
            )
        if distractor_group is not None:
            with distractor_group:
                rep.modify.pose(
                    position=rep.distribution.uniform(OBJECT_POS_MIN, OBJECT_POS_MAX),
                    rotation=rep.distribution.uniform((0, 0, 0), (0, 0, 360)),
                )
        # 7. floor and wall textures
        with rep.get.prims(path_pattern="SM_Floor"):
            rep.randomizer.texture(textures=textures, texture_rotate=rep.distribution.uniform(80, 95))
        with rep.get.prims(path_pattern="SM_Wall"):
            rep.randomizer.texture(textures=textures, texture_rotate=rep.distribution.uniform(80, 95))
        # added: lighting (intensity, colour temperature, position)
        with lights:
            rep.modify.pose(position=rep.distribution.uniform(LIGHT_POS_MIN, LIGHT_POS_MAX))
            rep.modify.attribute("intensity", rep.distribution.normal(35000.0, 8000.0))
            rep.modify.attribute("colorTemperature", rep.distribution.uniform(3500.0, 7500.0))  # verify: attr name

    # 8. KITTI writer
    writer = rep.WriterRegistry.get("KittiWriter")
    writer.initialize(output_dir=args.data_dir, omit_semantic_type=True)  # verify: KittiWriter kwargs in 5.0
    writer.attach([render_product])

    rep.orchestrator.run_until_complete()  # verify: blocks until the on_frame trigger is exhausted
    app.close()  # the tutorial: "once this finish Isaac Sim will close automatically" (35:08)
    return 0


if __name__ == "__main__":
    sys.exit(main())

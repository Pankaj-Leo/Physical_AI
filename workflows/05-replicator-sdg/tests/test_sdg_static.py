"""CPU checks for the GPU-only pieces of the SDG workflow.

- replicator_pallet_jack.py: imports without Isaac Sim; argparse defaults match the tutorial command; the three-run
  plan; the semantic-label filter on a real (in-memory) USD stage built with pxr.
- tao/*.txt: the 14% hold-out, the class name, and a DetectNet_v2-legal resolution are consistent across specs.
- train_torchvision.py: imports without torch; the KITTI-to-tensor helper.
- shell scripts: bash -n.
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

FOLDER = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, FOLDER / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


kt = _load("kitti_tools", "kitti_tools.py")
rp = _load("replicator_pallet_jack", "replicator_pallet_jack.py")
tv = _load("train_torchvision", "train_torchvision.py")


def test_replicator_args_match_the_tutorial_command():
    # --height 544 --width 960 --num_frames 2000 --distractors warehouse --data_dir <out> (Basic Training 05, 20:05)
    args = rp.parse_args(["--data_dir", "/tmp/x"])
    assert (args.width, args.height, args.num_frames, args.distractors) == (960, 544, 2000, "warehouse")
    assert sorted(rp.DISTRACTOR_SETS) == ["additional", "none", "warehouse"]
    assert kt.tao_resolution_problems(args.width, args.height) == []
    with pytest.raises(SystemExit):
        rp.parse_args(["--data_dir", "/tmp/x", "--distractors", "forest"])


def test_run_plan_three_sets_6000_frames_in_kitti_tools_layout():
    plan = rp.run_plan("/data/")
    assert [r.distractors for r in plan] == ["warehouse", "additional", "none"]
    assert [Path(r.data_dir).name for r in plan] == list(kt.DEFAULT_SETS)
    assert sum(r.num_frames for r in plan) == 6000
    assert rp.distractor_paths("none") == []
    assert len(rp.PALLETJACKS) == 3  # "three different types of pallet jacks" (28:36)


def test_strip_semantics_except_on_a_usd_stage():
    from pxr import Sdf, Usd, Vt

    stage = Usd.Stage.CreateInMemory()
    jack = stage.DefinePrim("/World/PalletJack", "Xform")
    rack = stage.DefinePrim("/World/Rack", "Xform")
    floor = stage.DefinePrim("/World/SM_Floor", "Mesh")
    stage.DefinePrim("/World/Unlabelled", "Xform")
    for prim, label in ((jack, "palletjack"), (rack, "rack")):  # legacy SemanticsAPI attribute layout
        prim.CreateAttribute("semantic:Semantics:params:semanticType", Sdf.ValueTypeNames.String).Set("class")
        prim.CreateAttribute("semantic:Semantics:params:semanticData", Sdf.ValueTypeNames.String).Set(label)
    floor.CreateAttribute("semantics:labels:class", Sdf.ValueTypeNames.TokenArray).Set(  # newer LabelsAPI layout
        Vt.TokenArray(["floor", "palletjack"])
    )

    assert rp.semantic_labels(jack) == [("class", "palletjack")]
    assert sorted(rp.semantic_labels(floor)) == [("class", "floor"), ("class", "palletjack")]
    changed = rp.strip_semantics_except(stage, keep=["palletjack"])
    assert changed == 2  # rack and floor
    assert rp.semantic_labels(jack) == [("class", "palletjack")]
    assert rp.semantic_labels(rack) == []
    assert rp.semantic_labels(floor) == [("class", "palletjack")]
    assert rp.strip_semantics_except(stage, keep=()) == 2  # before spawning: clear everything
    assert all(rp.semantic_labels(p) == [] for p in stage.Traverse())


def test_replicator_main_keeps_isaac_imports_inside_main():
    src = (FOLDER / "replicator_pallet_jack.py").read_text(encoding="utf-8")
    head = src[: src.index("def main(")]
    assert not re.search(r"^\s*(import|from)\s+(omni|isaacsim|carb)\b", head, flags=re.M)
    for needle in ("KittiWriter", "rep.trigger.on_frame", "randomizer.color", "randomizer.texture", "look_at"):
        assert needle in src


def _spec(name: str) -> str:
    return (FOLDER / "tao" / name).read_text(encoding="utf-8")


def test_tao_specs_are_consistent():
    convert = _spec("detectnet_v2_tfrecords_kitti.txt")
    train = _spec("detectnet_v2_train_resnet18_kitti.txt")
    infer = _spec("detectnet_v2_inference_kitti.txt")
    assert re.search(r"^\s*val_split:\s*14\s*$", convert, flags=re.M)  # the 14% hold-out
    assert re.search(r"^\s*validation_fold:\s*0\s*$", train, flags=re.M)
    assert 'label_dir_name: "object_detection"' in convert and 'image_dir_name: "rgb"' in convert
    for text in (train, infer):
        assert '"palletjack"' in text
        w = int(re.search(r"(?:output_)?image_width:\s*(\d+)", text).group(1))
        h = int(re.search(r"(?:output_)?image_height:\s*(\d+)", text).group(1))
        assert (w, h) == (960, 544)
    for s in kt.DEFAULT_SETS:
        assert f"tfrecords/{s}/" in train
    for text in (convert, train, infer):  # balanced braces, the cheapest protobuf-text sanity check
        body = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
        assert body.count("{") == body.count("}")


def test_torchvision_helpers_without_torch():
    objs = [
        kt.parse_line("palletjack 0 0 0 10 20 110 80 0 0 0 0 0 0 0"),
        kt.parse_line("forklift 0 0 0 10 20 110 80 0 0 0 0 0 0 0"),  # unknown class: dropped
        kt.parse_line("palletjack 0 0 0 50 50 50 90 0 0 0 0 0 0 0"),  # zero width: dropped
        kt.parse_line("palletjack 0 0 0 0 0 30 12 0 0 0 0 0 0 0"),  # 12 px tall: dropped with min_height=20
    ]
    boxes, labels = tv.objects_to_arrays(objs, {"palletjack": 1}, min_height=20)
    assert boxes.dtype == np.float32 and labels.dtype == np.int64
    np.testing.assert_array_equal(boxes, [[10, 20, 110, 80]])
    np.testing.assert_array_equal(labels, [1])
    empty_boxes, empty_labels = tv.objects_to_arrays([], {"palletjack": 1})
    assert empty_boxes.shape == (0, 4) and empty_labels.shape == (0,)
    assert tv.collate([("img1", "t1"), ("img2", "t2")]) == (("img1", "img2"), ("t1", "t2"))
    assert tv.CLASSES == ("__background__", "palletjack")


def test_shell_scripts_pass_bash_syntax_check():
    bash = shutil.which("bash")
    assert bash is not None
    for script in (FOLDER / "generate_data.sh", FOLDER / "tao/run_tao.sh"):
        subprocess.run([bash, "-n", str(script)], check=True)

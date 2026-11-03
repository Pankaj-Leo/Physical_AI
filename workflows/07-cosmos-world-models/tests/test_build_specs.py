"""Transfer specs: up to date, and field-for-field consistent with the Cosmos 3 cookbook specs."""

import json

import build_specs as bs
import pytest

COOKBOOK = bs.COSMOS_ROOT / "cookbooks/cosmos3/generator/transfer"
needs_cosmos = pytest.mark.skipif(not COOKBOOK.is_dir(), reason="cosmos-main snapshot not present")
PATH_KEYS = {"name", "negative_prompt_file", "prompt_path"}


@needs_cosmos
def test_committed_specs_are_up_to_date():
    assert bs.check() == []


@needs_cosmos
def test_av_spec_matches_cookbook_wsm():
    ref = json.loads((COOKBOOK / "specs/wsm.json").read_text())
    ours = json.loads((bs.SPEC_DIR / "av/av_snow.json").read_text())
    assert set(ours) == set(ref)
    for k in set(ref) - PATH_KEYS - {"wsm"}:
        assert ours[k] == ref[k], k
    assert set(ours["wsm"]) == set(ref["wsm"]) == {"control_path"}


@needs_cosmos
def test_robot_spec_matches_cookbook_depth_and_seg():
    depth = json.loads((COOKBOOK / "specs/depth.json").read_text())
    seg = json.loads((COOKBOOK / "specs/seg.json").read_text())
    ours = json.loads((bs.SPEC_DIR / "robot/robot_desert.json").read_text())
    assert set(depth) - {"depth"} <= set(ours)
    for k in set(depth) - PATH_KEYS - {"depth", "control_guidance"}:
        assert ours[k] == depth[k], k
    # multi-control: ssw.transfer uses the stricter seg control guidance (2.0) and Framework per-hint weights
    assert ours["control_guidance"] == seg["control_guidance"] == 2.0
    assert ours["depth"]["weight"] == ours["seg"]["weight"] == 0.5


@needs_cosmos
def test_av_prompt_follows_cookbook_caption_schema():
    ref = json.loads((COOKBOOK / "assets/wsm/prompt.json").read_text())
    ours = bs.build_av_prompt("night_rain")
    assert set(ours) <= set(ref)
    assert set(ref) - set(ours) == {"text_and_signage_elements", "audio_description"}
    assert set(ours["lighting"]) == set(ref["lighting"])
    assert ours["fps"] == 10 and ours["duration"] == "10s"


def test_vllm_form_from_spec_mirrors_cookbook_request():
    spec = bs.build_wsm_framework_spec("t", "controls/x.mp4", "p.json", "n.json")
    form = bs.vllm_form_from_spec(spec, {"a": 1}, {"b": 2}, "wsm")
    extra = json.loads(form["extra_params"])
    assert form["size"] == "1280x720" and form["num_frames"] == "101" and form["fps"] == "10"
    assert form["guidance_scale"] == "1.0" and form["flow_shift"] == "10.0" and form["num_inference_steps"] == "50"
    assert extra["wsm"] == {"control_path": "controls/x.mp4"} and extra["control_guidance"] == 3.0
    assert extra["max_frames"] == 101 and extra["guardrails"] is False
    kw = bs.diffusers_kwargs_from_spec(spec, "wsm")
    assert (kw["height"], kw["width"], kw["fps"], kw["guidance_scale"]) == (720, 1280, 10.0, 1.0)


def test_av_variants_differ_only_in_appearance():
    a, b = bs.build_av_prompt("clear_noon"), bs.build_av_prompt("snow")
    differing = {k for k in a if a[k] != b[k]}
    assert differing == {"background_setting", "lighting", "aesthetics", "temporal_caption"}

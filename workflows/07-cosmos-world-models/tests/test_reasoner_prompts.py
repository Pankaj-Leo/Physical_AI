"""Prompt catalogue: wording against the Cosmos 3 cookbook, request shape, and reply parsing (CPU, no server)."""

import json
import re
from pathlib import Path

import pytest
import reasoner_prompts as rp

GUIDE = Path(__file__).resolve().parents[4] / "cosmos-main/cookbooks/cosmos3/reasoner/reasoner_prompt_guide.md"

# The cookbook's own example values for every templated prompt.
COOKBOOK_FILLS = {
    "robot.plausibility": {},
    "robot.plan": {"task": "put flower into the red bottle"},
    "robot.trajectory": {"task": "Move the pink bowl to the right"},
    "robot.assisted_next_action": {
        "overall_task": "The student exchanges the black ink cartridge of the printer.",
        "instruction": "place old ink_cartridge.",
    },
    "av.grounding": {"target": "the load as a whole"},
    "warehouse.grounding": {"target": "the load as a whole"},
    "warehouse.timestamp_query": {
        "event": "A man in a white sweater walks out of a room carrying a box, closes the door behind him, walks on "
        "the floor, and turns left at the end near the wall."
    },
    "warehouse.interval": {"start": "00:05.64", "end": "00:17.49"},
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def test_catalogue_is_well_formed():
    assert {p.domain for p in rp.CATALOGUE.values()} == {"robot", "av", "warehouse"}
    for pid, p in rp.CATALOGUE.items():
        assert pid == p.id
        assert p.kind in rp.KINDS
        assert p.provenance in {"cookbook", "cookbook-template", "repo"}
        assert p.media in {"video", "image"}
        text = p.render(**{f: "x" for f in p.fields if f not in p.defaults})
        assert text and "{x}" not in text
        if p.kind == "choice":
            assert p.choices in {"AB", "ABC"}
            assert "(A)" in text
    assert '"start": t_start' in rp.CATALOGUE["robot.segments"].render()  # literal JSON braces survive


@pytest.mark.skipif(not GUIDE.is_file(), reason="cosmos-main snapshot not present")
def test_cookbook_prompts_are_verbatim():
    guide = _norm(GUIDE.read_text())
    checked = 0
    for pid, p in rp.CATALOGUE.items():
        if p.provenance == "repo":
            continue
        fills = COOKBOOK_FILLS.get(pid, {})
        if p.provenance == "cookbook-template" and pid not in COOKBOOK_FILLS:
            continue
        text = _norm(p.render(**fills))
        assert text in guide, f"{pid} drifted from the cookbook wording"
        checked += 1
    assert checked >= 15


def test_missing_field_raises():
    with pytest.raises(KeyError):
        rp.CATALOGUE["warehouse.grounding"].render()


def test_video_request_media_first_and_sampling():
    req = rp.build_request("warehouse.hazard.near_miss", "/data/clip.mp4")
    user = req["messages"][1]["content"]
    assert req["messages"][0] == {"role": "system", "content": "You are a helpful assistant."}
    assert user[0]["type"] == "video_url" and user[0]["video_url"]["url"] == "file:///data/clip.mp4"
    assert user[1]["type"] == "text" and user[1]["text"].endswith("immediately after the </think> tag.")
    assert (req["temperature"], req["top_p"], req["presence_penalty"]) == (0.6, 0.95, 0.0)
    assert req["extra_body"]["media_io_kwargs"] == {"video": {"num_frames": -1, "fps": 4.0}}
    meta = rp.sampling_metadata(req)
    assert meta["think_suffix"] and meta["top_k"] == 20 and meta["repetition_penalty"] == 1.0


def test_image_request_plain_sampling():
    req = rp.build_request("warehouse.grounding", "/data/f.jpg", target="the forklift")
    part = req["messages"][1]["content"][0]
    assert part == {"type": "image_url", "image_url": {"url": "file:///data/f.jpg"}}
    assert "media_io_kwargs" not in req["extra_body"]
    assert (req["temperature"], req["top_p"], req["presence_penalty"]) == (0.7, 0.8, 1.5)
    assert "<think>" not in req["messages"][1]["content"][1]["text"]


def test_parse_cookbook_grounding_answer():
    answer = '```json\n[\n    {"bbox_2d": [214, 141, 497, 722], "label": "load as a whole"}\n]\n```'
    (box,) = rp.parse_bboxes(answer)
    assert box.valid and box.as_list() == [214, 141, 497, 722]
    # grounding_2d.png is 624 x 416 (deep-dive notes); cookbook rule x/1000*W, y/1000*H
    assert box.to_pixels(624, 416) == (134, 59, 310, 300)


def test_parse_bboxes_flags_bad_boxes_and_tolerates_prose():
    text = (
        "<think>two objects</think>\n```json\n"
        '[{"bbox_2d": [10, 20, 1040, 900], "label": "a"}, {"bbox_2d": [500, 500, 400, 600], "label": "b"},'
        ' {"bbox_2d": [1, 2, 3], "label": "c"}, {"bbox_2d": [100, 100, 200, 200], "label": "d"}]\n```\nDone.'
    )
    boxes = rp.parse_bboxes(text)
    assert [b.issue for b in boxes] == ["outside 0-1000", "x2<=x1 or y2<=y1", "not four numbers", ""]
    assert rp.parse_bboxes("no json here") is None


def test_parse_cookbook_trajectory_and_timestamp():
    traj = (
        "<think>\nI will move my gripper ...\n</think>\n\n```json\n[\n"
        '  {"point_2d": [713, 680], "label": "gripper trajectory"},\n'
        '  {"point_2d": [552, 721], "label": "gripper trajectory"},\n'
        '  {"point_2d": [517, 755], "label": "gripper trajectory"},\n'
        '  {"point_2d": [592, 739], "label": "gripper trajectory"},\n'
        '  {"point_2d": [588, 680], "label": "gripper trajectory"}\n]\n```'
    )
    pts = rp.parse_points(traj)
    assert len(pts) == 5 and pts[0] == (713.0, 680.0, "gripper trajectory")
    ts = rp.parse_timestamp('```json\n{\n  "start": "00:13.30",\n  "end": "00:23.40"\n}\n```')
    assert ts == pytest.approx((13.3, 23.4))


def test_interpret_choice_and_truncation():
    ok = rp.interpret("warehouse.hazard.fallen_load", "<think>\nnothing falls\n</think>\n\n(C) No")
    assert ok["choice"] == "C" and ok["well_formed"]
    cut = rp.interpret("warehouse.hazard.fallen_load", "<think>\nThe camera is dark and")
    assert cut["choice"] is None and not cut["well_formed"]
    plain = rp.interpret("robot.plausibility", "A")
    assert plain["choice"] == "A" and plain["well_formed"]
    segs = rp.interpret("warehouse.event_timeline", '[{"start": "00:00.10", "end": "00:05.70", "caption": "x"}]')
    assert segs["segments"] == [{"start": pytest.approx(0.1), "end": pytest.approx(5.7), "caption": "x"}]


def test_cli(capsys):
    assert rp.main(["list", "--domain", "av"]) == 0
    assert "av.grounding" in capsys.readouterr().out
    assert rp.main(["request", "--prompt", "robot.task_success", "--media", "/x.mp4"]) == 0
    req = json.loads(capsys.readouterr().out)
    assert "walk forward across the terrain" in req["messages"][1]["content"][1]["text"]

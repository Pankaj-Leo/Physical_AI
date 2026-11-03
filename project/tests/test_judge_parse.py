"""Reasoner request building and reply parsing; no server or openai package needed."""

import json

import pytest

from ssw import judge, transfer


def test_parse_think_standard():
    ta = judge.parse_think("<think>\nThe robot sinks.\n</think>\n\nC")
    assert ta.well_formed
    assert ta.reasoning == "The robot sinks."
    assert ta.answer == "C"


def test_parse_think_missing_open_tag_and_answer_tags():
    ta = judge.parse_think("reasoning here</think> <answer>(A) Yes</answer>")
    assert ta.well_formed
    assert ta.reasoning == "reasoning here"
    assert ta.answer == "(A) Yes"


def test_parse_think_truncated_and_plain():
    ta = judge.parse_think("<think>I am still thinking and ran out of tok")
    assert not ta.well_formed
    assert ta.answer == ""
    plain = judge.parse_think("B")
    assert not plain.well_formed
    assert plain.answer == "B"


@pytest.mark.parametrize(
    ("answer", "valid", "expected"),
    [
        ("A", "ABC", "A"),
        ("(C)", "ABC", "C"),
        ("(B) Impossible to tell", "ABC", "B"),
        ("C. No, the robot fell.", "ABC", "C"),
        ("The answer is (A).", "ABC", "A"),
        ("Yes", "ABC", "A"),
        ("Impossible", "AB", "B"),
        ("No", "AB", None),  # "No" is option C, which does not exist for plausibility
        ("", "ABC", None),
        ("Possibly yes, possibly no", "ABC", None),
        ("D", "ABC", None),
    ],
)
def test_parse_choice(answer, valid, expected):
    assert judge.parse_choice(answer, valid) == expected


def test_parse_segments_with_fences_and_prose():
    text = (
        '```json\n[\n {"start": 0.1, "end": 2.7, "caption": "walks"},\n'
        ' {"start": "00:02.70", "end": 4.0, "caption": "sinks"}\n]\n```\nThat is all.'
    )
    segs = judge.parse_segments(text)
    assert segs == [
        {"start": 0.1, "end": 2.7, "caption": "walks"},
        {"start": 2.7, "end": 4.0, "caption": "sinks"},
    ]


def test_parse_segments_after_think_and_garbage():
    segs = judge.parse_segments('<think>x</think>{"start": 1, "end": 2, "caption": "fall"}')
    assert segs == [{"start": 1.0, "end": 2.0, "caption": "fall"}]
    assert judge.parse_segments("no json here") is None
    assert judge.parse_segments("[{broken") is None


def test_build_request_layout_and_sampling():
    req = judge.build_request("task_success", "https://example.org/clip.mp4", model="m")
    user = req["messages"][1]
    assert user["role"] == "user"
    # Media must come before text.
    assert [c["type"] for c in user["content"]] == ["video_url", "text"]
    assert "<think>" in user["content"][1]["text"]
    assert "(C) No" in user["content"][1]["text"]
    assert (req["temperature"], req["top_p"], req["presence_penalty"]) == (0.6, 0.95, 0.0)
    assert req["extra_body"]["media_io_kwargs"]["video"] == {"num_frames": -1, "fps": 4.0}
    seg = judge.build_request("segments", "https://example.org/clip.mp4")
    assert "<think>" not in seg["messages"][1]["content"][1]["text"]
    assert seg["temperature"] == 0.7
    json.dumps(req)  # serializable


def test_media_url_local_path(tmp_path):
    f = tmp_path / "c.mp4"
    f.write_bytes(b"\x00\x01")
    assert judge.media_url(str(f)).startswith("file:///")
    assert judge.media_url(str(f), embed=True).startswith("data:video/mp4;base64,")


def test_score_against_ground_truth():
    recs = [
        {"clip_meta": {"material": "sand", "fell": True}, "task_success": {"choice": "C"}},
        {"clip_meta": {"material": "sand", "fell": False, "forward_distance_m": 2.0}, "task_success": {"choice": "A"}},
        {"clip_meta": {"material": "snow", "fell": False, "forward_distance_m": 0.2}, "task_success": {"choice": "A"}},
        {"clip_meta": {"material": "snow", "fell": True}, "task_success": {"choice": "B"}},
    ]
    s = judge.score(recs)
    fd = s["failure_detection"]
    assert (fd["tp"], fd["fp"], fd["fn"], fd["tn"], fd["abstained"]) == (1, 0, 1, 1, 1)
    assert fd["precision"] == 1.0
    assert fd["recall"] == 0.5
    assert s["confusion_rows_truth_cols_pred"]["matrix"] == [[1, 0, 0], [1, 1, 1]]
    assert set(s["by_material"]) == {"sand", "snow"}


def test_dry_run_cli(tmp_path, capsys):
    m = tmp_path / "clips.jsonl"
    m.write_text(json.dumps({"clip": "http://h/c.mp4", "material": "sand"}) + "\n")
    assert judge.main(["ask", "--manifest", str(m), "--dry-run"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 3
    assert {json.loads(ln)["question"] for ln in lines} == set(judge.QUESTIONS)


def test_transfer_requests(tmp_path):
    prompt = transfer.build_prompt("desert", "sand")
    assert prompt["resolution"] == {"W": 1280, "H": 720}
    assert "sand" in prompt["subjects"][0]["action"]
    form = transfer.build_vllm_omni_form({"depth": "d.mp4", "seg": "s.mp4"}, prompt, {"x": "y"})
    extra = json.loads(form["extra_params"])
    assert form["size"] == "1280x720"
    assert form["num_frames"] == "121"
    assert extra["depth"] == {"control_path": "d.mp4"}
    assert extra["control_guidance"] == 2.0  # seg present
    assert extra["guardrails"] is False
    depth_only = json.loads(transfer.build_vllm_omni_form({"depth": "d.mp4"}, prompt, {})["extra_params"])
    assert depth_only["control_guidance"] == 1.5
    assert transfer.build_vllm_omni_form({"depth": "d"}, prompt, {}, resolution="480")["size"] == "853x480"
    with pytest.raises(ValueError):
        transfer.build_vllm_omni_form({"edge": "e.mp4"}, prompt, {})
    assert transfer.main(["--depth", "d.mp4", "--seg", "s.mp4", "--dry-run", "--out", str(tmp_path)]) == 0
    files = sorted(p.name for p in tmp_path.iterdir())
    assert "clip_desert_request.json" in files
    assert "negative_prompt.json" in files
    bundle = json.loads((tmp_path / "clip_snowfield_request.json").read_text())
    assert bundle["framework_spec"]["depth"]["weight"] == 0.5

"""Incident agent dry run on recorded mock replies: plan, log, scoring, failure modes (CPU, no server)."""

import json
import shutil
import subprocess
from pathlib import Path

import incident_agent as ia
import pytest

FIX = Path(__file__).resolve().parent / "fixtures"
FOLDER = FIX.parents[1]


@pytest.fixture
def dry_run(tmp_path):
    clips = ia.read_jsonl(FIX / "warehouse_clips.jsonl")
    client = ia.MockReasoner.from_file(FIX / "mock_responses.json")
    log = ia.run(clips, client, tmp_path, mock=True)
    return clips, client, log, tmp_path


def test_log_contents(dry_run):
    _, client, log, out = dry_run
    assert log["mode"] == "mock" and "not Cosmos output" in log["note"]
    assert client.missing == []
    got = {(i["clip_id"], i["type"]): i for i in log["incidents"]}
    assert set(got) == {("wh_01", "near_miss"), ("wh_02", "near_miss"), ("wh_02", "zone_violation")}
    a = got[("wh_01", "near_miss")]
    assert (a["start_s"], a["end_s"]) == (3.2, 5.9)
    assert a["boxes"][0]["bbox_px"] == [515, 229, 753, 620]  # [402, 318, 588, 861] on 1280 x 720
    assert a["flags"] == []
    b = got[("wh_02", "near_miss")]
    assert "timestamp_unparsed_used_full_clip" in b["flags"]
    assert "invalid_box:outside 0-1000" in b["flags"]
    assert (b["start_s"], b["end_s"]) == (0.0, 12.0)
    verdicts = {c["id"]: c["verdicts"] for c in log["clips"]}
    assert verdicts["wh_03"]["zone_violation"] is None  # truncated <think>
    assert verdicts["wh_04"]["fallen_load"] == "B"
    assert (out / "incident_log.json").is_file() and (out / "incident_log.md").is_file()


def test_requests_are_logged_media_first(dry_run):
    *_, out = dry_run
    lines = (out / "requests.jsonl").read_text().splitlines()
    assert len(lines) == 22  # 4 questions per clip + 2 follow-ups per detection (3 detections)
    for line in lines:
        user = json.loads(line)["request"]["messages"][1]["content"]
        assert user[0]["type"] in {"video_url", "image_url"} and user[1]["type"] == "text"


def test_score_and_failure_modes(dry_run):
    clips, _, log, _ = dry_run
    sc = ia.score(log, clips)
    o = sc["overall"]
    assert (o["tp"], o["fp"], o["fn"], o["tn"], o["abstained"]) == (2, 1, 1, 6, 2)
    assert o["precision"] == pytest.approx(2 / 3) and o["recall"] == pytest.approx(2 / 3)
    assert o["agreement"] == pytest.approx(8 / 12) and o["coverage"] == pytest.approx(10 / 12)
    assert o["confusion_rows_truth_cols_pred"]["matrix"] == [[2, 1, 1], [1, 6, 1]]
    assert set(sc["by_condition"]) == {"camera=ceiling", "camera=forklift", "lighting=day", "lighting=night"}
    assert sc["by_condition"]["lighting=night"]["fn"] == 1
    fm = sc["failure_modes"]
    assert fm["false_positives"] == ["wh_02/near_miss"]
    assert fm["false_negatives"] == ["wh_04/near_miss"]
    assert fm["abstentions"] == ["wh_03/zone_violation", "wh_04/fallen_load"]
    assert fm["malformed_replies"] == ["wh_03/hazard/zone_violation"]
    loc = {e["clip_id"]: e for e in sc["localization_on_true_positives"]}
    assert loc["wh_01"]["temporal_iou"] == pytest.approx(0.793, abs=1e-3)
    md = ia.to_markdown(log, sc)
    assert "MOCK" in md and "they say nothing about Cosmos" in md


def test_missing_reply_is_reported(tmp_path):
    clip = {"id": "x", "clip": "/x.mp4", "duration_s": 5.0}
    log = ia.run([clip], ia.MockReasoner({}), tmp_path, mock=True)
    assert len(log["clips"][0]["missing_replies"]) == 4 and log["incidents"] == []


def test_iou_helpers():
    assert ia.temporal_iou((0, 2), (1, 3)) == pytest.approx(1 / 3)
    assert ia.temporal_iou((0, 1), (2, 3)) == 0.0
    assert ia.box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)


def test_committed_example_is_reproducible(tmp_path):
    committed = FOLDER / "results" / "incident_mock"
    ia.main(["run", "--dry-run", "--manifest", str(FIX / "warehouse_clips.jsonl"), "--out", str(tmp_path)])
    for name in ("incident_log.json", "incident_log.md", "score.json"):
        assert (tmp_path / name).read_text() == (committed / name).read_text(), name


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_extract_keyframe(tmp_path):
    from PIL import Image

    video = tmp_path / "t.mp4"
    cmd = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x90:rate=10:duration=2", str(video)]
    subprocess.run(cmd, check=True)
    frame = ia.extract_keyframe(str(video), 1.0, tmp_path / "k" / "f.jpg")
    assert Image.open(frame).size == (160, 90)

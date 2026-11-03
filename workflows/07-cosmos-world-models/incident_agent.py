#!/usr/bin/env python3
"""Cosmos 3 Reasoner as a physical-operations agent: video clips in, incident log (JSON + Markdown) out.

Roadmap track T5. The pattern is the Metropolis "perceive, reason, act" loop from the GTC 2025 talk "Agentic AI for
Physical Operations" (proximity and zone violations, an evaluation set, final-response scoring). That talk used
SpatialRGPT, Sparse4D, and AgentIQ; here one Cosmos 3 Reasoner answers every question, so this is the pattern with
a different model, not a reproduction of the Metropolis stack.

Per clip the agent asks, in order:

1. ``warehouse.event_timeline``      -> timestamped events (context for the log).
2. ``warehouse.hazard.<type>``       -> (A) Yes / (B) Impossible to tell / (C) No, with ``<think>`` reasoning.
3. for each "Yes": ``warehouse.timestamp_query`` -> start/end seconds of the incident.
4. for each "Yes": a keyframe at the incident midpoint, then ``warehouse.grounding`` -> 0-1000 boxes -> pixels.

Then it writes ``incident_log.json`` and ``incident_log.md``. With ground-truth labels in the manifest, ``score``
reports precision, recall, agreement, and a confusion matrix per hazard and per condition, plus the failure modes
(abstentions, malformed replies, invalid boxes, timestamp fallbacks).

Usage::

    # Dry run: build every request, answer from recorded mock replies (tests/fixtures), no server, no GPU
    python incident_agent.py run --manifest tests/fixtures/warehouse_clips.jsonl \\
        --mock tests/fixtures/mock_responses.json --out results/incident_mock

    # Live: an OpenAI-compatible vLLM server with nvidia/Cosmos3-Nano (not executed: needs GPU)
    vllm serve nvidia/Cosmos3-Nano --async-scheduling --allowed-local-media-path / --port 8000
    python incident_agent.py run --manifest clips.jsonl --base-url http://HOST:8000/v1 --out results/live

    python incident_agent.py score --log results/incident_mock/incident_log.json \\
        --manifest tests/fixtures/warehouse_clips.jsonl

Manifest (JSON Lines), one clip per line::

    {"id": "wh_01", "clip": "/data/wh_01.mp4", "width": 1280, "height": 720, "duration_s": 12.0,
     "condition": {"lighting": "day", "camera": "ceiling"},
     "labels": {"near_miss": {"present": true, "start_s": 3.0, "end_s": 5.5, "bbox_2d": [400, 300, 620, 900]},
                "zone_violation": {"present": false}, "fallen_load": {"present": false}}}
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

_HERE = Path(__file__).resolve().parent
_SSW_SRC = _HERE.parents[1] / "project" / "src"
if _SSW_SRC.is_dir() and str(_SSW_SRC) not in sys.path:
    sys.path.insert(0, str(_SSW_SRC))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import reasoner_prompts as rp  # noqa: E402
from ssw import metrics  # noqa: E402


@dataclass(frozen=True)
class Hazard:
    name: str
    prompt_id: str
    event: str  # sentence for the timestamp query
    target: str  # grounding target ("the ..." phrase)
    severity: str


HAZARDS: dict[str, Hazard] = {
    "near_miss": Hazard(
        "near_miss",
        "warehouse.hazard.near_miss",
        "A person comes within about two meters of a moving forklift or mobile robot",
        "the person closest to the moving vehicle and that vehicle",
        "high",
    ),
    "zone_violation": Hazard(
        "zone_violation",
        "warehouse.hazard.zone_violation",
        "A person walks into a floor area marked as restricted",
        "the person inside the restricted zone",
        "medium",
    ),
    "fallen_load": Hazard(
        "fallen_load",
        "warehouse.hazard.fallen_load",
        "A box, pallet load, or other object falls onto the floor",
        "the fallen object",
        "medium",
    ),
}

MOCK_NOTE = "MOCK: replies are hand-written fixtures in tests/fixtures/, not Cosmos output"


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------


class Reasoner(Protocol):
    def complete(self, request: dict, key: str) -> str: ...


class MockReasoner:
    """Answers from a dict ``{"<clip_id>/<step>": reply}``. Missing keys return an empty reply (logged)."""

    def __init__(self, responses: dict[str, str]):
        self.responses = responses
        self.missing: list[str] = []

    @classmethod
    def from_file(cls, path: Path) -> MockReasoner:
        data = json.loads(Path(path).read_text())
        return cls({k: v for k, v in data.items() if not k.startswith("_")})

    def complete(self, request: dict, key: str) -> str:
        if key not in self.responses:
            self.missing.append(key)
            return ""
        return self.responses[key]


class LiveReasoner:
    """OpenAI-compatible client for a vLLM server. Not executed: needs GPU (a vLLM server with Cosmos3-Nano)."""

    def __init__(self, base_url: str, api_key: str = "EMPTY", model: str | None = None, timeout: float = 600.0):
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - live path
            raise SystemExit("pip install openai") from exc
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model or self.client.models.list().data[0].id

    def complete(self, request: dict, key: str) -> str:  # pragma: no cover - live path
        req = {**request, "model": self.model}
        resp = self.client.chat.completions.create(**req)
        return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Keyframes
# ---------------------------------------------------------------------------


def keyframe_path(out_dir: Path, clip_id: str, hazard: str) -> Path:
    return out_dir / "keyframes" / f"{clip_id}_{hazard}.jpg"


def extract_keyframe(video: str, t_s: float, out: Path) -> Path:
    """One JPEG at ``t_s`` seconds with ffmpeg (CPU). Raises if ffmpeg is missing or fails."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("ffmpeg not found")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [exe, "-v", "error", "-y", "-ss", f"{max(t_s, 0.0):.3f}", "-i", str(video), "-frames:v", "1", str(out)]
    subprocess.run(cmd, check=True)
    return out


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


def _ask(
    client: Reasoner, prompt_id: str, media: str, key: str, *, trace: list, model: str, fps: float, **values
) -> dict:
    req = rp.build_request(prompt_id, media, model=model, fps=fps, **values)
    text = client.complete(req, key)
    rec = rp.interpret(prompt_id, text)
    trace.append({"key": key, "request": req, "reply": text})
    rec["key"] = key
    rec["settings"] = rp.sampling_metadata(req)
    if not text:
        rec["missing_reply"] = True
    return rec


def analyze_clip(
    clip: dict,
    client: Reasoner,
    out_dir: Path,
    *,
    model: str = rp.DEFAULT_MODEL,
    fps: float = 4.0,
    extract_frames: bool = False,
    hazards: Sequence[str] = tuple(HAZARDS),
    trace: list | None = None,
) -> dict:
    """Run the question plan on one clip.

    Returns ``{"clip": ..., "timeline": ..., "answers": ..., "incidents": ...}``.
    """
    trace = [] if trace is None else trace
    cid, video = clip["id"], clip["clip"]
    width, height = int(clip.get("width", 1280)), int(clip.get("height", 720))
    duration = float(clip.get("duration_s", 0.0)) or None
    answers: dict = {}
    timeline = _ask(client, "warehouse.event_timeline", video, f"{cid}/timeline", trace=trace, model=model, fps=fps)
    answers["timeline"] = timeline
    incidents = []
    for hz_name in hazards:
        hz = HAZARDS[hz_name]
        verdict = _ask(client, hz.prompt_id, video, f"{cid}/hazard/{hz_name}", trace=trace, model=model, fps=fps)
        answers[f"hazard/{hz_name}"] = verdict
        if verdict.get("choice") != "A":
            continue
        flags: list[str] = []
        when = _ask(
            client,
            "warehouse.timestamp_query",
            video,
            f"{cid}/when/{hz_name}",
            trace=trace,
            model=model,
            fps=fps,
            event=hz.event,
        )
        answers[f"when/{hz_name}"] = when
        ts = when.get("timestamp")
        if ts is None:
            flags.append("timestamp_unparsed_used_full_clip")
            ts = (0.0, duration or 0.0)
        start, end = ts
        if end < start:
            flags.append("timestamp_reversed")
            start, end = end, start
        if duration is not None and (start < 0 or end > duration + 0.5):
            flags.append("timestamp_outside_clip")
        mid = 0.5 * (start + end)
        frame = keyframe_path(out_dir, cid, hz_name)
        if extract_frames:
            extract_keyframe(video, mid, frame)
        ground = _ask(
            client,
            "warehouse.grounding",
            str(frame),
            f"{cid}/ground/{hz_name}",
            trace=trace,
            model=model,
            fps=fps,
            target=hz.target,
        )
        answers[f"ground/{hz_name}"] = ground
        boxes = []
        for b in ground.get("boxes") or []:
            box = rp.Box(b["label"], *b["bbox_2d"], issue=b["issue"])
            entry = {"label": box.label, "bbox_2d": box.as_list(), "valid": box.valid}
            if box.valid:
                entry["bbox_px"] = list(box.to_pixels(width, height))
            else:
                entry["issue"] = box.issue
                flags.append(f"invalid_box:{box.issue}")
            boxes.append(entry)
        if ground.get("boxes") is None:
            flags.append("grounding_unparsed")
        for rec in (verdict, when, ground):
            if not rec.get("well_formed", True):
                flags.append(f"malformed:{rec['key']}")
        incidents.append(
            {
                "clip_id": cid,
                "type": hz_name,
                "severity": hz.severity,
                "start_s": round(start, 2),
                "end_s": round(end, 2),
                "keyframe": str(frame.relative_to(out_dir)) if frame.is_relative_to(out_dir) else str(frame),
                "keyframe_t_s": round(mid, 2),
                "boxes": boxes,
                "evidence": (verdict.get("reasoning") or verdict.get("answer") or "")[:400],
                "flags": flags,
            }
        )
    return {"clip": clip, "timeline": timeline.get("segments"), "answers": answers, "incidents": incidents}


def run(
    clips: Sequence[dict],
    client: Reasoner,
    out_dir: Path,
    *,
    mock: bool,
    model: str = rp.DEFAULT_MODEL,
    fps: float = 4.0,
    extract_frames: bool = False,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    trace: list = []
    per_clip = [
        analyze_clip(c, client, out_dir, model=model, fps=fps, extract_frames=extract_frames, trace=trace)
        for c in clips
    ]
    settings = {}
    for pc in per_clip:
        for k, rec in pc["answers"].items():
            settings.setdefault(k.split("/")[0], rec["settings"])
    log = {
        "agent": "incident_agent.py (Cosmos 3 Reasoner pattern, roadmap T5)",
        "mode": "mock" if mock else "live",
        "note": MOCK_NOTE if mock else "live replies from the server named in settings",
        "model": model,
        "settings_by_step": settings,
        "hazards": {k: {"prompt_id": h.prompt_id, "event": h.event, "target": h.target} for k, h in HAZARDS.items()},
        "clips": [
            {
                "id": pc["clip"]["id"],
                "clip": pc["clip"]["clip"],
                "condition": pc["clip"].get("condition", {}),
                "timeline": pc["timeline"],
                "verdicts": {
                    k.split("/", 1)[1]: v.get("choice") for k, v in pc["answers"].items() if k.startswith("hazard/")
                },
                "malformed": [v["key"] for v in pc["answers"].values() if not v.get("well_formed", True)],
                "missing_replies": [v["key"] for v in pc["answers"].values() if v.get("missing_reply")],
            }
            for pc in per_clip
        ],
        "incidents": [i for pc in per_clip for i in pc["incidents"]],
    }
    (out_dir / "incident_log.json").write_text(json.dumps(log, indent=2) + "\n")
    (out_dir / "incident_log.md").write_text(to_markdown(log))
    with (out_dir / "requests.jsonl").open("w") as f:
        for t in trace:
            f.write(json.dumps({"key": t["key"], "request": t["request"]}) + "\n")
    return log


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _label(present: bool) -> str:
    return "present" if present else "absent"


def _pred(choice: str | None) -> str | None:
    return {"A": "present", "C": "absent"}.get(choice or "")


def temporal_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _scores(y_true: list, y_pred: list) -> dict:
    bs = metrics.binary_scores(y_true, y_pred, positive="present")
    cm = metrics.confusion_matrix(y_true, y_pred, ["present", "absent"], ["present", "absent", None])
    return {
        "n": len(y_true),
        "tp": bs.tp,
        "fp": bs.fp,
        "fn": bs.fn,
        "tn": bs.tn,
        "abstained": bs.abstained,
        "precision": bs.precision,
        "recall": bs.recall,
        "coverage": bs.coverage,
        "agreement": metrics.agreement(y_true, y_pred),
        "confusion_rows_truth_cols_pred": {
            "rows": ["present", "absent"],
            "cols": ["present", "absent", "abstain"],
            "matrix": cm.tolist(),
        },
    }


def score(log: dict, clips: Sequence[dict]) -> dict:
    """Detection scores per hazard, overall, and per condition value; localization quality on true positives."""
    by_id = {c["id"]: c for c in clips}
    rows = []  # (clip, hazard, truth, pred)
    for c in log["clips"]:
        labels = by_id[c["id"]].get("labels", {})
        for hz, choice in c["verdicts"].items():
            if hz in labels:
                rows.append((c, hz, _label(bool(labels[hz]["present"])), _pred(choice)))
    out: dict = {"n_decisions": len(rows), "overall": _scores([r[2] for r in rows], [r[3] for r in rows])}
    out["by_hazard"] = {
        hz: _scores([r[2] for r in rows if r[1] == hz], [r[3] for r in rows if r[1] == hz])
        for hz in sorted({r[1] for r in rows})
    }
    by_cond: dict = {}
    for key in sorted({k for r in rows for k in r[0].get("condition", {})}):
        for val in sorted({r[0]["condition"].get(key) for r in rows if key in r[0].get("condition", {})}):
            sel = [r for r in rows if r[0].get("condition", {}).get(key) == val]
            by_cond[f"{key}={val}"] = _scores([r[2] for r in sel], [r[3] for r in sel])
    out["by_condition"] = by_cond
    loc = []
    for inc in log["incidents"]:
        lab = by_id[inc["clip_id"]].get("labels", {}).get(inc["type"], {})
        if not lab.get("present"):
            continue
        entry = {"clip_id": inc["clip_id"], "type": inc["type"]}
        if "start_s" in lab:
            entry["temporal_iou"] = round(
                temporal_iou((inc["start_s"], inc["end_s"]), (lab["start_s"], lab["end_s"])), 3
            )
        if "bbox_2d" in lab:
            ious = [box_iou(b["bbox_2d"], lab["bbox_2d"]) for b in inc["boxes"] if b["valid"]]
            entry["best_box_iou"] = round(max(ious), 3) if ious else 0.0
        loc.append(entry)
    out["localization_on_true_positives"] = loc
    out["failure_modes"] = failure_modes(log, rows)
    return out


def failure_modes(log: dict, rows: list) -> dict:
    fm = {
        "false_positives": [f"{r[0]['id']}/{r[1]}" for r in rows if r[2] == "absent" and r[3] == "present"],
        "false_negatives": [f"{r[0]['id']}/{r[1]}" for r in rows if r[2] == "present" and r[3] == "absent"],
        "abstentions": [f"{r[0]['id']}/{r[1]}" for r in rows if r[3] is None],
        "malformed_replies": [k for c in log["clips"] for k in c["malformed"]],
        "missing_replies": [k for c in log["clips"] for k in c["missing_replies"]],
        "incident_flags": [f"{i['clip_id']}/{i['type']}: {f}" for i in log["incidents"] for f in i["flags"]],
    }
    return fm


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _fmt_boxes(boxes: list) -> str:
    parts = []
    for b in boxes:
        if b["valid"]:
            parts.append(f"{b['label']} {b['bbox_px']}")
        else:
            parts.append(f"{b['label']} INVALID ({b['issue']})")
    return "; ".join(parts) or "none"


def to_markdown(log: dict, scores: dict | None = None) -> str:
    lines = ["# Incident log", ""]
    if log["mode"] == "mock":
        lines += [f"> **{log['note']}.** This file shows the log format and the failure handling only.", ""]
    s = next(iter(log["settings_by_step"].values()), {})
    lines += [
        f"Model: `{log['model']}`. Sampling per step is recorded in `incident_log.json` "
        f"(think steps: temperature 0.6, top_p 0.95; plain steps: temperature 0.7, top_p 0.8; top_k "
        f"{s.get('top_k')}). Boxes: model output in 0-1000 coordinates, converted to pixels of the clip size.",
        "",
        "| Clip | Incident | Severity | Time (s) | Boxes (px) | Flags |",
        "|---|---|---|---|---|---|",
    ]
    for i in log["incidents"]:
        flags = ", ".join(i["flags"]) or "-"
        lines.append(
            f"| {i['clip_id']} | {i['type']} | {i['severity']} | {i['start_s']:.2f}-{i['end_s']:.2f} | "
            f"{_fmt_boxes(i['boxes'])} | {flags} |"
        )
    if not log["incidents"]:
        lines.append("| - | none | - | - | - | - |")
    lines += ["", "## Verdicts per clip", "", "| Clip | Condition | " + " | ".join(HAZARDS) + " | Malformed |"]
    lines.append("|---|---|" + "---|" * len(HAZARDS) + "---|")
    for c in log["clips"]:
        cond = ", ".join(f"{k}={v}" for k, v in c["condition"].items()) or "-"
        verdicts = " | ".join(str(c["verdicts"].get(h) or "abstain") for h in HAZARDS)
        lines.append(f"| {c['id']} | {cond} | {verdicts} | {len(c['malformed'])} |")
    lines += ["", "Letters: A = yes (incident), C = no, abstain = (B) or an unparseable reply.", ""]
    lines += ["## Evidence (model reasoning, first 400 characters)", ""]
    for i in log["incidents"]:
        lines.append(f"- **{i['clip_id']} / {i['type']}**: {i['evidence'].strip() or '(none)'}")
    if scores:
        o = scores["overall"]
        lines += [
            "",
            "## Score against labels",
            "",
            *(
                [
                    "These numbers score the hand-written mock replies. They test the scoring code; they say nothing "
                    "about Cosmos.",
                    "",
                ]
                if log["mode"] == "mock"
                else []
            ),
            f"Decisions: {scores['n_decisions']}. Precision {o['precision']:.2f}, recall {o['recall']:.2f}, "
            f"agreement {o['agreement']:.2f}, coverage {o['coverage']:.2f}.",
            "",
            "| Slice | n | TP | FP | FN | TN | Abstain | Precision | Recall | Agreement |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        slices = {"overall": o, **{f"hazard {k}": v for k, v in scores["by_hazard"].items()}, **scores["by_condition"]}
        for name, v in slices.items():
            lines.append(
                f"| {name} | {v['n']} | {v['tp']} | {v['fp']} | {v['fn']} | {v['tn']} | {v['abstained']} | "
                f"{_f(v['precision'])} | {_f(v['recall'])} | {_f(v['agreement'])} |"
            )
        lines += ["", "Confusion matrix (rows truth present/absent; columns predicted present/absent/abstain):", ""]
        lines.append(f"`{o['confusion_rows_truth_cols_pred']['matrix']}`")
        lines += ["", "## Failure modes", ""]
        for k, v in scores["failure_modes"].items():
            lines.append(f"- {k}: {', '.join(v) if v else 'none'}")
        if scores["localization_on_true_positives"]:
            lines += ["", "Localization on true positives:", ""]
            for e in scores["localization_on_true_positives"]:
                lines.append(
                    f"- {e['clip_id']}/{e['type']}: temporal IoU {e.get('temporal_iou', 'n/a')}, "
                    f"best box IoU {e.get('best_box_iou', 'n/a')}"
                )
    return "\n".join(lines) + "\n"


def _f(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{x:.2f}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="analyze every clip in a manifest")
    r.add_argument("--manifest", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--mock", type=Path, help="recorded replies (dry run); default: tests/fixtures/mock_responses.json")
    r.add_argument("--dry-run", action="store_true", help="use --mock replies; print requests; no server")
    r.add_argument("--base-url", default="http://localhost:8000/v1")
    r.add_argument("--model", default=rp.DEFAULT_MODEL)
    r.add_argument("--fps", type=float, default=4.0)
    r.add_argument("--extract-frames", action="store_true", help="cut keyframes with ffmpeg (live clips only)")
    r.add_argument("--print-requests", action="store_true")
    s = sub.add_parser("score", help="score a log against manifest labels")
    s.add_argument("--log", type=Path, required=True)
    s.add_argument("--manifest", type=Path, required=True)
    s.add_argument("--markdown", type=Path, help="rewrite this Markdown log with the score appended")
    args = ap.parse_args(argv)

    if args.cmd == "score":
        log = json.loads(args.log.read_text())
        sc = score(log, read_jsonl(args.manifest))
        print(json.dumps(sc, indent=2))
        if args.markdown:
            args.markdown.write_text(to_markdown(log, sc))
        return 0

    clips = read_jsonl(args.manifest)
    mock = args.dry_run or args.mock is not None
    if mock:
        client: Reasoner = MockReasoner.from_file(args.mock or _HERE / "tests" / "fixtures" / "mock_responses.json")
    else:
        client = LiveReasoner(args.base_url, model=None if args.model == rp.DEFAULT_MODEL else args.model)
    log = run(clips, client, args.out, mock=mock, model=args.model, fps=args.fps, extract_frames=args.extract_frames)
    if args.print_requests:
        print((args.out / "requests.jsonl").read_text(), end="")
    if any("labels" in c for c in clips):
        sc = score(log, clips)
        (args.out / "score.json").write_text(json.dumps(sc, indent=2) + "\n")
        (args.out / "incident_log.md").write_text(to_markdown(log, sc))
    print(f"[incident_agent] {len(log['incidents'])} incidents from {len(clips)} clips -> {args.out}/incident_log.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

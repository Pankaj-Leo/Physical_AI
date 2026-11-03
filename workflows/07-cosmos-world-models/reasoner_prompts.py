#!/usr/bin/env python3
"""Cosmos 3 Reasoner prompt catalogue for robot, AV, and warehouse video, with request builders and parsers.

Every prompt records where its wording comes from:

* ``cookbook``         - verbatim from ``cosmos-main/cookbooks/cosmos3/reasoner/reasoner_prompt_guide.md``
                         (Cosmos 3; examples there were tested with Cosmos3-Super-Reasoner).
* ``cookbook-template`` - the cookbook sentence with one slot filled (task, target, event, setup sentence).
* ``repo``             - written here in the cookbook's multiple-choice style; not tested by NVIDIA.

Request shape, sampling, and parsing reuse ``ssw.judge`` (system prompt, think suffix, the two sampling presets,
``media_url``, ``parse_think``, ``parse_choice``, ``parse_segments``) so the flagship judge and this catalogue
cannot drift apart. Media goes before text in the user turn. Boxes and points are normalized 0-1000 per axis,
origin top-left, x right, y down (cookbook "2D Grounding and Pointing Coordinate Conventions").

Usage::

    python reasoner_prompts.py list                                  # print the catalogue
    python reasoner_prompts.py request --prompt warehouse.grounding \\
        --media frame.jpg --field target="the forklift"              # print one request (dry run, no server)

Building and parsing are tested on CPU. Sending a request needs a vLLM server with ``nvidia/Cosmos3-Nano``
(not executed: needs GPU); see ``incident_agent.py`` for the live client.
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
from dataclasses import dataclass, field
from pathlib import Path

_SSW_SRC = Path(__file__).resolve().parents[2] / "project" / "src"
if _SSW_SRC.is_dir() and str(_SSW_SRC) not in sys.path:
    sys.path.insert(0, str(_SSW_SRC))

from ssw import judge  # noqa: E402

DEFAULT_MODEL = "nvidia/Cosmos3-Nano"

# Output kinds the parser understands.
KINDS = ("choice", "segments", "timestamp", "bbox", "points", "subjects", "text")


@dataclass(frozen=True)
class PromptSpec:
    id: str
    domain: str  # robot | av | warehouse
    task: str
    template: str  # str.format template; slots listed in ``fields``
    kind: str  # one of KINDS
    reasoning: bool  # default: append the <think> suffix
    media: str = "video"  # video | image
    choices: str = ""  # valid letters for kind == "choice"
    provenance: str = "cookbook"
    defaults: dict = field(default_factory=dict)

    @property
    def fields(self) -> list[str]:
        return [f for _, f, _, _ in string.Formatter().parse(self.template) if f]

    def render(self, **values) -> str:
        merged = {**self.defaults, **values}
        missing = [f for f in self.fields if f not in merged]
        if missing:
            raise KeyError(f"prompt {self.id!r} needs field(s) {missing}")
        return self.template.format(**merged)


# cookbook verbatim, already in ssw.judge; braces escaped because templates go through str.format
_TEMPORAL_LOCALIZATION = judge.QUESTIONS["segments"].replace("{", "{{").replace("}", "}}")
_EVENT_TIMELINE = (
    "Describe the notable events in the provided video. Provide the result in json format with 'mm:ss.ff' format "
    "for time depiction for each event. Use keywords 'start', 'end' and 'caption' in the json output."
)
_TIMESTAMP_QUERY = (
    "When is \"{event}\" depicted in the video? Please provide the result in json format with 'mm:ss.ff' format "
    "for time depiction for the event. Use keywords 'start', 'end' in the json output."
)
_PLAUSIBILITY = (
    "Is this video physically plausible/possible according to your understanding of e.g. object permanence, "
    "shape constancy (objects maintain shape over time), continuous trajectories of objects? Assume it is the "
    "normal laws of physics.\n"
    "Your answer should be based on the events in the video and ignore the quality of the simulation engine. "
    "{setup}\n(A) Possible\n(B) Impossible"
)
_GROUNDING = "Locate the accurate bounding box of {target}. Return a json."
_TRAJECTORY = (
    'You are given the task "{task}". Specify the 2D trajectory your end effector should follow in pixel space. '
    'Return the trajectory coordinates in JSON format like this: {{"point_2d": [x, y], "label": "gripper trajectory"}}.'
)

# A hazard question shared by the warehouse incident agent. Hazard types follow the Metropolis talk
# ("proximity violation between human and humanoids, robots and humans and also zone violations");
# the wording is ours.
_HAZARD = "{question}\n(A) Yes\n(B) Impossible to tell\n(C) No"

CATALOGUE: dict[str, PromptSpec] = {
    p.id: p
    for p in [
        # ------------------------------------------------------------------ robot
        PromptSpec(
            "robot.task_success",
            "robot",
            "task success verdict",
            "The robot's task is to {task}. Did the robot complete the task?\n(A) Yes\n(B) Impossible to tell\n(C) No",
            "choice",
            True,
            choices="ABC",
            provenance="repo",
            defaults={"task": "walk forward across the terrain without falling"},
        ),
        PromptSpec(
            "robot.plausibility",
            "robot",
            "physical plausibility",
            _PLAUSIBILITY,
            "choice",
            False,
            choices="AB",
            provenance="cookbook-template",
            defaults={
                "setup": "The rising wall is part of the experiment setup and should not be judged for plausibility."
            },
        ),
        PromptSpec(
            "robot.segments", "robot", "temporal localization (seconds)", _TEMPORAL_LOCALIZATION, "segments", False
        ),
        PromptSpec(
            "robot.next_action",
            "robot",
            "next immediate action",
            "What can be the next immediate action?",
            "text",
            True,
        ),
        PromptSpec(
            "robot.plan",
            "robot",
            "subtask plan",
            "The task is to {task}. Generate a plan consisting of subtasks for accomplish the task.",
            "text",
            False,
            media="image",
            provenance="cookbook-template",
        ),
        PromptSpec(
            "robot.trajectory",
            "robot",
            "action chain-of-thought trajectory (0-1000 points)",
            _TRAJECTORY,
            "points",
            True,
            media="image",
            provenance="cookbook-template",
        ),
        PromptSpec(
            "robot.assisted_next_action",
            "robot",
            "assisted-task next action",
            'This is the overall task that the agent is trying to complete: "{overall_task}"\n'
            "In the video, the agent is trying to follow the instruction (a single step out of many to complete the "
            'overall task): "{instruction}"\nWhat should be the next action of the agent?',
            "text",
            True,
            provenance="cookbook-template",
        ),
        # ------------------------------------------------------------------ AV
        PromptSpec(
            "av.next_action",
            "av",
            "drive-scene next action and trajectory",
            "You are an autonomous vehicle planning system. The video depicts the observation from the vehicle's "
            "camera. You need to observe the critical objects in the environment and reason your next action and the "
            "driving trajectory ahead.",
            "text",
            False,
        ),
        PromptSpec(
            "av.critical_objects",
            "av",
            "objects critical for safe navigation",
            "The video depicts the observation from the vehicle's camera. You need to think step by step and identify "
            "the objects in the scene that are critical for safe navigation.",
            "text",
            True,
        ),
        PromptSpec("av.event_timeline", "av", "event timeline (mm:ss.ff)", _EVENT_TIMELINE, "segments", False),
        PromptSpec(
            "av.grounding",
            "av",
            "2D grounding (0-1000 boxes)",
            _GROUNDING,
            "bbox",
            False,
            media="image",
            provenance="cookbook-template",
        ),
        PromptSpec(
            "av.plausibility",
            "av",
            "physical plausibility of generated driving video",
            _PLAUSIBILITY,
            "choice",
            False,
            choices="AB",
            provenance="cookbook-template",
            defaults={
                "setup": "The vehicle hood at the bottom of the frame and the dashcam lens are part of the "
                "experiment setup and should not be judged for plausibility."
            },
        ),
        # ------------------------------------------------------------------ warehouse
        PromptSpec("warehouse.caption", "warehouse", "dense caption", "Describe the video in detail.", "text", False),
        PromptSpec(
            "warehouse.event_timeline", "warehouse", "event timeline (mm:ss.ff)", _EVENT_TIMELINE, "segments", False
        ),
        PromptSpec(
            "warehouse.timestamp_query",
            "warehouse",
            "when does an event happen",
            _TIMESTAMP_QUERY,
            "timestamp",
            False,
            provenance="cookbook-template",
        ),
        PromptSpec(
            "warehouse.interval",
            "warehouse",
            "what happened in an interval",
            "What happened between {start} and {end}?",
            "text",
            False,
            provenance="cookbook-template",
        ),
        PromptSpec(
            "warehouse.grounding",
            "warehouse",
            "2D grounding (0-1000 boxes)",
            _GROUNDING,
            "bbox",
            False,
            media="image",
            provenance="cookbook-template",
        ),
        PromptSpec(
            "warehouse.hazard.near_miss",
            "warehouse",
            "person close to a moving vehicle",
            _HAZARD,
            "choice",
            True,
            choices="ABC",
            provenance="repo",
            defaults={
                "question": "Does a person come within about two meters of a moving forklift or mobile robot at any "
                "time in the video?"
            },
        ),
        PromptSpec(
            "warehouse.hazard.zone_violation",
            "warehouse",
            "person inside a marked restricted zone",
            _HAZARD,
            "choice",
            True,
            choices="ABC",
            provenance="repo",
            defaults={
                "question": "Does a person walk into a floor area marked as restricted (painted hatching, a taped "
                "box, or a fenced robot cell) at any time in the video?"
            },
        ),
        PromptSpec(
            "warehouse.hazard.fallen_load",
            "warehouse",
            "load falls or spills",
            _HAZARD,
            "choice",
            True,
            choices="ABC",
            provenance="repo",
            defaults={"question": "Does any box, pallet load, or other object fall onto the floor during the video?"},
        ),
    ]
}


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


def build_request(
    prompt_id: str,
    media: str,
    *,
    model: str = DEFAULT_MODEL,
    reasoning: bool | None = None,
    fps: float = 4.0,
    max_tokens: int = 4096,
    embed: bool = False,
    **values,
) -> dict:
    """Keyword arguments for ``client.chat.completions.create`` (OpenAI-compatible vLLM server).

    Same shape as ``ssw.judge.build_request``: system prompt, media part first, then text; cookbook sampling
    preset chosen by whether the think suffix is appended.
    """
    spec = CATALOGUE[prompt_id]
    reasoning = spec.reasoning if reasoning is None else reasoning
    text = spec.render(**values) + ("\n\n" + judge.THINK_SUFFIX if reasoning else "")
    s = judge.SAMPLING[reasoning]
    url = judge.media_url(media, embed=embed)
    part = (
        {"type": "image_url", "image_url": {"url": url}}
        if spec.media == "image"
        else {"type": "video_url", "video_url": {"url": url}}
    )
    extra_body: dict = {"top_k": s["top_k"], "repetition_penalty": s["repetition_penalty"]}
    if spec.media == "video":
        extra_body["media_io_kwargs"] = {"video": {"num_frames": -1, "fps": fps}}
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": judge.SYSTEM_PROMPT},
            {"role": "user", "content": [part, {"type": "text", "text": text}]},
        ],
        "max_tokens": max_tokens,
        "temperature": s["temperature"],
        "top_p": s["top_p"],
        "presence_penalty": s["presence_penalty"],
        "extra_body": extra_body,
    }


def sampling_metadata(request: dict) -> dict:
    """Every setting that affects the reply, for result metadata."""
    eb = request.get("extra_body", {})
    return {
        "model": request["model"],
        "max_tokens": request["max_tokens"],
        "temperature": request["temperature"],
        "top_p": request["top_p"],
        "presence_penalty": request["presence_penalty"],
        "top_k": eb.get("top_k"),
        "repetition_penalty": eb.get("repetition_penalty"),
        "video_sampling": eb.get("media_io_kwargs", {}).get("video"),
        "think_suffix": judge.THINK_SUFFIX in request["messages"][1]["content"][1]["text"],
    }


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def norm_to_px(v: float, size: int) -> int:
    """One 0-1000 coordinate to pixels, clamped to the image."""
    return round(min(max(v, 0.0), 1000.0) / 1000.0 * size)


@dataclass(frozen=True)
class Box:
    label: str
    x1: float
    y1: float
    x2: float
    y2: float
    issue: str = ""  # empty when valid

    @property
    def valid(self) -> bool:
        return not self.issue

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Cookbook conversion: ``x / 1000 * width``, ``y / 1000 * height`` (clamped to the image)."""
        return (
            norm_to_px(self.x1, width),
            norm_to_px(self.y1, height),
            norm_to_px(self.x2, width),
            norm_to_px(self.y2, height),
        )

    def as_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


def extract_json(text: str):
    """The notebook recipe: drop everything up to ``</think>``, strip code fences, ``raw_decode`` from the first
    ``[`` or ``{`` (the model may add prose after the JSON). Returns the object or ``None``."""
    body = judge.parse_think(text).answer if "think>" in (text or "").lower() else (text or "")
    body = re.sub(r"```(?:json)?", "", body)
    starts = [i for i in (body.find("["), body.find("{")) if i >= 0]
    if not starts:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(body[min(starts) :])
    except json.JSONDecodeError:
        return None
    return obj


def parse_bboxes(text: str) -> list[Box] | None:
    """``[{"bbox_2d": [x1, y1, x2, y2], "label": ...}]`` in 0-1000 coordinates. Invalid boxes are kept with an
    ``issue`` so failure modes stay visible instead of being silently dropped."""
    obj = extract_json(text)
    if obj is None:
        return None
    items = obj if isinstance(obj, list) else [obj]
    out: list[Box] = []
    for it in items:
        if not isinstance(it, dict) or "bbox_2d" not in it:
            continue
        label = str(it.get("label", ""))
        coords = it["bbox_2d"]
        try:
            x1, y1, x2, y2 = (float(v) for v in coords)
        except (TypeError, ValueError):
            out.append(Box(label, float("nan"), float("nan"), float("nan"), float("nan"), "not four numbers"))
            continue
        issue = ""
        if min(x1, y1, x2, y2) < 0 or max(x1, y1, x2, y2) > 1000:
            issue = "outside 0-1000"
        elif x2 <= x1 or y2 <= y1:
            issue = "x2<=x1 or y2<=y1"
        out.append(Box(label, x1, y1, x2, y2, issue))
    return out


def parse_points(text: str) -> list[tuple[float, float, str]] | None:
    obj = extract_json(text)
    if obj is None:
        return None
    items = obj if isinstance(obj, list) else [obj]
    pts = []
    for it in items:
        if isinstance(it, dict) and isinstance(it.get("point_2d"), list | tuple) and len(it["point_2d"]) == 2:
            try:
                pts.append((float(it["point_2d"][0]), float(it["point_2d"][1]), str(it.get("label", ""))))
            except (TypeError, ValueError):
                continue
    return pts


def parse_timestamp(text: str) -> tuple[float, float] | None:
    """``{"start": "00:13.30", "end": "00:23.40"}`` -> seconds, via ``ssw.judge.parse_segments``."""
    segs = judge.parse_segments(text)
    if not segs:
        return None
    return segs[0]["start"], segs[0]["end"]


def interpret(prompt_id: str, text: str) -> dict:
    """Parse one raw reply into a record: reasoning, answer, well_formed, plus a kind-specific field."""
    spec = CATALOGUE[prompt_id]
    ta = judge.parse_think(text)
    rec: dict = {"prompt_id": prompt_id, "raw": text, "reasoning": ta.reasoning, "answer": ta.answer}
    # A reply without think tags is fine when the think suffix was not requested.
    rec["well_formed"] = ta.well_formed or ("<think>" not in (text or "").lower())
    if spec.kind == "choice":
        rec["choice"] = judge.parse_choice(ta.answer, spec.choices)
    elif spec.kind == "segments":
        rec["segments"] = judge.parse_segments(text)
    elif spec.kind == "timestamp":
        rec["timestamp"] = parse_timestamp(text)
    elif spec.kind == "bbox":
        boxes = parse_bboxes(text)
        rec["boxes"] = (
            None if boxes is None else [{"label": b.label, "bbox_2d": b.as_list(), "issue": b.issue} for b in boxes]
        )
    elif spec.kind == "points":
        rec["points"] = parse_points(text)
    elif spec.kind == "subjects":
        rec["subjects"] = extract_json(text)
    return rec


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_fields(pairs: list[str]) -> dict:
    out = {}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"--field expects key=value, got {p!r}")
        k, v = p.split("=", 1)
        out[k] = v
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list", help="print the catalogue")
    ls.add_argument("--domain", choices=["robot", "av", "warehouse"])
    rq = sub.add_parser("request", help="print one request (dry run; no server needed)")
    rq.add_argument("--prompt", required=True, choices=sorted(CATALOGUE))
    rq.add_argument("--media", required=True)
    rq.add_argument("--field", action="append", default=[], help="template slot, key=value")
    rq.add_argument("--model", default=DEFAULT_MODEL)
    rq.add_argument("--fps", type=float, default=4.0)
    rq.add_argument("--reasoning", choices=["default", "on", "off"], default="default")
    args = ap.parse_args(argv)

    if args.cmd == "list":
        for p in CATALOGUE.values():
            if args.domain and p.domain != args.domain:
                continue
            think = "think" if p.reasoning else "plain"
            print(f"{p.id:34s} {p.kind:9s} {p.media:5s} {think:5s} {p.provenance:17s} {p.task}")
        return 0
    reasoning = {"default": None, "on": True, "off": False}[args.reasoning]
    req = build_request(
        args.prompt, args.media, model=args.model, fps=args.fps, reasoning=reasoning, **_parse_fields(args.field)
    )
    print(json.dumps(req, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

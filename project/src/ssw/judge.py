"""Phase 2: Cosmos 3 Reasoner as a critic of simulated rollouts.

Sends each Phase 1 clip to a vLLM-served ``nvidia/Cosmos3-Nano`` through the OpenAI-compatible
chat API and asks three questions:

* ``task_success``  - did the robot walk forward without falling? (A) Yes (B) Impossible to tell (C) No
* ``plausibility``  - is the motion physically plausible? (A) Possible (B) Impossible
* ``segments``      - list action segments as JSON ``[{"start", "end", "caption"}]`` in seconds

Prompts, sampling settings and the ``<think>...</think>`` answer format follow the Cosmos 3
cookbook's ``reasoner_prompt_guide.md``. Media goes before text in the user turn.

Usage::

    # Offline: print the exact requests, no server or openai package needed (tested)
    python -m ssw.judge ask --manifest clips.jsonl --dry-run

    # Online: needs a running server (not executed in this repo: needs a CUDA GPU host)
    vllm serve nvidia/Cosmos3-Nano --async-scheduling --allowed-local-media-path / --port 8000
    python -m ssw.judge ask --manifest clips.jsonl --base-url http://HOST:8000/v1 --out verdicts.jsonl

    # Score verdicts against simulator ground truth (tested on synthetic data)
    python -m ssw.judge score --verdicts verdicts.jsonl

The manifest is JSON Lines, one clip per line, e.g.
``{"clip": "/data/clips/sand_s0.mp4", "material": "sand", "seed": 0, "fell": false, "forward_distance_m": 2.1}``.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from ssw import metrics

SYSTEM_PROMPT = "You are a helpful assistant."

THINK_SUFFIX = (
    "Answer the question using the following format:\n\n<think>\nYour reasoning.\n</think>\n\n"
    "Write your final answer immediately after the </think> tag."
)

# Sampling from the cookbook (reasoner_prompt_guide.md).
SAMPLING = {
    True: {"temperature": 0.6, "top_p": 0.95, "presence_penalty": 0.0, "top_k": 20, "repetition_penalty": 1.0},
    False: {"temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5, "top_k": 20, "repetition_penalty": 1.0},
}

QUESTIONS = {
    "task_success": (
        "The robot's task is to walk forward across the terrain without falling. "
        "Did the robot complete the task?\n(A) Yes\n(B) Impossible to tell\n(C) No"
    ),
    # Cookbook plausibility prompt, with the experiment-setup sentence adapted to our scene.
    "plausibility": (
        "Is this video physically plausible/possible according to your understanding of e.g. object "
        "permanence, shape constancy (objects maintain shape over time), continuous trajectories of objects? "
        "Assume it is the normal laws of physics.\n"
        "Your answer should be based on the events in the video and ignore the quality of the simulation "
        "engine. The flat untextured ground and the particle rendering of the terrain are part of the "
        "experiment setup and should not be judged for plausibility.\n"
        "(A) Possible\n(B) Impossible"
    ),
    # Cookbook temporal-localization prompt, verbatim.
    "segments": (
        "List all action segments in the video.\n\n"
        "Provide the result in json format with 'seconds' for time depiction for each event. "
        "Use keywords 'start', 'end' and 'caption' in the json output. Please list multiple events if applicable.\n\n"
        '```json\n[\n{\n  "start": t_start,\n  "end": t_end,\n  "caption": EVENT1\n},\n{\n  "start": t_start,\n'
        '  "end": t_end,\n  "caption": EVENT2\n},\n...\n]\n```'
    ),
}

# Which questions use the think format by default. The segment prompt asks for plain JSON.
REASONING_DEFAULT = {"task_success": True, "plausibility": True, "segments": False}
VALID_CHOICES = {"task_success": "ABC", "plausibility": "AB"}

# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------


def media_url(path_or_url: str, embed: bool = False) -> str:
    """``file://`` URI for local paths (server must allow the path), passthrough for URLs.

    With ``embed=True`` the file is inlined as a base64 ``data:`` URI (what NIM expects).
    """
    if re.match(r"^(https?|file|data):", path_or_url):
        return path_or_url
    p = Path(path_or_url).expanduser().resolve()
    if embed:
        mime = "video/mp4" if p.suffix.lower() == ".mp4" else "application/octet-stream"
        return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode("ascii")
    return p.as_uri()


def build_request(
    question: str,
    video: str,
    *,
    model: str = "nvidia/Cosmos3-Nano",
    reasoning: bool | None = None,
    fps: float = 4.0,
    max_tokens: int = 4096,
    embed: bool = False,
) -> dict:
    """Keyword arguments for ``client.chat.completions.create`` for one clip and question."""
    if question not in QUESTIONS:
        raise KeyError(f"unknown question {question!r}; choose from {sorted(QUESTIONS)}")
    reasoning = REASONING_DEFAULT[question] if reasoning is None else reasoning
    text = QUESTIONS[question] + ("\n\n" + THINK_SUFFIX if reasoning else "")
    s = SAMPLING[reasoning]
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": media_url(video, embed=embed)}},
                    {"type": "text", "text": text},
                ],
            },
        ],
        "max_tokens": max_tokens,
        "temperature": s["temperature"],
        "top_p": s["top_p"],
        "presence_penalty": s["presence_penalty"],
        "extra_body": {
            "media_io_kwargs": {"video": {"num_frames": -1, "fps": fps}},
            "top_k": s["top_k"],
            "repetition_penalty": s["repetition_penalty"],
        },
    }


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThinkAnswer:
    reasoning: str
    answer: str
    well_formed: bool  # True if a closing </think> tag was found


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def parse_think(text: str) -> ThinkAnswer:
    """Split a Reasoner reply into reasoning and final answer.

    Handles ``<think>..</think>answer``, a missing opening tag (some chat templates inject
    it), an optional ``<answer>..</answer>`` wrapper (NIM variant), and plain answers.
    A reply truncated inside ``<think>`` returns an empty answer and ``well_formed=False``.
    """
    text = text or ""
    m = _THINK_RE.search(text)
    if m:
        reasoning, rest, ok = m.group(1).strip(), text[m.end() :], True
    elif "</think>" in text.lower():
        idx = text.lower().index("</think>")
        reasoning, rest, ok = text[:idx].replace("<think>", "").strip(), text[idx + len("</think>") :], True
    elif "<think>" in text.lower():
        return ThinkAnswer(reasoning=text.split(">", 1)[-1].strip(), answer="", well_formed=False)
    else:
        reasoning, rest, ok = "", text, False
    a = _ANSWER_TAG_RE.search(rest)
    answer = a.group(1) if a else rest
    return ThinkAnswer(reasoning=reasoning, answer=answer.strip(), well_formed=ok)


_CHOICE_PATTERNS = (
    r"^\s*\(?([A-Z])\)?\s*$",  # "A", "(A)"
    r"^\s*\(?([A-Z])\)[\s.:]",  # "(A) Yes", "A) Yes"
    r"^\s*([A-Z])[.:]\s",  # "A. Yes", "A: Yes"
    r"(?:answer|option|choice)\s*(?:is)?\s*[:\-]?\s*\(?([A-Z])\)?\b",  # "The answer is (C)"
    r"\(([A-Z])\)",  # first parenthesized letter anywhere
)


def parse_choice(answer: str, valid: str = "ABC") -> str | None:
    """Extract a multiple-choice letter from a final answer, or ``None`` if ambiguous."""
    text = (answer or "").strip()
    if not text:
        return None
    first_line = text.splitlines()[0]
    for pat in _CHOICE_PATTERNS:
        for candidate in (first_line, text):
            m = re.search(pat, candidate, flags=re.IGNORECASE)
            if m and m.group(1).upper() in valid:
                return m.group(1).upper()
    # Fall back to option words ("Yes"/"No"/"Possible"/"Impossible") when unambiguous.
    words = {"yes": "A", "no": "C", "possible": "A", "impossible": "B"}
    found = {words[w] for w in re.findall(r"\b(yes|no|possible|impossible)\b", text.lower())}
    if len(found) == 1:
        letter = found.pop()
        if valid == "AB" and letter == "C":
            return None
        return letter if letter in valid else None
    return None


def parse_segments(text: str) -> list[dict] | None:
    """Parse the temporal-segments JSON (after any ``</think>``), tolerating fences and trailing prose.

    Returns a list of ``{"start": float, "end": float, "caption": str}`` or ``None``.
    """
    body = parse_think(text).answer if "think>" in (text or "").lower() else (text or "")
    body = re.sub(r"```(?:json)?", "", body)
    starts = [i for i in (body.find("["), body.find("{")) if i >= 0]
    if not starts:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(body[min(starts) :])
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict):
        obj = [obj]
    out = []
    for seg in obj:
        if not isinstance(seg, dict) or "start" not in seg or "end" not in seg:
            continue
        try:
            out.append(
                {
                    "start": _to_seconds(seg["start"]),
                    "end": _to_seconds(seg["end"]),
                    "caption": str(seg.get("caption", "")),
                }
            )
        except ValueError:
            continue
    return out


def _to_seconds(v) -> float:
    """Accept seconds as number/str or ``mm:ss.ff``."""
    if isinstance(v, int | float):
        return float(v)
    s = str(v).strip()
    if ":" in s:
        mm, ss = s.split(":", 1)
        return 60.0 * float(mm) + float(ss)
    return float(s)


# ---------------------------------------------------------------------------
# Verdicts and scoring
# ---------------------------------------------------------------------------


def task_label(choice: str | None) -> str | None:
    """Map the task-success choice to ``"success"``/``"failure"``; B and unparseable abstain."""
    return {"A": "success", "C": "failure"}.get(choice or "")


def ground_truth_label(clip: dict, min_distance_m: float = 1.0) -> str:
    """Simulator ground truth: success iff the robot did not fall and walked far enough."""
    ok = (not clip.get("fell", False)) and float(clip.get("forward_distance_m", 0.0)) >= min_distance_m
    return "success" if ok else "failure"


def score(records: Sequence[dict], min_distance_m: float = 1.0) -> dict:
    """Precision/recall of the judge for detecting *failures*, overall and per material."""
    y_true = [ground_truth_label(r["clip_meta"], min_distance_m) for r in records]
    y_pred = [task_label(r.get("task_success", {}).get("choice")) for r in records]
    labels_pred = ["success", "failure", None]
    out = {
        "n": len(records),
        "failure_detection": asdict_scores(metrics.binary_scores(y_true, y_pred, positive="failure")),
        "agreement": metrics.agreement(y_true, y_pred),
        "confusion_rows_truth_cols_pred": {
            "labels_true": ["success", "failure"],
            "labels_pred": ["success", "failure", "abstain"],
            "matrix": metrics.confusion_matrix(y_true, y_pred, ["success", "failure"], labels_pred).tolist(),
        },
        "plausible_rate": _rate(r.get("plausibility", {}).get("choice") == "A" for r in records),
        "by_material": {},
    }
    for mat in sorted({r["clip_meta"].get("material", "?") for r in records}):
        idx = [i for i, r in enumerate(records) if r["clip_meta"].get("material", "?") == mat]
        bs = metrics.binary_scores([y_true[i] for i in idx], [y_pred[i] for i in idx], positive="failure")
        out["by_material"][mat] = {
            **asdict_scores(bs),
            "agreement": metrics.agreement([y_true[i] for i in idx], [y_pred[i] for i in idx]),
        }
    return out


def asdict_scores(bs: metrics.BinaryScores) -> dict:
    return {
        **asdict(bs),
        "precision": bs.precision,
        "recall": bs.recall,
        "f1": bs.f1,
        "accuracy": bs.accuracy,
        "coverage": bs.coverage,
    }


def _rate(flags: Iterable[bool]) -> float:
    flags = list(flags)
    return sum(flags) / len(flags) if flags else float("nan")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class Judge:
    """Thin wrapper around an OpenAI-compatible client. Not executed: needs a vLLM server on a CUDA GPU."""

    def __init__(self, base_url: str, api_key: str = "EMPTY", model: str | None = None, timeout: float = 600.0):
        try:
            import openai
        except ImportError as exc:
            raise SystemExit("pip install 'sand-solvers-worldmodels[cosmos]' (needs the openai package)") from exc
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model or self.client.models.list().data[0].id

    def ask(self, question: str, video: str, **kw) -> dict:
        req = build_request(question, video, model=self.model, **kw)
        resp = self.client.chat.completions.create(**req)
        text = resp.choices[0].message.content or ""
        return interpret(question, text)


def interpret(question: str, text: str) -> dict:
    """Parse one raw reply into a verdict record."""
    ta = parse_think(text)
    rec = {"raw": text, "reasoning": ta.reasoning, "answer": ta.answer, "well_formed": ta.well_formed}
    if question in VALID_CHOICES:
        rec["choice"] = parse_choice(ta.answer, VALID_CHOICES[question])
    elif question == "segments":
        rec["segments"] = parse_segments(text)
    return rec


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask", help="query the Reasoner for every clip in a manifest")
    a.add_argument("--manifest", type=Path, required=True)
    a.add_argument("--base-url", default="http://localhost:8000/v1")
    a.add_argument("--model", default=None, help="defaults to the first model the server lists")
    a.add_argument("--questions", nargs="+", choices=sorted(QUESTIONS), default=list(QUESTIONS))
    a.add_argument("--fps", type=float, default=4.0)
    a.add_argument("--embed", action="store_true", help="inline videos as base64 data URIs (NIM)")
    a.add_argument("--dry-run", action="store_true", help="print requests; no server, no openai import")
    a.add_argument("--out", type=Path, default=None)
    sc = sub.add_parser("score", help="score verdicts against simulator ground truth")
    sc.add_argument("--verdicts", type=Path, required=True)
    sc.add_argument("--min-distance", type=float, default=1.0)
    args = p.parse_args(argv)

    if args.cmd == "score":
        print(json.dumps(score(_read_jsonl(args.verdicts), args.min_distance), indent=2))
        return 0

    clips = _read_jsonl(args.manifest)
    if args.dry_run:
        for clip in clips:
            for q in args.questions:
                req = build_request(q, clip["clip"], model=args.model or "nvidia/Cosmos3-Nano", fps=args.fps)
                print(json.dumps({"clip": clip["clip"], "question": q, "request": req}))
        return 0

    judge = Judge(args.base_url, model=args.model)
    out = args.out.open("w") if args.out else sys.stdout
    try:
        for clip in clips:
            rec = {"clip_meta": clip}
            for q in args.questions:
                rec[q] = judge.ask(q, clip["clip"], fps=args.fps, embed=args.embed)
            out.write(json.dumps(rec) + "\n")
            out.flush()
    finally:
        if out is not sys.stdout:
            out.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

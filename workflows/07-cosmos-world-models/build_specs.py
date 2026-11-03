#!/usr/bin/env python3
"""Write the Cosmos 3 transfer requests in ``specs/`` (robot depth+seg, AV world-scenario-map) with prompt variants.

(a) Robot rollouts: depth + segmentation control videos rendered from a simulated quadruped rollout (the flagship's
    Phase 1 renders or workflow 06). Built with ``ssw.transfer`` (structured JSON prompt, appearance variants desert /
    snowfield / muddy construction site, cookbook sampling: 121 frames at 30 fps, 50 steps, guidance 3.0, shift 10).
(b) AV: the Cosmos 3 cookbook has no "HD map + boxes" hint. Its closest control is ``wsm`` (world scenario map), which
    the Cosmos 2.5 paper describes as HD-map lanes, poles, lights and colour/heading-coded cuboids, and whose cookbook
    sample is a dashcam-view render of lanes and vehicle boxes. The AV spec mirrors ``specs/wsm.json`` field for field
    (101 frames at 10 fps, guidance 1.0, control guidance 3.0). The control video can come from a real WSM renderer
    or, for plumbing tests, from ``dreams_loop.py export-wsm --encode`` (a toy look-alike, not the official format).

Each variant gets three files:

* ``<stem>_prompt.json``    - the structured JSON caption (cookbook schema).
* ``<stem>.json``           - a Cosmos Framework spec, runnable with ``cosmos_framework.scripts.inference -i``.
* ``<stem>_request.json``   - the same request for vLLM-Omni (``POST /v1/videos/sync`` form) and Diffusers kwargs.

Usage::

    python build_specs.py               # (re)write specs/
    python build_specs.py --check       # exit 1 if specs/ differs from what this script would write

Writing the JSON is tested on CPU. Running any of it needs Cosmos3-Nano on an 80 GB GPU (not executed: needs GPU).
"""

from __future__ import annotations

import argparse
import copy
import filecmp
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
_SSW_SRC = HERE.parents[1] / "project" / "src"
if _SSW_SRC.is_dir() and str(_SSW_SRC) not in sys.path:
    sys.path.insert(0, str(_SSW_SRC))

from ssw import transfer  # noqa: E402

COSMOS_ROOT = REPO / "cosmos-main"
SPEC_DIR = HERE / "specs"

ROBOT_CONTROLS = {"depth": "controls/robot_rollout_depth.mp4", "seg": "controls/robot_rollout_seg.mp4"}
ROBOT_MATERIAL = "sand"
AV_CONTROL = "controls/av_wsm.mp4"

# Mirrors cosmos-main/cookbooks/cosmos3/generator/transfer/specs/wsm.json (Cosmos 3 cookbook).
WSM_SETTINGS = {
    "model_mode": "video2video",
    "resolution": "720",
    "aspect_ratio": "16,9",
    "num_frames": 101,
    "fps": 10,
    "shift": 10.0,
    "num_steps": 50,
    "seed": 2026,
    "num_video_frames_per_chunk": 101,
    "num_conditional_frames": 1,
    "num_first_chunk_conditional_frames": 0,
    "share_vision_temporal_positions": True,
    "negative_metadata_mode": "none",
    "negative_prompt_keep_metadata": False,
    "guidance": 1.0,
    "control_guidance": 3.0,
}

# ---------------------------------------------------------------------------
# AV prompts: one scene (the dreams_loop toy scenario), five lighting/weather variants
# ---------------------------------------------------------------------------

AV_BASE = {
    "subjects": [
        {
            "description": "A dark gray four-door sedan driving ahead in the same lane",
            "appearance_details": "Clean paint, red tail lights, standard alloy wheels",
            "relationship": "Lead vehicle followed by the recording vehicle",
            "location": "Center of frame, in the right lane ahead",
            "relative_size": "Small to medium within frame",
            "orientation": "Facing away from camera",
            "pose": "Vehicle in motion",
            "action": "Drives ahead, then brakes firmly and slows down",
            "state_changes": "Brake lights turn on and the gap to the recording vehicle shrinks",
            "number_of_subjects": 1,
            "number_of_arms": 0,
            "number_of_legs": 0,
        },
        {
            "description": "A white compact SUV in the opposite lane",
            "appearance_details": "White paint, headlights on",
            "relationship": "Oncoming traffic",
            "location": "Left lane, approaching from the distance",
            "relative_size": "Small, growing as it approaches",
            "orientation": "Facing the camera",
            "pose": "Vehicle in motion",
            "action": "Passes the recording vehicle on the left",
            "state_changes": "Approaches and leaves the frame on the left",
            "number_of_subjects": 1,
            "number_of_arms": 0,
            "number_of_legs": 0,
        },
        {
            "description": "The hood of the recording vehicle at the bottom of the frame",
            "appearance_details": "Dark glossy hood",
            "relationship": "The host vehicle carrying the dashcam",
            "location": "Lower foreground",
            "relative_size": "Large within frame",
            "orientation": "Facing forward",
            "pose": "Moving forward",
            "action": "Follows the gently curving road",
            "state_changes": "Slows down when the lead vehicle brakes",
            "number_of_subjects": 1,
            "number_of_arms": 0,
            "number_of_legs": 0,
        },
    ],
    "cinematography": {
        "camera_motion": "Forward-moving vehicle-mounted camera",
        "framing": "Wide shot",
        "camera_angle": "Eye-level",
        "depth_of_field": "Deep",
        "focus": "Sharp focus on the road ahead",
        "lens_focal_length": "Wide-angle dashcam lens",
    },
    "style_medium": "Live-action video",
    "artistic_style": "Documentary dashcam realism",
    "context": "Dashcam footage from a car following another car on a curving two-lane suburban road",
    "actions": [
        {"time": "0:00-0:04", "description": "The recording vehicle follows the sedan along the curving road."},
        {"time": "0:04-0:07", "description": "The sedan brakes firmly; the recording vehicle slows behind it."},
        {"time": "0:07-0:10", "description": "An oncoming SUV passes on the left while both cars drive on slowly."},
    ],
    "segments": [
        {
            "segment_index": 0,
            "time_range": "0:00-0:10",
            "description": "Continuous forward dashcam view following a sedan that brakes, with oncoming traffic.",
            "key_changes": "Lead vehicle brake event",
            "camera": "Vehicle-mounted, moving forward",
        }
    ],
    "transitions": [],
    "resolution": {"W": 1280, "H": 720},
    "aspect_ratio": "16,9",
    "duration": "10s",
    "fps": 10,
}

AV_VARIANTS = {
    "clear_noon": {
        "background_setting": "A two-lane suburban road with a double yellow center line, sidewalks, low houses, "
        "and leafy trees under a clear blue sky.",
        "lighting": {
            "conditions": "Clear midday sun",
            "direction": "High overhead",
            "shadows": "Short, crisp shadows under the cars and trees",
            "illumination_effect": "Bright, saturated, high-contrast scene",
        },
        "aesthetics": {"color_scheme": "Saturated greens and blues, gray asphalt", "mood_atmosphere": "Calm commute"},
        "weather": "dry",
    },
    "overcast_drizzle": {
        "background_setting": "The same suburban road under a uniformly gray sky, the asphalt darkened by drizzle.",
        "lighting": {
            "conditions": "Overcast with light drizzle",
            "direction": "Diffuse from a gray sky",
            "shadows": "Almost none",
            "illumination_effect": "Flat light, faint reflections on the damp road",
        },
        "aesthetics": {"color_scheme": "Muted grays and dull greens", "mood_atmosphere": "Quiet, damp afternoon"},
        "weather": "drizzle",
    },
    "night_rain": {
        "background_setting": "The same suburban road at night in steady rain, wet asphalt reflecting street lights.",
        "lighting": {
            "conditions": "Night, street lights and car headlights",
            "direction": "Point sources from street lamps and headlights",
            "shadows": "Hard, sparse shadows near lamps",
            "illumination_effect": "Glare and streaks of reflected light on the wet road, rain drops on the lens",
        },
        "aesthetics": {
            "color_scheme": "Dark blues and blacks, orange lamp light, red tail lights",
            "mood_atmosphere": "Tense",
        },
        "weather": "rain",
    },
    "snow": {
        "background_setting": "The same suburban road after snowfall: snow banks along the curbs, white lawns, "
        "packed snow in the wheel tracks.",
        "lighting": {
            "conditions": "Bright overcast winter light",
            "direction": "Diffuse",
            "shadows": "Soft, faint",
            "illumination_effect": "High-key scene with bright snow",
        },
        "aesthetics": {"color_scheme": "Whites and cool grays, dark cars", "mood_atmosphere": "Cold, careful driving"},
        "weather": "snow",
    },
    "dusk_fog": {
        "background_setting": "The same suburban road at dusk in fog; houses and trees fade into gray "
        "beyond 50 meters.",
        "lighting": {
            "conditions": "Dusk with dense fog",
            "direction": "Low, diffuse",
            "shadows": "None",
            "illumination_effect": "Low contrast, headlights scattered by fog",
        },
        "aesthetics": {"color_scheme": "Desaturated blue-grays", "mood_atmosphere": "Low visibility"},
        "weather": "fog",
    },
}


def build_av_prompt(variant: str) -> dict:
    """Cookbook caption schema (same top-level keys as the wsm sample prompt) for one weather/lighting variant."""
    v = AV_VARIANTS[variant]
    p = copy.deepcopy(AV_BASE)
    p["background_setting"] = v["background_setting"]
    p["lighting"] = copy.deepcopy(v["lighting"])
    p["aesthetics"] = {
        "composition": "Dashcam framing with the hood in the lower third and the road vanishing at the center",
        **v["aesthetics"],
    }
    weather = {
        "dry": "",
        "drizzle": " in light drizzle",
        "rain": " in the rain at night",
        "snow": " on a snowy road",
        "fog": " in fog at dusk",
    }[v["weather"]]
    p["temporal_caption"] = (
        f"From 0:00 to 0:04 the car follows a gray sedan along a curving suburban road{weather}. From 0:04 to 0:07 "
        "the sedan brakes and the gap closes as the car slows. From 0:07 to 0:10 a white SUV passes in the "
        "opposite lane."
    )
    order = ["subjects", "background_setting", "lighting", "aesthetics", "cinematography", "style_medium",
             "artistic_style", "context", "actions", "segments", "transitions", "temporal_caption", "resolution",
             "aspect_ratio", "duration", "fps"]  # fmt: skip
    return {k: p[k] for k in order}


def build_wsm_framework_spec(name: str, control_path: str, prompt_path: str, negative_path: str) -> dict:
    return {
        "name": name,
        **{k: WSM_SETTINGS[k] for k in WSM_SETTINGS},
        "negative_prompt_file": negative_path,
        "prompt_path": prompt_path,
        "wsm": {"control_path": control_path},
        "emphasize_control_in_prompt": False,
    }


def _compact(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"))


def vllm_form_from_spec(spec: dict, prompt: dict, negative: dict, control: str, *, guardrails: bool = False) -> dict:
    """``POST /v1/videos/sync`` form, built from a Framework spec the way the cookbook's vLLM-Omni notebook does."""
    h = int(spec["resolution"])
    wr, hr = (int(x) for x in spec["aspect_ratio"].split(",", 1))
    extra = {
        "use_resolution_template": False,
        "use_duration_template": False,
        "guardrails": guardrails,
        control: dict(spec[control]),
        "resolution": spec["resolution"],
        "control_guidance": spec["control_guidance"],
        "num_video_frames_per_chunk": spec["num_video_frames_per_chunk"],
        "num_conditional_frames": spec["num_conditional_frames"],
        "num_first_chunk_conditional_frames": spec["num_first_chunk_conditional_frames"],
        "share_vision_temporal_positions": spec["share_vision_temporal_positions"],
        "max_frames": spec["num_frames"],
    }
    return {
        "prompt": _compact(prompt),
        "negative_prompt": _compact(negative),
        "size": f"{round(h * wr / hr)}x{h}",
        "num_frames": str(spec["num_frames"]),
        "fps": str(spec["fps"]),
        "num_inference_steps": str(spec["num_steps"]),
        "guidance_scale": str(spec["guidance"]),
        "flow_shift": str(spec["shift"]),
        "seed": str(spec["seed"]),
        "extra_params": json.dumps(extra, separators=(",", ":")),
    }


def diffusers_kwargs_from_spec(spec: dict, control: str) -> dict:
    """Arguments of ``Cosmos3OmniModularPipeline(...)`` as the cookbook's Diffusers notebook passes them.

    Load ``control_videos`` with ``diffusers.utils.load_video`` and set the scheduler's ``flow_shift`` from the spec
    before calling (not executed: needs GPU).
    """
    h = int(spec["resolution"])
    wr, hr = (int(x) for x in spec["aspect_ratio"].split(",", 1))
    return {
        "control_videos": {control: spec[control]["control_path"]},
        "control_guidance": spec["control_guidance"],
        "num_frames": spec["num_frames"],
        "num_video_frames_per_chunk": spec["num_video_frames_per_chunk"],
        "num_conditional_frames": spec["num_conditional_frames"],
        "height": h,
        "width": round(h * wr / hr),
        "fps": float(spec["fps"]),
        "num_inference_steps": spec["num_steps"],
        "guidance_scale": spec["guidance"],
        "seed": spec["seed"],
        "scheduler_flow_shift": spec["shift"],
        "output": "videos",
    }


def _metadata(kind: str, guardrails: bool) -> dict:
    return {
        "cosmos_generation": "Cosmos 3 (cookbook cookbooks/cosmos3/generator/transfer)",
        "model": "nvidia/Cosmos3-Nano",
        "kind": kind,
        "guardrails": guardrails,
        "guardrails_note": "Without access to the gated nvidia/Cosmos-1.0-Guardrail repo, start vLLM-Omni with "
        "--no-guardrails; per-request guardrails:false does not avoid the startup load (cookbook).",
        "status": "not executed: needs GPU (80 GB class for Nano transfer)",
    }


def write_all(out: Path, cosmos_root: Path | None = COSMOS_ROOT, guardrails: bool = False) -> list[Path]:
    """Write every spec file under ``out``. Returns the paths written."""
    written: list[Path] = []

    def dump(path: Path, obj: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2) + "\n")
        written.append(path)

    negative = transfer.default_negative_prompt(cosmos_root if cosmos_root and cosmos_root.is_dir() else None)
    dump(out / "negative_prompt.json", negative)

    # (a) robot: depth + seg, built by ssw.transfer
    weights = {k: 0.5 for k in ROBOT_CONTROLS}
    for variant in transfer.VARIANTS:
        stem = f"robot_{variant}"
        prompt = transfer.build_prompt(variant, ROBOT_MATERIAL)
        spec = transfer.build_framework_spec(
            f"transfer_{stem}", ROBOT_CONTROLS, f"{stem}_prompt.json", "../negative_prompt.json", weights=weights
        )
        dump(out / "robot" / f"{stem}_prompt.json", prompt)
        dump(out / "robot" / f"{stem}.json", spec)
        dump(
            out / "robot" / f"{stem}_request.json",
            {
                "metadata": {**_metadata("robot depth+seg multi-control", guardrails), "variant": variant},
                "vllm_omni_form": transfer.build_vllm_omni_form(
                    ROBOT_CONTROLS, prompt, negative, guardrails=guardrails
                ),
                "diffusers_kwargs": transfer.build_diffusers_kwargs(ROBOT_CONTROLS, prompt, negative),
            },
        )

    # (b) AV: world scenario map
    for variant in AV_VARIANTS:
        stem = f"av_{variant}"
        prompt = build_av_prompt(variant)
        spec = build_wsm_framework_spec(
            f"transfer_{stem}", AV_CONTROL, f"{stem}_prompt.json", "../negative_prompt.json"
        )
        dump(out / "av" / f"{stem}_prompt.json", prompt)
        dump(out / "av" / f"{stem}.json", spec)
        dump(
            out / "av" / f"{stem}_request.json",
            {
                "metadata": {**_metadata("AV world-scenario-map (wsm) transfer", guardrails), "variant": variant},
                "vllm_omni_form": vllm_form_from_spec(spec, prompt, negative, "wsm", guardrails=guardrails),
                "diffusers_kwargs": diffusers_kwargs_from_spec(spec, "wsm"),
            },
        )
    return written


def check(out: Path = SPEC_DIR) -> list[str]:
    """Names of files that differ from a fresh build (empty list when up to date)."""
    with tempfile.TemporaryDirectory() as tmp:
        fresh = write_all(Path(tmp))
        diffs = []
        for f in fresh:
            rel = f.relative_to(tmp)
            if not (out / rel).is_file() or not filecmp.cmp(f, out / rel, shallow=False):
                diffs.append(str(rel))
        return diffs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=SPEC_DIR)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--guardrails", action="store_true", help="request guardrails (needs the gated HF repo)")
    args = ap.parse_args(argv)
    if args.check:
        diffs = check(args.out)
        print("specs up to date" if not diffs else "stale: " + ", ".join(diffs))
        return 1 if diffs else 0
    paths = write_all(args.out, guardrails=args.guardrails)
    print(f"[build_specs] wrote {len(paths)} files under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Phase 2: build Cosmos 3 transfer (controlled video-to-video) requests from Newton renders.

For each Phase 1 clip we have a depth video and a semantic-segmentation video rendered by
``SensorTiledCamera``. This module turns them into requests for ``nvidia/Cosmos3-Nano``
transfer mode, with a structured JSON prompt whose ``lighting``, ``background_setting`` and
``aesthetics`` fields are varied to produce photoreal variants (desert, snowfield, muddy
construction site).

Three request formats are produced, mirroring the Cosmos cookbook
(``cookbooks/cosmos3/generator/transfer``):

* ``vllm_omni``  - multipart form for ``POST /v1/videos/sync`` on a vLLM-Omni server.
* ``diffusers``  - keyword arguments for ``Cosmos3OmniModularPipeline.__call__``.
* ``framework``  - a Cosmos Framework spec JSON (the only format with per-hint ``weight``).

Usage::

    # Offline: write request JSON files, no GPU or server needed (tested)
    python -m ssw.transfer --depth clips/sand_s0_depth.mp4 --seg clips/sand_s0_seg.mp4 \\
        --material sand --variants desert snowfield --dry-run --out requests/

    # Online: POST to a vLLM-Omni server (not executed in this repo: needs an 80 GB CUDA GPU)
    python -m ssw.transfer --depth ... --seg ... --base-url http://HOST:8000/v1 --out videos/

Settings follow the cookbook specs: 121 frames at 30 fps, 50 steps, guidance 3.0, flow shift
10.0, control guidance 1.5 for depth and 2.0 for segmentation, seed 2026.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Sequence
from pathlib import Path

DEFAULTS = {
    "resolution": "720",
    "aspect_ratio": "16,9",
    "num_frames": 121,
    "fps": 30,
    "num_steps": 50,
    "guidance": 3.0,
    "shift": 10.0,
    "seed": 2026,
    "num_video_frames_per_chunk": 121,
    "num_conditional_frames": 1,
    "num_first_chunk_conditional_frames": 0,
}
CONTROL_GUIDANCE = {"depth": 1.5, "seg": 2.0}

# Scene variants: only the appearance fields change; geometry and motion come from the controls.
VARIANTS = {
    "desert": {
        "background_setting": (
            "An open desert of fine, wind-rippled golden sand under a wide sky, low dunes on the horizon, "
            "scattered dry shrubs and small rocks."
        ),
        "lighting": {
            "conditions": "Harsh midday sun, clear sky",
            "direction": "High overhead sun, slightly from the left",
            "shadows": "Short, crisp shadows under the robot body and legs",
            "illumination_effect": "Bright, high-contrast scene with heat shimmer near the ground",
        },
        "aesthetics": {
            "composition": "Robot centered, walking toward the right of frame along a straight path",
            "color_scheme": "Warm sand yellows and ochres against a pale blue sky; matte gray robot",
            "mood_atmosphere": "Arid, quiet field test",
        },
    },
    "snowfield": {
        "background_setting": (
            "A flat snowfield of fresh, powdery snow with faint boot tracks, a line of conifers and gray "
            "mountains in the distance."
        ),
        "lighting": {
            "conditions": "Overcast winter daylight",
            "direction": "Diffuse, top-down skylight",
            "shadows": "Very soft, low-contrast shadows",
            "illumination_effect": "Even, cool illumination with bright snow highlights",
        },
        "aesthetics": {
            "composition": "Robot centered, walking toward the right of frame",
            "color_scheme": "Cool whites and blue-grays; dark green trees; gray robot",
            "mood_atmosphere": "Cold, calm outdoor trial",
        },
    },
    "muddy_construction_site": {
        "background_setting": (
            "A muddy construction site with wet brown clay, puddles, tire ruts, orange safety fencing, "
            "stacked concrete blocks and a parked excavator in the background."
        ),
        "lighting": {
            "conditions": "Late-afternoon sun breaking through clouds",
            "direction": "Low sun from behind the camera, slightly to the right",
            "shadows": "Long, medium-soft shadows",
            "illumination_effect": "Glossy reflections on wet mud and puddles",
        },
        "aesthetics": {
            "composition": "Robot centered, walking toward the right of frame",
            "color_scheme": "Earthy browns, safety orange accents, gray robot",
            "mood_atmosphere": "Gritty, industrial field test",
        },
    },
}

MATERIAL_TEXT = {
    "rigid": "firm, flat ground",
    "sand": "loose sand that deforms and leaves footprints",
    "snow": "soft snow that compacts under each foot",
    "clay": "sticky wet clay that the feet sink into",
}


def _mmss(seconds: float) -> str:
    s = round(seconds)
    return f"{s // 60}:{s % 60:02d}"


def build_prompt(variant: str, material: str = "sand", duration_s: float = 121 / 30) -> dict:
    """Structured JSON prompt in the cookbook's caption schema, for one appearance variant."""
    if variant not in VARIANTS:
        raise KeyError(f"unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
    v = VARIANTS[variant]
    terrain = MATERIAL_TEXT.get(material, material)
    return {
        "subjects": [
            {
                "description": "A four-legged industrial inspection robot with a boxy gray body and four jointed legs",
                "appearance_details": "Matte gray shell, black joints, sensor heads at the front and rear",
                "relationship": "The only moving subject in the scene",
                "location": "Center of frame, on the ground",
                "relative_size": "Medium within frame",
                "orientation": "Side view, facing right",
                "pose": "Standing on four legs, trotting",
                "action": f"Walking forward across {terrain}",
                "state_changes": "Moves steadily from left to right; feet press into the ground surface.",
                "number_of_subjects": 1,
                "number_of_legs": 4,
            }
        ],
        "background_setting": v["background_setting"],
        "lighting": copy.deepcopy(v["lighting"]),
        "aesthetics": copy.deepcopy(v["aesthetics"]),
        "cinematography": {
            "camera_motion": "Static camera",
            "framing": "Wide shot with the whole robot and the ground around it visible",
            "camera_angle": "Slightly elevated, three-quarter side view",
            "depth_of_field": "Deep",
            "focus": "Sharp focus on the robot",
        },
        "style_medium": "Live-action video",
        "artistic_style": "Documentary, observational realism",
        "context": "Field test of a legged robot on deformable terrain",
        "actions": [{"time": f"0:00-{_mmss(duration_s)}", "description": f"The robot walks forward across {terrain}."}],
        "segments": [
            {
                "segment_index": 0,
                "time_range": f"0:00-{_mmss(duration_s)}",
                "description": f"A continuous static shot of the quadruped robot walking across {terrain}.",
            }
        ],
        "transitions": [],
        "temporal_caption": f"Throughout the clip the quadruped robot walks steadily forward across {terrain}.",
        "resolution": {"W": 1280, "H": 720},
        "aspect_ratio": "16,9",
        "duration": f"{round(duration_s)}s",
        "fps": 30,
    }


def default_negative_prompt(cosmos_root: Path | None = None) -> dict:
    """The cookbook's negative prompt if the Cosmos repo is available, else a short fallback."""
    if cosmos_root is not None:
        p = cosmos_root / "cookbooks/cosmos3/generator/transfer/assets/negative_prompt.json"
        if p.is_file():
            return json.loads(p.read_text())
    return {
        "physical_realism": "Objects pass through each other, limbs bend impossibly, feet float above the ground.",
        "aesthetics": "Blurry, low detail, compression artifacts, flickering textures, color bleeding.",
    }


def _size(resolution: str, aspect_ratio: str) -> str:
    h = int(resolution)
    wr, hr = (int(x) for x in aspect_ratio.split(",", 1))
    return f"{round(h * wr / hr)}x{h}"


def build_vllm_omni_form(
    controls: dict[str, str],
    prompt: dict,
    negative: dict,
    *,
    control_guidance: float | None = None,
    guardrails: bool = False,
    **overrides,
) -> dict[str, str]:
    """Multipart form fields for ``POST /v1/videos/sync`` (as in the cookbook's vLLM-Omni notebook).

    ``controls`` maps hint key (``"depth"``, ``"seg"``) to a control-video path that the server
    can read (``--allowed-local-media-path``). vLLM-Omni accepts several hints but no per-hint
    weights; with both depth and seg the stricter seg guidance (2.0) is used unless overridden.
    """
    if not controls:
        raise ValueError("at least one control video is required")
    unknown = set(controls) - set(CONTROL_GUIDANCE)
    if unknown:
        raise ValueError(f"unsupported control(s) {sorted(unknown)}; use {sorted(CONTROL_GUIDANCE)}")
    cfg = {**DEFAULTS, **overrides}
    cg = control_guidance if control_guidance is not None else max(CONTROL_GUIDANCE[k] for k in controls)
    extra = {
        "use_resolution_template": False,
        "use_duration_template": False,
        "guardrails": guardrails,
        **{k: {"control_path": str(v)} for k, v in controls.items()},
        "resolution": cfg["resolution"],
        "control_guidance": cg,
        "num_video_frames_per_chunk": cfg["num_video_frames_per_chunk"],
        "num_conditional_frames": cfg["num_conditional_frames"],
        "num_first_chunk_conditional_frames": cfg["num_first_chunk_conditional_frames"],
        "share_vision_temporal_positions": True,
        "max_frames": cfg["num_frames"],
    }
    return {
        "prompt": json.dumps(prompt, ensure_ascii=True, separators=(",", ":")),
        "negative_prompt": json.dumps(negative, ensure_ascii=True, separators=(",", ":")),
        "size": _size(cfg["resolution"], cfg["aspect_ratio"]),
        "num_frames": str(cfg["num_frames"]),
        "fps": str(cfg["fps"]),
        "num_inference_steps": str(cfg["num_steps"]),
        "guidance_scale": str(cfg["guidance"]),
        "flow_shift": str(cfg["shift"]),
        "seed": str(cfg["seed"]),
        "extra_params": json.dumps(extra, separators=(",", ":")),
    }


def build_diffusers_kwargs(controls: dict[str, str], prompt: dict, negative: dict, **overrides) -> dict:
    """JSON-serializable kwargs for ``Cosmos3OmniModularPipeline(...)``.

    Control videos are given as paths; load them with ``diffusers.utils.load_video`` and pass
    a ``torch.Generator(device="cuda").manual_seed(seed)`` before calling. Building the dict is tested;
    calling the pipeline with it is not executed: needs CUDA.
    """
    cfg = {**DEFAULTS, **overrides}
    h = int(cfg["resolution"])
    w = int(_size(cfg["resolution"], cfg["aspect_ratio"]).split("x")[0])
    return {
        "prompt": json.dumps(prompt),
        "negative_prompt": json.dumps(negative),
        "control_videos": dict(controls),
        "control_guidance": max(CONTROL_GUIDANCE[k] for k in controls),
        "num_frames": cfg["num_frames"],
        "num_video_frames_per_chunk": cfg["num_video_frames_per_chunk"],
        "num_conditional_frames": cfg["num_conditional_frames"],
        "height": h,
        "width": w,
        "fps": float(cfg["fps"]),
        "num_inference_steps": cfg["num_steps"],
        "guidance_scale": cfg["guidance"],
        "seed": cfg["seed"],
        "output": "videos",
    }


def build_framework_spec(
    name: str, controls: dict[str, str], prompt_path: str, negative_path: str, weights: dict[str, float] | None = None
) -> dict:
    """Cosmos Framework spec (``specs/depth.json`` schema), with optional per-hint weights."""
    spec = {
        "name": name,
        "model_mode": "video2video",
        "resolution": DEFAULTS["resolution"],
        "aspect_ratio": DEFAULTS["aspect_ratio"],
        "num_frames": DEFAULTS["num_frames"],
        "fps": DEFAULTS["fps"],
        "shift": DEFAULTS["shift"],
        "num_steps": DEFAULTS["num_steps"],
        "seed": DEFAULTS["seed"],
        "num_video_frames_per_chunk": DEFAULTS["num_video_frames_per_chunk"],
        "num_conditional_frames": DEFAULTS["num_conditional_frames"],
        "num_first_chunk_conditional_frames": DEFAULTS["num_first_chunk_conditional_frames"],
        "share_vision_temporal_positions": True,
        "negative_metadata_mode": "none",
        "negative_prompt_keep_metadata": False,
        "guidance": DEFAULTS["guidance"],
        "control_guidance": max(CONTROL_GUIDANCE[k] for k in controls),
        "negative_prompt_file": negative_path,
        "prompt_path": prompt_path,
        "emphasize_control_in_prompt": False,
    }
    for k, path in controls.items():
        spec[k] = {"control_path": path}
        if weights and k in weights:
            spec[k]["weight"] = weights[k]
    return spec


def post_vllm_omni(base_url: str, form: dict[str, str], out_path: Path, timeout: float = 3600.0) -> Path:
    """POST one request and save the returned MP4. Not executed: needs a vLLM-Omni server on CUDA."""
    import requests

    url = base_url.rstrip("/")
    url = (url if url.endswith("/v1") else url + "/v1") + "/videos/sync"
    r = requests.post(url, data=form, headers={"Accept": "video/mp4"}, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"{r.status_code}: {r.text[:500]}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(r.content)
    return out_path


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--depth", help="depth control video (server-visible path)")
    p.add_argument("--seg", help="segmentation control video (server-visible path)")
    p.add_argument("--material", default="sand", choices=sorted(MATERIAL_TEXT))
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=sorted(VARIANTS))
    p.add_argument("--clip-id", default="clip")
    p.add_argument("--cosmos-root", type=Path, default=None, help="Cosmos repo, for the cookbook negative prompt")
    p.add_argument("--resolution", default="720", choices=["480", "720"])
    p.add_argument("--guardrails", action="store_true")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--dry-run", action="store_true", help="write request JSON only")
    p.add_argument("--out", type=Path, default=Path("transfer_requests"))
    args = p.parse_args(argv)

    controls = {k: v for k, v in (("depth", args.depth), ("seg", args.seg)) if v}
    if not controls:
        p.error("give at least one of --depth/--seg")
    negative = default_negative_prompt(args.cosmos_root)
    args.out.mkdir(parents=True, exist_ok=True)
    for variant in args.variants:
        prompt = build_prompt(variant, args.material)
        stem = f"{args.clip_id}_{variant}"
        form = build_vllm_omni_form(controls, prompt, negative, guardrails=args.guardrails, resolution=args.resolution)
        bundle = {
            "clip_id": args.clip_id,
            "variant": variant,
            "material": args.material,
            "vllm_omni_form": form,
            "diffusers_kwargs": build_diffusers_kwargs(controls, prompt, negative, resolution=args.resolution),
            "framework_spec": build_framework_spec(
                f"transfer_{stem}",
                controls,
                f"{stem}_prompt.json",
                "negative_prompt.json",
                weights={k: 1.0 / len(controls) for k in controls} if len(controls) > 1 else None,
            ),
        }
        (args.out / f"{stem}_request.json").write_text(json.dumps(bundle, indent=2) + "\n")
        (args.out / f"{stem}_prompt.json").write_text(json.dumps(prompt, indent=2) + "\n")
        print(f"[transfer] wrote {args.out / (stem + '_request.json')}")
        if not args.dry_run:
            out = post_vllm_omni(args.base_url, form, args.out / f"{stem}.mp4")
            print(f"[transfer] saved {out}")
    (args.out / "negative_prompt.json").write_text(json.dumps(negative, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

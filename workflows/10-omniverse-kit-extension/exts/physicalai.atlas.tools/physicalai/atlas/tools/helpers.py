"""Pure-Python helpers for the Atlas Tools extension: no omni, no pxr. Executed here (tests/test_helpers.py).

Kept apart from ``extension.py`` so the logic behind each button can be tested without Kit, and so a syntax error in
UI code (which makes the extension window vanish, Introduction 03 and Training Series 005) cannot hide a logic bug.
"""

from __future__ import annotations

import os
import random
import re
from collections.abc import Iterable, Mapping, Sequence

DEFAULT_SEED = 2026
ROBOT_USD_ENV = "ATLAS_ROBOT_USD"
# Variant-set names that usually hold looks; the first one present wins when the user leaves the field empty.
LOOK_SET_PRIORITY = ("material", "materials", "look", "looks", "color", "colour", "paint", "leather")

# Lighting ranges for "Randomize lighting" (stage units: metres for SimReady assets).
INTENSITY_RANGE = (20_000.0, 60_000.0)
TEMPERATURE_RANGE_K = (3200.0, 7500.0)
POSITION_MIN = (-4.0, -4.0, 2.5)
POSITION_MAX = (4.0, 4.0, 5.0)

_IDENT = re.compile(r"[^A-Za-z0-9_]")


def default_robot_usd(env: Mapping[str, str] | None = None) -> str:
    """Robot USD for the "Load robot" field: ``$ATLAS_ROBOT_USD`` if set, else an empty string.

    Point it at the SimReady robot asset from workflow folder 01 (``01-openusd-simready``) or any robot USD/URL.
    """
    env = os.environ if env is None else env
    return env.get(ROBOT_USD_ENV, "")


def prim_name_from_asset(asset_path: str, fallback: str = "Robot") -> str:
    """A valid USD prim name from a file name or URL: ``.../franka_panda.usd`` -> ``franka_panda``."""
    stem = re.split(r"[\\/]", asset_path.rstrip("/\\"))[-1]
    stem = stem.split("?", 1)[0]
    stem = stem.rsplit(".", 1)[0] if "." in stem else stem
    name = _IDENT.sub("_", stem).strip("_")
    if not name:
        return fallback
    return name if not name[0].isdigit() else f"_{name}"


def unique_prim_path(existing: Iterable[str], base: str) -> str:
    """``base`` if unused, else ``base_01``, ``base_02``, ... (clicking twice must not stack two robots on one path)."""
    taken = set(existing)
    if base not in taken:
        return base
    i = 1
    while f"{base}_{i:02d}" in taken:
        i += 1
    return f"{base}_{i:02d}"


def choose_variant_set(variant_sets: Mapping[str, Sequence[str]], preferred: str = "") -> str | None:
    """Which variant set a button acts on: the user's choice if present, else the first look-like set, else the
    alphabetically first set. ``None`` when the prim has no variant sets with at least one variant."""
    usable = {k: v for k, v in variant_sets.items() if v}
    if not usable:
        return None
    if preferred:
        return preferred if preferred in usable else None
    lowered = {k.lower(): k for k in usable}
    for cand in LOOK_SET_PRIORITY:
        if cand in lowered:
            return lowered[cand]
    return sorted(usable)[0]


def next_variant(names: Sequence[str], current: str | None) -> str:
    """The variant after ``current`` in authored order, wrapping; the first one if ``current`` is unset or unknown."""
    if not names:
        raise ValueError("variant set has no variants")
    if current not in names:
        return names[0]
    return names[(list(names).index(current) + 1) % len(names)]


def random_variant(names: Sequence[str], current: str | None, rng: random.Random) -> str:
    """A random variant that differs from ``current`` whenever there are two or more (a button press should change
    something)."""
    if not names:
        raise ValueError("variant set has no variants")
    pool = [n for n in names if n != current] or list(names)
    return rng.choice(pool)


VARIANT_EVENT = "atlas_switch_variant"  # message-bus event a streamed web page sends (see streaming.md)


def parse_variant_event(payload: Mapping[str, object]) -> tuple[str, str]:
    """Validate a ``atlas_switch_variant`` payload from the browser: returns (mode, variant set name).

    Accepts ``{"mode": "next" | "random", "variant_set": "<name>"}``; both keys optional. Basic Training 08's React
    button failed silently because event and attribute names did not line up, so bad payloads raise with a message.
    """
    mode = str(payload.get("mode", "next"))
    if mode not in {"next", "random"}:
        raise ValueError(f"mode must be 'next' or 'random', got {mode!r}")
    variant_set = payload.get("variant_set", "")
    if not isinstance(variant_set, str):
        raise ValueError("variant_set must be a string")
    return mode, variant_set.strip()


def sample_light_params(rng: random.Random, count: int = 3) -> list[dict]:
    """``count`` light settings: intensity, colour temperature (K) and position, uniform in the ranges above."""
    if count < 1:
        raise ValueError("count must be >= 1")
    out = []
    for _ in range(count):
        out.append(
            {
                "intensity": rng.uniform(*INTENSITY_RANGE),
                "temperature": rng.uniform(*TEMPERATURE_RANGE_K),
                "position": tuple(rng.uniform(lo, hi) for lo, hi in zip(POSITION_MIN, POSITION_MAX, strict=True)),
            }
        )
    return out

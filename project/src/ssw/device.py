"""Device selection shared by the bake-off and the terrain sweep.

The project targets an NVIDIA GPU (Warp ``cuda:0``). ``--device auto`` (the default) picks
``cuda:0`` when Warp sees a CUDA device and falls back to ``cpu`` with a warning, so the same
command runs a GPU job on a GPU box and a smoke test on a laptop. An explicit ``cuda`` request
on a machine without CUDA fails loudly instead of silently measuring the CPU.
"""

from __future__ import annotations

import warnings

AUTO = "auto"


def resolve_device(spec: str = AUTO) -> str:
    """Return a concrete Warp device string for ``spec`` (``auto``, ``cpu``, ``cuda``, ``cuda:N``)."""
    import warp as wp

    if spec == AUTO:
        if wp.get_cuda_device_count() > 0:
            return "cuda:0"
        warnings.warn(
            "No CUDA device found; falling back to the Warp CPU device. Results are a smoke test, not GPU numbers.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "cpu"
    if spec.startswith("cuda") and wp.get_cuda_device_count() == 0:
        raise SystemExit(f"--device {spec} requested but Warp found no CUDA device. Use --device cpu or auto.")
    return spec


def can_capture_graph(device) -> bool:
    """CUDA-graph capture needs a CUDA device with the stream-ordered mempool enabled."""
    return bool(device.is_cuda and device.is_mempool_enabled)


def describe(device) -> dict:
    """Small, JSON-safe description of the device for run metadata."""
    info = {"device": str(device), "is_cuda": bool(device.is_cuda)}
    if device.is_cuda:
        info.update(
            name=device.name,
            arch=f"sm_{device.arch}",
            total_memory_gb=round(device.total_memory / 2**30, 2),
            mempool=bool(device.is_mempool_enabled),
        )
    return info

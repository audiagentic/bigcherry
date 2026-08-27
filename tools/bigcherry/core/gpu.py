"""HI130: real GPU VRAM queries and a fail-closed preflight check.

A real single-GPU R9700 27B tune run this session OOM'd because the tune
binary inherited the model's own (huge) default context, leaving no VRAM
headroom for the tuner's per-candidate timing workspace. The fix is not to
catch the crash and silently retry with a smaller context -- that changes
the measured workload without the operator ever seeing it happen. Instead,
check free VRAM against a named runtime profile's declared requirement
BEFORE launching anything, and refuse with a clear, actionable error if it
would not fit.
"""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import config as campaign_config


class GpuError(RuntimeError):
    pass


class GpuQueryError(GpuError):
    pass


class GpuPreflightError(GpuError):
    pass


def free_vram_bytes(device_indices: tuple[int, ...]) -> dict[int, int]:
    """Real free VRAM (total - used) per requested device index, via
    ``rocm-smi --showmeminfo vram --json``. Never returns a fabricated
    value for a device it could not query -- raises instead.
    """
    try:
        completed = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise GpuQueryError(f"rocm-smi query failed: {exc}") from exc
    try:
        raw = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise GpuQueryError(f"rocm-smi produced unparseable output: {exc}") from exc

    result: dict[int, int] = {}
    for index in device_indices:
        key = f"card{index}"
        card = raw.get(key)
        if not isinstance(card, dict):
            raise GpuQueryError(f"rocm-smi output has no entry for {key!r}: {raw!r}")
        try:
            total = int(card["VRAM Total Memory (B)"])
            used = int(card["VRAM Total Used Memory (B)"])
        except (KeyError, TypeError, ValueError) as exc:
            raise GpuQueryError(
                f"rocm-smi output for {key!r} is missing/malformed VRAM fields: {card!r}"
            ) from exc
        result[index] = total - used
    return result


def preflight_context(
    *,
    profile: "campaign_config.RuntimeProfile",
    devices: tuple[int, ...],
    stage: str,
) -> None:
    """Raise :class:`GpuPreflightError` if any requested device does not
    have ``profile.min_free_vram_bytes_per_device`` free right now.

    A tensor-split run needs headroom on EVERY participating device, not
    just one -- checked per-device, not summed. Never auto-shrinks the
    request; the operator must pick a different --runtime-profile or free
    VRAM themselves.
    """
    free = free_vram_bytes(devices)
    failures = [
        f"GPU {index}: {free[index] / (1 << 30):.1f}GiB free, need >= "
        f"{profile.min_free_vram_bytes_per_device / (1 << 30):.1f}GiB for "
        f"runtime-profile {profile.name!r}'s {stage} stage"
        for index in devices
        if free[index] < profile.min_free_vram_bytes_per_device
    ]
    if failures:
        raise GpuPreflightError("; ".join(failures))

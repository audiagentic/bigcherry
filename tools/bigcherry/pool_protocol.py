"""Offline validation for the HI54 pool-cache measurement protocol.

The CUDA implementation deliberately keeps pool-cache isolation local to
workspace evidence measurements.  This module validates event traces emitted
by tooling or tests without requiring a HIP device.  It is intentionally
small: it checks ordering and stage ownership, not allocator performance.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


class PoolProtocolError(ValueError):
    """Raised when a pool-cache measurement trace violates HI54."""


_ISOLATED = "isolated_workspace"
_INTERLEAVED = {"final", "confirmation"}
_EVENTS = {
    "clear_cache",
    "warmup_begin",
    "warmup_complete",
    "synchronize",
    "rebase_peak",
    "timed_sample_begin",
    "timed_sample_end",
}


def validate_pool_protocol(events: Iterable[Mapping[str, Any]]) -> None:
    """Validate one candidate's pool-cache protocol trace.

    Each event must provide ``stage`` and ``event``.  Isolated workspace
    traces require ``clear_cache -> warmup -> synchronize -> rebase_peak``;
    the rebase must precede timed samples.  Final and confirmation traces are
    intentionally interleaved and may not clear or rebase the shared pool.
    """

    rows = list(events)
    if not rows:
        raise PoolProtocolError("pool protocol trace is empty")

    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise PoolProtocolError(f"event {index} is not an object")
        stage = row.get("stage")
        event = row.get("event")
        if not isinstance(stage, str) or not stage:
            raise PoolProtocolError(f"event {index} has no stage")
        if event not in _EVENTS:
            raise PoolProtocolError(f"event {index} has unknown event")
        if stage not in {_ISOLATED, *_INTERLEAVED}:
            raise PoolProtocolError(f"event {index} has unknown pool protocol stage")
        if stage in _INTERLEAVED and event in {"clear_cache", "rebase_peak"}:
            raise PoolProtocolError(
                f"{stage} stage may not {event.replace('_', ' ')}"
            )

    stages = {row["stage"] for row in rows}
    if _ISOLATED not in stages:
        # A trace containing only interleaved timing is valid, provided the
        # forbidden operations were rejected above.
        return
    if stages - {_ISOLATED} - _INTERLEAVED:
        raise PoolProtocolError("unknown pool protocol stage")

    isolated = [row["event"] for row in rows if row["stage"] == _ISOLATED]
    required = [
        "clear_cache",
        "warmup_begin",
        "warmup_complete",
        "synchronize",
        "rebase_peak",
    ]
    positions: dict[str, int] = {}
    for event in required:
        if isolated.count(event) != 1:
            raise PoolProtocolError(
                f"isolated workspace trace must contain exactly one {event}"
            )
        try:
            positions[event] = isolated.index(event)
        except ValueError as exc:
            raise PoolProtocolError(
                f"isolated workspace trace missing {event}"
            ) from exc
    if any(positions[left] >= positions[right]
           for left, right in zip(required, required[1:])):
        raise PoolProtocolError("isolated workspace events are out of order")

    rebase = positions["rebase_peak"]
    if any(event == "timed_sample_begin" for event in isolated[:rebase + 1]):
        raise PoolProtocolError("timed sample begins before pool rebase")
    timed = isolated[rebase + 1:]
    if not timed:
        raise PoolProtocolError(
            "isolated workspace trace has no timed samples after pool rebase"
        )
    if any(event in required for event in timed):
        raise PoolProtocolError(
            "pool lifecycle event occurs after isolated workspace rebase"
        )
    timing_open = False
    for event in timed:
        if event == "timed_sample_begin":
            if timing_open:
                raise PoolProtocolError("timed samples overlap")
            timing_open = True
        elif event == "timed_sample_end":
            if not timing_open:
                raise PoolProtocolError("timed sample ends without a begin")
            timing_open = False
        else:
            raise PoolProtocolError(
                f"unexpected {event} event after isolated workspace rebase"
            )
    if timing_open:
        raise PoolProtocolError("timed sample begins without an end")

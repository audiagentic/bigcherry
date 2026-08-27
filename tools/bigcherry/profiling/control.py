"""PROF01/HI132: unprofiled control blocks.

Real HTTP requests against a real, already-running server (no profiler
attached) -- measures wall-clock tokens/sec directly rather than trusting
any one field name in the server's own /completion timings payload, so
this stays correct even if that payload's shape changes upstream.
"""

from __future__ import annotations

import statistics
import time

from .schema import ControlBlock
from ..tuning.server_runner import ServerRunner


def run_control_block(
    *, runner: ServerRunner, label: str, reps: int, prompt: str, n_predict: int,
) -> ControlBlock:
    tps_values: list[float] = []
    for _ in range(reps):
        started = time.monotonic()
        runner.run_completion(prompt, n_predict=n_predict)
        elapsed = time.monotonic() - started
        if elapsed > 0:
            tps_values.append(n_predict / elapsed)
    mean = statistics.fmean(tps_values) if tps_values else 0.0
    stddev = statistics.stdev(tps_values) if len(tps_values) > 1 else 0.0
    return ControlBlock(
        label=label, reps=reps, tg_tps_values=tuple(tps_values),
        tg_tps_mean=mean, tg_tps_stddev=stddev,
    )

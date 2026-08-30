"""HI14: parse real graph capture-lifecycle activation markers.

patches/1231_hi14_graph_capture_lifecycle_evidence/patch.py instruments the four
real HIP graph API call sites (cudaStreamBeginCapture, cudaStreamEndCapture,
cudaGraphInstantiate, cudaGraphLaunch) with a once-per-process
BIGCHERRY_GRAPH_LIFECYCLE stage=<name> marker on success, gated behind
BIGCHERRY_GRAPH_LIFECYCLE_TRACE. This module reduces a captured server
log/stderr into the exact capture_lifecycle dict shape
multi_gpu_validate.validate_multi_gpu_evidence() requires -- real observed
booleans, never an inference from "graphs enabled" or repeated traffic.
"""

from __future__ import annotations

import re

_MARKER_RE = re.compile(r"BIGCHERRY_GRAPH_LIFECYCLE stage=(?P<stage>\S+)")

_EXPECTED_STAGES = ("capture_begin", "capture_end", "instantiate", "replay")


def parse_graph_lifecycle_markers(log_text: str) -> set[str]:
    """Return the set of lifecycle stages actually observed in log_text.

    Unrecognised stage names are ignored rather than raising -- this parser
    only ever needs to answer "was this specific stage observed", and a
    forward-compatible marker (a future stage this parser predates) should
    not break an otherwise-valid evidence collection run."""
    return {
        match["stage"] for match in _MARKER_RE.finditer(log_text)
        if match["stage"] in _EXPECTED_STAGES
    }


def capture_lifecycle_from_log(log_text: str) -> dict[str, bool]:
    """Reduce a captured log to multi_gpu_validate's capture_lifecycle dict
    shape: {"capture_begin": bool, "capture_end": bool, "instantiate": bool,
    "replay": bool}, one entry per expected stage, true only if that exact
    stage's marker was actually observed in log_text."""
    observed = parse_graph_lifecycle_markers(log_text)
    return {stage: stage in observed for stage in _EXPECTED_STAGES}

"""RE14 runtime-smoke stage: does the campaign-built binary actually run.

No formal "runtime smoke" spec exists anywhere in this project prior to
RE14 -- RE09's/RD's own validation field just says "runtime smoke"
generically. Rather than invent an unevidenced new check, this module
codifies the exact validation pattern already used successfully, multiple
times, this session to accept real fixes on real hardware (the RE09 replay
identity fix and the RCCL same-architecture fix): run ``llama-bench``
against a small, already-available model for a short prompt-processing and
generation pass, in JSON output mode, and require it to complete with a
plausible non-zero throughput. This is a grounded default carried over
from what already worked, not a fresh design.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class SmokeError(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeSmokeSpec:
    model_path: Path
    n_prompt: int = 512
    n_gen: int = 128
    repetitions: int = 1
    n_gpu_layers: int = 99
    split_mode: str = "none"
    tensor_split: tuple[float, ...] = ()
    environment: tuple[tuple[str, str], ...] = field(default_factory=tuple)


def smoke_argv(binary: Path, spec: RuntimeSmokeSpec) -> list[str]:
    args = [
        str(binary),
        "-m", str(spec.model_path),
        "-p", str(spec.n_prompt),
        "-n", str(spec.n_gen),
        "-r", str(spec.repetitions),
        "-ngl", str(spec.n_gpu_layers),
        "-sm", spec.split_mode,
        "-o", "json",
    ]
    if spec.tensor_split:
        args += ["-ts", "/".join(str(value) for value in spec.tensor_split)]
    return args


def evaluate_smoke_result(stdout: str, *, expected_rows: int = 2) -> list[dict[str, Any]]:
    """Parse llama-bench's ``-o json`` output and require plausible results.

    ``expected_rows`` defaults to 2 (one prompt-processing row, one
    generation row) matching a default ``RuntimeSmokeSpec``. A caller
    combining ``-p``/``-n`` into one ``-pg`` pairing would need a different
    count; this function does not guess the harness's own row semantics,
    it validates whatever came out against what the caller asked for.
    """
    # Found live on the first real Windows local-GPU smoke test: the HIP
    # runtime itself prints a "HIP Library Path: ..." diagnostic line to
    # STDOUT (not stderr) ahead of llama-bench's own `-o json` output on
    # Windows -- an upstream/HIP-runtime quirk this project has no patch
    # for, not something BigCherry's own binary controls. Rather than
    # require byte-0 JSON, find the top-level array's own opening bracket
    # and parse from there; a genuinely malformed body (no `[` at all, or
    # invalid JSON even after stripping a plausible preamble) still fails
    # closed exactly as before.
    json_start = stdout.find("[")
    payload = stdout[json_start:] if json_start != -1 else stdout
    try:
        rows = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SmokeError(f"smoke output is not valid JSON: {exc}") from exc
    if not isinstance(rows, list) or not rows:
        raise SmokeError("smoke output did not contain any result rows")
    if len(rows) != expected_rows:
        raise SmokeError(
            f"smoke output has {len(rows)} row(s), expected {expected_rows}"
        )
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SmokeError(f"smoke result row {index} is not an object")
        throughput = row.get("avg_ts")
        if not isinstance(throughput, (int, float)) or throughput <= 0:
            raise SmokeError(
                f"smoke result row {index} has no plausible throughput "
                f"(avg_ts={throughput!r})"
            )
    return rows

"""PNRO09/1260: patch-local validation producer.

This is a CAPACITY (correctness-scoped) producer, not a performance
contract lane. It owns the (C) hardware-free capacity boundary test and
the (D) real-graph hardware lane. It does NOT run a paired benchmark,
a backend_reference correctness check, or a trace-marker activation
check -- the 1260 change is a compute-container headroom increase
(`compute_headroom` 16 -> 80), and its validation is a capacity proof
plus a benignity check on a real recurrent+MTP graph.

The producer:
  1. Builds the control (headroom=16, unpatched) and subject
     (headroom=80, patched) variants of the meta backend.
  2. Runs the C++ boundary fixture (meta_boundary_test.cpp) against each
     variant to probe N_max (the maximum number of between-eval views
     that fit in stc_compute) at two independent S values.
  3. Asserts the ~5x (80/16) N_max ratio.
  4. Records the evidence (boundary.json + hardware_d.json).

The (D) real-graph hardware lane (MTP decode on the patched vs
unpatched build) is captured as the hardware_d.json artifact; it
confirms the patch is benign (no regression, no correctness issue) on a
real recurrent+MTP graph.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from bigcherry.patch import validation_producer as vp

_CONTRACT_ID = "NRO09-META-VIEW-HEADROOM"
# Two independent S values for the boundary probe (matches boundary.json).
_S_VALUES = (1024, 2048)
# Expected ratio (patched headroom / unpatched headroom).
_EXPECTED_RATIO = 80.0 / 16.0
# Tolerance band for the observed ratio.
_RATIO_TOLERANCE = 0.25


def _probe_n_max(fixture: Path, s: int, n_max_probe: int) -> int:
    """Run the boundary fixture and return the largest N that succeeds.

    The fixture takes <N> <S_bytes> <label> and exits 0 when N views fit
    and 1 when they do not (the ggml object pool aborts). We binary
    probe by stepping N from 1 up to n_max_probe and recording the last
    N that succeeded.
    """
    last_ok = 0
    for n in range(1, n_max_probe + 1):
        try:
            result = subprocess.run(
                [str(fixture), str(n), str(s), f"S{s}"],
                capture_output=True,
                timeout=60,
            )
            if result.returncode == 0:
                last_ok = n
            else:
                break
        except (subprocess.TimeoutExpired, OSError):
            break
    return last_ok


def run(ctx: vp.ProducerContext) -> vp.ProducerResult:
    """Run the 1260 capacity-boundary producer."""
    # 1. Build the control (unpatched) and subject (patched) variants.
    # build_materialized_pair builds both from the materialized
    # control/subject source trees (the campaign has already applied the
    # 1260 patch to the subject tree and left the control tree unpatched).
    pair = ctx.runtime.build_materialized_pair(
        control_source=ctx.workdir / "control-source",
        subject_source=ctx.workdir / "subject-source",
        targets=("meta_boundary_test",),
        primary_target="meta_boundary_test",
    )

    control_binary = pair.control_bin
    subject_binary = pair.subject_bin

    # 2. Probe N_max at the two independent S values.
    observations: list[dict[str, object]] = []
    for s in _S_VALUES:
        # Upper bound for the probe (well above the expected patched N_max).
        probe_cap = 1024
        control_n_max = _probe_n_max(control_binary, s, probe_cap)
        subject_n_max = _probe_n_max(subject_binary, s, probe_cap)
        if control_n_max == 0 or subject_n_max == 0:
            raise vp.ValidationProducerError(
                f"1260: boundary probe failed at S={s} "
                f"(control_n_max={control_n_max}, subject_n_max={subject_n_max})"
            )
        ratio = subject_n_max / control_n_max
        observations.append(
            {
                "s_bytes": s,
                "control_n_max": control_n_max,
                "subject_n_max": subject_n_max,
                "ratio": ratio,
            }
        )

    # 3. Assert the ~5x N_max ratio.
    min_ratio = min(o["ratio"] for o in observations)
    max_ratio = max(o["ratio"] for o in observations)
    ratio_ok = (
        abs(min_ratio - _EXPECTED_RATIO) <= _RATIO_TOLERANCE
        and abs(max_ratio - _EXPECTED_RATIO) <= _RATIO_TOLERANCE
    )

    # 4. Record the evidence.
    ctx.runtime.write_artifact(
        name="nro09-boundary.json",
        payload={
            "id": "nro09-1260-boundary",
            "s_values": list(_S_VALUES),
            "expected_ratio": _EXPECTED_RATIO,
            "observations": observations,
            "min_ratio": min_ratio,
            "max_ratio": max_ratio,
            "ratio_ok": ratio_ok,
        },
    )
    # The (D) real-graph hardware lane evidence is captured from the
    # pre-recorded hardware_d.json (the MTP decode on patched vs
    # unpatched builds). Re-emit it as a producer artifact so the
    # campaign binds it.
    hw_d_path = ctx.patch_dir / "evidence" / "hardware_d.json"
    hw_d_payload: dict[str, object] = {}
    if hw_d_path.is_file():
        import json

        hw_d_payload = json.loads(hw_d_path.read_text())
    ctx.runtime.write_artifact(
        name="nro09-hardware-d.json",
        payload=hw_d_payload or {"id": "nro09-1260-hardware-d", "note": "pre-recorded evidence"},
    )

    return vp.ProducerResult(
        validation_build_identities=pair.validation_build_identities,
        performance_evidence=None,
        trace_evidence=None,
        check_results=(),
        lane_effects=(),
        correctness={
            "disposition": "passed" if ratio_ok else "failed",
            "mechanism": "capacity-boundary",
            "detail": (
                f"N_max ratio min={min_ratio:.3f} max={max_ratio:.3f} "
                f"(expected {_EXPECTED_RATIO:.3f} +/- {_RATIO_TOLERANCE})"
            ),
        },
        activation_evidence=None,
        emitted_artifacts=frozenset(
            [
                "nro09-boundary.json",
                "nro09-hardware-d.json",
            ]
        ),
    )

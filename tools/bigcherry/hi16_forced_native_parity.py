"""HI16: forced-native bitwise-parity check, driven against a real
test-backend-ops binary.

HI24's double-native twin and J_best-forced-candidate canary (already
shipped, hardware-validated 2026-08-23) already perform and record exactly
the check this item's acceptance criteria ask for: a candidate configured
to equal native's own resolved parameters must produce BITWISE-IDENTICAL
output to native, per type and shape class -- "forced J == native J_best ->
bitwise-identical output". Every real GGML_HIP_TUNE_DOUBLE_NATIVE=1 tuning
run already writes the evidence (nmse/max_abs per candidate, including the
"<name>#twin" and J_best-forced rows) into its measurements JSONL; this
turns that into a repeatable, asserted check instead of an ad hoc read of
the file.

Deliberately reuses the existing dispatch/tuner machinery rather than new
C++ test infrastructure: the check already runs in production code on
every tuning sweep, this only exercises and asserts on it deliberately per
representative signature.

On Windows/WDDM, set HIP_LAUNCH_BLOCKING=1 (HI64's own documented mitigation
for a transient single-sample timing flake unrelated to correctness) --
observed directly (2026-08-23) on the local RX 7900 GRE: an mmvq case's twin
came back status=launch_failed without it, and passed cleanly with it, on
the identical case. This script's status=='ok' check is deliberately strict
and will surface that flake rather than silently skip it; retry under
HIP_LAUNCH_BLOCKING=1 before treating a launch_failed result as a real
finding.

Usage:
    python -m bigcherry.hi16_forced_native_parity \\
        --binary /c/bcw/bin/test-backend-ops.exe \\
        --case 'type_a=f32,type_b=f32,m=16,n=1,k=256,bs=\\[1,1\\],nr=\\[1,1\\]' \\
        --case 'type_a=q4_0,type_b=f32,m=16,n=16,k=256,bs=\\[1,1\\],nr=\\[1,1\\]'
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_TWIN_SUFFIX = "#twin"


class ParityError(RuntimeError):
    pass


def parity_rows(candidates: list[dict[str, Any]], *, canary_pair: str) -> list[dict[str, Any]]:
    """The rows this check is actually about: the double-native twin (always,
    when present) and the J_best-forced candidate this row's own canary_pair
    named (when the canary compared against a real registry candidate rather
    than the twin). Anything else in `candidates` is an ordinary challenger
    and out of scope for a parity check -- it is allowed to disagree with
    native; only a row claiming to reproduce native's own resolved value is
    required to match exactly."""
    rows = [c for c in candidates if c.get("name", "").endswith(_TWIN_SUFFIX)]
    if canary_pair and not canary_pair.endswith(_TWIN_SUFFIX):
        rows += [c for c in candidates if c.get("name") == canary_pair]
    return rows


def check_result_row(row: dict[str, Any]) -> list[str]:
    """Returns a list of failure descriptions (empty = parity holds) for one
    measurements JSONL result row. Fails closed: a row with NO parity
    candidate at all (no twin, no named J_best pair) is itself a failure --
    a signature this check was asked to cover that produced no evidence is
    not a pass, it is a gap."""
    candidates = row.get("candidates", [])
    rows = parity_rows(candidates, canary_pair=row.get("canary_pair", ""))
    if not rows:
        return [
            f"dispatch {row.get('dispatch', '?')}: no double-native twin or "
            "J_best-forced candidate found -- nothing to check parity against "
            "(GGML_HIP_TUNE_DOUBLE_NATIVE may be disabled, or this signature's "
            "native has no same-kernel forced-variant pair)"
        ]
    failures = []
    for candidate in rows:
        if candidate.get("status") != "ok":
            failures.append(
                f"dispatch {row.get('dispatch', '?')}: {candidate['name']} "
                f"status={candidate.get('status')!r}, expected 'ok'"
            )
            continue
        nmse = candidate.get("nmse", 0.0)
        max_abs = candidate.get("max_abs", 0.0)
        if nmse != 0.0 or max_abs != 0.0:
            failures.append(
                f"dispatch {row.get('dispatch', '?')}: {candidate['name']} "
                f"nmse={nmse!r} max_abs={max_abs!r}, expected exactly 0.0 "
                "(bitwise parity) for a forced-native-equivalent candidate"
            )
    return failures


def check_measurements_file(path: Path) -> list[str]:
    """All parity failures across every result row in a measurements JSONL."""
    failures: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("kind") != "result":
            continue
        failures.extend(check_result_row(row))
    return failures


def run_case(
    binary: Path, case: str, *, dispatch_db: Path, screen_samples: int, final_samples: int,
    runner=subprocess.run,
) -> Path:
    """Run one real test-backend-ops sweep in tune mode with double-native
    enabled, returning the measurements JSONL path it wrote."""
    env = os.environ.copy()
    env.update({
        "GGML_HIP_DISPATCH_MODE": "tune",
        "GGML_HIP_DISPATCH_DB": str(dispatch_db),
        "GGML_HIP_TUNE_DOUBLE_NATIVE": "1",
        "GGML_HIP_TUNE_SCREEN_SAMPLES": str(screen_samples),
        "GGML_HIP_TUNE_FINAL_SAMPLES": str(final_samples),
    })
    result = runner(
        [str(binary), "-o", "MUL_MAT", "-p", case],
        capture_output=True, text=True, env=env,
    )
    measurements = Path(f"{dispatch_db}.measurements.jsonl")
    if result.returncode != 0 or not measurements.is_file():
        raise ParityError(
            f"case {case!r} failed to produce a measurements file "
            f"(exit {result.returncode}): {result.stderr[-2000:]}"
        )
    return measurements


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True, help="test-backend-ops binary")
    parser.add_argument(
        "--case", action="append", required=True, dest="cases",
        help="test-backend-ops -p filter -- this is matched as a REGEX "
        "(std::regex_search against the test case's vars() string), same as "
        "test-backend-ops itself and signature_correctness_mapping.py's "
        "op_filter, so literal brackets in a shape like bs=[1,1] must be "
        "escaped as bs=\\[1,1\\] or they are parsed as a regex character "
        "class and silently match something else (or nothing). Repeatable, "
        "one per representative family/type/shape class.",
    )
    parser.add_argument("--screen-samples", type=int, default=3)
    parser.add_argument("--final-samples", type=int, default=5)
    args = parser.parse_args(argv)

    failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        for index, case in enumerate(args.cases):
            dispatch_db = Path(tmp) / f"case{index}.jsonl"
            try:
                measurements = run_case(
                    args.binary, case, dispatch_db=dispatch_db,
                    screen_samples=args.screen_samples, final_samples=args.final_samples,
                )
                failures = check_measurements_file(measurements)
            except ParityError as exc:
                failures = [str(exc)]
            if failures:
                failed += 1
                print(f"FAIL  {case}", file=sys.stderr)
                for failure in failures:
                    print(f"      {failure}", file=sys.stderr)
            else:
                print(f"PASS  {case}", file=sys.stderr)

    if failed:
        print(f"hi16-forced-native-parity: {failed}/{len(args.cases)} case(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

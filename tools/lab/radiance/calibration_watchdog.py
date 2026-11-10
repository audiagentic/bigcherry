"""RHP19 calibration loop envelope and marker validation; no GPU access.

This is a launcher-side safety bound, not a replacement for helper convergence.
The final bc-serve calibration-ingest remains the strict manifest/key validator.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

# Match serving/live_policy.cpp::live_policy_record_complete. Optional effect
# quantities (including abort_quiescent_ns) cannot block production calibration.
REQUIRED = frozenset((
    "txn_p50_ns", "txn_p95_ns", "copy_bytes_per_s", "heat_gap_max_ns",
    "heat_watermarked", "hot_set_k", "marginal_error_ppm",
    "marginal_observations", "drift_half_life_observations",
    "observation_rate_per_s", "tail_share_gain_ppm",
    "tail_lane_service_ns_per_s", "tail_lane_row_ns", "lane_lost_ns",
))
MARKER = "RHP16_CALIBRATION_JSON "
DRIFT_MARKER = "RHP19_DRIFT_PROGRESS_JSON "


class CalibrationError(ValueError):
    pass


def _positive(policy: dict, key: str) -> int:
    value = policy.get(key)
    if type(value) is not int or value <= 0 or value >= 2**63:
        raise CalibrationError(f"{key} must be a positive policy-derived integer")
    return value


def measured_drift(log: str) -> dict | None:
    """Latest helper-measured drift basis, not guessed from policy cadence."""
    rows = [line[len(DRIFT_MARKER):] for line in log.splitlines()
            if line.startswith(DRIFT_MARKER)]
    if not rows:
        return None
    try:
        row = json.loads(rows[-1])
    except json.JSONDecodeError as e:
        raise CalibrationError(f"invalid helper drift progress: {e}") from e
    if not isinstance(row, dict):
        raise CalibrationError("helper drift progress is not an object")
    for key in ("capacity", "intervals", "horizon_observations",
                "observation_rate_per_s", "accepted_gap_ns", "accepted_heat_deltas"):
        value = row.get(key)
        if type(value) is not int or value < 0 or value >= 2**63:
            raise CalibrationError(f"invalid helper drift progress {key}")
    if not row["capacity"] or row["intervals"] > row["capacity"]:
        raise CalibrationError("invalid helper drift retention capacity/intervals")
    return row


def limits(policy: dict, drift: dict | None = None) -> tuple[int, int]:
    """Finite A/B/B/A envelope including the helper's measured drift basis.

    Before any accepted heat gap, a policy-only deadline applies. Once the
    helper measures its gap, its capacity and actual retained observation
    horizon extend the bound. The launcher locks this extension exactly once;
    a continuously coarsening drift ring cannot slide the deadline forever.
    """
    cadence = _positive(policy, "cadence_ns")
    age = _positive(policy, "max_heat_age_ns")
    residency = _positive(policy, "min_residency_ns")
    samples = _positive(policy, "min_samples")
    phase_ns = (samples + 1) * cadence + age + residency
    if drift is not None and drift["accepted_heat_deltas"] > 1 and drift["accepted_gap_ns"]:
        # CalibrationProgressTimeout::wait_span: measured accepted gap and
        # helper's retained ring capacity, not policy min_samples.
        drift_ns = drift["capacity"] * max(cadence, age, drift["accepted_gap_ns"])
        if drift["observation_rate_per_s"] and drift["horizon_observations"]:
            observed_ns = (drift["horizon_observations"] * 1_000_000_000 +
                           drift["observation_rate_per_s"] - 1) // drift["observation_rate_per_s"]
            drift_ns = max(drift_ns, observed_ns)
        phase_ns += drift_ns
    return math.ceil(4 * phase_ns / 1_000_000_000), math.ceil(max(cadence, age) / 1_000_000_000)


def record_in_log(log: str) -> dict | None:
    markers = [line[len(MARKER):] for line in log.splitlines() if line.startswith(MARKER)]
    if not markers:
        return None
    if len(markers) != 1:
        raise CalibrationError(f"expected exactly one complete calibration marker, found {len(markers)}")
    try:
        row = json.loads(markers[0])
    except json.JSONDecodeError as e:
        raise CalibrationError(f"malformed calibration marker: {e}") from e
    if not isinstance(row, dict) or not isinstance(row.get("measurements"), dict):
        raise CalibrationError("calibration record has no measurements object")
    missing = sorted(REQUIRED - row["measurements"].keys())
    if missing:
        absent = row.get("absent", {})
        reasons = ", ".join(f"{name}: {absent.get(name, 'not reported')}" for name in missing)
        raise CalibrationError(f"partial record is NOT production calibration: {reasons}")
    return row


def waiting_reason(log: str) -> str:
    lines = [line for line in log.splitlines() if "calibration waiting:" in line]
    if not lines:
        return "heat_watermarked: no measurable helper progress report yet"
    return lines[-1].split("calibration waiting:", 1)[1].strip()


def main(argv: list[str]) -> int:
    try:
        if (len(argv) not in (3, 4) or argv[1] not in {"limits", "marker", "waiting"}
                or (len(argv) == 4 and argv[1] != "limits")):
            raise CalibrationError("usage: calibration_watchdog.py limits POLICY.json [HELPER.log] | marker|waiting HELPER.log")
        if argv[1] == "limits":
            policy = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
            if not isinstance(policy, dict):
                raise CalibrationError("policy must be a JSON object")
            drift = measured_drift(Path(argv[3]).read_text(encoding="utf-8")) if len(argv) == 4 else None
            print(*limits(policy, drift))
        else:
            log = Path(argv[2]).read_text(encoding="utf-8")
            if argv[1] == "waiting":
                print(waiting_reason(log))
            else:
                row = record_in_log(log)
                if row is None:
                    return 10
                print("calibration: once-only complete planner record validated", file=sys.stderr)
        return 0
    except (CalibrationError, OSError, json.JSONDecodeError) as e:
        print(f"calibration: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

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


class CalibrationError(ValueError):
    pass


def _positive(policy: dict, key: str) -> int:
    value = policy.get(key)
    if type(value) is not int or value <= 0 or value >= 2**63:
        raise CalibrationError(f"{key} must be a positive policy-derived integer")
    return value


def limits(policy: dict) -> tuple[int, int]:
    """Four A,B,B,A phases, each allowed a full warmup, drift-age and residency.

    Everything comes from the explicitly selected policy/derived measurements;
    no guessed wall-clock timeout or request count is introduced.
    """
    cadence = _positive(policy, "cadence_ns")
    age = _positive(policy, "max_heat_age_ns")
    residency = _positive(policy, "min_residency_ns")
    samples = _positive(policy, "min_samples")
    phase_ns = (samples + 1) * cadence + age + residency
    return math.ceil(4 * phase_ns / 1_000_000_000), math.ceil(max(cadence, age) / 1_000_000_000)



def workload_limits(policy: dict, prompts_a: Path, prompts_b: Path,
                    request_timeout_s: int) -> tuple[int, int, int, int]:
    """Bound the full A,B,B,A workload, plus the policy warmup envelope.

    Each request has a transport deadline. A policy-only 240-second bound
    cannot fit 120 long-prefill requests, or the helper's drift retention.
    """
    if type(request_timeout_s) is not int or request_timeout_s <= 0:
        raise CalibrationError("request timeout must be positive")
    def count(path: Path) -> int:
        with path.open(encoding="utf-8") as handle:
            count = sum(bool(line.strip()) for line in handle)
        if not count:
            raise CalibrationError(f"empty calibration workload: {path}")
        return count
    base, report = limits(policy)
    request_budget = 2 * (count(prompts_a) + count(prompts_b)) * request_timeout_s
    return base + request_budget, report, base, request_budget


PROGRESS = "RHP19_CALIBRATION_PROGRESS "
PROGRESS_KEYS = frozenset(("accepted_heat_deltas", "drift_intervals",
    "drift_capacity", "drift_horizon_observations",
    "observation_rate_per_s", "heat_gap_max_ns"))


def drift_evidence(policy: dict, log: str) -> tuple[int, bool]:
    """Measured horizon from the helper's live drift tracker and heat gaps.

    Return (additional wall seconds, usable feed). Missing measurements
    cannot be replaced with configured cadence or an invented ring size.
    """
    lines = [line[len(PROGRESS):] for line in log.splitlines()
             if line.startswith(PROGRESS)]
    if not lines:
        return 0, False
    fields = lines[-1].split()
    if any(field.count("=") != 1 for field in fields):
        raise CalibrationError("malformed helper drift progress")
    raw = dict(field.split("=", 1) for field in fields)
    if len(raw) != len(fields) or set(raw) != PROGRESS_KEYS:
        raise CalibrationError("incomplete helper drift progress")
    try:
        values = {key: int(value) for key, value in raw.items()}
    except ValueError as e:
        raise CalibrationError("non-integer helper drift progress") from e
    if any(value < 0 or value >= 2**63 for value in values.values()):
        raise CalibrationError("invalid helper drift progress")
    if values["drift_intervals"] > values["drift_capacity"]:
        raise CalibrationError("helper drift intervals exceed capacity")
    if not (values["accepted_heat_deltas"] and values["drift_capacity"]
            and values["heat_gap_max_ns"] and values["observation_rate_per_s"]):
        return 0, False
    gap = max(values["heat_gap_max_ns"], _positive(policy, "cadence_ns"),
              _positive(policy, "max_heat_age_ns"))
    ring_ns = gap * values["drift_capacity"]
    observed_ns = (values["drift_horizon_observations"] * 1_000_000_000
                   + values["observation_rate_per_s"] - 1) // values["observation_rate_per_s"]
    return math.ceil(max(ring_ns, observed_ns) / 1_000_000_000), True


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
        if len(argv) < 3 or argv[1] not in {"limits", "drift", "marker", "waiting"}:
            raise CalibrationError("usage: calibration_watchdog.py limits POLICY [A B REQUEST_TIMEOUT_S] | drift POLICY HELPER.log | marker|waiting HELPER.log")
        if argv[1] == "limits":
            if len(argv) not in (3, 6):
                raise CalibrationError("limits needs POLICY and optionally A B REQUEST_TIMEOUT_S")
            policy = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
            if not isinstance(policy, dict):
                raise CalibrationError("policy must be a JSON object")
            if len(argv) == 3:
                print(*limits(policy))
            else:
                print(*workload_limits(policy, Path(argv[3]), Path(argv[4]), int(argv[5])))
        elif argv[1] == "drift":
            if len(argv) != 4:
                raise CalibrationError("drift needs POLICY HELPER.log")
            policy = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
            if not isinstance(policy, dict):
                raise CalibrationError("policy must be a JSON object")
            seconds, ready = drift_evidence(policy, Path(argv[3]).read_text(encoding="utf-8"))
            print(seconds, int(ready))
        else:
            if len(argv) != 3:
                raise CalibrationError("marker/waiting needs HELPER.log")
            log = Path(argv[2]).read_text(encoding="utf-8")
            if argv[1] == "waiting":
                print(waiting_reason(log))
            else:
                row = record_in_log(log)
                if row is None:
                    return 10
                print("calibration: once-only complete planner record validated", file=sys.stderr)
        return 0
    except (CalibrationError, OSError, json.JSONDecodeError, ValueError) as e:
        print(f"calibration: {e}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

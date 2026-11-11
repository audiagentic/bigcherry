"""Lease-gated S8 calibration measurement-run producer.

Executes real hardware runner arms and records the applied values and raw
observations. It does NOT infer hardware qualification from synthetic fixtures,
replay old v1 defaults, or choose a value without all matched gates passing.
"""
from __future__ import annotations

import argparse
import hashlib
import statistics
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from ..hardware import lease
from .calibration_rebind import read_native_scope
from .calibration_store import (
    KEYS,
    MAX_INT,
    CalibrationInvalid,
    _read,
    canonical,
    emit_from_run,
    validate_record,
)

_HANDOFF = frozenset(KEYS) - {"host_ledger_margin.mib"}
_TELEMETRY = frozenset(("handoff.histogram_period_s", "handoff.live_cadence_ns"))
_MIB = 1024 * 1024
_METRIC_KEYS = frozenset((
    "throughput_milli_tok_s", "tail_latency_us", "overflow_count",
    "histogram_write_us", "max_heat_age_ns",
))
_OBSERVATION = frozenset((
    "schema_version", "kind", "key", "candidate", "scope", "workload_digest",
    "applied_value", "output_correct", "health_ok", "thermal_ok",
    "effect_window_ok", "migration_stable", "metrics", "raw_evidence",
))
_HOST_OBSERVATION = frozenset((
    "schema_version", "kind", "key", "scope", "planning_key",
    "required_bytes", "available_bytes", "planned_reserve_bytes",
    "shortfall_bytes", "raw_evidence",
))
_GATES = frozenset(("min_throughput_milli_tok_s", "max_tail_latency_us",
                    "max_histogram_write_us", "max_heat_age_ns"))


def _int(value, name, *, positive=False) -> int:
    if type(value) is not int or value < (1 if positive else 0) or value > MAX_INT:
        raise CalibrationInvalid(f"S8 lab {name}: nonnegative I-JSON integer required")
    return value


def _digest(value, name) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise CalibrationInvalid(f"S8 lab {name}: SHA-256 digest required")
    return value


def _raw_evidence(out: Path, name: object) -> str:
    if not isinstance(name, str) or not name or "/" in name or name in (".", ".."):
        raise CalibrationInvalid("S8 lab observation must name an evidence file in the run directory")
    path = out / name
    if path.is_symlink() or not path.is_file() or not path.read_bytes():
        raise CalibrationInvalid("S8 lab raw hardware evidence missing or empty: " + str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_one(runner: tuple[str, ...], *, key: str, scope_file: Path, out: Path,
             arm: str, candidate: int | None, run: Callable) -> dict:
    output = out / (arm + ".json")
    argv = [*runner, "--s8-key", key, "--s8-scope", str(scope_file),
            "--s8-output", str(output)]
    if candidate is None:
        argv.append("--s8-measure-host-room")
    else:
        argv.extend(("--s8-value", str(candidate)))
    result = run(argv, check=False, text=True, capture_output=True)
    (out / (arm + ".stdout.txt")).write_text(result.stdout or "", encoding="utf-8")
    (out / (arm + ".stderr.txt")).write_text(result.stderr or "", encoding="utf-8")
    if result.returncode != 0 or not output.is_file():
        raise CalibrationInvalid(
            f"S8 lab {arm}: hardware runner failed (exit {result.returncode}); "
            "no calibration run file emitted"
        )
    record = _read(output)
    return record


def _gates(path: Path, key: str) -> dict:
    gates = _read(path)
    required = {"min_throughput_milli_tok_s", "max_tail_latency_us"}
    if key in _TELEMETRY:
        required.add("max_histogram_write_us")
    if key == "handoff.live_cadence_ns":
        required.add("max_heat_age_ns")
    if not isinstance(gates, dict) or set(gates) != required:
        raise CalibrationInvalid("S8 lab explicit metric gates missing or unexpected")
    for field, value in gates.items():
        _int(value, field, positive=True)
    return gates


def _validate_handoff(record: dict, *, key: str, candidate: int, scope: dict,
                      gates: dict, out: Path, workload: str | None) -> str:
    if not isinstance(record, dict) or set(record) != _OBSERVATION:
        raise CalibrationInvalid("S8 lab observation is not the closed hardware result")
    if (record["kind"] != "s8_hardware_observation" or
            type(record["schema_version"]) is not int or record["schema_version"] != 1 or
            record["key"] != key or type(record["candidate"]) is not int or
            record["candidate"] != candidate or type(record["applied_value"]) is not int or
            record["applied_value"] != candidate or record["scope"] != scope):
        raise CalibrationInvalid("S8 lab runner did not attest the exact selected hardware arm")
    _digest(record["workload_digest"], "workload")
    if workload is not None and record["workload_digest"] != workload:
        raise CalibrationInvalid("S8 lab changed workload between candidate arms")
    for name in ("output_correct", "health_ok", "thermal_ok",
                 "effect_window_ok", "migration_stable"):
        if type(record[name]) is not bool or not record[name]:
            raise CalibrationInvalid("S8 lab hardware gate failed: " + name)
    metrics = record["metrics"]
    if not isinstance(metrics, dict) or set(metrics) != _METRIC_KEYS:
        raise CalibrationInvalid("S8 lab hardware metrics incomplete")
    for name, value in metrics.items():
        _int(value, name, positive=name in ("throughput_milli_tok_s", "tail_latency_us"))
    if (metrics["overflow_count"] != 0 or
            metrics["throughput_milli_tok_s"] < gates["min_throughput_milli_tok_s"] or
            metrics["tail_latency_us"] > gates["max_tail_latency_us"]):
        raise CalibrationInvalid("S8 lab overflow/throughput/tail latency gate failed")
    if key in _TELEMETRY and metrics["histogram_write_us"] > gates["max_histogram_write_us"]:
        raise CalibrationInvalid("S8 lab histogram write-cost gate failed")
    if (key == "handoff.live_cadence_ns" and
            metrics["max_heat_age_ns"] > gates["max_heat_age_ns"]):
        raise CalibrationInvalid("S8 lab live heat freshness gate failed")
    return _raw_evidence(out, record["raw_evidence"])


def _validate_host(record: dict, *, scope: dict, out: Path) -> tuple[int, str]:
    if not isinstance(record, dict) or set(record) != _HOST_OBSERVATION:
        raise CalibrationInvalid("S8 lab host room observation incomplete")
    if (record["kind"] != "s8_host_admission_observation" or
            type(record["schema_version"]) is not int or record["schema_version"] != 1 or
            record["key"] != "host_ledger_margin.mib" or record["scope"] != scope):
        raise CalibrationInvalid("S8 lab host room does not match requested scope")
    _digest(record["planning_key"], "planning key")
    for field in ("required_bytes", "available_bytes",
                  "planned_reserve_bytes", "shortfall_bytes"):
        _int(record[field], field)
    if record["shortfall_bytes"] != max(0, record["required_bytes"] - record["available_bytes"]):
        raise CalibrationInvalid("S8 lab host room shortfall contradicts live admission")
    # Do not conflate VRAM refusal with a host RAM margin observation.
    return (record["shortfall_bytes"] + _MIB - 1) // _MIB, _raw_evidence(out, record["raw_evidence"])


def capture_s8_run(
    *, key: str, scope_file: Path, out: Path, runner: tuple[str, ...],
    valid_until_unix_s: int, reason: str, candidates: tuple[int, ...] = (),
    gates_file: Path | None = None, objective: str = "throughput",
    prior_margin_record: Path | None = None, now_unix_s: int | None = None,
    lease_ok: Callable[[], bool] = lease.held_by_this_process,
    run: Callable = subprocess.run,
) -> Path:
    """Execute ABBA hardware arms and emit one P2-accepted measured run file."""
    if not lease_ok():
        raise CalibrationInvalid("S8 lab hardware run refused: GPU lease not held by this process")
    if key not in KEYS:
        raise CalibrationInvalid("S8 lab unrecognized S8 calibration key")
    if not runner or any(not isinstance(arg, str) or not arg for arg in runner):
        raise CalibrationInvalid("S8 lab requires the actual hardware runner argv")
    if not isinstance(reason, str) or len(reason.strip()) < 12:
        raise CalibrationInvalid("S8 lab requires explicit method/qualification rationale")
    now = int(time.time()) if now_unix_s is None else _int(now_unix_s, "measurement time", positive=True)
    _int(valid_until_unix_s, "valid until", positive=True)
    if valid_until_unix_s <= now:
        raise CalibrationInvalid("S8 lab cannot emit an already expired run")
    scope = read_native_scope(scope_file)
    if objective not in ("throughput", "latency"):
        raise CalibrationInvalid("S8 lab objective must be throughput or latency")
    out = Path(out)
    if out.exists():
        raise CalibrationInvalid("S8 lab refuses to overwrite prior raw evidence")
    gate = None
    if key in _HANDOFF:
        if (len(candidates) < 2 or len(set(candidates)) != len(candidates) or
                any(type(c) is not int or c < 0 or c > MAX_INT for c in candidates)):
            raise CalibrationInvalid("S8 lab requires two or more distinct measured candidates")
        if key == "handoff.moe_down_tile":
            if any(c not in (0, 1, 2) for c in candidates):
                raise CalibrationInvalid("S8 lab down tile candidates must be v2 modes 0/1/2")
        elif any(c == 0 for c in candidates):
            raise CalibrationInvalid("S8 lab handoff/cadence candidates must be positive")
        if gates_file is None:
            raise CalibrationInvalid("S8 lab requires operator-defined throughput/latency gates")
        gate = _gates(gates_file, key)
    else:
        if candidates or gates_file is not None:
            raise CalibrationInvalid("S8 lab host margin is derived from measured room, not candidate guesses")
        if prior_margin_record is not None:
            prior = validate_record(_read(prior_margin_record), scope=scope, now_unix_s=now)
            if prior["key"] != "host_ledger_margin.mib":
                raise CalibrationInvalid("S8 lab previous record is not the matching host margin")
        else:
            prior = None
    out.mkdir(parents=True)
    runner_scope = out / "scope.json"
    runner_scope.write_bytes(canonical(scope) + b"\n")
    rows: list[dict] = []
    workload = None
    plan_key = None
    if key in _HANDOFF:
        # Four matched ABBA pairs (forward, reverse, reverse, forward).
        schedule = (list(candidates) + list(reversed(candidates)) +
                    list(reversed(candidates)) + list(candidates)) * 4
    else:
        # Multiple independent, real pre-weight admission measurements.
        schedule = [None] * 4
    for index, candidate in enumerate(schedule):
        if not lease_ok():
            raise CalibrationInvalid("S8 lab GPU lease lost during measurement")
        arm = f"arm-{index:03d}"
        observation = _run_one(
            runner, key=key, scope_file=runner_scope, out=out,
            arm=arm, candidate=candidate, run=run,
        )
        if key in _HANDOFF:
            raw_digest = _validate_handoff(
                observation, key=key, candidate=candidate,
                scope=scope, gates=gate, out=out, workload=workload,
            )
            workload = observation["workload_digest"]
        else:
            shortage, raw_digest = _validate_host(observation, scope=scope, out=out)
            if plan_key is not None and observation["planning_key"] != plan_key:
                raise CalibrationInvalid("S8 lab host sample changed planning key")
            plan_key = observation["planning_key"]
        row = {"arm": arm, "candidate": candidate, "observation": observation,
               "raw_evidence_sha256": raw_digest}
        if key == "host_ledger_margin.mib":
            row["observed_shortfall_mib"] = shortage
        rows.append(row)
    raw_path = out / "raw-observations.json"
    raw_path.write_bytes(canonical(rows) + b"\n")
    if key in _HANDOFF:
        medians = {}
        for candidate in candidates:
            results = [r["observation"]["metrics"] for r in rows if r["candidate"] == candidate]
            if len(results) != 16:
                raise CalibrationInvalid("S8 lab candidate coverage is not four ABBA pairs")
            medians[candidate] = (
                statistics.median_low(r["throughput_milli_tok_s"] for r in results),
                statistics.median_low(r["tail_latency_us"] for r in results),
            )
        selected = (
            max(candidates, key=lambda c: (medians[c][0], -medians[c][1]))
            if objective == "throughput" else
            min(candidates, key=lambda c: (medians[c][1], -medians[c][0]))
        )
        measured_candidates = list(candidates)
    else:
        margin_floor = prior["value"] if prior is not None else 0
        shortages = [r["observed_shortfall_mib"] for r in rows]
        selected = margin_floor + max(shortages)
        _int(selected, "measured host margin")
        measured_candidates = sorted({selected, *shortages})
    def candidate_evidence(candidate: int) -> list[dict]:
        if key != "host_ledger_margin.mib":
            return [row for row in rows if row["candidate"] == candidate]
        # Refit adds the previous qualified margin to the maximum observed
        # shortfall. The resulting value is supported by the full host window.
        return (rows if candidate == selected else
                [row for row in rows if row["observed_shortfall_mib"] == candidate])

    samples = [{
        "candidate": c,
        "evidence_ref": "sha256:" + hashlib.sha256(
            canonical(candidate_evidence(c))
        ).hexdigest(),
    } for c in measured_candidates]
    measured = {
        "schema_version": 1, "kind": "s8_calibration_measurement",
        "key": key, "scope": scope, "raw_run": str(raw_path),
        "evidence_id": "sha256:" + hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "method": "hardware ABBA 4 pairs; " + objective + "; " + reason.strip(),
        "measured_at_unix_s": now, "valid_until_unix_s": valid_until_unix_s,
        "source": "cache", "selected_value": selected, "samples": samples,
    }
    dest = out / "s8_calibration_measurement.json"
    dest.write_bytes(canonical(measured) + b"\n")
    return dest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="S8 hardware candidate sweep; must hold GPU lease")
    parser.add_argument("--key", choices=sorted(KEYS), required=True)
    parser.add_argument("--native-measurement", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--runner-arg", action="append", default=[])
    parser.add_argument("--candidate", action="append", type=int, default=[])
    parser.add_argument("--gates", type=Path)
    parser.add_argument("--objective", choices=("throughput", "latency"), default="throughput")
    parser.add_argument("--valid-until-unix-s", type=int, required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--prior-margin-record", type=Path)
    parser.add_argument("--store", type=Path, help="optional immutable P2 record store")
    args = parser.parse_args(argv)
    try:
        runfile = capture_s8_run(
            key=args.key, scope_file=args.native_measurement, out=args.out,
            runner=(args.runner, *args.runner_arg),
            candidates=tuple(args.candidate), gates_file=args.gates,
            valid_until_unix_s=args.valid_until_unix_s,
            reason=args.reason, objective=args.objective,
            prior_margin_record=args.prior_margin_record,
        )
        print(runfile)
        if args.store:
            print(emit_from_run(runfile, args.store))
        return 0
    except (CalibrationInvalid, OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

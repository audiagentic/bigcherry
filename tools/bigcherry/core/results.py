"""Engine-agnostic baseline, comparison and history over verified MRC01 records.

The baseline is portable by value: the exact canonical run records and every
referenced provenance blob are embedded and revalidated without a run directory
or even a RunRecordStore. Engines contribute metric directions/repeat groups in
their run records; no engine names or lab-specific metric lists live here.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import statistics
from collections.abc import Sequence
from copy import deepcopy
from typing import Any

from . import run_records

BASELINE_KIND = "bigcherry-results-baseline"
PHASES = ("decode", "prefill", "ttft")
_SENSITIVE = ("password", "secret", "token", "credential", "api_key", "private_key")


class ResultsError(ValueError):
    """An unsafe cohort, baseline or comparison request was refused."""


def _as_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ResultsError(f"invalid JSON value: {exc}") from exc


def _spread(values: Sequence[float]) -> float:
    """Parity semantics: range / absolute arithmetic mean (not stdev)."""
    if len(values) < 2:
        return 0.0
    centre = statistics.fmean(values)
    width = max(values) - min(values)
    if not centre:
        if not width:
            return 0.0
        raise ResultsError("relative spread undefined around zero; choose a stable metric")
    return width / abs(centre)


def _verified_rows(store: run_records.RunRecordStore, rows: Sequence[dict]) -> list[dict]:
    checked = []
    for row in rows:
        try:
            digest = row["sha256"]
            record = store.get_record(digest)
            if row.get("record") != record:
                raise ResultsError(f"record sha256 {digest}: query row differs from verified store")
            if record.get("validity", {}).get("state") in ("invalid", "unverified"):
                raise ResultsError(f"record sha256 {digest}: invalid or unverified execution attestation")
        except (KeyError, TypeError, run_records.RecordError) as exc:
            raise ResultsError(f"invalid or unreadable record: {exc}") from exc
        checked.append({"sha256": digest, "record": record})
    return checked


def _context(rows: Sequence[dict]) -> dict:
    launch_ids = [row["record"]["run_id"] for row in rows]
    if len(rows) < 2 or len(set(launch_ids)) < 2:
        raise ResultsError("at least two distinct launches (run_id values) are required")
    if len(set(launch_ids)) != len(launch_ids):
        raise ResultsError("duplicate launch run_id: each measurement must be an independent launch")
    first = rows[0]["record"]
    key = {
        "engine": first["identity"]["engine"],
        "layout": first["identity"]["layout"],
        "model": first["identity"]["model"],
        "workload": deepcopy(first["workload"]),
    }
    for row in rows[1:]:
        rec = row["record"]
        for field in ("engine", "layout", "model"):
            if rec["identity"][field] != key[field]:
                raise ResultsError(f"cohort has mixed {field} identities")
        if rec["workload"] != key["workload"]:
            raise ResultsError("cohort has mixed workload names or definitions")
    return key


def _metric_specs(rec: dict) -> dict[str, dict]:
    specs = rec.get("metric_specs")
    if not isinstance(specs, dict):
        raise ResultsError("metric_specs with explicit metric direction required in every run record")
    found = {}
    for phase in PHASES:
        for metric, spec in specs[phase].items():
            if (
                set(spec) - {"direction", "gated", "repeat_groups", "covariate"}
                or spec["direction"] not in ("higher", "lower")
                or type(spec["gated"]) is not bool
            ):
                raise ResultsError(f"invalid metric direction/gate for {phase}.{metric}")
            found[f"{phase}.{metric}"] = spec
    return found


def _values(rows: Sequence[dict], metric: str) -> list[float | None]:
    phase, name = metric.split(".", 1)
    return [row["record"]["metrics"][phase].get(name) for row in rows]


def _fit_covariate(ys: Sequence[float], xs: Sequence[float]) -> tuple[float, float]:
    """(slope, mean of x): the least-squares slope of a metric on its covariate over the baseline launches."""
    xbar = statistics.fmean(xs)
    sxx = sum((x - xbar) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0, xbar
    ybar = statistics.fmean(ys)
    return sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / sxx, xbar


def _adjust(y: float, x: float, slope: float, xbar: float) -> float:
    return y - slope * (x - xbar)


# A covariate needs enough launches for a slope to mean something; below this the metric is compared as it is.
COVARIATE_MIN_LAUNCHES = 4


def _metrics(rows: Sequence[dict]) -> dict[str, dict]:
    # All metric meaning is supplied by records. Missing gated values fail
    # closed; a completely absent phase is not implicitly measured.
    catalog: dict[str, dict] = {}
    for row in rows:
        specs = _metric_specs(row["record"])
        for metric, spec in specs.items():
            if metric in catalog and (
                catalog[metric]["direction"] != spec["direction"]
                or catalog[metric]["gated"] != spec["gated"]
            ):
                raise ResultsError(f"metric direction or gated state differs for {metric}")
            catalog[metric] = {
                "direction": spec["direction"], "gated": spec["gated"],
            }
            if spec.get("covariate"):
                catalog[metric]["covariate"] = spec["covariate"]
    if not catalog:
        raise ResultsError("baseline has no recorded metric directions")
    result = {}
    for metric, identity in sorted(catalog.items()):
        samples = _values(rows, metric)
        valid = [v for v in samples if v is not None]
        if identity["gated"] and len(valid) != len(rows):
            raise ResultsError(f"gated metric {metric} is missing in one or more launches")
        if not valid:
            raise ResultsError(f"no measured value for {metric}")
        entry = {**identity, "value": statistics.median(valid), "launches": samples}
        adjusted = valid
        covariate = identity.get("covariate")
        if covariate and identity["gated"] and len(rows) >= COVARIATE_MIN_LAUNCHES:
            xs = _values(rows, covariate)
            if any(x is None for x in xs):
                raise ResultsError(f"covariate {covariate} of {metric} is missing in one or more launches")
            slope, xbar = _fit_covariate(valid, xs)
            adjusted = [_adjust(y, x, slope, xbar) for y, x in zip(valid, xs)]
            entry.update({"value": statistics.median(adjusted),
                          "covariate_fit": {"metric": covariate, "slope": slope, "mean": xbar,
                                            "raw_value": statistics.median(valid)}})
        if identity["gated"]:
            across = _spread(adjusted)
            repeated = []
            for row in rows:
                for group in _metric_specs(row["record"]).get(metric, {}).get("repeat_groups", []):
                    if len(group) >= 2:
                        repeated.append(_spread(group))
            repeat = max(repeated) if repeated else None
            entry.update({
                "spread_across_launches": across,
                "spread_repeats": repeat,
                "tolerance": max(across, repeat or 0.0),
            })
        result[metric] = entry
    return result


def _store_blobs(store: run_records.RunRecordStore, rows: Sequence[dict]) -> list[dict]:
    unique = {}
    for row in rows:
        for ref in run_records.blob_references(row["record"]):
            digest = ref["sha256"]
            raw = store.get_blob(ref)
            if digest not in unique:
                unique[digest] = {
                    "sha256": digest, "size_bytes": ref["size_bytes"],
                    "base64": base64.b64encode(raw).decode("ascii"),
                }
    return [unique[k] for k in sorted(unique)]


def make_baseline(
    store: run_records.RunRecordStore, rows: Sequence[dict], name: str
) -> dict:
    """Build a reproducible baseline from independently launched stored records.

    Each provenance input is copied by content, not by original file path.
    True prompt-repeat spreads, when supplied by the record, can widen the
    observed launch spread. Distinct workload prompts never count as repeats.
    """
    if not isinstance(name, str) or not name.strip():
        raise ResultsError("baseline name must not be empty")
    verified = _verified_rows(store, rows)
    context = _context(verified)
    metrics = _metrics(verified)
    result = {
        "schema_version": 1, "kind": BASELINE_KIND, "name": name,
        "identity": context,
        "records": deepcopy(verified),
        "blobs": _store_blobs(store, verified),
        "metrics": metrics,
    }
    return validate_baseline(result)


def validate_baseline(baseline: dict) -> dict:
    """Validate hashes, all required blobs and the derived gates offline.

    Do not trust a baseline's metric table, record digest or referenced blob
    just because it was read from a JSON file.
    """
    if not isinstance(baseline, dict) or set(baseline) != {
        "schema_version", "kind", "name", "identity", "records", "blobs", "metrics"
    } or baseline["schema_version"] != 1 or baseline["kind"] != BASELINE_KIND:
        raise ResultsError("unsupported baseline format")
    rows = baseline["records"]
    if not isinstance(rows, list):
        raise ResultsError("baseline records must be an array")
    for row in rows:
        try:
            rec = row["record"]
            expected = row["sha256"]
            raw = run_records.canonical_record_bytes(rec)
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ResultsError("baseline record sha256 mismatch")
        except (KeyError, TypeError, run_records.RecordError) as exc:
            raise ResultsError(f"invalid baseline record: {exc}") from exc
    if len({r["sha256"] for r in rows}) != len(rows):
        raise ResultsError("baseline includes duplicate record sha256")
    known = {}
    if not isinstance(baseline["blobs"], list):
        raise ResultsError("baseline blobs must be an array")
    for blob in baseline["blobs"]:
        try:
            raw = base64.b64decode(blob["base64"], validate=True)
            digest = blob["sha256"]
            if hashlib.sha256(raw).hexdigest() != digest or len(raw) != blob["size_bytes"]:
                raise ResultsError("baseline blob sha256 or size mismatch")
            if digest in known:
                raise ResultsError("duplicate baseline blob")
            known[digest] = raw
        except (KeyError, TypeError, ValueError, binascii.Error) as exc:
            raise ResultsError(f"invalid baseline blob: {exc}") from exc
    needed = {}
    for row in rows:
        for ref in run_records.blob_references(row["record"]):
            if ref["sha256"] not in known or len(known[ref["sha256"]]) != ref["size_bytes"]:
                raise ResultsError(f"missing baseline provenance blob {ref['sha256']}")
            needed[ref["sha256"]] = True
    if set(known) != set(needed):
        raise ResultsError("unreferenced blobs in baseline")
    if any(row["record"].get("validity", {}).get("state") in
           ("invalid", "unverified") for row in rows):
        raise ResultsError("baseline contains invalid or unverified execution attestation")
    if _context(rows) != baseline["identity"]:
        raise ResultsError("baseline identity/workload differs from source records")
    if _metrics(rows) != baseline["metrics"]:
        raise ResultsError("baseline metric gates differ from source records")
    _as_json(baseline)
    return baseline


def _worse(now: float, ref: float, direction: str, tolerance: float) -> bool:
    allowance = abs(ref) * tolerance
    return (now - ref if direction == "lower" else ref - now) > allowance


def _flat(value: Any, prefix: str = "", depth: int = 0) -> dict[str, Any]:
    if depth >= 6:
        return {prefix: "<nested>"}
    if isinstance(value, dict):
        result = {}
        for key, child in sorted(value.items()):
            path = f"{prefix}.{key}" if prefix else key
            if any(s in key.lower() for s in _SENSITIVE):
                result[path] = "<redacted>"
            else:
                result.update(_flat(child, path, depth + 1))
        return result
    if isinstance(value, list):
        result = {}
        for i, child in enumerate(value[:64]):
            result.update(_flat(child, f"{prefix}[{i}]", depth + 1))
        if len(value) > 64:
            result[prefix + ".length"] = len(value)
        return result
    return {prefix: value}


def _decoded_blob(raw: bytes) -> dict:
    if len(raw) > 256 * 1024:
        return {}
    try:
        parsed = json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        return _flat(parsed) if isinstance(parsed, (dict, list)) else {}
    except (ValueError, UnicodeError):
        return {}


def _provenance_fields(rec: dict, read_blob) -> dict[tuple[str, str], Any]:
    fields = {}
    for field in ("effective_config", "argv", "environment", "power_settings"):
        ref = rec["provenance"][field]
        category = {
            "effective_config": "config", "argv": "argv",
            "environment": "environment", "power_settings": "power",
        }[field]
        fields[(category, field + ".sha256")] = ref["sha256"]
        if field in ("effective_config", "power_settings"):
            for path, value in _decoded_blob(read_blob(ref)).items():
                fields[(category, field + "." + path)] = value
    inputs = {}
    for item in rec["provenance"]["inputs"]:
        inputs.setdefault(item["role"], []).append(item["blob"]["sha256"])
    for role, hashes in inputs.items():
        group = "heat" if "heat" in role.lower() else (
            "policy" if "policy" in role.lower() or "calibration" in role.lower()
            else "inputs"
        )
        fields[(group, "inputs." + role)] = sorted(hashes)
    flat_host = _flat(rec["host_facts"])
    for key, value in flat_host.items():
        fields[("host_facts", key)] = value
        if "kernel" in key.lower():
            fields[("kernel", key)] = value
        if "power" in key.lower() or "cap_uw" in key.lower():
            fields[("power", "host_facts." + key)] = value
    for state in rec["feature_states"]:
        name = state["engine"] + "." + state["feature_id"]
        for key in ("state", "reason_code", "reason_detail"):
            fields[("feature_states", name + "." + key)] = state[key]
            if "policy" in state["feature_id"]:
                fields[("policy", name + "." + key)] = state[key]
    for field in ("build_id", "source_revision", "binary_sha256"):
        fields[("build", field)] = rec["identity"][field]
    return fields


def _differences(baseline: dict, rows: Sequence[dict], store: run_records.RunRecordStore) -> list[dict]:
    embedded = {b["sha256"]: base64.b64decode(b["base64"]) for b in baseline["blobs"]}
    old = [
        _provenance_fields(row["record"], lambda ref: embedded[ref["sha256"]])
        for row in baseline["records"]
    ]
    new = [
        _provenance_fields(row["record"], store.get_blob) for row in rows
    ]
    all_keys = set().union(*(set(d) for d in old + new))
    rank = ["host_facts", "feature_states", "kernel", "config", "heat",
            "policy", "power", "inputs", "environment", "argv", "build"]
    differences = []
    for category, path in sorted(all_keys, key=lambda k: (
        rank.index(k[0]) if k[0] in rank else len(rank), k[1]
    )):
        def unique(cohort, key=(category, path)):
            values = {_as_json(f.get(key, "<missing>")) for f in cohort}
            return [json.loads(x) for x in sorted(values)]
        before, after = unique(old), unique(new)
        if _as_json(before) != _as_json(after):
            differences.append({
                "category": category, "path": path,
                "baseline": before, "now": after,
            })
    return differences


def compare_records(
    store: run_records.RunRecordStore, baseline: dict, rows: Sequence[dict]
) -> dict:
    """Compare independent launches; show confounding provenance before gates."""
    validate_baseline(baseline)
    current = _verified_rows(store, rows)
    if _context(current) != baseline["identity"]:
        raise ResultsError("comparison engine/layout/model/workload differs from baseline")
    current_specs = [_metric_specs(r["record"]) for r in current]
    output = {}
    failed = False
    for name, base in baseline["metrics"].items():
        values = _values(current, name)
        for spec in current_specs:
            if name in spec and (
                spec[name]["direction"] != base["direction"]
                or spec[name]["gated"] != base["gated"]
            ):
                raise ResultsError(f"comparison metric direction/gate mismatch for {name}")
        missing = any(v is None for v in values)
        fit = base.get("covariate_fit")
        if fit and not missing:
            xs = _values(current, fit["metric"])
            if any(x is None for x in xs):
                missing = True
            else:
                values = [_adjust(y, x, fit["slope"], fit["mean"]) for y, x in zip(values, xs)]
        now = statistics.median(v for v in values if v is not None) if not missing else None
        ref = base["value"]
        change = ((now - ref) / abs(ref) * 100) if now is not None and ref else None
        if not base["gated"]:
            verdict = "info"
        elif missing:
            verdict = "MISSING"
        elif _worse(now, ref, base["direction"], base["tolerance"]):
            verdict = "REGRESSION"
        elif (now < ref if base["direction"] == "lower" else now > ref):
            verdict = "better"
        else:
            verdict = "ok"
        failed = failed or verdict in ("REGRESSION", "MISSING")
        output[name] = {
            "direction": base["direction"], "gated": base["gated"],
            "baseline": ref, "now": now, "change_percent": change,
            "tolerance": base.get("tolerance"), "verdict": verdict,
        }
    # A candidate-only metric must never disappear from the report. It has
    # no baseline gate, so mark it NEW rather than inventing a pass threshold.
    extras = {}
    for specs in current_specs:
        for metric, spec in specs.items():
            if metric in baseline["metrics"]:
                continue
            identity = (spec["direction"], spec["gated"])
            if metric in extras and extras[metric] != identity:
                raise ResultsError(f"candidate metric direction/gate differs for {metric}")
            extras[metric] = identity
    for metric, (direction, gated) in sorted(extras.items()):
        values = _values(current, metric)
        available = [value for value in values if value is not None]
        output[metric] = {
            "direction": direction, "gated": gated, "baseline": None,
            "now": statistics.median(available) if available else None,
            "change_percent": None, "tolerance": None, "verdict": "NEW",
        }
    old_sessions = {r["record"].get("session_id") for r in baseline["records"]}
    now_sessions = {r["record"].get("session_id") for r in current}
    warnings = []
    if None in old_sessions or None in now_sessions:
        warnings.append("session_id missing: cannot verify same-session A/B; same-session comparisons are preferred")
    elif len(old_sessions) != 1 or len(now_sessions) != 1 or old_sessions != now_sessions:
        warnings.append("different or mixed sessions: same-session A/B is preferred; inspect host-facts and feature-state differences")
    return {
        "baseline": baseline["name"], "launches": len(current),
        "warnings": warnings,
        "provenance_differences": _differences(baseline, current, store),
        "metrics": output, "failed": failed,
    }


def history(
    store: run_records.RunRecordStore, *, engine: str, layout: str,
    workload: str, metric: str, baseline: dict | None = None,
    trailing_window: int = 3, since: str | None = None, until: str | None = None
) -> dict:
    """One phase-qualified metric over time, with two-launch drift signals.

    Each signal compares the latest two launches, never a single launch, to
    either the reproducible baseline or an earlier non-overlapping trailing
    reference window of at least two launches. These are drift flags, not
    replacements for a provenance-aware cohort compare.
    """
    if not run_records._METRIC.fullmatch(metric):
        raise ResultsError("metric must be phase.name (decode, prefill or ttft)")
    if trailing_window < 2:
        raise ResultsError("trailing window must contain at least two launches")
    rows = store.query(engine=engine, layout=layout, workload=workload,
                       metric=metric, since=since, until=until)
    if not rows:
        return {"engine": engine, "layout": layout, "workload": workload,
                "metric": metric, "points": []}
    for row in rows:
        if row["record"].get("validity", {}).get("state") in ("invalid", "unverified"):
            raise ResultsError("history refuses invalid or unverified execution attestation")
    run_ids = [row["record"]["run_id"] for row in rows]
    if len(set(run_ids)) != len(run_ids):
        raise ResultsError("history contains duplicate launch run_id; cannot count one launch twice")
    first = rows[0]["record"]
    for row in rows:
        rec = row["record"]
        if rec["workload"] != first["workload"] or rec["identity"]["model"] != first["identity"]["model"]:
            raise ResultsError("history has mixed workload definitions or models")
        if metric not in _metric_specs(rec):
            raise ResultsError(f"missing metric direction in record {row['sha256']}")
    directions = {_metric_specs(row["record"])[metric]["direction"] for row in rows}
    if len(directions) != 1:
        raise ResultsError(f"history has mixed metric direction for {metric}")
    direction = directions.pop()
    base_metric = None
    if baseline is not None:
        validate_baseline(baseline)
        if baseline["identity"] != {
            "engine": engine, "layout": layout, "model": first["identity"]["model"],
            "workload": first["workload"],
        }:
            raise ResultsError("history baseline identity/workload mismatch")
        base_metric = baseline["metrics"].get(metric)
        if base_metric is None or base_metric["direction"] != direction:
            raise ResultsError("history baseline metric direction mismatch")
    values = [row["record"]["metrics"][metric.split(".")[0]][metric.split(".")[1]] for row in rows]
    points = []
    for i, (row, value) in enumerate(zip(rows, values)):
        pair = values[i-1:i+1] if i >= 1 else []
        base_drift = None
        if len(pair) == 2 and base_metric is not None and base_metric["gated"]:
            base_drift = _worse(statistics.median(pair), base_metric["value"],
                                direction, base_metric["tolerance"])
        reference = values[max(0, i - 1 - trailing_window):i-1] if i >= 3 else []
        trailing_drift = None
        trailing_median = None
        trailing_tolerance = None
        if len(pair) == 2 and len(reference) >= 2:
            trailing_median = statistics.median(reference)
            trailing_tolerance = _spread(reference)
            trailing_drift = _worse(statistics.median(pair), trailing_median,
                                    direction, trailing_tolerance)
        points.append({
            "recorded_at": row["record"]["recorded_at"], "sha256": row["sha256"],
            "run_id": row["record"]["run_id"], "value": value,
            "baseline_drift": base_drift, "trailing_drift": trailing_drift,
            "trailing_median": trailing_median, "trailing_tolerance": trailing_tolerance,
        })
    return {
        "engine": engine, "layout": layout, "workload": workload,
        "model": first["identity"]["model"], "metric": metric,
        "direction": direction, "points": points,
    }

"""MRC06 llama.cpp run_bench row -> platform v1, independent of upstream harness."""

from __future__ import annotations

import hashlib
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..core import host_facts, measurement_producers, run_records
from ..experiment import attestation

_HEX = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ENV = {
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GGML_HIP_DISPATCH_MODE",
    "GGML_HIP_DISPATCH_CACHE",
    "GGML_HIP_TUNE_WORKLOAD",
}


def _metrics(row: dict) -> tuple[dict[str, dict], dict[str, dict]]:
    phases: dict[str, dict[str, float]] = {
        name: {} for name in ("decode", "prefill", "ttft")
    }
    specs: dict[str, dict[str, dict]] = {name: {} for name in phases}
    for key, raw in row["results"].items():
        if key.endswith("_tps"):
            phase = "prefill" if key.startswith("pp") else "decode"
            direction = "higher"
        elif "ttft" in key and key.endswith("_ms"):
            phase = "ttft"
            direction = "lower"
        else:
            # Other output measures (e.g. ppl, variance, accuracy, power)
            # remain present, without remapping to a false phase, in the
            # untouched source-row blob.
            continue
        try:
            number = float(raw)
        except (TypeError, ValueError) as exc:
            raise run_records.RecordError(f"non-numeric llama metric {key}") from exc
        if not math.isfinite(number):
            raise run_records.RecordError(f"non-finite llama metric {key}")
        phases[phase][key] = number
        specs[phase][key] = {"direction": direction, "gated": True}
    return phases, specs


def _missing_host_start() -> dict:
    raise run_records.RecordError("llama benchmark requires a wrapper start host-facts receipt")


def publish_row(
    row: dict[str, Any],
    *,
    work_root: Path | None = None,
    expected: attestation.ExecutionIdentity | None = None,
    observed: attestation.ExecutionAttestation | None = None,
    host_snapshot: dict | None = None,
    recorded_at: str | None = None,
) -> str:
    """One measured run_bench result, with VA25 process-observed validity."""
    if not isinstance(row, dict) or not isinstance(row.get("results"), dict):
        raise run_records.RecordError("llama.cpp row requires measured results")
    meta = row.get("build_metadata") or {}
    binary_path = row.get("binary_path")
    if isinstance(binary_path, str) and binary_path and Path(binary_path).is_file():
        binary_hash = hashlib.sha256(Path(binary_path).read_bytes()).hexdigest()
    else:
        binary_hash = row.get("binary_sha256") or meta.get("binary_sha256")
    if not isinstance(binary_hash, str) or not _HEX.fullmatch(binary_hash):
        raise run_records.RecordError(
            "llama.cpp row lacks a verified executable SHA-256"
        )
    model = row.get("model") or {}
    if not isinstance(model, dict) or not (model.get("sha256") or model.get("path")):
        raise run_records.RecordError("llama.cpp model identity missing")
    source = meta.get("source_commit") or meta.get("ref") or row.get("source_revision")
    build_id = meta.get("build_id") or meta.get("ref")
    if (
        not isinstance(source, str)
        or not source
        or not isinstance(build_id, str)
        or not build_id
    ):
        raise run_records.RecordError("llama.cpp build identity/ref missing")
    metrics, metric_specs = _metrics(row)
    if not any(metrics.values()) and not row["results"]:
        raise run_records.RecordError("llama.cpp row contains no measured values")
    proof, validity = attestation.record_execution_evidence(expected, observed)
    env = row.get("env_vars_set") or {}
    if not isinstance(env, dict):
        raise run_records.RecordError("llama.cpp environment must be a mapping")
    if any(not isinstance(k, str) or not isinstance(v, str) for k, v in env.items()):
        raise run_records.RecordError("llama.cpp environment must contain strings")
    if any(k not in _SAFE_ENV for k in env):
        raise run_records.RecordError(
            "llama.cpp result environment has non-allowlisted keys"
        )
    command = row.get("command_line")
    if not isinstance(command, str) or not command:
        raise run_records.RecordError("llama.cpp row lacks actual command line")
    workload = {
        "name": str(row.get("bench_type") or "llama-bench"),
        "definition": {
            "config_name": str(row.get("config_name") or ""),
            "bench_type": row.get("bench_type"),
            "toggles_applied": row.get("toggles_applied") or {},
        },
    }
    measured_when = recorded_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = measurement_producers.canonical_blob(row)
    return measurement_producers.publish(
        work_root=work_root,
        run_id=str(row.get("run_id") or hashlib.sha256(payload).hexdigest()),
        recorded_at=measured_when,
        identity={
            "engine": "llama.cpp",
            "build_id": build_id,
            "binary_sha256": binary_hash,
            "source_revision": source,
            "model": model.get("sha256") or model["path"],
            "layout": (row.get("platform") or {}).get("name") or row.get("device"),
        },
        workload=workload,
        metrics=metrics,
        metric_specs=metric_specs,
        effective_config=measurement_producers.canonical_blob(
            {
                "config": row.get("config", {}),
                "toggles_applied": row.get("toggles_applied", {}),
                "build_metadata": meta,
            }
        ),
        argv=[command],
        environment=env,
        power_settings=row.get("power_state") or {},
        inputs={
            "original_llama_row": payload,
            "execution_attestation": measurement_producers.canonical_blob(proof),
        },
        host_facts=host_facts.validate_start_snapshot(host_snapshot)
        if host_snapshot is not None else _missing_host_start(),
        feature_states=[],
        validity=validity,
    )

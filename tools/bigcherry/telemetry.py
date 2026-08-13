"""Strict ingestion for telemetry-only BLAS and SPLIT_REDUCE JSONL.

Telemetry is diagnostic evidence, not a tuning/replay or release artifact.  The
loader therefore normalizes neither missing fields nor malformed events: a
partial or mixed-provenance file must be rejected before analysis.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable


class TelemetryError(ValueError):
    """Raised when telemetry cannot be trusted as one artifact."""


_HEX32 = set("0123456789abcdefABCDEF")
_BLAS_APIS = {"cublasSgemm", "cublasGemmEx", "cublasGemmStridedBatchedEx",
              "cublasGemmBatchedEx"}
_BLAS_WRAPPERS = {"ggml_cuda_mul_mat_cublas"}
_PROVIDERS = {"internal", "rccl", "meta", "unknown", "provider_declined"}
_HANDOFFS = {"none", "provider_declined_handoff_meta"}
_UNAVAILABLE = {"unavailable", "unknown", "not_collected"}
_BLAS_CONVERSIONS = {"direct", "contiguous", "non_contiguous"}
_BLAS_OUTPUT_CONVERSIONS = {"direct", "temporary_to_f32"}
_BLAS_PROVIDERS = {"hipblas", "unknown"}
_BLAS_BACKENDS = {"unknown", "rocblas", "hipblaslt"}
_REDUCTION_ALGORITHM_ALIASES = {
    # The runtime telemetry calls the RCCL-backed implementation "rccl";
    # HI18's candidate identity names the algorithm "nccl" after the
    # upstream entry point.  Keep that translation in the offline evidence
    # layer instead of changing the producer's source vocabulary.
    "rccl": "nccl",
    "meta": "meta",
    "internal": "internal",
    "unknown": "unknown",
    "provider_declined": "provider_declined",
}


def _text(value: Any, field: str, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TelemetryError(f"{where}: {field} must be a non-empty string")
    return value.strip()


def _digest(value: Any, field: str, where: str) -> str:
    value = _text(value, field, where)
    if len(value) != 32 or any(char not in _HEX32 for char in value):
        raise TelemetryError(f"{where}: {field} must be a 32-digit hex digest")
    return value.lower()


def _nonnegative(value: Any, field: str, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TelemetryError(f"{where}: {field} must be a non-negative integer")
    return value


def _positive(value: Any, field: str, where: str) -> int:
    value = _nonnegative(value, field, where)
    if value == 0:
        raise TelemetryError(f"{where}: {field} must be positive")
    return value


def _provenance(value: Any, where: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise TelemetryError(f"{where}: provenance must be an object")
    result = {
        "source_revision": _text(value.get("source_revision"), "source_revision", where),
        "manifest_hash": _text(value.get("manifest_hash"), "manifest_hash", where).lower(),
        "variant_set": _text(value.get("variant_set"), "variant_set", where),
    }
    if len(result["source_revision"]) < 7 or len(result["manifest_hash"]) < 8:
        raise TelemetryError(f"{where}: provenance digest fields are malformed")
    return result


def _same_provenance(actual: dict[str, str], expected: dict[str, str], where: str) -> None:
    if actual != expected:
        raise TelemetryError(f"{where}: telemetry has mixed or unexpected provenance")


def _availability(value: Any, field: str, where: str) -> str:
    value = _text(value, field, where)
    if value in _UNAVAILABLE:
        return value
    return value


def _validate_blas_metadata(row: dict[str, Any], where: str) -> dict[str, Any]:
    metadata = row.get("blas_metadata")
    if not isinstance(metadata, dict):
        raise TelemetryError(f"{where}: BLAS observation requires blas_metadata")
    normalized = {
        "operand_a_type": _text(metadata.get("operand_a_type"), "operand_a_type", where),
        "operand_b_type": _text(metadata.get("operand_b_type"), "operand_b_type", where),
        "output_type": _text(metadata.get("output_type"), "output_type", where),
        "accumulation_type": _text(metadata.get("accumulation_type"), "accumulation_type", where),
        "source_a_conversion": _text(metadata.get("source_a_conversion"), "source_a_conversion", where),
        "source_b_conversion": _text(metadata.get("source_b_conversion"), "source_b_conversion", where),
        "output_conversion": _text(metadata.get("output_conversion"), "output_conversion", where),
        "requested_precision": _text(metadata.get("requested_precision"), "requested_precision", where),
        "effective_call_api": _text(metadata.get("effective_call_api"), "effective_call_api", where),
        "effective_provider": _text(metadata.get("effective_provider"), "effective_provider", where),
        "effective_backend": _text(metadata.get("effective_backend"), "effective_backend", where),
        "source_a_temp_bytes": _nonnegative(metadata.get("source_a_temp_bytes"), "source_a_temp_bytes", where),
        "source_b_temp_bytes": _nonnegative(metadata.get("source_b_temp_bytes"), "source_b_temp_bytes", where),
        "output_temp_bytes": _nonnegative(metadata.get("output_temp_bytes"), "output_temp_bytes", where),
    }
    if normalized["source_a_conversion"] not in _BLAS_CONVERSIONS:
        raise TelemetryError(f"{where}: unsupported source_a_conversion")
    if normalized["source_b_conversion"] not in _BLAS_CONVERSIONS:
        raise TelemetryError(f"{where}: unsupported source_b_conversion")
    if normalized["output_conversion"] not in _BLAS_OUTPUT_CONVERSIONS:
        raise TelemetryError(f"{where}: unsupported output_conversion")
    if normalized["effective_call_api"] not in _BLAS_APIS:
        raise TelemetryError(f"{where}: unsupported effective_call_api")
    if normalized["effective_provider"] not in _BLAS_PROVIDERS:
        raise TelemetryError(f"{where}: unsupported effective_provider")
    if normalized["effective_backend"] not in _BLAS_BACKENDS:
        raise TelemetryError(f"{where}: unsupported effective_backend")
    return normalized


def _validate_blas(row: dict[str, Any], where: str) -> tuple[str, dict[str, Any]]:
    signature = _digest(row.get("signature"), "signature", where)
    hardware = _digest(row.get("hardware"), "hardware", where)
    _text(row.get("native"), "native", where)
    calls = _nonnegative(row.get("calls"), "calls", where)
    _nonnegative(row.get("est_bytes"), "est_bytes", where)
    _nonnegative(row.get("workspace_bytes"), "workspace_bytes", where)
    # ``effective_api`` identifies the established upstream wrapper.  The
    # exact hipBLAS branch is deliberately a separate observation recorded at
    # the call site, because the wrapper can select one of several APIs.
    effective = _availability(row.get("effective_api"), "effective_api", where)
    call_api = _availability(row.get("effective_call_api"), "effective_call_api", where)
    if effective not in _BLAS_WRAPPERS and effective not in _UNAVAILABLE:
        raise TelemetryError(f"{where}: unsupported effective_api {effective!r}")
    if call_api not in _BLAS_APIS and call_api not in _UNAVAILABLE:
        raise TelemetryError(f"{where}: unsupported effective_call_api {call_api!r}")
    if effective in _BLAS_WRAPPERS and call_api not in _BLAS_APIS:
        raise TelemetryError(
            f"{where}: BLAS wrapper observation requires an exact effective_call_api")
    blas_metadata = _validate_blas_metadata(row, where)
    devices = row.get("devices")
    if not isinstance(devices, list) or not devices or any(
            isinstance(device, bool) or not isinstance(device, int) or device < 0
            for device in devices):
        raise TelemetryError(f"{where}: devices must be a non-empty list of ordinals")
    if len(set(devices)) != len(devices):
        raise TelemetryError(f"{where}: duplicate device ordinal")
    return f"blas:{signature}:{hardware}", {**row, "signature": signature, "hardware": hardware,
                                             "devices": list(devices), "calls": calls,
                                             "blas_metadata": blas_metadata}


def normalize_split_reduce_observation(row: dict[str, Any], where: str) -> dict[str, Any]:
    """Attach the HI18 candidate identity implied by validated telemetry.

    This is evidence normalization, not runtime selection.  A handoff is
    classified as ``upstream_default`` and is explicitly non-promotable for
    algorithm-level conclusions because the requested provider did not own the
    complete path.  No reduction signature or topology key is invented here.
    """
    requested = row["requested_provider"]
    effective = row["effective_provider"]
    handoff = row["handoff"]
    if handoff == "none" and effective != requested:
        raise TelemetryError(
            f"{where}: none fallback policy requires requested/effective provider match")

    preferred_algorithm = _REDUCTION_ALGORITHM_ALIASES[requested]
    effective_algorithm = _REDUCTION_ALGORITHM_ALIASES[effective]
    fallback_policy = "none" if handoff == "none" else "upstream_default"
    promotable = (
        fallback_policy == "none"
        and effective == requested
        and preferred_algorithm not in {"unknown", "provider_declined"}
    )
    return {
        "preferred_algorithm": preferred_algorithm,
        "effective_algorithm": effective_algorithm,
        "fallback_policy": fallback_policy,
        "candidate_identity": (
            f"split_reduce:{preferred_algorithm}:{fallback_policy}:v1"),
        "promotable": promotable,
    }


def _validate_split(row: dict[str, Any], where: str) -> tuple[str, dict[str, Any]]:
    # SPLIT_REDUCE is a multi-device evidence channel.  A zero/one-device
    # record cannot establish collective topology and must not enter later
    # attribution or tuning analysis.
    timestamp_us = _positive(row.get("timestamp_us"), "timestamp_us", where)
    requested = _text(row.get("requested_provider"), "requested_provider", where)
    effective = _text(row.get("effective_provider"), "effective_provider", where)
    handoff = _text(row.get("handoff"), "handoff", where)
    if requested not in _PROVIDERS or effective not in _PROVIDERS:
        raise TelemetryError(f"{where}: unknown reduction provider")
    if handoff not in _HANDOFFS:
        raise TelemetryError(f"{where}: unknown handoff {handoff!r}")
    depth = _nonnegative(row.get("fallback_depth"), "fallback_depth", where)
    if handoff == "none" and depth != 0:
        raise TelemetryError(f"{where}: fallback depth requires a handoff")
    if handoff != "none" and depth == 0:
        raise TelemetryError(f"{where}: handoff requires positive fallback depth")
    _nonnegative(row.get("element_count"), "element_count", where)
    _text(row.get("element_type"), "element_type", where)
    count = _positive(row.get("device_count"), "device_count", where)
    if count < 2:
        raise TelemetryError(f"{where}: device_count must be at least 2")
    devices = row.get("devices")
    if not isinstance(devices, list) or len(devices) != count:
        raise TelemetryError(f"{where}: device_count does not match devices")
    if any(isinstance(device, bool) or not isinstance(device, int) or device < 0 for device in devices):
        raise TelemetryError(f"{where}: devices must contain non-negative ordinals")
    if len(set(devices)) != len(devices):
        raise TelemetryError(f"{where}: duplicate device ordinal")
    event_id = _text(row.get("event_id"), "event_id", where) if "event_id" in row else None
    key = event_id or f"split:{timestamp_us}:{requested}:{effective}:{devices}"
    normalized = {**row, "timestamp_us": timestamp_us, "device_count": count,
                  "devices": list(devices), "event_id": event_id}
    normalized.update(normalize_split_reduce_observation(normalized, where))
    return key, normalized


def load_telemetry(paths: str | Path | Iterable[str | Path], *,
                   expected_provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load BLAS record and/or SPLIT_REDUCE telemetry JSONL strictly.

    BLAS files require one header.  Header provenance is inherited by its
    observations.  SPLIT_REDUCE's current channel has no header; callers must
    provide ``expected_provenance`` to bind those events to a build.  Duplicate
    logical events and duplicate headers are rejected across all input files.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    expected = _provenance(expected_provenance, "expected_provenance") if expected_provenance else None
    result = {"provenance": None, "blas": [], "split_reduce": []}
    seen: set[str] = set()
    headers = 0
    for path_value in paths:
        path = Path(path_value)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise TelemetryError(f"{path}: cannot read telemetry") from exc
        for number, line in enumerate(lines, 1):
            where = f"{path}:{number}"
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TelemetryError(f"{where}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise TelemetryError(f"{where}: event must be an object")
            kind = row.get("kind")
            if kind == "header":
                headers += 1
                if headers != 1:
                    raise TelemetryError(f"{where}: duplicate header")
                actual = _provenance(row, where)
                if expected is not None:
                    _same_provenance(actual, expected, where)
                result["provenance"] = actual
                continue
            if kind == "observation":
                # Record mode writes a single inventory containing every
                # dispatch family.  A pair of empty BLAS fields denotes a
                # non-BLAS observation, which is outside this telemetry-only
                # loader rather than malformed BLAS evidence.  A partial pair
                # is never safe to ignore: it means the producer lost one half
                # of the effective-path attribution.
                effective_api = row.get("effective_api")
                effective_call_api = row.get("effective_call_api")
                empty_api = effective_api in (None, "")
                empty_call_api = effective_call_api in (None, "")
                if empty_api and empty_call_api:
                    continue
                if empty_api != empty_call_api:
                    raise TelemetryError(
                        f"{where}: partial BLAS effective-path telemetry")
                key, normalized = _validate_blas(row, where)
                result_kind = "blas"
            elif kind == "split_reduce_observation":
                key, normalized = _validate_split(row, where)
                result_kind = "split_reduce"
            else:
                raise TelemetryError(f"{where}: unknown telemetry kind {kind!r}")
            if key in seen:
                raise TelemetryError(f"{where}: duplicate telemetry event {key}")
            seen.add(key)
            if result["provenance"] is None:
                if expected is None:
                    raise TelemetryError(f"{where}: telemetry provenance is required")
                result["provenance"] = expected
            elif expected is not None:
                _same_provenance(result["provenance"], expected, where)
            result[result_kind].append(normalized)
    if not result["blas"] and not result["split_reduce"]:
        raise TelemetryError("telemetry artifact contains no events")
    return result

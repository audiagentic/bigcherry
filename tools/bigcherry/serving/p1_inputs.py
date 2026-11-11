"""Inactive P1 native bc-serve measurement consumer for S11.

The input is an engine-produced measurement, never a calibration target or a
v1 manifest. No hardware access occurs during decoding. S13 will supply the
command that obtains a fresh scoped record for each planning attempt.
"""
from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable, Sequence

from ..engine_adapters.radiance_resources import MIB, HostLimits, ModelMemory, Targets
from .measured_plan import GpuDemand, MeasuredPlanRefused, PlanBounds, PlanMeasurement

_MAX_WIRE = (1 << 53) - 1
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_RECORD = frozenset(("schema_version", "kind", "planning_key", "provenance",
                     "resource_inputs", "gpu", "concurrent_sessions",
                     "host_reserve_bytes", "scope"))
_RESOURCE = frozenset(("host_embedding_bytes", "unowned_expert_bytes",
                       "staging_bytes", "ngram_bytes", "checkpoint_bytes",
                       "resident_other_bytes", "kv_bytes_per_token",
                       "context_target_tokens", "retention_target_tokens",
                       "mem_available_bytes", "cgroup_headroom_bytes",
                       "pin_available_bytes", "owner_map_digest",
                       "host_pool_margin_calibration", "helper_footprints"))
_SCOPE = frozenset(("model_digest", "engine_build_digest", "helper_build_digest",
                    "layout_digest", "inventory_digest", "owner_map_digest",
                    "context_target_tokens", "retention_target_tokens",
                    "kv_bytes_per_token"))
_HELPER = frozenset(("role", "bdf", "lanes_bytes", "pointer_banks_bytes",
                     "slot_map_bytes", "incoming_slot_bytes", "total_bytes"))
_GPU = frozenset(("bdf", "required_bytes", "fixed_bytes"))
_MARGIN = frozenset(("key", "source", "evidence", "mib"))


def _object(value: object, fields: frozenset[str], where: str) -> dict:
    if not isinstance(value, dict) or set(value) != fields:
        raise MeasuredPlanRefused(f"P1 {where}: missing/unknown fields")
    return value


def _bytes(name: str, value: object, *, positive: bool = False) -> int:
    if type(value) is not int or not (0 <= value <= _MAX_WIRE) or (positive and not value):
        raise MeasuredPlanRefused(f"P1 {name}: measured wire integer required")
    return value


def _digest(name: str, value: object) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise MeasuredPlanRefused(f"P1 {name}: SHA-256 identity required")
    return value


def _pairs(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise MeasuredPlanRefused(f"P1 duplicate JSON member {key}")
        value[key] = item
    return value


def _no_fraction(_: str) -> None:
    raise MeasuredPlanRefused("P1 only I-JSON-safe integer numbers are valid")


def parse_p1_json(text: str, layout) -> PlanMeasurement:
    """Decode one strict, scoped, engine-native publication for an S11 attempt."""
    try:
        raw = json.loads(text, object_pairs_hook=_pairs, parse_float=_no_fraction,
                         parse_constant=_no_fraction)
    except (ValueError, TypeError) as exc:
        raise MeasuredPlanRefused("P1 invalid measurement JSON") from exc
    return decode_p1_record(raw, layout)


def decode_p1_record(raw: object, layout) -> PlanMeasurement:
    record = _object(raw, _RECORD, "record")
    if (record["schema_version"] != 1 or type(record["schema_version"]) is not int
            or record["kind"] != "bc_serve_v2_measured_inputs"):
        raise MeasuredPlanRefused("P1 publication has wrong identity")
    key = _digest("planning_key", record["planning_key"])
    provenance = record["provenance"]
    if not isinstance(provenance, str) or not provenance.strip():
        raise MeasuredPlanRefused("P1 measurement provenance is missing")
    scope = _object(record["scope"], _SCOPE, "scope")
    for name in ("model_digest", "engine_build_digest", "helper_build_digest",
                 "layout_digest", "inventory_digest", "owner_map_digest"):
        _digest(f"scope.{name}", scope[name])
    r = _object(record["resource_inputs"], _RESOURCE, "resource_inputs")
    digest = _digest("owner_map_digest", r["owner_map_digest"])
    if digest != scope["owner_map_digest"]:
        raise MeasuredPlanRefused("P1 owner map changed between scope and engine publication")
    for name in _RESOURCE - {"owner_map_digest", "helper_footprints",
                             "host_pool_margin_calibration"}:
        _bytes(name, r[name], positive=name in (
            "kv_bytes_per_token", "context_target_tokens", "retention_target_tokens",
            "mem_available_bytes", "cgroup_headroom_bytes", "pin_available_bytes"))
    for name in ("context_target_tokens", "retention_target_tokens", "kv_bytes_per_token"):
        if r[name] != scope[name]:
            raise MeasuredPlanRefused(f"P1 scope disagrees on {name}")

    margin = _object(r["host_pool_margin_calibration"], _MARGIN, "margin")
    if (margin["key"] != "host_ledger_margin.mib" or
            margin["source"] not in ("seed", "cache") or
            not isinstance(margin["evidence"], str) or not margin["evidence"].strip()):
        raise MeasuredPlanRefused("P1 qualified host-margin calibration missing")
    margin_mib = _bytes("margin.mib", margin["mib"])
    if margin_mib > _MAX_WIRE // MIB:
        raise MeasuredPlanRefused("P1 host margin overflows byte range")
    session_count = _bytes("concurrent_sessions", record["concurrent_sessions"],
                           positive=True)
    host_reserve = _bytes("host_reserve_bytes", record["host_reserve_bytes"])
    if r["retention_target_tokens"] > r["context_target_tokens"] * session_count:
        raise MeasuredPlanRefused("P1 retained KV target exceeds service capacity")

    if not isinstance(r["helper_footprints"], list):
        raise MeasuredPlanRefused("P1 helper measurements must be a list")
    worker_bytes = {}
    roles = set()
    for item in r["helper_footprints"]:
        helper = _object(item, _HELPER, "helper")
        bdf = helper["bdf"]
        role = helper["role"]
        if (not isinstance(bdf, str) or not isinstance(role, str) or not role
                or bdf in worker_bytes or role in roles):
            raise MeasuredPlanRefused("P1 duplicate/unidentified helper measurement")
        roles.add(role)
        parts = tuple(_bytes(f"helper.{name}", helper[name]) for name in
                      ("lanes_bytes", "pointer_banks_bytes", "slot_map_bytes",
                       "incoming_slot_bytes"))
        total = _bytes("helper.total_bytes", helper["total_bytes"], positive=True)
        if sum(parts) != total:
            raise MeasuredPlanRefused(f"P1 {bdf}: worker component sum disagrees")
        worker_bytes[bdf] = total
    if set(worker_bytes) != set(layout.helper_bdfs):
        raise MeasuredPlanRefused("P1 helper footprint BDF set differs from selected layout")
    if roles != {f"helper{i}" for i in range(len(layout.helper_bdfs))}:
        raise MeasuredPlanRefused("P1 helper role set differs from selected layout")

    if not isinstance(record["gpu"], list):
        raise MeasuredPlanRefused("P1 GPU measurements must be a list")
    gpu = []
    for value in record["gpu"]:
        item = _object(value, _GPU, "GPU")
        bdf = item["bdf"]
        if not isinstance(bdf, str):
            raise MeasuredPlanRefused("P1 engine BDF is missing")
        required = _bytes("GPU required", item["required_bytes"], positive=True)
        fixed = _bytes("GPU fixed", item["fixed_bytes"])
        if fixed > required:
            raise MeasuredPlanRefused("P1 fixed GPU bytes exceed required bytes")
        gpu.append(GpuDemand(bdf, required, fixed))
    if tuple(g.bdf for g in gpu) != tuple(layout.engine_bdfs):
        raise MeasuredPlanRefused("P1 engine GPU rank/BDF order differs from layout")

    model = ModelMemory(
        owner_map_digest=digest,
        host_embedding_bytes=r["host_embedding_bytes"],
        unowned_expert_bytes=r["unowned_expert_bytes"],
        host_staging_bytes=r["staging_bytes"],
        ngram_ram_bytes=r["ngram_bytes"],
        checkpoint_bytes=r["checkpoint_bytes"],
        other_engine_bytes=r["resident_other_bytes"],
        kv_bytes_per_token=r["kv_bytes_per_token"],
    )
    targets = Targets(
        context_tokens=r["context_target_tokens"],
        concurrent_sessions=session_count,
        retained_kv_tokens=r["retention_target_tokens"],
        host_pool_margin_bytes=margin_mib * MIB,
        host_reserve_bytes=host_reserve,
    )
    host = HostLimits(
        mem_available_bytes=r["mem_available_bytes"],
        cgroup_headroom_bytes=r["cgroup_headroom_bytes"],
        pin_available_bytes=r["pin_available_bytes"],
    )
    return PlanMeasurement(key, provenance, model, targets, host, worker_bytes,
                           tuple(gpu), dict(r), dict(scope))


class P1MeasurementSource:
    """S11 source adapter: a fresh native command for every bounded replan.

    S13 must inject a command builder that points to bc-serve's P1 producer,
    with model/layout/target identity and explicit PlanBounds. Never reuse a
    captured record for a second attempt; it may be stale.
    """

    def __init__(self, command: Callable[[object, PlanBounds], Sequence[str]],
                 *, run: Callable = subprocess.run):
        self.command = command
        self.run = run

    def measure(self, layout, bounds: PlanBounds) -> PlanMeasurement:
        argv = self.command(layout, bounds)
        if (not isinstance(argv, (list, tuple)) or not argv or
                any(not isinstance(a, str) or not a for a in argv)):
            raise MeasuredPlanRefused("P1 command builder omitted native measurement command")
        result = self.run(argv, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            raise MeasuredPlanRefused(
                f"P1 native measurement failed (exit {result.returncode}): "
                "refit requires a fresh engine publication")
        return parse_p1_json(result.stdout, layout)

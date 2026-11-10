"""GPU-free RSC03 resource derivation for the future v2 serving plan.

The values in :class:`ModelMemory` are authoritative measured inputs, not
estimates of expert ownership or worker allocations. In particular, the
host expert bytes must come from the engine's actual owner map, and each
helper runtime footprint from worker_footprint.h. A missing measurement
refuses; this module never opens a GPU, model container, or manifest.
The v1 resolver is intentionally not a caller (RSC11 S13 owns cutover).
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .radiance_inventory import Layout

MIB = 1024 * 1024
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")


class ResourceError(ValueError):
    """Insufficient or unmeasured resources: no fixed-size fallback."""


def _size(name: str, value: int) -> int:
    if type(value) is not int or value < 0:
        raise ResourceError(f"{name} must be a measured, non-negative byte count")
    return value


def _positive(name: str, value: int) -> int:
    if _size(name, value) == 0:
        raise ResourceError(f"{name} must be positive")
    return value


def _ceil_mib(byte_count: int) -> int:
    return (byte_count + MIB - 1) // MIB


@dataclass(frozen=True)
class ModelMemory:
    """Published from the real model/owner map and engine plan, all in bytes."""

    owner_map_digest: str
    host_embedding_bytes: int
    unowned_expert_bytes: int
    host_staging_bytes: int
    ngram_ram_bytes: int
    checkpoint_bytes: int
    other_engine_bytes: int
    kv_bytes_per_token: int


@dataclass(frozen=True)
class Targets:
    """Per-session context/retention objective and calibrated headroom."""

    context_tokens: int
    concurrent_sessions: int
    retained_kv_tokens: int
    host_pool_margin_bytes: int
    host_reserve_bytes: int


@dataclass(frozen=True)
class HostLimits:
    """At admission time: MemAvailable, cgroup headroom, and pinable memory."""

    mem_available_bytes: int
    cgroup_headroom_bytes: int
    pin_available_bytes: int


@dataclass(frozen=True)
class Breakdown:
    name: str
    bytes: int
    source: str


@dataclass(frozen=True)
class HelperBudget:
    bdf: str
    arch: str
    vram_mib: int
    reserve_mib: int
    worker_footprint_mib: int
    expert_budget_mib: int
    slack_bytes: int


@dataclass(frozen=True)
class ResourcePlan:
    owner_map_digest: str
    host_pool_mib: int
    host_kv_mib: int
    helpers: tuple[HelperBudget, ...]
    host_components: tuple[Breakdown, ...]
    host_available_bytes: int
    host_committed_bytes: int
    host_remaining_bytes: int
    pin_available_bytes: int
    pin_committed_bytes: int

    def explain(self) -> dict:
        return {
            "owner_map_digest": self.owner_map_digest,
            "host_pool_mib": self.host_pool_mib,
            "host_kv_mib": self.host_kv_mib,
            "host": {
                "components": [vars(part) for part in self.host_components],
                "available_bytes": self.host_available_bytes,
                "committed_bytes": self.host_committed_bytes,
                "remaining_bytes": self.host_remaining_bytes,
                "pin_available_bytes": self.pin_available_bytes,
                "pin_committed_bytes": self.pin_committed_bytes,
            },
            "helpers": [vars(helper) for helper in self.helpers],
        }


def derive(
    layout: Layout,
    model: ModelMemory,
    targets: Targets,
    host: HostLimits,
    *,
    worker_footprints_bytes: Mapping[str, int],
) -> ResourcePlan:
    """Derive v2 budgets from S2 BDF roles and authoritative model measurements.

    `worker_footprints_bytes` must include every selected helper's complete
    non-expert VRAM footprint (lanes, bank pointers, slot map, incoming slot).
    A different inventory or role assignment therefore changes the plan.
    """
    if not _SHA256.fullmatch(model.owner_map_digest):
        raise ResourceError("owner_map_digest must identify the actual ownership plan (sha256)")
    for name in ("host_embedding_bytes", "unowned_expert_bytes", "host_staging_bytes",
                 "ngram_ram_bytes", "checkpoint_bytes", "other_engine_bytes"):
        _size(f"model.{name}", getattr(model, name))
    _positive("model.kv_bytes_per_token", model.kv_bytes_per_token)
    _positive("targets.context_tokens", targets.context_tokens)
    _positive("targets.concurrent_sessions", targets.concurrent_sessions)
    _size("targets.retained_kv_tokens", targets.retained_kv_tokens)
    if targets.retained_kv_tokens > targets.context_tokens * targets.concurrent_sessions:
        raise ResourceError("retained KV target exceeds concurrent_sessions * context_tokens")
    _size("targets.host_pool_margin_bytes", targets.host_pool_margin_bytes)
    _size("targets.host_reserve_bytes", targets.host_reserve_bytes)
    for key in ("mem_available_bytes", "cgroup_headroom_bytes", "pin_available_bytes"):
        _positive(f"host.{key}", getattr(host, key))
    if not isinstance(worker_footprints_bytes, Mapping):
        raise ResourceError("worker_footprints_bytes must be BDF-keyed measurements")

    needed = set(layout.helper_bdfs)
    if set(worker_footprints_bytes) != needed:
        raise ResourceError(
            "worker footprint BDFs differ from selected helpers: "
            f"missing={sorted(needed - set(worker_footprints_bytes))}, "
            f"unexpected={sorted(set(worker_footprints_bytes) - needed)}"
        )
    if len(needed) != len(layout.resolved.helpers):
        raise ResourceError("helper role/BDF mismatch")

    helpers = []
    for bdf, helper in zip(layout.helper_bdfs, layout.resolved.helpers):
        gpu = layout.inventory.by_bdf(bdf)
        if gpu.arch != helper.arch.name or gpu.ordinal != helper.index:
            raise ResourceError(f"helper role changed since inventory detection: {bdf}")
        footprint = _positive(f"worker_footprints_bytes[{bdf}]", worker_footprints_bytes[bdf])
        available = gpu.vram_mib * MIB - helper.arch.reserve_mib * MIB - footprint
        if available < MIB:
            raise ResourceError(f"{bdf}: no expert VRAM after architecture reserve and worker footprint")
        budget = available // MIB
        helpers.append(HelperBudget(
            bdf, gpu.arch, gpu.vram_mib, helper.arch.reserve_mib,
            _ceil_mib(footprint), budget, available - budget * MIB
        ))

    pool_parts = (
        Breakdown("embedding", model.host_embedding_bytes, "engine plan"),
        Breakdown("unowned_experts", model.unowned_expert_bytes, "published owner map"),
        Breakdown("staging", model.host_staging_bytes, "engine plan"),
        Breakdown("pool_margin", targets.host_pool_margin_bytes, "calibrated margin"),
    )
    pool = _ceil_mib(sum(part.bytes for part in pool_parts))
    kv_bytes = targets.retained_kv_tokens * model.kv_bytes_per_token
    kv = _ceil_mib(kv_bytes)
    components = pool_parts + (
        Breakdown("ngram_ram", model.ngram_ram_bytes, "engine plan"),
        Breakdown("checkpoints", model.checkpoint_bytes, "engine plan"),
        Breakdown("other_engine", model.other_engine_bytes, "engine plan"),
        Breakdown("retained_kv", kv_bytes, "target * bytes per token"),
        Breakdown("host_reserve", targets.host_reserve_bytes, "host reserve policy"),
    )
    available = min(host.mem_available_bytes, host.cgroup_headroom_bytes)
    committed = ((pool + kv) * MIB + model.ngram_ram_bytes +
                 model.checkpoint_bytes + model.other_engine_bytes + targets.host_reserve_bytes)
    pinned = (pool + kv) * MIB
    if committed > available:
        raise ResourceError(
            f"host RAM admission refused: {committed} bytes required, "
            f"{available} bytes available (including reserve)"
        )
    if pinned > host.pin_available_bytes:
        raise ResourceError(
            f"pin admission refused: {pinned} bytes required, "
            f"{host.pin_available_bytes} bytes available"
        )
    return ResourcePlan(model.owner_map_digest, pool, kv, tuple(helpers),
                        components, available, committed, available - committed,
                        host.pin_available_bytes, pinned)

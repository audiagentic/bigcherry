"""Inactive RSC11 S11 v2 measured planner: RSC03 derivation + RSC10 refit.

S5 owns the engine's structured pre-weight admission result. S10/S13 own the
native measurement source, v2 writer/strict reader and actual process adapter.
No GPU access, guessed resources, v1 path or production CLI activation here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Protocol

from ..engine_adapters.radiance_inventory import Layout
from ..engine_adapters.radiance_resources import (
    MIB,
    HostLimits,
    ModelMemory,
    ResourceError,
    ResourcePlan,
    Targets,
)
from ..hardware.power_receipt import write_json_atomic
from .supervisor import ServeRefused, ServingPlan


class MeasuredPlanRefused(ServeRefused):
    """No measured, admitted v2 plan can safely be handed to the supervisor."""


RoomKind = Literal["host_room", "gpu_room", "pin_room"]

# Stable S5-to-S11 retryable resource classification and the exact knobs
# S11 can remeasure. A refusal claiming other knobs is not safe to retry.
_ROOM_RESOURCES = {
    "host_room": frozenset(("host_ram", "host_kv")),
    "gpu_room": frozenset(("vram",)),
    "pin_room": frozenset(("host_pool", "host_kv")),
}
_ROOM_DEFAULT_RESOURCE = {
    "host_room": "host_ram",
    "gpu_room": "vram",
    "pin_room": "host_pool",
}
_ROOM_CHANGES = {
    "host_room": ("host_ledger_margin", "host_kv"),
    "gpu_room": ("gpu_elastic_kv",),
    "pin_room": ("host_kv",),
}
S5_WIRE_PREFIX = "BC_SERVING_ADMISSION_REFUSAL="


@dataclass(frozen=True)
class RoomRefusal:
    """Strict S5 pre-weight wire format. Extra diagnostics are never parsed from text."""

    reason_code: RoomKind
    available_bytes: int
    required_bytes: int
    allocations_still_to_come_bytes: int
    headroom_bytes: int
    device_bdf: str | None = None
    resource: str | None = None
    component_breakdown: tuple[tuple[str, int], ...] = ()
    may_change: tuple[str, ...] = ()

    @property
    def shortfall_bytes(self) -> int:
        return self.required_bytes - self.available_bytes

    def to_record(self) -> dict:
        # Existing S11 fake constructors remain valid; the actual engine
        # supplies a named measured component for every allocation.
        components = self.component_breakdown or (
            ("allocation", self.allocations_still_to_come_bytes),
        )
        return {
            "schema_version": 1,
            "kind": "serving_admission_refusal",
            "phase": "before_weights",
            "reason_code": self.reason_code,
            "resource": self.resource or _ROOM_DEFAULT_RESOURCE[self.reason_code],
            "device_bdf": self.device_bdf,
            "available_bytes": self.available_bytes,
            "required_bytes": self.required_bytes,
            "shortfall_bytes": self.shortfall_bytes,
            "allocations_still_to_come_bytes": self.allocations_still_to_come_bytes,
            "headroom_bytes": self.headroom_bytes,
            "component_breakdown": dict(components),
            "may_change": list(self.may_change or _ROOM_CHANGES[self.reason_code]),
        }

    @classmethod
    def from_record(cls, raw: object) -> RoomRefusal:
        required = {
            "schema_version",
            "kind",
            "phase",
            "reason_code",
            "resource",
            "device_bdf",
            "available_bytes",
            "required_bytes",
            "shortfall_bytes",
            "allocations_still_to_come_bytes",
            "headroom_bytes",
            "component_breakdown",
            "may_change",
        }
        if not isinstance(raw, dict) or set(raw) != required:
            raise MeasuredPlanRefused("S5 refusal missing/unknown structured fields")
        if (
            type(raw["schema_version"]) is not int
            or raw["schema_version"] != 1
            or raw["kind"] != "serving_admission_refusal"
            or raw["phase"] != "before_weights"
        ):
            raise MeasuredPlanRefused("S5 refusal lacks v1 pre-weight envelope")
        kind = raw["reason_code"]
        if not isinstance(kind, str) or kind not in _ROOM_RESOURCES:
            raise MeasuredPlanRefused(f"unknown S5 admission reason {kind!r}")
        resource = raw["resource"]
        if not isinstance(resource, str) or resource not in _ROOM_RESOURCES[kind]:
            raise MeasuredPlanRefused("S5 resource is not retryable under its reason")
        bdf = raw["device_bdf"]
        if kind == "gpu_room":
            if not isinstance(bdf, str) or re.fullmatch(
                r"[0-9a-f]{4}:[0-9a-f]{2}:[0-1][0-9a-f]\.[0-7]", bdf
            ) is None:
                raise MeasuredPlanRefused("S5 GPU refusal needs a valid device BDF")
        elif bdf is not None:
            raise MeasuredPlanRefused("S5 host refusal must have a null device BDF")
        keys = (
            "available_bytes",
            "required_bytes",
            "shortfall_bytes",
            "allocations_still_to_come_bytes",
            "headroom_bytes",
        )
        if any(type(raw[k]) is not int or raw[k] < 0 for k in keys):
            raise MeasuredPlanRefused("S5 byte fields must be nonnegative integers")
        if raw["required_bytes"] != (
            raw["allocations_still_to_come_bytes"] + raw["headroom_bytes"]
        ):
            raise MeasuredPlanRefused(
                "S5 required bytes differ from remaining + headroom"
            )
        if raw["required_bytes"] <= raw["available_bytes"]:
            raise MeasuredPlanRefused("S5 refusal has no measured shortfall")
        if raw["shortfall_bytes"] != raw["required_bytes"] - raw["available_bytes"]:
            raise MeasuredPlanRefused("S5 shortfall disagrees with measured bytes")
        parts = raw["component_breakdown"]
        if (
            not isinstance(parts, dict)
            or not parts
            or any(
                not isinstance(k, str)
                or not k
                or type(v) is not int
                or v < 0
                for k, v in parts.items()
            )
            or sum(parts.values()) != raw["allocations_still_to_come_bytes"]
        ):
            raise MeasuredPlanRefused(
                "S5 component breakdown does not sum to remaining allocations"
            )
        changes = raw["may_change"]
        if (
            not isinstance(changes, list)
            or changes != list(_ROOM_CHANGES[kind])
        ):
            raise MeasuredPlanRefused("S5 may_change differs from S11 refit controls")
        return cls(
            kind,
            raw["available_bytes"],
            raw["required_bytes"],
            raw["allocations_still_to_come_bytes"],
            raw["headroom_bytes"],
            bdf,
            raw["resource"],
            tuple(parts.items()),
            tuple(changes),
        )


class EngineRoomRefused(Exception):
    """Raised ONLY for one authenticated S5 pre-weight exit-4 refusal record."""

    def __init__(self, room: RoomRefusal):
        self.room = room
        super().__init__(str(room.to_record()))

    @classmethod
    def from_process_exit(cls, exit_code: int, stderr: str) -> EngineRoomRefused:
        """S13 adapter entry: no substring/error-message inference or fallback."""
        if exit_code != 4:
            raise MeasuredPlanRefused(
                f"S5 structured refusal requires admission exit 4, got {exit_code}"
            )
        lines = [
            line[len(S5_WIRE_PREFIX):]
            for line in stderr.splitlines()
            if line.startswith(S5_WIRE_PREFIX)
        ]
        if len(lines) != 1:
            raise MeasuredPlanRefused(
                "S5 admission exit has no unique machine refusal (text-only rejected)"
            )
        try:
            record = json.loads(lines[0])
        except (ValueError, TypeError) as exc:
            raise MeasuredPlanRefused("S5 admission record is not JSON") from exc
        return cls(RoomRefusal.from_record(record))


@dataclass(frozen=True)
class GpuDemand:
    """Engine-measured GPU requirement and non-reclaimable floor (bytes)."""

    bdf: str
    required_bytes: int
    fixed_bytes: int


@dataclass(frozen=True)
class PlanBounds:
    """Refit constraints, all derived from the engine's measured shortage."""

    gpu_ceiling: tuple[tuple[str, int], ...] = ()
    retained_kv_tokens: int | None = None
    extra_host_margin_bytes: int = 0


@dataclass(frozen=True)
class PlanMeasurement:
    """S7/S8 handoff from bc-serve and engine --plan-memory; no estimates."""

    planning_key: str
    provenance: str
    model: ModelMemory
    targets: Targets
    host: HostLimits
    worker_footprints_bytes: dict[str, int]
    gpu: tuple[GpuDemand, ...]
    resource_inputs: dict | None = None
    scope: dict | None = None


@dataclass(frozen=True)
class AdmissionReport:
    """Actual engine startup admission figure, for the plan-cache receipt."""

    available_bytes: int
    required_bytes: int

    def __post_init__(self) -> None:
        if (
            type(self.available_bytes) is not int
            or self.available_bytes < 0
            or type(self.required_bytes) is not int
            or self.required_bytes < 0
            or self.required_bytes > self.available_bytes
        ):
            raise MeasuredPlanRefused("S5 success has invalid room measurements")


class MeasurementSource(Protocol):
    def measure(self, layout: Layout, bounds: PlanBounds) -> PlanMeasurement: ...


class ManifestWriter(Protocol):
    def write(
        self,
        layout: Layout,
        run_id: str,
        attempt: int,
        measurement: PlanMeasurement,
        resources: ResourcePlan,
        bounds: PlanBounds,
    ) -> ServingPlan: ...


class EngineValidator(Protocol):
    def validate(
        self,
        layout: Layout,
        plan: ServingPlan,
        measurement: PlanMeasurement,
        resources: ResourcePlan,
    ) -> AdmissionReport: ...


class MeasuredPlanValidator:
    """Real S11 implementation of Supervisor.plan(layout, run_id).

    Source must remeasure with GPU ceilings (never just relabel an old plan).
    Writer must emit a fresh strictly checked v2 manifest for each candidate,
    with changed budget/calibration and unchanged planning key.
    Validator must use the S5 pre-weight structured interface, with no live
    weights; generic process failures cannot trigger a refit.
    Exactly two attempts, as required by RSC10 and RSC11's admission gate.
    """

    def __init__(
        self,
        *,
        source: MeasurementSource,
        writer: ManifestWriter,
        validator: EngineValidator,
        record_dir: Path,
    ):
        self.source = source
        self.writer = writer
        self.validator = validator
        self.record_dir = Path(record_dir)

    @staticmethod
    def _measured(layout: Layout, m: PlanMeasurement, bounds: PlanBounds) -> None:
        if (
            not isinstance(m.planning_key, str)
            or len(m.planning_key) != 64
            or any(c not in "0123456789abcdef" for c in m.planning_key)
            or not isinstance(m.provenance, str)
            or not m.provenance.strip()
        ):
            raise MeasuredPlanRefused("measurement lacks key/provenance")
        if tuple(g.bdf for g in m.gpu) != layout.engine_bdfs:
            raise MeasuredPlanRefused("measured GPU BDFs do not match bc-serve layout")
        for g in m.gpu:
            if (
                type(g.required_bytes) is not int
                or type(g.fixed_bytes) is not int
                or g.required_bytes < 0
                or g.fixed_bytes < 0
                or g.fixed_bytes > g.required_bytes
            ):
                raise MeasuredPlanRefused(f"{g.bdf}: malformed measured GPU footprint")
        ceilings = dict(bounds.gpu_ceiling)
        if len(ceilings) != len(bounds.gpu_ceiling):
            raise MeasuredPlanRefused("duplicate GPU ceiling")
        for bdf, ceiling in ceilings.items():
            if bdf not in layout.engine_bdfs or type(ceiling) is not int or ceiling < 0:
                raise MeasuredPlanRefused(f"{bdf}: invalid measured GPU ceiling")
            gpu = next(g for g in m.gpu if g.bdf == bdf)
            if gpu.required_bytes > ceiling:
                raise MeasuredPlanRefused(
                    f"{bdf}: engine failed to replan under measured ceiling "
                    f"({gpu.required_bytes} required, {ceiling} claimable); "
                    f"free at least {gpu.required_bytes - ceiling} GPU bytes"
                )

    @staticmethod
    def _fingerprint(m: PlanMeasurement, targets: Targets, r: ResourcePlan) -> tuple:
        return (
            m.model,
            targets,
            r.host_pool_mib,
            r.host_kv_mib,
            r.host_committed_bytes,
            r.pin_committed_bytes,
            tuple((g.bdf, g.required_bytes, g.fixed_bytes) for g in m.gpu),
        )

    @staticmethod
    def _refit(m: PlanMeasurement, b: PlanBounds, room: RoomRefusal) -> PlanBounds:
        short = room.shortfall_bytes
        if room.reason_code == "gpu_room":
            g = next((g for g in m.gpu if g.bdf == room.device_bdf), None)
            if g is None:
                raise MeasuredPlanRefused("S5 GPU refusal targets a nonselected BDF")
            if g.required_bytes <= room.available_bytes:
                raise MeasuredPlanRefused(
                    "S5 GPU refusal disagrees with measured GPU plan"
                )
            if g.fixed_bytes > room.available_bytes:
                raise MeasuredPlanRefused(
                    f"{g.bdf}: fixed GPU demand {g.fixed_bytes} exceeds claimable "
                    f"{room.available_bytes}; free at least "
                    f"{g.fixed_bytes - room.available_bytes} GPU bytes"
                )
            return replace(b, gpu_ceiling=((g.bdf, room.available_bytes),))

        # A host shortage raises calibrated reserve and releases real KV
        # tokens. A pin shortage releases tokens without raising RAM reserve.
        tokens = (
            m.targets.retained_kv_tokens
            if b.retained_kv_tokens is None
            else b.retained_kv_tokens
        )
        kv_unit = m.model.kv_bytes_per_token
        if type(kv_unit) is not int or kv_unit <= 0:
            raise MeasuredPlanRefused("missing measured KV bytes/token")

        allocated = ((tokens * kv_unit + MIB - 1) // MIB) * MIB
        release = ((short + MIB - 1) // MIB) * MIB
        drop = tokens - min(tokens, (allocated - release) // kv_unit)
        if release > allocated:
            domain = "pinned RAM" if room.reason_code == "pin_room" else "host RAM"
            raise MeasuredPlanRefused(
                f"{room.reason_code}: cannot release {short} bytes from "
                f"{allocated} allocated KV bytes; free at least "
                f"{short - allocated} bytes of {domain}"
            )
        return replace(
            b,
            retained_kv_tokens=tokens - drop,
            extra_host_margin_bytes=(
                b.extra_host_margin_bytes + short
                if room.reason_code == "host_room"
                else b.extra_host_margin_bytes
            ),
        )

    def __call__(self, layout: Layout, run_id: str) -> ServingPlan:
        bounds = PlanBounds()
        key: str | None = None
        owner: str | None = None
        previous: tuple | None = None
        attempts: list[dict] = []
        refusals: list[RoomRefusal] = []
        receipt = self.record_dir / f"{run_id}.plan.json"
        if receipt.exists():
            raise MeasuredPlanRefused(f"measured-plan record already exists: {receipt}")

        def persist(state: str) -> None:
            write_json_atomic(
                receipt,
                {
                    "schema_version": 1,
                    "run_id": run_id,
                    "state": state,
                    "planning_key": key,
                    "owner_map_digest": owner,
                    "attempts": attempts,
                },
            )

        try:
            for attempt_index in range(2):
                m = self.source.measure(layout, bounds)
                self._measured(layout, m, bounds)
                if key is not None and (
                    key != m.planning_key or owner != m.model.owner_map_digest
                ):
                    raise MeasuredPlanRefused(
                        "planning key/owner map changed during calibration-only refit"
                    )
                key, owner = m.planning_key, m.model.owner_map_digest
                targets = replace(
                    m.targets,
                    retained_kv_tokens=(
                        m.targets.retained_kv_tokens
                        if bounds.retained_kv_tokens is None
                        else bounds.retained_kv_tokens
                    ),
                    host_reserve_bytes=(
                        m.targets.host_reserve_bytes + bounds.extra_host_margin_bytes
                    ),
                )
                r = layout.derive_resources(
                    m.model,
                    targets,
                    m.host,
                    worker_footprints_bytes=m.worker_footprints_bytes,
                )
                fingerprint = self._fingerprint(m, targets, r)
                if previous == fingerprint:
                    raise MeasuredPlanRefused(
                        "no progress: remeasurement produced the same resource plan"
                    )
                previous = fingerprint
                candidate = self.writer.write(
                    layout,
                    run_id,
                    attempt_index,
                    replace(m, targets=targets),
                    r,
                    bounds,
                )
                entry = {
                    "attempt": attempt_index + 1,
                    "manifest": str(candidate.manifest),
                    "provenance": m.provenance,
                    "gpu": [vars(g) for g in m.gpu],
                    "host": r.explain(),
                    "bounds": {
                        "gpu_ceiling": dict(bounds.gpu_ceiling),
                        "retained_kv_tokens": targets.retained_kv_tokens,
                        "host_margin_increase_bytes": bounds.extra_host_margin_bytes,
                    },
                }
                attempts.append(entry)
                persist("validating")
                try:
                    admitted = self.validator.validate(
                        layout, candidate, replace(m, targets=targets), r
                    )
                except EngineRoomRefused as exc:
                    room = RoomRefusal.from_record(exc.room.to_record())
                    refusals.append(room)
                    entry["refusal"] = {
                        **room.to_record(),
                        "shortfall_bytes": room.shortfall_bytes,
                    }
                    persist("refused")
                    if attempt_index:
                        first, second = refusals
                        domain = (
                            "GPU VRAM"
                            if room.reason_code == "gpu_room"
                            else "pinned RAM"
                            if room.reason_code == "pin_room"
                            else "host RAM"
                        )
                        raise MeasuredPlanRefused(
                            "second structured admission refusal: first "
                            f"available={first.available_bytes} required={first.required_bytes}; "
                            f"second available={second.available_bytes} "
                            f"required={second.required_bytes}; free at least "
                            f"{second.shortfall_bytes} bytes of {domain}"
                        ) from exc
                    bounds = self._refit(m, bounds, room)
                    continue
                if not isinstance(admitted, AdmissionReport):
                    raise MeasuredPlanRefused(
                        "validator provided no measured admission report"
                    )
                entry["admitted"] = {
                    "available_bytes": admitted.available_bytes,
                    "required_bytes": admitted.required_bytes,
                    "spare_bytes": admitted.available_bytes - admitted.required_bytes,
                }
                persist("admitted")
                return candidate
            raise MeasuredPlanRefused("no measured admission result")
        except Exception as exc:
            attempts.append({"terminal_refusal": str(exc)})
            persist("terminal_refusal")
            if isinstance(exc, MeasuredPlanRefused):
                raise
            if isinstance(exc, (ResourceError, ValueError, OSError)):
                raise MeasuredPlanRefused(str(exc)) from exc
            raise

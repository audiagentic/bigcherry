"""Capability-derived acceptance matrix for job-service cutover.

RCD10 acceptance is generated from portable BatchSpecs plus accepted hardware,
never from physical slot numbers.  Unsupported cases are retained in the
matrix with an explicit reason so absent hardware cannot be mistaken for an
accepted capability.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Iterable, Mapping

from bigcherry.hardware.inventory import HardwareBindingError, bind_gpu_requirement
from bigcherry.hardware.model import HardwareInventory

from .model import BatchSpec, GpuRequirement, digest


@dataclass(frozen=True)
class AcceptanceCase:
    case_id: str
    executor_id: str
    host_id: str | None
    platform_family: str | None
    patch: str
    architecture: str
    model: str
    producer: str | None
    hip_path: str | None
    gpu_count: int
    min_vram_bytes: int
    require_peer_access: bool
    production_lane: bool
    common_patches: tuple[str, ...]
    supported: bool
    reason: str | None
    selected_device_ids: tuple[str, ...] = ()
    hardware_cohort_hash: str | None = None
    accepted_inventory_hash: str | None = None

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def _case_material(batch: BatchSpec, architecture: str) -> dict[str, object]:
    return {
        "executor_id": batch.target.executor_id,
        "host_id": batch.target.host_id,
        "platform_family": batch.target.platform_family,
        "patch": batch.patch,
        "architecture": architecture,
        "model": batch.model,
        "producer": batch.producer,
        "hip_path": batch.hip_path,
        "gpu_count": batch.gpu_count,
        "min_vram_bytes": batch.min_vram_bytes,
        "require_peer_access": batch.require_peer_access,
        "production_lane": batch.production_lane,
        "common_patches": tuple(sorted(batch.common_patches)),
    }


def acceptance_matrix(
    batches: Iterable[BatchSpec],
    *,
    inventories: Mapping[str, HardwareInventory],
) -> tuple[AcceptanceCase, ...]:
    """Return the deterministic capability matrix for a proposed cutover.

    One case is emitted for every distinct active capability combination.
    Hardware binding is attempted against the target executor's accepted
    inventory. Missing executors, target host/platform mismatches and
    unsatisfied GPU requirements remain as ``supported=False`` cases instead
    of being dropped.
    """
    by_id: dict[str, AcceptanceCase] = {}
    for batch in batches:
        for architecture in batch.architectures:
            material = _case_material(batch, architecture)
            case_id = "a-" + digest(material, person=b"bc-accept-case")
            if case_id in by_id:
                continue
            inventory = inventories.get(batch.target.executor_id)
            reason: str | None = None
            selected: tuple[str, ...] = ()
            cohort: str | None = None
            accepted_hash: str | None = None
            host_id = batch.target.host_id
            platform_family = batch.target.platform_family
            if inventory is None:
                reason = f"no accepted inventory for executor {batch.target.executor_id!r}"
            else:
                host_id = inventory.host_id
                platform_family = inventory.platform_family
                if batch.target.host_id and batch.target.host_id != inventory.host_id:
                    reason = (
                        f"target host {batch.target.host_id!r} does not match accepted "
                        f"host {inventory.host_id!r}"
                    )
                elif (
                    batch.target.platform_family
                    and batch.target.platform_family != inventory.platform_family
                ):
                    reason = (
                        f"target platform {batch.target.platform_family!r} does not match "
                        f"accepted platform {inventory.platform_family!r}"
                    )
                else:
                    requirement = GpuRequirement(
                        architecture=architecture,
                        count=batch.gpu_count,
                        min_vram_bytes=batch.min_vram_bytes,
                        require_peer_access=batch.require_peer_access,
                    )
                    try:
                        binding = bind_gpu_requirement(requirement, inventory)
                    except HardwareBindingError as exc:
                        reason = str(exc)
                    else:
                        selected = binding.selected_device_ids
                        cohort = binding.hardware_cohort_hash
                        accepted_hash = binding.accepted_inventory_hash
            by_id[case_id] = AcceptanceCase(
                case_id=case_id,
                executor_id=batch.target.executor_id,
                host_id=host_id,
                platform_family=platform_family,
                patch=batch.patch,
                architecture=architecture,
                model=batch.model,
                producer=batch.producer,
                hip_path=batch.hip_path,
                gpu_count=batch.gpu_count,
                min_vram_bytes=batch.min_vram_bytes,
                require_peer_access=batch.require_peer_access,
                production_lane=batch.production_lane,
                common_patches=tuple(sorted(batch.common_patches)),
                supported=reason is None,
                reason=reason,
                selected_device_ids=selected,
                hardware_cohort_hash=cohort,
                accepted_inventory_hash=accepted_hash,
            )
    return tuple(by_id[key] for key in sorted(by_id))


def acceptance_document(cases: Iterable[AcceptanceCase]) -> dict[str, object]:
    rows = tuple(cases)
    return {
        "schema": "bigcherry.jobs.acceptance-matrix.v1",
        "cases": [case.to_dict() for case in rows],
        "supported": sum(1 for case in rows if case.supported),
        "unsupported": sum(1 for case in rows if not case.supported),
        "ready": bool(rows) and all(case.supported for case in rows),
    }

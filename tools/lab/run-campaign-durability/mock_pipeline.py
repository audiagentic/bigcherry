"""RCD planning simulator.

Pure-Python falsification harness for the JOBS_ORCHESTRATOR design. This is
plan-validation code under tools/lab, not the production jobs implementation.
It deliberately has no Slurm/ROCm/llama-swap imports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True)
class DeviceRecord:
    device_id: str
    identity_source: str
    arch: str
    model: str
    vram_bytes: int
    pci_bdf: str | None
    render_node: str | None
    numa_node: int | None
    driver_version: str


@dataclass(frozen=True)
class GpuRequirement:
    architecture: str
    count: int = 1
    min_vram_bytes: int = 0
    homogeneous_model: bool = True
    require_peer_access: bool = False
    exact_device_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProductionSnapshot:
    config_hash: str
    potential_devices: frozenset[str] | None  # None == all/fail-closed
    running_devices: frozenset[str]
    observed_devices: frozenset[str]


@dataclass(frozen=True)
class Allocation:
    device_ids: tuple[str, ...]


@dataclass(frozen=True)
class ExecutionEnvironment:
    os_family: str
    os_version: str
    kernel_or_build: str
    runtime_family: str
    runtime_version: str
    hip_runtime_version: str
    compiler: str
    driver_version: str


class GateMode(str, Enum):
    EXCLUSIVE_WINDOW = "exclusive-production-window"
    IDLE_ATTESTATION = "production-idle-attestation"


class RetryAction(str, Enum):
    COMPLETE = "complete"
    SAME_COMMIT_REQUEUE = "same-commit-requeue"
    NEW_ATTEMPT = "new-attempt"
    BLOCK = "block"


@dataclass
class FakeExecution:
    native_id: str
    execution_id: str
    dependencies: tuple[str, ...]
    state: str = "queued"


class FakeExecutor:
    """Minimal scheduler double: dependency readiness and durable native handles."""

    def __init__(self) -> None:
        self._counter = 0
        self._runs: dict[str, FakeExecution] = {}

    def submit(self, execution_id: str, dependencies: tuple[str, ...] = ()) -> str:
        self._counter += 1
        native_id = f"fake-{self._counter:04d}"
        self._runs[native_id] = FakeExecution(native_id, execution_id, dependencies)
        return native_id

    def start_ready(self) -> list[str]:
        started: list[str] = []
        completed_execution_ids = {
            run.execution_id for run in self._runs.values() if run.state == "completed"
        }
        for run in self._runs.values():
            if run.state == "queued" and set(run.dependencies) <= completed_execution_ids:
                run.state = "running"
                started.append(run.native_id)
        return started

    def complete(self, native_id: str) -> None:
        assert self._runs[native_id].state == "running"
        self._runs[native_id].state = "completed"

    def status(self, native_id: str) -> str:
        return self._runs[native_id].state


def resolve_requirement(
    requirement: GpuRequirement,
    devices: Iterable[DeviceRecord],
    *,
    peer_pairs: set[frozenset[str]] | None = None,
) -> tuple[DeviceRecord, ...]:
    device_list = tuple(devices)
    by_id = {device.device_id: device for device in device_list}

    if requirement.exact_device_ids:
        try:
            candidates = tuple(by_id[item] for item in requirement.exact_device_ids)
        except KeyError as exc:
            raise ValueError(f"exact device missing: {exc.args[0]}") from exc
    else:
        candidates = tuple(
            device
            for device in device_list
            if device.arch == requirement.architecture
            and device.vram_bytes >= requirement.min_vram_bytes
        )

    if any(device.arch != requirement.architecture for device in candidates):
        raise ValueError("exact device violates architecture")
    if any(device.vram_bytes < requirement.min_vram_bytes for device in candidates):
        raise ValueError("device violates minimum VRAM")

    if requirement.homogeneous_model and candidates:
        model = candidates[0].model
        candidates = tuple(device for device in candidates if device.model == model)

    if len(candidates) < requirement.count:
        raise ValueError("insufficient matching GPUs")

    if requirement.require_peer_access:
        if requirement.count != 2:
            raise ValueError("planning simulator currently models peer pairs only")
        valid_pairs = peer_pairs or set()
        for index, left in enumerate(candidates):
            for right in candidates[index + 1 :]:
                if frozenset((left.device_id, right.device_id)) in valid_pairs:
                    return (left, right)
        raise ValueError("no peer-capable pair satisfies requirement")

    return candidates[: requirement.count]


def requires_all_of_arch_reservation(
    requirement: GpuRequirement, all_devices: Iterable[DeviceRecord]
) -> bool:
    """Whether Slurm arch/count cannot safely encode the BigCherry subset constraint."""
    same_arch = tuple(device for device in all_devices if device.arch == requirement.architecture)
    if requirement.exact_device_ids:
        return set(requirement.exact_device_ids) != {device.device_id for device in same_arch}
    if any(device.vram_bytes < requirement.min_vram_bytes for device in same_arch):
        return True
    if requirement.require_peer_access:
        return True
    if requirement.homogeneous_model and len({device.model for device in same_arch}) > 1:
        return True
    return False


def production_gate(snapshot: ProductionSnapshot, allocation: Allocation) -> GateMode:
    if snapshot.potential_devices is None:
        return GateMode.EXCLUSIVE_WINDOW
    if set(allocation.device_ids) & set(snapshot.potential_devices):
        return GateMode.EXCLUSIVE_WINDOW
    return GateMode.IDLE_ATTESTATION


def platform_environment_hash(environment: ExecutionEnvironment) -> str:
    return _digest(asdict(environment))


def hardware_cohort_hash(
    devices: Iterable[DeviceRecord],
    *,
    topology_fingerprint: str,
) -> str:
    identity = [
        {
            "device_id": device.device_id,
            "identity_source": device.identity_source,
            "arch": device.arch,
            "model": device.model,
            "vram_bytes": device.vram_bytes,
        }
        for device in sorted(devices, key=lambda item: item.device_id)
    ]
    return _digest({"devices": identity, "topology_fingerprint": topology_fingerprint})


def inventory_material_hash(devices: Iterable[DeviceRecord], *, topology_fingerprint: str) -> str:
    return _digest(
        {
            "devices": [asdict(item) for item in sorted(devices, key=lambda item: item.device_id)],
            "topology_fingerprint": topology_fingerprint,
        }
    )


def retry_action(exit_code: int) -> RetryAction:
    return {
        0: RetryAction.COMPLETE,
        75: RetryAction.SAME_COMMIT_REQUEUE,
        76: RetryAction.NEW_ATTEMPT,
        77: RetryAction.BLOCK,
    }.get(exit_code, RetryAction.NEW_ATTEMPT)


def _fixture_devices() -> tuple[DeviceRecord, ...]:
    gib = 1024**3
    return (
        DeviceRecord("gpu-A", "amd_uuid", "gfx1100", "XTX", 24 * gib, "0000:01:00.0", "/dev/dri/renderD128", 0, "drv"),
        DeviceRecord("gpu-B", "amd_uuid", "gfx1100", "XTX", 24 * gib, "0000:02:00.0", "/dev/dri/renderD129", 0, "drv"),
        DeviceRecord("gpu-C", "amd_uuid", "gfx1201", "R9700", 32 * gib, "0000:03:00.0", "/dev/dri/renderD130", 0, "drv"),
        DeviceRecord("gpu-D", "amd_uuid", "gfx1030", "6800", 16 * gib, "0000:04:00.0", "/dev/dri/renderD131", 0, "drv"),
    )


def self_test() -> None:
    devices = _fixture_devices()
    gib = 1024**3

    one = resolve_requirement(GpuRequirement("gfx1201", min_vram_bytes=30 * gib), devices)
    assert tuple(device.device_id for device in one) == ("gpu-C",)

    pair = resolve_requirement(
        GpuRequirement("gfx1100", count=2, require_peer_access=True),
        devices,
        peer_pairs={frozenset(("gpu-A", "gpu-B"))},
    )
    assert {device.device_id for device in pair} == {"gpu-A", "gpu-B"}
    assert requires_all_of_arch_reservation(
        GpuRequirement("gfx1100", count=2, require_peer_access=True), devices
    )

    exact = resolve_requirement(
        GpuRequirement("gfx1100", exact_device_ids=("gpu-B",)), devices
    )
    assert exact[0].device_id == "gpu-B"

    no_conflict = ProductionSnapshot("cfg1", frozenset({"gpu-A"}), frozenset(), frozenset())
    assert production_gate(no_conflict, Allocation(("gpu-C",))) == GateMode.IDLE_ATTESTATION
    assert production_gate(no_conflict, Allocation(("gpu-A",))) == GateMode.EXCLUSIVE_WINDOW
    assert (
        production_gate(
            ProductionSnapshot("cfg1", None, frozenset(), frozenset()),
            Allocation(("gpu-C",)),
        )
        == GateMode.EXCLUSIVE_WINDOW
    )

    linux = ExecutionEnvironment("linux", "ubuntu-24.04", "6.x", "rocm-linux", "7.2", "7.2", "clang", "drv")
    windows = ExecutionEnvironment("windows", "11", "26100", "hip-sdk-windows", "7.2", "7.2", "clang-cl", "drv")
    assert platform_environment_hash(linux) != platform_environment_hash(windows)

    cohort = hardware_cohort_hash(devices[:2], topology_fingerprint="peer:A-B")
    moved = (
        DeviceRecord("gpu-A", "amd_uuid", "gfx1100", "XTX", 24 * gib, "0000:09:00.0", "/dev/dri/renderD140", 1, "drv"),
        devices[1],
    )
    assert hardware_cohort_hash(moved, topology_fingerprint="peer:A-B") == cohort
    assert inventory_material_hash(moved, topology_fingerprint="peer:A-B") != inventory_material_hash(
        devices[:2], topology_fingerprint="peer:A-B"
    )
    assert hardware_cohort_hash(moved, topology_fingerprint="peer:A-B:numa1") != cohort
    replacement = (
        DeviceRecord("gpu-X", "amd_uuid", "gfx1100", "XTX", 24 * gib, "0000:01:00.0", "/dev/dri/renderD128", 0, "drv"),
        devices[1],
    )
    assert hardware_cohort_hash(replacement, topology_fingerprint="peer:A-B") != cohort

    assert retry_action(75) == RetryAction.SAME_COMMIT_REQUEUE
    assert retry_action(76) == RetryAction.NEW_ATTEMPT
    assert retry_action(77) == RetryAction.BLOCK

    executor = FakeExecutor()
    prepare = executor.submit("prepare")
    execute = executor.submit("execute", ("prepare",))
    assert executor.start_ready() == [prepare]
    assert executor.status(execute) == "queued"
    executor.complete(prepare)
    assert executor.start_ready() == [execute]
    executor.complete(execute)
    assert executor.status(execute) == "completed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print(json.dumps({"ok": True, "checks": 16}, sort_keys=True))
        return 0
    parser.error("--self-test is required for this planning harness")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

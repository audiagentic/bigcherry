"""RCD planning simulator.

Pure-Python falsification harness for JOBS_ORCHESTRATOR. Lab-only: no Slurm,
ROCm, llama-swap, repository, or evidence authority imports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from itertools import combinations
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
    model: str | None = None
    require_peer_access: bool = False
    exact_device_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SeriesGpuBinding:
    selected_device_ids: tuple[str, ...]
    reserve_all_of_arch: bool
    hardware_cohort_hash: str


@dataclass(frozen=True)
class ProductionSnapshot:
    config_hash: str
    potential_devices: frozenset[str] | None  # None == ALL/fail-closed
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


class DriftKind(str, Enum):
    SAME = "same"
    LOCATOR_ONLY = "locator-only"
    TOPOLOGY_CHANGE = "topology-change"
    DEVICE_SET_CHANGE = "device-set-change"


@dataclass
class FakeExecution:
    native_id: str
    execution_id: str
    dependencies: tuple[str, ...]
    state: str = "queued"


class FakeExecutor:
    def __init__(self) -> None:
        self._counter = 0
        self._runs: dict[str, FakeExecution] = {}

    def submit(self, execution_id: str, dependencies: tuple[str, ...] = ()) -> str:
        self._counter += 1
        native_id = f"fake-{self._counter:04d}"
        self._runs[native_id] = FakeExecution(native_id, execution_id, dependencies)
        return native_id

    def start_ready(self) -> list[str]:
        complete = {run.execution_id for run in self._runs.values() if run.state == "completed"}
        started: list[str] = []
        for run in self._runs.values():
            if run.state == "queued" and set(run.dependencies) <= complete:
                run.state = "running"
                started.append(run.native_id)
        return started

    def complete(self, native_id: str) -> None:
        assert self._runs[native_id].state == "running"
        self._runs[native_id].state = "completed"

    def status(self, native_id: str) -> str:
        return self._runs[native_id].state


def _eligible(req: GpuRequirement, devices: Iterable[DeviceRecord]) -> tuple[DeviceRecord, ...]:
    by_id = {device.device_id: device for device in devices}
    if req.exact_device_ids:
        try:
            candidates = tuple(by_id[item] for item in req.exact_device_ids)
        except KeyError as exc:
            raise ValueError(f"exact device missing: {exc.args[0]}") from exc
    else:
        candidates = tuple(
            device for device in by_id.values()
            if device.arch == req.architecture
            and device.vram_bytes >= req.min_vram_bytes
            and (req.model is None or device.model == req.model)
        )
    candidates = tuple(sorted(candidates, key=lambda device: device.device_id))
    if any(device.arch != req.architecture for device in candidates):
        raise ValueError("exact device violates architecture")
    if any(device.vram_bytes < req.min_vram_bytes for device in candidates):
        raise ValueError("device violates minimum VRAM")
    if req.model is not None and any(device.model != req.model for device in candidates):
        raise ValueError("device violates model requirement")
    if req.homogeneous_model and req.model is None and not req.exact_device_ids:
        groups: dict[str, list[DeviceRecord]] = {}
        for device in candidates:
            groups.setdefault(device.model, []).append(device)
        viable = [tuple(group) for _, group in sorted(groups.items()) if len(group) >= req.count]
        if len(viable) > 1:
            raise ValueError("ambiguous homogeneous model; specify model or exact ids")
        if viable:
            candidates = tuple(sorted(viable[0], key=lambda device: device.device_id))
    if len(candidates) < req.count:
        raise ValueError("insufficient matching GPUs")
    return candidates


def resolve_requirement(
    req: GpuRequirement,
    devices: Iterable[DeviceRecord],
    *,
    peer_pairs: set[frozenset[str]] | None = None,
) -> tuple[DeviceRecord, ...]:
    candidates = _eligible(req, devices)
    if not req.require_peer_access:
        return candidates[: req.count]
    peers = peer_pairs or set()
    options: list[tuple[DeviceRecord, ...]] = []
    for choice in combinations(candidates, req.count):
        ids = [device.device_id for device in choice]
        if all(frozenset(pair) in peers for pair in combinations(ids, 2)):
            options.append(choice)
    if not options:
        raise ValueError("no peer-capable cohort satisfies requirement")
    return min(options, key=lambda choice: tuple(device.device_id for device in choice))


def hardware_cohort_hash(devices: Iterable[DeviceRecord], *, topology_fingerprint: str) -> str:
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


def bind_series_gpu(
    req: GpuRequirement,
    devices: Iterable[DeviceRecord],
    *,
    topology_fingerprint: str,
    peer_pairs: set[frozenset[str]] | None = None,
) -> SeriesGpuBinding:
    all_devices = tuple(devices)
    selected = resolve_requirement(req, all_devices, peer_pairs=peer_pairs)
    all_arch_ids = {device.device_id for device in all_devices if device.arch == req.architecture}
    selected_ids = tuple(device.device_id for device in selected)
    # Architecture/count GRES cannot guarantee which card is returned. Once a
    # scientific series is bound to exact stable IDs, any proper subset of an
    # architecture must reserve the whole architecture pool in v1 and narrow
    # visibility inside that exclusive allocation.
    reserve_all = set(selected_ids) != all_arch_ids
    return SeriesGpuBinding(
        selected_ids,
        reserve_all,
        hardware_cohort_hash(selected, topology_fingerprint=topology_fingerprint),
    )


def verify_series_allocation(binding: SeriesGpuBinding, allocation: Allocation) -> tuple[str, ...]:
    selected = set(binding.selected_device_ids)
    if not selected or not selected <= set(allocation.device_ids):
        raise ValueError("series-bound devices are not all inside allocation")
    return binding.selected_device_ids


def parse_production_claim(claim: str | None, devices: Iterable[DeviceRecord]) -> frozenset[str] | None:
    if claim is None or not claim.strip() or claim.strip() == "all":
        return None
    text = claim.strip()
    by_id = {device.device_id: device for device in devices}
    if text.startswith("uuid:"):
        ids: list[str] = []
        for part in (part.strip() for part in text.split(",")):
            if not part.startswith("uuid:") or not part[5:] or part[5:] not in by_id:
                return None
            ids.append(part[5:])
        return frozenset(ids)
    if text.startswith("arch:"):
        fields = [part.strip() for part in text.split(",")]
        arch = fields[0][5:]
        count = None
        for part in fields[1:]:
            key, sep, value = part.partition("=")
            if key == "count" and sep and value.isdigit():
                count = int(value)
            else:
                return None
        candidates = sorted(device.device_id for device in by_id.values() if device.arch == arch)
        if not arch or count is None or count < 1 or len(candidates) < count:
            return None
        return frozenset(candidates)
    return None


def production_gate(snapshot: ProductionSnapshot, allocation: Allocation) -> GateMode:
    if snapshot.potential_devices is None:
        return GateMode.EXCLUSIVE_WINDOW
    if set(allocation.device_ids) & set(snapshot.potential_devices):
        return GateMode.EXCLUSIVE_WINDOW
    return GateMode.IDLE_ATTESTATION


def classify_inventory_drift(
    accepted: Iterable[DeviceRecord],
    observed: Iterable[DeviceRecord],
    *,
    accepted_topology: str,
    observed_topology: str,
) -> DriftKind:
    old = {device.device_id: device for device in accepted}
    new = {device.device_id: device for device in observed}
    if set(old) != set(new):
        return DriftKind.DEVICE_SET_CHANGE
    if accepted_topology != observed_topology:
        return DriftKind.TOPOLOGY_CHANGE
    locator_fields = ("pci_bdf", "render_node", "numa_node")
    if any(
        any(getattr(old[key], field) != getattr(new[key], field) for field in locator_fields)
        for key in old
    ):
        return DriftKind.LOCATOR_ONLY
    return DriftKind.SAME


def inventory_material_hash(devices: Iterable[DeviceRecord], *, topology_fingerprint: str) -> str:
    return _digest({
        "devices": [asdict(device) for device in sorted(devices, key=lambda item: item.device_id)],
        "topology_fingerprint": topology_fingerprint,
    })


def platform_environment_hash(environment: ExecutionEnvironment) -> str:
    return _digest(asdict(environment))


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
        DeviceRecord("gpu-B", "amd_uuid", "gfx1100", "XTX", 24*gib, "0000:02:00.0", "/dev/dri/renderD129", 0, "drv"),
        DeviceRecord("gpu-A", "amd_uuid", "gfx1100", "XTX", 24*gib, "0000:01:00.0", "/dev/dri/renderD128", 0, "drv"),
        DeviceRecord("gpu-C", "amd_uuid", "gfx1201", "R9700", 32*gib, "0000:03:00.0", "/dev/dri/renderD130", 0, "drv"),
        DeviceRecord("gpu-D", "amd_uuid", "gfx1030", "6800", 16*gib, "0000:04:00.0", "/dev/dri/renderD131", 0, "drv"),
    )


def self_test() -> int:
    checks = 0

    def check(value: bool) -> None:
        nonlocal checks
        assert value
        checks += 1

    devices = _fixture_devices()
    gib = 1024**3
    one = resolve_requirement(GpuRequirement("gfx1201", min_vram_bytes=30*gib), devices)
    check(tuple(device.device_id for device in one) == ("gpu-C",))

    peers = {frozenset(("gpu-A", "gpu-B"))}
    pair = resolve_requirement(
        GpuRequirement("gfx1100", count=2, require_peer_access=True), devices, peer_pairs=peers
    )
    check(tuple(device.device_id for device in pair) == ("gpu-A", "gpu-B"))

    b1 = bind_series_gpu(GpuRequirement("gfx1100"), devices[:2], topology_fingerprint="peer:A-B")
    b2 = bind_series_gpu(GpuRequirement("gfx1100"), reversed(devices[:2]), topology_fingerprint="peer:A-B")
    check(b1 == b2)
    check(b1.selected_device_ids == ("gpu-A",))
    check(b1.reserve_all_of_arch is True)
    check(verify_series_allocation(b1, Allocation(("gpu-A", "gpu-B"))) == ("gpu-A",))
    try:
        verify_series_allocation(b1, Allocation(("gpu-B",)))
    except ValueError:
        checks += 1
    else:
        raise AssertionError("series rebound to different card")

    whole = bind_series_gpu(
        GpuRequirement("gfx1100", count=2, require_peer_access=True),
        devices[:2], topology_fingerprint="peer:A-B", peer_pairs=peers,
    )
    check(whole.reserve_all_of_arch is False)

    mixed = devices + (
        DeviceRecord("gpu-E", "amd_uuid", "gfx1100", "Other", 24*gib, None, None, None, "drv"),
    )
    try:
        bind_series_gpu(GpuRequirement("gfx1100"), mixed, topology_fingerprint="t")
    except ValueError as exc:
        check("ambiguous" in str(exc))
    else:
        raise AssertionError("mixed-model ambiguity silently selected a model")
    check(
        bind_series_gpu(GpuRequirement("gfx1100", model="XTX"), mixed, topology_fingerprint="t")
        .selected_device_ids == ("gpu-A",)
    )

    snapshot = ProductionSnapshot("cfg", frozenset({"gpu-A"}), frozenset(), frozenset())
    check(production_gate(snapshot, Allocation(("gpu-C",))) == GateMode.IDLE_ATTESTATION)
    check(production_gate(snapshot, Allocation(("gpu-A",))) == GateMode.EXCLUSIVE_WINDOW)
    check(
        production_gate(ProductionSnapshot("cfg", None, frozenset(), frozenset()), Allocation(("gpu-C",)))
        == GateMode.EXCLUSIVE_WINDOW
    )
    check(parse_production_claim("uuid:gpu-A,uuid:gpu-B", devices) == frozenset({"gpu-A", "gpu-B"}))
    check(parse_production_claim("arch:gfx1100,count=2", devices) == frozenset({"gpu-A", "gpu-B"}))
    check(parse_production_claim("uuid:missing", devices) is None)
    check(parse_production_claim(None, devices) is None)

    linux = ExecutionEnvironment("linux", "ubuntu-24.04", "6.x", "rocm-linux", "7.2", "7.2", "clang", "drv")
    windows = ExecutionEnvironment("windows", "11", "26100", "hip-sdk-windows", "7.2", "7.2", "clang-cl", "drv")
    check(platform_environment_hash(linux) != platform_environment_hash(windows))

    original = tuple(sorted(devices[:2], key=lambda device: device.device_id))
    moved = (
        DeviceRecord("gpu-A", "amd_uuid", "gfx1100", "XTX", 24*gib, "0000:09:00.0", "/dev/dri/renderD140", 1, "drv"),
        next(device for device in devices if device.device_id == "gpu-B"),
    )
    cohort = hardware_cohort_hash(original, topology_fingerprint="peer:A-B")
    check(hardware_cohort_hash(moved, topology_fingerprint="peer:A-B") == cohort)
    check(
        inventory_material_hash(moved, topology_fingerprint="peer:A-B")
        != inventory_material_hash(original, topology_fingerprint="peer:A-B")
    )
    check(
        classify_inventory_drift(original, moved, accepted_topology="peer:A-B", observed_topology="peer:A-B")
        == DriftKind.LOCATOR_ONLY
    )
    check(
        classify_inventory_drift(original, moved, accepted_topology="peer:A-B", observed_topology="peer:A-B:numa1")
        == DriftKind.TOPOLOGY_CHANGE
    )
    check(hardware_cohort_hash(moved, topology_fingerprint="peer:A-B:numa1") != cohort)
    replacement = (
        DeviceRecord("gpu-X", "amd_uuid", "gfx1100", "XTX", 24*gib, "0000:01:00.0", "/dev/dri/renderD128", 0, "drv"),
        original[1],
    )
    check(hardware_cohort_hash(replacement, topology_fingerprint="peer:A-B") != cohort)
    check(
        classify_inventory_drift(original, replacement, accepted_topology="peer:A-B", observed_topology="peer:A-B")
        == DriftKind.DEVICE_SET_CHANGE
    )

    check(retry_action(75) == RetryAction.SAME_COMMIT_REQUEUE)
    check(retry_action(76) == RetryAction.NEW_ATTEMPT)
    check(retry_action(77) == RetryAction.BLOCK)

    executor = FakeExecutor()
    prepare = executor.submit("prepare")
    execute = executor.submit("execute", ("prepare",))
    check(executor.start_ready() == [prepare])
    check(executor.status(execute) == "queued")
    executor.complete(prepare)
    check(executor.start_ready() == [execute])
    executor.complete(execute)
    check(executor.status(execute) == "completed")
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("--self-test is required for this planning harness")
    checks = self_test()
    print(json.dumps({"checks": checks, "ok": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

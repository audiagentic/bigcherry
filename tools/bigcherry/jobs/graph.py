"""Static deterministic campaign graph for durable operation scheduling.

The graph carries BigCherry operation identity/dependency semantics only.
Executor/Slurm remains the scheduling/resource authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .executor import ResourceRequest, SchedulerGpuRequest
from .operations import OperationSpec


class CampaignGraphError(ValueError):
    pass


@dataclass(frozen=True)
class CampaignGraph:
    operations: tuple[OperationSpec, ...]

    def __post_init__(self) -> None:
        ids = [operation.operation_id for operation in self.operations]
        if len(ids) != len(set(ids)):
            raise CampaignGraphError("campaign graph operation IDs must be unique")
        known = set(ids)
        for operation in self.operations:
            missing = sorted(set(operation.dependencies) - known)
            if missing:
                raise CampaignGraphError(
                    f"operation {operation.operation_id!r} names missing dependencies {missing!r}"
                )
        self.topological_order()

    @property
    def by_id(self) -> Mapping[str, OperationSpec]:
        return {operation.operation_id: operation for operation in self.operations}

    def topological_order(self) -> tuple[OperationSpec, ...]:
        """Stable topological order independent of caller/dict insertion order."""
        by_id = {operation.operation_id: operation for operation in self.operations}
        remaining = {
            operation_id: set(operation.dependencies)
            for operation_id, operation in by_id.items()
        }
        emitted: list[OperationSpec] = []
        done: set[str] = set()
        while remaining:
            ready = sorted(
                operation_id
                for operation_id, dependencies in remaining.items()
                if dependencies <= done
            )
            if not ready:
                cycle = ", ".join(sorted(remaining))
                raise CampaignGraphError(
                    f"campaign operation dependency cycle among: {cycle}"
                )
            for operation_id in ready:
                emitted.append(by_id[operation_id])
                done.add(operation_id)
                del remaining[operation_id]
        return tuple(emitted)

    def descendants(self, operation_id: str) -> tuple[str, ...]:
        if operation_id not in self.by_id:
            raise KeyError(operation_id)
        descendants: set[str] = set()
        changed = True
        while changed:
            changed = False
            for operation in self.operations:
                if operation.operation_id in descendants:
                    continue
                if operation_id in operation.dependencies or any(
                    dependency in descendants for dependency in operation.dependencies
                ):
                    descendants.add(operation.operation_id)
                    changed = True
        return tuple(sorted(descendants))

    def ready(self, succeeded: Iterable[str]) -> tuple[OperationSpec, ...]:
        succeeded_set = set(succeeded)
        unknown = succeeded_set - set(self.by_id)
        if unknown:
            raise CampaignGraphError(
                f"ready() received unknown completed operation IDs: {sorted(unknown)!r}"
            )
        return tuple(
            operation
            for operation in self.topological_order()
            if operation.operation_id not in succeeded_set
            and set(operation.dependencies) <= succeeded_set
        )


def compile_validation_graph(
    *,
    architecture: str,
    reserved_gpu_count: int,
    semantic_request: Mapping[str, object],
    include_production_lane: bool,
    cpu_slots: int = 1,
    build_timeout_seconds: int = 2700,
    measure_timeout_seconds: int = 2700,
) -> CampaignGraph:
    """Compile the fixed RCD07 validation graph.

    ``semantic_request`` is the already-frozen behavior-affecting campaign
    request (patch/composition/model/producer/toolchain/etc.). Each operation
    copies it into its command semantics plus a stage token. There is no
    producer-name branching here; specialized producers later contribute
    additional typed stages before this compiler is called.
    """
    if reserved_gpu_count < 1:
        raise ValueError("reserved_gpu_count must be >= 1")
    gpu = SchedulerGpuRequest(architecture, reserved_gpu_count)

    def make(
        operation_id: str,
        kind: str,
        activity_class: str,
        dependencies: tuple[str, ...],
        outputs: tuple[str, ...],
        *,
        needs_gpu: bool,
        timeout: int,
    ) -> OperationSpec:
        return OperationSpec(
            operation_id=operation_id,
            kind=kind,
            command_semantics={
                "stage": operation_id,
                "request": dict(semantic_request),
            },
            environment_semantics=(),
            resources=ResourceRequest(
                cpu_slots=cpu_slots,
                gpu=gpu if needs_gpu else None,
                activity_class=activity_class,
                memory_bytes=None,
                timeout_seconds=timeout,
            ),
            dependencies=dependencies,
            declared_outputs=outputs,
        )

    operations: list[OperationSpec] = [
        make(
            "prepare",
            "prepare",
            "build",
            (),
            ("prepared-manifest",),
            needs_gpu=False,
            timeout=build_timeout_seconds,
        ),
        make(
            "correctness-activation",
            "correctness-activation",
            "correctness",
            ("prepare",),
            ("correctness", "activation"),
            needs_gpu=True,
            timeout=measure_timeout_seconds,
        ),
        make(
            "timed-performance",
            "timed-performance",
            "timed-measure",
            ("correctness-activation",),
            ("performance",),
            needs_gpu=True,
            timeout=measure_timeout_seconds,
        ),
        make(
            "reference-ladder",
            "reference-ladder",
            "timed-measure",
            ("timed-performance",),
            ("reference-ladder",),
            needs_gpu=True,
            timeout=measure_timeout_seconds,
        ),
    ]
    final_measure = "reference-ladder"
    if include_production_lane:
        operations.append(
            make(
                "production-lane",
                "production-lane",
                "timed-measure",
                ("reference-ladder",),
                ("production-lane",),
                needs_gpu=True,
                timeout=measure_timeout_seconds,
            )
        )
        final_measure = "production-lane"
    operations.extend(
        (
            make(
                "evidence-finalize",
                "evidence-finalize",
                "harvest",
                (final_measure,),
                ("validation-evidence",),
                needs_gpu=False,
                timeout=300,
            ),
            make(
                "harvest",
                "harvest",
                "harvest",
                ("evidence-finalize",),
                ("harvest-manifest",),
                needs_gpu=False,
                timeout=300,
            ),
            make(
                "report",
                "report",
                "harvest",
                ("harvest",),
                ("series-report",),
                needs_gpu=False,
                timeout=300,
            ),
        )
    )
    return CampaignGraph(tuple(operations))


def graph_document(graph: CampaignGraph) -> dict[str, object]:
    return {
        "schema": "bigcherry.campaign-graph.v1",
        "operations": [operation.document() for operation in graph.topological_order()],
    }

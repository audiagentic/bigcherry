"""Static deterministic campaign graph for durable operation scheduling.

The graph carries BigCherry operation identity/dependency semantics only.
Executor/Slurm remains the scheduling/resource authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

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
        self.topological_order()  # validates cycles immediately

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


def graph_document(graph: CampaignGraph) -> dict[str, object]:
    return {
        "schema": "bigcherry.campaign-graph.v1",
        "operations": [operation.document() for operation in graph.topological_order()],
    }

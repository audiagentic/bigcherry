"""Typed campaign stage graph; no generic workflow language."""

from __future__ import annotations

from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter


@dataclass(frozen=True)
class ResourceClaim:
    kind: str
    resource_id: str
    exclusive: bool = True


@dataclass(frozen=True)
class StageNode:
    stage_id: str
    kind: str
    source_slice_id: str | None
    build_plan_id: str | None
    workload_id: str | None
    dependencies: tuple[str, ...] = ()
    resources: tuple[ResourceClaim, ...] = ()


class CampaignGraphError(ValueError):
    pass


class CampaignGraph:
    def __init__(self, nodes: tuple[StageNode, ...]):
        self.nodes = {node.stage_id: node for node in nodes}
        if len(self.nodes) != len(nodes):
            raise CampaignGraphError("duplicate stage ID")
        for node in nodes:
            unknown = set(node.dependencies) - set(self.nodes)
            if unknown:
                raise CampaignGraphError(f"{node.stage_id} has unknown dependencies: {sorted(unknown)}")
        self._sorter = TopologicalSorter({
            node.stage_id: set(node.dependencies) for node in nodes
        })
        try:
            self._order = tuple(self._sorter.static_order())
        except (CycleError, ValueError) as exc:
            raise CampaignGraphError("campaign stage graph is cyclic") from exc

    @property
    def order(self) -> tuple[str, ...]:
        return self._order

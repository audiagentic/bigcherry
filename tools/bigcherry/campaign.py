"""Campaign run state and hash-aware stage execution."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .campaign_graph import CampaignGraph
from .resources import ResourceLock


def _spec_hash(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


@dataclass
class StageRecord:
    stage_id: str
    state: str
    spec_hash: str
    output_hashes: tuple[str, ...] = ()


class CampaignRun:
    def __init__(self, graph: CampaignGraph, root: Path, run_id: str | None = None):
        self.graph = graph
        self.root = root.resolve()
        self.run_id = run_id or uuid.uuid4().hex
        self.records: dict[str, StageRecord] = {}

    def execute(
        self,
        executor: Callable[[str], tuple[str, ...]],
        *,
        resource_root: Path,
        reuse: Callable[[StageRecord], bool] | None = None,
    ) -> dict[str, StageRecord]:
        for stage_id in self.graph.order:
            node = self.graph.nodes[stage_id]
            spec = {
                "stage_id": node.stage_id, "kind": node.kind,
                "source_slice_id": node.source_slice_id,
                "build_plan_id": node.build_plan_id,
                "workload_id": node.workload_id,
                "dependencies": node.dependencies,
            }
            digest = _spec_hash(spec)
            if any(self.records.get(dep, StageRecord(dep, "blocked", "")).state in {"failed", "blocked"}
                   for dep in node.dependencies):
                self.records[stage_id] = StageRecord(stage_id, "blocked", digest)
                continue
            previous = self.records.get(stage_id)
            # previous.state in {"succeeded", "reused"}, not only
            # "succeeded": after one successful reuse, previous.state
            # becomes "reused" -- requiring exactly "succeeded" would force
            # a third pass through the same unchanged stage to execute for
            # real again, for no identity reason, purely because it had
            # already been reused once.
            if (previous and previous.spec_hash == digest
                    and previous.state in {"succeeded", "reused"}
                    and reuse and reuse(previous)):
                previous.state = "reused"
                continue
            locks = [ResourceLock(resource_root, claim.resource_id)
                     for claim in node.resources if claim.exclusive]
            try:
                for lock in locks:
                    lock.acquire()
                self.records[stage_id] = StageRecord(stage_id, "running", digest)
                outputs = executor(stage_id)
                self.records[stage_id] = StageRecord(stage_id, "succeeded", digest, tuple(outputs))
            except Exception:
                self.records[stage_id] = StageRecord(stage_id, "failed", digest)
            finally:
                for lock in reversed(locks):
                    if lock.inspect() is not None:
                        lock.release()
        return self.records

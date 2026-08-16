"""RE14: the stage executor CampaignRun/PipelineService actually call.

Two validation layers, deliberately not merged (per gpt-auto-agent's review,
verified against source before adoption): :class:`ArtifactStore` proves an
``ArtifactRef``'s bytes really match its ``content_hash`` -- something
neither :class:`PipelineService` nor :class:`CampaignRun` do on their own.
``PipelineService.run(expected=...)`` proves the namespace fields a stage is
permitted to consume/produce agree with the campaign's identity -- but it
checks the SAME ``expected`` dict against every input and every output of a
call, which makes it an invariant checker across a stage boundary, not a
place to assert an identity a stage is only now establishing.

That distinction is why the generate stage uses a narrower ``expected``
than build/runtime-smoke: workload identity does not exist yet when the
inventory artifact (generate's only real input) was produced -- it becomes
meaningful only once generation interprets that inventory and selects a
variant set. Assigning ``workload.workload_id`` onto the inventory's own
provenance after the fact would make it claim an identity it never had.
Build and runtime-smoke, whose inputs and outputs are both already
workload-scoped, use the wider envelope including ``workload.workload_id``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from . import provenance
from .artifacts import ArtifactStore
from .campaign_graph import CampaignGraph
from .pipeline import ArtifactRef, PipelineService


class CampaignExecutionError(RuntimeError):
    pass


def source_expected(source_slice_id: str) -> dict[str, str]:
    return {"source.source_slice_id": source_slice_id}


def build_expected(source_slice_id: str, build_plan_id: str) -> dict[str, str]:
    return {
        "source.source_slice_id": source_slice_id,
        "build.build_plan_id": build_plan_id,
    }


def workload_expected(
    source_slice_id: str, build_plan_id: str, workload_id: str, run_id: str
) -> dict[str, str]:
    """The wider envelope: build and runtime-smoke, where every participating
    artifact is already workload-scoped. Never used for generate -- see
    module docstring.

    Includes campaign.run_id, unlike source_expected/build_expected: build
    and runtime-smoke are same-run stage transitions, where every input and
    output genuinely belongs to this run -- an artifact from a different
    campaign run sharing the same source/build/workload identity by
    coincidence is exactly the cross-campaign-artifact case RE14's own
    negative-parity plan calls out, and should fail closed here rather than
    pass silently. Generate is different: its inventory input may
    legitimately have been produced by an earlier, separate record-mode
    campaign run, so run_id is not part of generate's own envelope.
    """
    return {
        "source.source_slice_id": source_slice_id,
        "build.build_plan_id": build_plan_id,
        "workload.workload_id": workload_id,
        "campaign.run_id": run_id,
    }


def _empty_namespace_provenance(
    *, source: dict[str, Any], build: dict[str, Any],
    workload: dict[str, Any], campaign: dict[str, Any],
) -> dict[str, Any]:
    """``provenance.validate`` requires all five namespaces present as
    dicts, even when a stage has nothing meaningful to say for one of them
    yet (e.g. generate's inputs have no real workload identity). An empty
    dict satisfies that structural requirement without asserting a value
    ``require_compatible`` could ever be asked to check, since an empty
    ``expected`` for that namespace never occurs -- see module docstring.
    """
    return provenance.make(
        project={}, source=source, build=build, workload=workload, campaign=campaign,
    )


class CampaignStageExecutor:
    """A stateful callable matching ``CampaignRun.execute``'s executor
    contract (``Callable[[str], tuple[str, ...]]``), backed internally by
    ``PipelineService`` per stage. State is necessary because
    ``CampaignRun`` only retains ``tuple[str, ...]`` output hashes, while
    later stages need the actual ``ArtifactRef``s (path, provenance) of
    their dependencies' outputs, not just their hashes.
    """

    def __init__(
        self,
        *,
        graph: CampaignGraph,
        store: ArtifactStore,
        run_id: str,
        materialize: Callable[[], dict[str, Any]],
        generate: Callable[[tuple[ArtifactRef, ...]], dict[str, Any]],
        source_slice_id_holder: list[str | None],
        build_plan_id: str | None = None,
        inventory_ref: ArtifactRef | None = None,
        workload_id: str | None = None,
        build: Callable[[tuple[ArtifactRef, ...]], tuple[ArtifactRef, ...]] | None = None,
        smoke: Callable[[tuple[ArtifactRef, ...]], tuple[ArtifactRef, ...]] | None = None,
    ) -> None:
        self.graph = graph
        self.store = store
        self.run_id = run_id
        self._materialize = materialize
        self._generate = generate
        self._source_slice_id_holder = source_slice_id_holder
        self.build_plan_id = build_plan_id
        self.inventory_ref = inventory_ref
        self.workload_id = workload_id
        self._build = build
        self._smoke = smoke
        self.outputs: dict[str, tuple[ArtifactRef, ...]] = {}

    def _require_stored_bytes(self, artifact: ArtifactRef) -> None:
        try:
            relative = artifact.path.resolve().relative_to(self.store.root)
        except ValueError as exc:
            raise CampaignExecutionError(
                f"{artifact.kind} at {artifact.path} is not owned by this "
                f"campaign's ArtifactStore ({self.store.root})"
            ) from exc
        if not self.store.verify(relative, artifact.content_hash):
            raise CampaignExecutionError(
                f"{artifact.kind} bytes do not match its own content_hash "
                f"{artifact.content_hash}"
            )

    def _publish(
        self, relative: Path, value: dict[str, Any], *, kind: str, doc: dict[str, Any]
    ) -> ArtifactRef:
        digest = self.store.publish_json(relative, value)
        if not self.store.verify(relative, digest):
            raise CampaignExecutionError(
                f"published artifact {relative} failed immediate verification"
            )
        return ArtifactRef(
            kind=kind, path=self.store.resolve(relative),
            content_hash=digest, provenance=doc,
        )

    def _run_materialize(self, stage_id: str) -> tuple[ArtifactRef, ...]:
        def execute(_stage: str, _inputs: tuple[ArtifactRef, ...]) -> tuple[ArtifactRef, ...]:
            metadata = self._materialize()
            source_slice_id = metadata.get("source_slice_id")
            if not isinstance(source_slice_id, str) or not source_slice_id:
                raise CampaignExecutionError(
                    "materialize did not return a source_slice_id"
                )
            self._source_slice_id_holder[0] = source_slice_id
            doc = _empty_namespace_provenance(
                source={"source_slice_id": source_slice_id},
                build={}, workload={}, campaign={"run_id": self.run_id},
            )
            relative = Path("runs") / self.run_id / "materialize" / "source-metadata.json"
            return (self._publish(relative, metadata, kind="source-metadata", doc=doc),)

        # No prior identity exists for materialize: nothing to check inputs
        # or outputs against beyond structural provenance validity, so
        # expected is empty. PipelineService.run still validates the five
        # namespaces are present via provenance.validate.
        pipeline = PipelineService(execute)
        outputs = pipeline.run(stage_id, inputs=(), expected={})
        self.outputs[stage_id] = outputs
        return outputs

    def _run_generate(self, stage_id: str) -> tuple[ArtifactRef, ...]:
        source_slice_id = self._source_slice_id_holder[0]
        if not source_slice_id:
            raise CampaignExecutionError(
                "generate stage requires source_slice_id from a completed "
                "materialize stage"
            )
        if self.build_plan_id is None:
            raise CampaignExecutionError("generate stage requires build_plan_id")
        if self.inventory_ref is None:
            raise CampaignExecutionError("generate stage requires an inventory artifact")
        self._require_node_identity_agreement(
            self.graph.nodes[stage_id], source_slice_id=source_slice_id,
            build_plan_id=self.build_plan_id,
        )

        self._require_stored_bytes(self.inventory_ref)

        def execute(_stage: str, inputs: tuple[ArtifactRef, ...]) -> tuple[ArtifactRef, ...]:
            result = self._generate(inputs)
            manifest = result["manifest"]
            workload_id = result.get("workload_id")
            if not isinstance(workload_id, str) or not workload_id:
                raise CampaignExecutionError(
                    "generate did not establish a workload_id for its output"
                )
            doc = _empty_namespace_provenance(
                source={"source_slice_id": source_slice_id},
                build={"build_plan_id": self.build_plan_id},
                workload={"workload_id": workload_id},
                campaign={"run_id": self.run_id},
            )
            relative = Path("runs") / self.run_id / "generate" / "hip-autotune-manifest.json"
            return (self._publish(relative, manifest, kind="manifest", doc=doc),)

        # Narrowest envelope: source_slice_id only. workload_id does not
        # exist on the inventory input (see module docstring) -- and
        # neither does build_plan_id: the inventory artifact was produced
        # by a DIFFERENT build entirely (a record-mode run), not by the
        # tune build_plan_id this generate call is scoped to. The only
        # identity genuinely invariant across generate's stage boundary is
        # the source both the inventory and the new manifest were built
        # against.
        pipeline = PipelineService(execute)
        outputs = pipeline.run(
            stage_id, inputs=(self.inventory_ref,),
            expected=source_expected(source_slice_id),
        )
        self.outputs[stage_id] = outputs
        return outputs

    @staticmethod
    def _require_node_identity_agreement(
        node, *, source_slice_id: str, build_plan_id: str | None = None,
        workload_id: str | None = None,
    ) -> None:
        """CampaignRun computes its stage spec_hash from the frozen
        StageNode's own source_slice_id/build_plan_id/workload_id fields
        -- but nothing previously checked those against the executor's
        actual, mutable identity state at execution time. A caller that
        builds the graph before an identity is fully known (e.g. seeding
        StageNode.workload_id with a placeholder before generate has run)
        and then updates executor state without rebuilding the graph gets
        a real split-brain: CampaignRun's own scheduling/reuse decisions
        would be keyed on the stale placeholder while every artifact this
        executor actually produces carries the real value. This check
        makes that disagreement a hard failure instead of a silent
        divergence between what CampaignRun thinks it ran and what
        actually got produced.
        """
        if node.source_slice_id is not None and node.source_slice_id != source_slice_id:
            raise CampaignExecutionError(
                f"stage {node.stage_id!r} node.source_slice_id "
                f"{node.source_slice_id!r} disagrees with the executor's "
                f"actual source_slice_id {source_slice_id!r}"
            )
        if (build_plan_id is not None and node.build_plan_id is not None
                and node.build_plan_id != build_plan_id):
            raise CampaignExecutionError(
                f"stage {node.stage_id!r} node.build_plan_id "
                f"{node.build_plan_id!r} disagrees with the executor's "
                f"actual build_plan_id {build_plan_id!r}"
            )
        if (workload_id is not None and node.workload_id is not None
                and node.workload_id != workload_id):
            raise CampaignExecutionError(
                f"stage {node.stage_id!r} node.workload_id "
                f"{node.workload_id!r} disagrees with the executor's "
                f"actual workload_id {workload_id!r}"
            )

    def _dependency_outputs(self, node) -> tuple[ArtifactRef, ...]:
        gathered: list[ArtifactRef] = []
        for dep in node.dependencies:
            if dep not in self.outputs:
                raise CampaignExecutionError(
                    f"stage {node.stage_id!r} depends on {dep!r}, which has "
                    f"not produced outputs yet"
                )
            gathered.extend(self.outputs[dep])
        return tuple(gathered)

    def _run_workload_scoped(
        self,
        stage_id: str,
        node,
        worker: Callable[[tuple[ArtifactRef, ...]], tuple[ArtifactRef, ...]] | None,
        *,
        kind_label: str,
    ) -> tuple[ArtifactRef, ...]:
        """Shared shape for build and runtime-smoke: both use the wider
        workload-scoped envelope (source + build + workload, all genuinely
        invariant across their input and output -- unlike generate), and
        both delegate to an already-publishing worker callable rather than
        building ArtifactRefs inline the way materialize/generate do, since
        their real implementations (campaign_build.execute_build_stage,
        the runtime-smoke harness) already own their own ArtifactStore
        publication.
        """
        source_slice_id = self._source_slice_id_holder[0]
        if not source_slice_id:
            raise CampaignExecutionError(
                f"{kind_label} stage requires source_slice_id from a "
                f"completed materialize stage"
            )
        if self.build_plan_id is None:
            raise CampaignExecutionError(f"{kind_label} stage requires build_plan_id")
        if not self.workload_id:
            raise CampaignExecutionError(f"{kind_label} stage requires workload_id")
        if worker is None:
            raise CampaignExecutionError(f"no {kind_label} worker was configured")
        self._require_node_identity_agreement(
            node, source_slice_id=source_slice_id, build_plan_id=self.build_plan_id,
            workload_id=self.workload_id,
        )

        inputs = self._dependency_outputs(node)
        for artifact in inputs:
            self._require_stored_bytes(artifact)

        def execute(_stage: str, refs: tuple[ArtifactRef, ...]) -> tuple[ArtifactRef, ...]:
            produced = worker(refs)
            for artifact in produced:
                self._require_stored_bytes(artifact)
            return produced

        pipeline = PipelineService(execute)
        outputs = pipeline.run(
            stage_id, inputs=inputs,
            expected=workload_expected(
                source_slice_id, self.build_plan_id, self.workload_id, self.run_id),
        )
        self.outputs[stage_id] = outputs
        return outputs

    def __call__(self, stage_id: str) -> tuple[str, ...]:
        node = self.graph.nodes[stage_id]
        if node.kind == "materialize":
            outputs = self._run_materialize(stage_id)
        elif node.kind == "generate":
            outputs = self._run_generate(stage_id)
        elif node.kind == "build":
            outputs = self._run_workload_scoped(stage_id, node, self._build, kind_label="build")
        elif node.kind == "runtime-smoke":
            outputs = self._run_workload_scoped(stage_id, node, self._smoke, kind_label="runtime-smoke")
        else:
            raise CampaignExecutionError(
                f"stage kind {node.kind!r} is not recognised by "
                f"CampaignStageExecutor"
            )
        return tuple(artifact.content_hash for artifact in outputs)

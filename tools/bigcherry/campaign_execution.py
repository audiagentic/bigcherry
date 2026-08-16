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
from typing import Any, Callable, Mapping

from . import provenance
from .artifacts import ArtifactStore
from .campaign import StageRecord
from .campaign_graph import CampaignGraph
from .pipeline import ArtifactRef, PipelineService


class CampaignExecutionError(RuntimeError):
    pass


def require_campaign_success(
    records: dict[str, "StageRecord"], *, label: str,
) -> None:
    """CampaignRun.execute() catches every executor exception itself and
    records the stage as "failed" rather than re-raising -- a caller that
    wraps ``run.execute(...)`` in a try/except for the executor's own
    exception types will never see one, and would otherwise silently treat
    a campaign with failed/blocked stages as a success. Call this
    immediately after every ``CampaignRun.execute()`` call.
    """
    bad = {
        stage_id: record.state for stage_id, record in records.items()
        if record.state not in {"succeeded", "reused"}
    }
    if bad:
        detail = ", ".join(f"{stage_id}={state}" for stage_id, state in sorted(bad.items()))
        raise CampaignExecutionError(f"{label} campaign failed: {detail}")


def make_artifact_reuse_checker(
    *, executor: "CampaignStageExecutor", store: ArtifactStore,
) -> Callable[["StageRecord"], bool]:
    """A real reuse callback for CampaignRun.execute(reuse=...), proving
    reuse against ArtifactStore rather than trusting StageRecord's own
    output_hashes tuple (which has no path/kind, and is not itself
    verified against anything by CampaignRun).

    Only valid within the same process/executor instance: ``executor``
    only knows about ArtifactRefs it has itself produced via
    ``executor.outputs`` in this run. Cross-process resume (rerunning the
    harness later and skipping already-built stages) needs persisted
    StageRecords and a way to rehydrate ArtifactRefs from disk, neither of
    which exists yet -- CampaignRun.root is not currently used to persist
    or reload records.
    """

    def reusable(record: "StageRecord") -> bool:
        refs = executor.outputs.get(record.stage_id)
        if refs is None:
            return False
        if tuple(ref.content_hash for ref in refs) != record.output_hashes:
            return False
        for ref in refs:
            try:
                relative = ref.path.resolve().relative_to(store.root)
            except ValueError:
                return False
            if not store.verify(relative, ref.content_hash):
                return False
        return True

    return reusable


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
    return execution_expected(source_slice_id, build_plan_id, run_id, workload_id)


def execution_expected(
    source_slice_id: str, build_plan_id: str, run_id: str, workload_id: str | None,
) -> dict[str, str]:
    """RE17: workload_expected(), generalized for builds that have no
    workload at all (needs=[] -- stock/control/record/audit). Omitting the
    workload.workload_id key entirely when workload_id is None (rather than
    some placeholder value) is correct because
    ``provenance.require_compatible`` only checks keys actually present in
    ``expected`` -- a key that isn't there is simply never checked, exactly
    the right behavior for "this build has no workload identity to assert".
    """
    expected = {
        "source.source_slice_id": source_slice_id,
        "build.build_plan_id": build_plan_id,
        "campaign.run_id": run_id,
    }
    if workload_id is not None:
        expected["workload.workload_id"] = workload_id
    return expected


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
        generate: Callable[[Mapping[str, ArtifactRef]], dict[str, Any]] | None = None,
        generate_inputs: Mapping[str, ArtifactRef] | None = None,
        generate_needs: frozenset[str] = frozenset(),
        source_slice_id_holder: list[str | None] | None = None,
        build_plan_id: str | None = None,
        workload_id: str | None = None,
        build: Callable[[tuple[ArtifactRef, ...]], tuple[ArtifactRef, ...]] | None = None,
        smoke: Callable[[tuple[ArtifactRef, ...]], tuple[ArtifactRef, ...]] | None = None,
    ) -> None:
        self.graph = graph
        self.store = store
        self.run_id = run_id
        self._materialize = materialize
        self._generate = generate
        self._generate_inputs = dict(generate_inputs or {})
        self._generate_needs = frozenset(generate_needs)
        self._source_slice_id_holder = source_slice_id_holder if source_slice_id_holder is not None else [None]
        self.build_plan_id = build_plan_id
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
        if self._generate is None:
            raise CampaignExecutionError("no generate worker was configured")
        # RE17: generate's real inputs are whatever config.Build.needs says
        # (possibly none at all -- stock/control/record/audit generate with
        # no external input), not always exactly one inventory artifact.
        # Validated as an exact set, same as the build worker's own need
        # validation, so a caller cannot silently supply the wrong input
        # under the right key or the right input under the wrong key.
        actual_needs = frozenset(self._generate_inputs)
        if actual_needs != self._generate_needs:
            raise CampaignExecutionError(
                f"generate stage input keys disagree with declared needs: "
                f"required={sorted(self._generate_needs)}, "
                f"actual={sorted(actual_needs)}"
            )
        for name, ref in self._generate_inputs.items():
            if ref.kind != name:
                raise CampaignExecutionError(
                    f"generate input {name!r} has artifact kind {ref.kind!r}"
                )
            self._require_stored_bytes(ref)

        self._require_node_identity_agreement(
            self.graph.nodes[stage_id], source_slice_id=source_slice_id,
            build_plan_id=self.build_plan_id,
        )

        # Deterministic order for a stable PipelineService input tuple,
        # independent of dict insertion order.
        ordered_names = tuple(sorted(self._generate_inputs))
        ordered_inputs = tuple(self._generate_inputs[name] for name in ordered_names)

        def execute(_stage: str, inputs: tuple[ArtifactRef, ...]) -> tuple[ArtifactRef, ...]:
            by_kind = {ref.kind: ref for ref in inputs}
            result = self._generate(by_kind)
            manifest = result["manifest"]
            # workload_id is no longer discovered here: the lane already
            # knows it (the inventory ref's own content_hash, computed
            # before this stage ever runs) or knows there isn't one
            # (needs=[] builds). Generate "establishing" it was backwards.
            doc = _empty_namespace_provenance(
                source={"source_slice_id": source_slice_id},
                build={"build_plan_id": self.build_plan_id},
                workload=({"workload_id": self.workload_id}
                         if self.workload_id is not None else {}),
                campaign={"run_id": self.run_id},
            )
            manifest_relative = Path("runs") / self.run_id / "generate" / "hip-autotune-manifest.json"
            manifest_ref = self._publish(manifest_relative, manifest, kind="manifest", doc=doc)

            generated_tree_doc = result.get("generated_tree")
            if not isinstance(generated_tree_doc, dict):
                raise CampaignExecutionError(
                    "generate did not return a generated_tree manifest"
                )
            tree_relative = Path("runs") / self.run_id / "generate" / "generated-tree.json"
            tree_ref = self._publish(tree_relative, generated_tree_doc, kind="generated-tree", doc=doc)

            return (manifest_ref, tree_ref)

        # Narrowest envelope: source_slice_id only, for every generated
        # build including needs=[]. Not build_plan_id/workload_id/run_id:
        # generate's real inputs (inventory, promoted-winners) may
        # legitimately have been produced by a different build_plan or an
        # earlier, separate campaign run -- the only identity genuinely
        # invariant across generate's stage boundary is the source both
        # those inputs and the new manifest were built against.
        pipeline = PipelineService(execute)
        outputs = pipeline.run(
            stage_id, inputs=ordered_inputs,
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

    def _run_build_scoped(
        self,
        stage_id: str,
        node,
        worker: Callable[[tuple[ArtifactRef, ...]], tuple[ArtifactRef, ...]] | None,
        *,
        kind_label: str,
    ) -> tuple[ArtifactRef, ...]:
        """Shared shape for build and runtime-smoke: both use the wider
        build-scoped envelope (source + build [+ workload, when the build
        actually has one] + run_id, all genuinely invariant across their
        input and output -- unlike generate), and both delegate to an
        already-publishing worker callable rather than building ArtifactRefs
        inline the way materialize/generate do, since their real
        implementations (campaign_build.execute_build_stage, the
        runtime-smoke harness) already own their own ArtifactStore
        publication.

        RE17: workload_id is genuinely optional here (stock/control/record/
        audit builds have none at all) -- unlike source_slice_id/
        build_plan_id, which every build has.
        """
        source_slice_id = self._source_slice_id_holder[0]
        if not source_slice_id:
            raise CampaignExecutionError(
                f"{kind_label} stage requires source_slice_id from a "
                f"completed materialize stage"
            )
        if self.build_plan_id is None:
            raise CampaignExecutionError(f"{kind_label} stage requires build_plan_id")
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
            expected=execution_expected(
                source_slice_id, self.build_plan_id, self.run_id, self.workload_id),
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
            outputs = self._run_build_scoped(stage_id, node, self._build, kind_label="build")
        elif node.kind == "runtime-smoke":
            outputs = self._run_build_scoped(stage_id, node, self._smoke, kind_label="runtime-smoke")
        else:
            raise CampaignExecutionError(
                f"stage kind {node.kind!r} is not recognised by "
                f"CampaignStageExecutor"
            )
        return tuple(artifact.content_hash for artifact in outputs)

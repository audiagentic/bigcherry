"""RE14: real generate/build/smoke workers for CampaignStageExecutor.

Wires the previously test-only executor to the actual implementations:
``autotune_catalog.build_manifest``/``emit`` for generate,
``campaign_build`` for build, and a real ``llama-bench`` subprocess for
runtime smoke. This is deliberately kept separate from
``campaign_execution.py`` -- that module defines the stage-boundary/
provenance contract; this module supplies the real work behind it, so the
contract can be (and was) unit-tested without needing a compiler or a GPU.

Known simplification, not yet hardened: the real generate worker writes
the full generated/ directory (registry.inc, build-hash.h, etc, not only
the manifest) to a run-scoped filesystem path that the build worker then
reads directly by convention (same run_id), rather than fully
reconstructing those files from ArtifactStore-verified bytes before every
configure. The manifest itself is published and verified through
ArtifactStore as usual; the generated/ directory beside it is not. Treat
that directory as trusted only within one run, not as portable evidence.
"""

from __future__ import annotations

import stat
import subprocess
from pathlib import Path
from typing import Any

from . import autotune_catalog, campaign_build, provenance, runtime_smoke
from .artifacts import ArtifactStore
from .builds import BuildPlan, binary_hash, build_directory
from .context import ProjectContext
from .pipeline import ArtifactRef


def make_generate_worker(
    *,
    context: ProjectContext,
    source_root: Path,
    run_id: str,
    variant_set: str,
    architectures: list[str],
    upstream_revision: str,
):
    """Returns the callable ``_run_generate`` expects: takes the inventory
    ArtifactRef tuple, returns ``{"manifest": ..., "workload_id": ...}``.

    ``workload_id`` is the inventory artifact's own content_hash: the same
    inventory bytes always yield the same workload_id, and different
    inventory content always yields a different one. No other workload
    identity convention exists in this project to defer to.
    """

    def generate(inputs: tuple[ArtifactRef, ...]) -> dict[str, Any]:
        if len(inputs) != 1:
            raise campaign_build.CampaignBuildError(
                f"generate worker expects exactly one inventory input, got {len(inputs)}"
            )
        inventory_ref = inputs[0]
        inventory = autotune_catalog.Inventory.from_json(inventory_ref.path)

        manifest = autotune_catalog.build_manifest(
            source_root, variant_set=variant_set, architectures=architectures,
            inventory=inventory, source_revision=upstream_revision,
        )

        stage_root = context.work_root / "runs" / run_id / "generate"
        artifact_root = stage_root / "catalog"
        generated_root = stage_root / "generated"
        autotune_catalog.emit(manifest, source_root, artifact_root, generated_root=generated_root)

        return {"manifest": manifest, "workload_id": inventory_ref.content_hash}

    return generate


def make_build_worker(
    *,
    context: ProjectContext,
    source_root: Path,
    run_id: str,
    build_plan: BuildPlan,
    platform,
    build,
    store: ArtifactStore,
    binary_relative_path: str,
    source_slice_id: str,
    workload_id: str,
    cmake_targets: tuple[str, ...] = (),
    inventory_path: Path | None = None,
):
    """Returns the callable ``_run_workload_scoped`` expects for the build
    stage: takes generate's output ArtifactRef tuple, configures and
    compiles for real, returns already-published+verified ArtifactRefs.
    """

    def run_build(inputs: tuple[ArtifactRef, ...]) -> tuple[ArtifactRef, ...]:
        if len(inputs) != 1 or inputs[0].kind != "manifest":
            raise campaign_build.CampaignBuildError(
                f"build worker expects exactly one 'manifest' input from "
                f"the generate stage, got {[ref.kind for ref in inputs]}"
            )

        generated_root = context.work_root / "runs" / run_id / "generate" / "generated"
        if not generated_root.is_dir():
            raise campaign_build.CampaignBuildError(
                f"expected generate stage's generated/ directory at "
                f"{generated_root}, but it does not exist"
            )

        build_dir = build_directory(context, source_slice_id, build_plan)
        build_dir.mkdir(parents=True, exist_ok=True)

        configure_args = campaign_build.cmake_configure_args(
            build, platform, source_root, build_dir,
            generated_root=generated_root, inventory=inventory_path,
            c_compiler=platform.c_compiler, cxx_compiler=platform.cxx_compiler,
        )
        subprocess.run(configure_args, cwd=source_root, check=True)
        subprocess.run(
            campaign_build.cmake_build_args(build_dir, targets=cmake_targets),
            cwd=source_root, check=True,
        )

        binary = build_dir / binary_relative_path
        if not binary.is_file():
            raise campaign_build.CampaignBuildError(
                f"build did not produce the expected binary: {binary}"
            )

        doc = provenance.make(
            project={}, source={"source_slice_id": source_slice_id},
            build={"build_plan_id": build_plan.build_plan_id},
            workload={"workload_id": workload_id},
            campaign={"run_id": run_id},
        )

        prefix = f"builds/{source_slice_id}/{build_plan.build_plan_id}"
        # Earlier version of this function re-published the manifest under
        # a second, build-scoped path -- redundant (it is already published
        # and verified by generate) and unsafe (the raw manifest JSON
        # embeds a real-time generated_at not covered by manifest_hash, so
        # a second publish attempt could collide with itself on re-run,
        # exactly as ArtifactStore's immutability check is supposed to
        # catch, and did). The fix was to stop re-publishing it at all --
        # not just to reference the existing ArtifactRef, but to stop
        # returning it from build entirely. Build did not produce the
        # manifest; generate did, and it remains available via
        # executor.outputs[generate_stage_id]. Returning it again from
        # build serves no purpose and adds a needless kind="manifest"
        # collision risk if build ever gains a second dependent stage that
        # also produces a "manifest"-kind artifact.
        refs: list[ArtifactRef] = []

        binary_relative = f"{prefix}/{binary.name}"
        binary_digest = store.publish_file(binary_relative, binary)
        if binary_digest != binary_hash(binary):
            raise campaign_build.CampaignBuildError(
                "published binary digest differs from build-tree binary"
            )
        if not store.verify(binary_relative, binary_digest):
            raise campaign_build.CampaignBuildError(f"published {binary_relative} failed verification")
        published_binary = store.resolve(binary_relative)
        # ArtifactStore.publish_file copies bytes only; it makes no promise
        # about executable mode, correctly -- most published artifacts are
        # data (JSON, manifests), not executables. A binary artifact is the
        # one kind that specifically needs its execute bit set here, by the
        # caller that knows it is a binary, not inside the general-purpose
        # store. Owner-execute only: this store is not (yet) a shared
        # multi-user artifact store, and mkstemp's default 0600 becoming
        # 0700 is the narrowest change that makes the binary runnable by
        # the same process/user that just published it.
        published_binary.chmod(published_binary.stat().st_mode | stat.S_IXUSR)
        refs.append(ArtifactRef(kind="binary", path=published_binary,
                                content_hash=binary_digest, provenance=doc))

        return tuple(refs)

    return run_build


def make_smoke_worker(
    *,
    run_id: str,
    store: ArtifactStore,
    source_slice_id: str,
    build_plan_id: str,
    workload_id: str,
    spec: runtime_smoke.RuntimeSmokeSpec,
    environment: dict[str, str] | None = None,
):
    """Returns the callable ``_run_workload_scoped`` expects for the
    runtime-smoke stage: takes build's output ArtifactRef tuple, runs the
    real binary against ``spec``, publishes the smoke result.
    """

    def run_smoke(inputs: tuple[ArtifactRef, ...]) -> tuple[ArtifactRef, ...]:
        # Exact cardinality, not "take whichever comes first": the graph
        # does not (yet) forbid a node from depending on more than one
        # upstream stage or from having overlapping artifact kinds, so a
        # silent next()-style pick would be wrong under a future fan-in
        # graph even though today's linear build->smoke edge makes it safe
        # in practice. Matches the same-shaped guard the build worker
        # already has on its own single manifest input.
        binaries = [ref for ref in inputs if ref.kind == "binary"]
        if len(binaries) != 1:
            raise campaign_build.CampaignBuildError(
                f"runtime-smoke worker expects exactly one 'binary' input, "
                f"found {len(binaries)}"
            )
        binary_ref = binaries[0]

        argv = runtime_smoke.smoke_argv(binary_ref.path, spec)
        completed = subprocess.run(
            argv, capture_output=True, text=True, env=environment,
        )
        if completed.returncode != 0:
            raise runtime_smoke.SmokeError(
                f"runtime smoke exited {completed.returncode}: {completed.stderr[-2000:]}"
            )
        rows = runtime_smoke.evaluate_smoke_result(completed.stdout)

        doc = provenance.make(
            project={}, source={"source_slice_id": source_slice_id},
            build={"build_plan_id": build_plan_id},
            workload={"workload_id": workload_id},
            campaign={"run_id": run_id},
        )
        relative = f"runs/{run_id}/smoke/result.json"
        # A smoke result naming only binary_hash does not actually pin down
        # what it proves: the same binary smoked against two different
        # models, or two different -p/-n/-sm settings, would otherwise be
        # indistinguishable evidence. Record the real inputs alongside the
        # binary hash so the result is self-describing even before a
        # first-class smoke_spec_id exists on StageNode's own identity.
        result = {
            "rows": rows,
            "binary_hash": binary_ref.content_hash,
            "model_hash": binary_hash(spec.model_path),
            "model_path": str(spec.model_path),
            "n_prompt": spec.n_prompt,
            "n_gen": spec.n_gen,
            "split_mode": spec.split_mode,
        }
        digest = store.publish_json(relative, result)
        if not store.verify(relative, digest):
            raise campaign_build.CampaignBuildError(f"published {relative} failed verification")
        return (ArtifactRef(kind="smoke-result", path=store.resolve(relative),
                            content_hash=digest, provenance=doc),)

    return run_smoke

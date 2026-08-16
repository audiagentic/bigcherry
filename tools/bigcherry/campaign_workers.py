"""RE14: real generate/build/smoke workers for CampaignStageExecutor.

Wires the previously test-only executor to the actual implementations:
``autotune_catalog.build_manifest``/``emit`` for generate,
``campaign_build`` for build, and a real ``llama-bench`` subprocess for
runtime smoke. This is deliberately kept separate from
``campaign_execution.py`` -- that module defines the stage-boundary/
provenance contract; this module supplies the real work behind it, so the
contract can be (and was) unit-tested without needing a compiler or a GPU.

The generate worker writes the full generated/ directory (registry.inc,
build-hash.h, etc, not only the manifest) to a run-scoped filesystem path
that the build worker then reads by convention (same run_id). That
filesystem convention is not itself the trust boundary, though: generate
also hashes every file under generated/ (see generated_tree.py) and
publishes that hash manifest as its own verified ArtifactRef, and build
re-verifies the real directory against it before trusting it -- once always,
and again after compiling when a compile actually ran (the reuse path below
never invokes the compiler, so there is nothing a second check there could
catch). A file that changed on disk between generate and build, or during
the compile itself, is a hard failure, not a silent pass-through.

The build worker also implements real cross-invocation build reuse via
builds.validate_reuse(): build_directory() is content-addressed by
(source_slice_id, build_plan_id), so a directory that already carries a
prior build's metadata and still checks out against that recorded identity
is trusted without recompiling. A directory that carries metadata which does
NOT check out is a hard failure, not a silent rebuild -- that would hide
real corruption or tampering of build output behind an unremarkable
recompile. This is a different, lower layer than CampaignRun's own
StageRecord-based reuse (campaign.py): that reuse skips re-executing a stage
within one CampaignRun/run_id; this reuse lets a fresh process/run_id skip
recompiling entirely when a prior run (any run_id) already produced the
identical build.
"""

from __future__ import annotations

import json
import stat
import subprocess
from pathlib import Path
from typing import Any

from . import autotune_catalog, campaign_build, generated_tree, provenance, runtime_smoke
from .artifacts import ArtifactStore
from .builds import (BuildIdentityError, BuildPlan, binary_hash, build_directory,
                     effective_build_id, parse_effective_configure, validate_reuse)
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
        emit_result = autotune_catalog.emit(
            manifest, source_root, artifact_root, generated_root=generated_root)

        tree_manifest = generated_tree.build_manifest(
            generated_root, compile_inputs=emit_result.compile_input_paths)

        return {
            "manifest": manifest,
            "generated_tree": tree_manifest,
            "workload_id": inventory_ref.content_hash,
        }

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
        by_kind = {ref.kind: ref for ref in inputs}
        if len(inputs) != 2 or set(by_kind) != {"manifest", "generated-tree"}:
            raise campaign_build.CampaignBuildError(
                f"build worker expects exactly one 'manifest' and one "
                f"'generated-tree' input from the generate stage, got "
                f"{[ref.kind for ref in inputs]}"
            )
        generated_tree_document = json.loads(
            by_kind["generated-tree"].path.read_text(encoding="utf-8"))

        generated_root = context.work_root / "runs" / run_id / "generate" / "generated"
        if not generated_root.is_dir():
            raise campaign_build.CampaignBuildError(
                f"expected generate stage's generated/ directory at "
                f"{generated_root}, but it does not exist"
            )
        # Prove the compile inputs on disk are still exactly what generate
        # published, BEFORE trusting them to configure/compile -- this is
        # the actual gap that made generated/ an unverified side-channel:
        # build previously read this directory by run_id convention alone,
        # with nothing checking it hadn't been modified since generate ran.
        generated_tree.verify_tree(generated_root, generated_tree_document)

        build_dir = build_directory(context, source_slice_id, build_plan)
        build_dir.mkdir(parents=True, exist_ok=True)
        binary = build_dir / binary_relative_path
        # toolchain_request is a tuple of tuples; metadata is read back from
        # JSON, which only knows lists. Normalize once, to the same
        # JSON-round-tripped shape on both the write and the reuse-check
        # side, rather than comparing a tuple to a list and always losing.
        expected_toolchain = [list(pair) for pair in build_plan.toolchain_request]
        # Named per-binary, not one metadata file per build_dir: cmake_targets
        # can request a different binary from the same build_plan_id/build_dir
        # (a shared configure cache across targets is legitimate), and a
        # single metadata file would describe only whichever target was
        # built first.
        metadata_path = build_dir / f"bigcherry-build-metadata-{binary.name}.json"

        reused = False
        if metadata_path.is_file():
            # A prior build claims to have already produced this exact
            # (source_slice_id, build_plan_id, binary) triple -- content-
            # addressed by construction (build_directory() is keyed on
            # both). A metadata file that fails its own recorded identity
            # here means real corruption or tampering, not "just rebuild
            # it": silently recompiling over a directory whose provenance
            # doesn't check out would hide exactly the kind of divergence
            # RE14's fail-closed philosophy exists to catch.
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                validate_reuse(metadata, build_plan, binary=binary,
                               expected_toolchain=expected_toolchain)
            except (BuildIdentityError, json.JSONDecodeError, OSError) as exc:
                raise campaign_build.CampaignBuildError(
                    f"build directory {build_dir} has an existing "
                    f"{metadata_path.name} but it failed reuse validation "
                    f"-- refusing to silently rebuild over it: {exc}"
                ) from exc
            reused = True

        if not reused:
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

            if not binary.is_file():
                raise campaign_build.CampaignBuildError(
                    f"build did not produce the expected binary: {binary}"
                )
            # Verify again, after compiling: a pass before configure alone
            # would still miss a modification that happened WHILE the
            # compiler was running, between the pre-configure check and
            # publication. Only meaningful when a compile actually ran --
            # the reuse path never invokes the compiler.
            generated_tree.verify_tree(generated_root, generated_tree_document)

            effective_configure = parse_effective_configure(build_dir / "CMakeCache.txt")
            metadata = {
                "source_slice_id": source_slice_id,
                "build_plan_id": build_plan.build_plan_id,
                "effective_configure": effective_configure,
                "build_id": effective_build_id(effective_configure),
                "toolchain": expected_toolchain,
                "binary_hash": binary_hash(binary),
            }
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

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

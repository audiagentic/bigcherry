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
NOT check out against its OWN recorded identity (validate_reuse()) is a hard
failure, not a silent rebuild -- that would hide real corruption or
tampering of build output behind an unremarkable recompile. This is a
different, lower layer than CampaignRun's own StageRecord-based reuse
(campaign.py): that reuse skips re-executing a stage within one
CampaignRun/run_id; this reuse lets a fresh process/run_id skip recompiling
entirely when a prior run (any run_id) already produced the identical build.

validate_reuse() alone is not sufficient, though: BuildPlan does not (and
currently cannot cheaply) depend on the catalog's candidate architectures,
so two generate runs given different --arch values can produce different
generated/ catalogs while still sharing a build_plan_id and build_dir.
generated_tree.py's compile_inputs_hash exists specifically to distinguish
that case (see its docstring); the build worker additionally requires the
cached metadata's recorded compile_inputs_hash to match the CURRENT
generate stage's before trusting a cache hit. A mismatch there is not
tampering -- it is a legitimate cache miss, so it falls through to a fresh
build rather than raising.

binary_hash alone is also not sufficient: RE09's own investigation
established that the meaningful HIP dispatch logic lives in
libggml-hip.so, not the tiny llama-bench launcher stub validate_reuse()
was originally checking. The build worker additionally requires a
runtime_bundle_hash (builds.resolve_runtime_artifacts/runtime_bundle_hash)
covering every shared library this build produced alongside the launcher
to match too -- UNLIKE the compile_inputs_hash check above, a mismatch
here (given source/build/compile-inputs identity all already agreed) means
something modified files inside what should be an untouched cached
directory, so it IS treated as tampering: a hard failure, not a fresh
rebuild.
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
                     effective_build_id, parse_effective_configure,
                     resolve_runtime_artifacts, runtime_bundle_hash, validate_reuse)
from .context import ProjectContext
from .pipeline import ArtifactRef


#: Need names the generate/build workers know how to interpret. Fail
#: closed on anything else -- BuildPlan has no generic identity slot for
#: an unknown need kind yet (only inventory_hash/winners_hash), so a need
#: name outside this set has nowhere safe to record its identity.
SUPPORTED_BUILD_NEEDS = frozenset({"inventory", "promoted-winners"})


def make_generate_worker(
    *,
    context: ProjectContext,
    source_root: Path,
    run_id: str,
    variant_set: str,
    architectures: list[str],
    upstream_revision: str,
    required_needs: frozenset[str] = frozenset({"inventory"}),
):
    """Returns the callable ``_run_generate`` expects: takes a by-kind
    ``{need_name: ArtifactRef}`` mapping (RE17 -- config.Build.needs is the
    authority for what a build kind requires, not a hardcoded single
    inventory input), returns ``{"manifest": ..., "generated_tree": ...}``.

    No ``workload_id`` in the result any more: the lane already knows it
    (the inventory ref's own content_hash, computed before generate ever
    runs) or knows there isn't one (needs=[] builds) -- generate
    "discovering" it was backwards.
    """

    def generate(inputs: Any) -> dict[str, Any]:
        actual = frozenset(inputs)
        if actual != required_needs:
            raise campaign_build.CampaignBuildError(
                f"generate worker expected needs {sorted(required_needs)}, "
                f"got {sorted(actual)}"
            )
        unsupported = actual - SUPPORTED_BUILD_NEEDS
        if unsupported:
            raise campaign_build.CampaignBuildError(
                f"generate worker cannot interpret need(s): {sorted(unsupported)}"
            )

        inventory_ref = inputs.get("inventory")
        winners_ref = inputs.get("promoted-winners")
        inventory = (
            autotune_catalog.Inventory.from_json(inventory_ref.path)
            if inventory_ref is not None else None
        )
        # winners feeds BOTH the winners= filtering param (replay-slim's
        # variant reduction) AND correctness_source= (production variant
        # sets consume the same evidence as correctness input) -- matching
        # what the legacy CLI already does with the same file; do not
        # assume winners only matters for replay-slim's own filtering.
        winners = (
            autotune_catalog.read_winners(winners_ref.path)
            if winners_ref is not None else None
        )

        manifest = autotune_catalog.build_manifest(
            source_root, variant_set=variant_set, architectures=architectures,
            inventory=inventory, source_revision=upstream_revision,
            winners=winners,
            correctness_source=winners_ref.path if winners_ref is not None else None,
        )

        stage_root = context.work_root / "runs" / run_id / "generate"
        artifact_root = stage_root / "catalog"
        generated_root = stage_root / "generated"
        emit_result = autotune_catalog.emit(
            manifest, source_root, artifact_root, generated_root=generated_root)

        tree_manifest = generated_tree.build_manifest(
            generated_root, compile_inputs=emit_result.compile_input_paths)

        return {"manifest": manifest, "generated_tree": tree_manifest}

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
    workload_id: str | None,
    lane_inputs: dict[str, ArtifactRef] | None = None,
    has_generate_stage: bool = True,
    cmake_targets: tuple[str, ...] = (),
):
    """Returns the callable ``_run_build_scoped`` expects for the build
    stage: takes generate's output ArtifactRef tuple (or none at all, for
    builds with no generate stage -- stock), configures and compiles for
    real, returns already-published+verified ArtifactRefs.

    ``lane_inputs`` is the same by-kind mapping the generate worker
    received (e.g. ``{"inventory": ref}``) -- the build worker needs the
    inventory's real, verified path for ``GGML_HIP_AUTOTUNE_SIGNATURE_FILE``,
    independent of whether generation happened, so it is threaded through
    here rather than re-derived from generate's stage inputs.
    """
    lane_inputs = lane_inputs or {}

    def run_build(inputs: tuple[ArtifactRef, ...]) -> tuple[ArtifactRef, ...]:
        if has_generate_stage:
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
            # Prove the compile inputs on disk are still exactly what
            # generate published, BEFORE trusting them to configure/compile
            # -- this is the actual gap that made generated/ an unverified
            # side-channel: build previously read this directory by run_id
            # convention alone, with nothing checking it hadn't been
            # modified since generate ran.
            generated_tree.verify_tree(generated_root, generated_tree_document)
            compile_inputs_hash = generated_tree_document["compile_inputs_hash"]
        else:
            # A build with no generate stage (stock: needs=[], no
            # variant_set) has nothing to receive from a prior stage --
            # absence of generation is a real graph property, not a
            # zero-file generation run to fabricate an empty artifact for.
            if inputs:
                raise campaign_build.CampaignBuildError(
                    f"build worker for a non-generated build expects no "
                    f"stage inputs, got {[ref.kind for ref in inputs]}"
                )
            generated_tree_document = None
            generated_root = None
            compile_inputs_hash = None

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
                # Computed from whatever is CURRENTLY on disk next to the
                # cached binary, not trusted from metadata -- validate_reuse
                # compares this against the recorded runtime_bundle_hash, so
                # a library modified since the metadata was written (even
                # with the launcher itself untouched) is caught here.
                current_runtime_hash = None
                if binary.is_file():
                    current_runtime_hash = runtime_bundle_hash({
                        artifact.name: binary_hash(artifact)
                        for artifact in resolve_runtime_artifacts(binary)
                    })
                validate_reuse(metadata, build_plan, binary=binary,
                               expected_toolchain=expected_toolchain,
                               runtime_bundle_hash=current_runtime_hash)
            except (BuildIdentityError, json.JSONDecodeError, OSError) as exc:
                raise campaign_build.CampaignBuildError(
                    f"build directory {build_dir} has an existing "
                    f"{metadata_path.name} but it failed reuse validation "
                    f"-- refusing to silently rebuild over it: {exc}"
                ) from exc
            # validate_reuse() only proves this metadata's OWN recorded
            # identity is self-consistent and matches the requested
            # BuildPlan/toolchain/binary -- it says nothing about whether
            # the cached binary was compiled from the SAME generated
            # catalog this run's generate stage just produced. BuildPlan
            # (and therefore build_plan_id / build_directory()) does not
            # depend on the catalog's candidate architectures, so two
            # generate runs asked for different --arch values can produce
            # different registry.inc/candidate sets while still sharing a
            # build_dir. generated_tree.py's own compile_inputs_hash exists
            # specifically to answer that question (see its docstring) --
            # a mismatch here is a real gap gpt-auto-agent review found:
            # without this check, the second run would silently reuse a
            # binary built from an entirely different candidate catalog.
            #
            # This is NOT tampering of build_dir's own identity (that's
            # what validate_reuse() already checked and raises on above) --
            # it is a legitimate cache miss: the same requested BuildPlan
            # genuinely was built from a different generated catalog. Fall
            # through to a fresh build rather than failing closed here.
            # None == None for a non-generated build is correct: there is
            # no generated catalog to diverge, so a stock build's metadata
            # matching (source_slice_id, build_plan_id, toolchain, binary,
            # runtime bundle) is already the complete identity -- nothing
            # else could legitimately differ between two "reuses" of it.
            reused = metadata.get("generated_compile_inputs_hash") == compile_inputs_hash

        if not reused:
            inventory_ref = lane_inputs.get("inventory")
            configure_args = campaign_build.cmake_configure_args(
                build, platform, source_root, build_dir,
                generated_root=generated_root,
                inventory=inventory_ref.path if inventory_ref is not None else None,
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
            if has_generate_stage:
                # Verify again, after compiling: a pass before configure
                # alone would still miss a modification that happened WHILE
                # the compiler was running, between the pre-configure check
                # and publication. Only meaningful when a compile actually
                # ran AND there was a generated tree to begin with.
                generated_tree.verify_tree(generated_root, generated_tree_document)

            effective_configure = parse_effective_configure(build_dir / "CMakeCache.txt")
            runtime_artifacts = {
                artifact.name: binary_hash(artifact)
                for artifact in resolve_runtime_artifacts(binary)
            }
            metadata = {
                "source_slice_id": source_slice_id,
                "build_plan_id": build_plan.build_plan_id,
                "effective_configure": effective_configure,
                "build_id": effective_build_id(effective_configure),
                "toolchain": expected_toolchain,
                "binary_hash": binary_hash(binary),
                "generated_compile_inputs_hash": compile_inputs_hash,
                "runtime_artifacts": runtime_artifacts,
                "runtime_bundle_hash": runtime_bundle_hash(runtime_artifacts),
            }
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        doc = provenance.make(
            project={}, source={"source_slice_id": source_slice_id},
            build={"build_plan_id": build_plan.build_plan_id},
            workload={"workload_id": workload_id} if workload_id is not None else {},
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
    workload_id: str | None,
    spec: runtime_smoke.RuntimeSmokeSpec,
    environment: dict[str, str] | None = None,
):
    """Returns the callable ``_run_build_scoped`` expects for the
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
            workload={"workload_id": workload_id} if workload_id is not None else {},
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

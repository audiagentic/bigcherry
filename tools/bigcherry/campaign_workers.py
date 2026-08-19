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
import os
import stat
import subprocess
from pathlib import Path
from typing import Any

from . import (
    autotune_catalog,
    campaign_build,
    generated_tree,
    provenance,
    runtime_smoke,
)
from .artifacts import ArtifactStore
from .builds import (
    BuildIdentityError,
    BuildPlan,
    binary_hash,
    build_directory,
    effective_build_id,
    parse_effective_configure,
    resolve_runtime_artifacts,
    runtime_bundle_hash,
    validate_reuse,
)
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
            if inventory_ref is not None
            else None
        )
        # winners feeds BOTH the winners= filtering param (replay-slim's
        # variant reduction) AND correctness_source= (production variant
        # sets consume the same evidence as correctness input) -- matching
        # what the legacy CLI already does with the same file; do not
        # assume winners only matters for replay-slim's own filtering.
        winners = (
            autotune_catalog.read_winners(winners_ref.path)
            if winners_ref is not None
            else None
        )

        manifest = autotune_catalog.build_manifest(
            source_root,
            variant_set=variant_set,
            architectures=architectures,
            inventory=inventory,
            source_revision=upstream_revision,
            winners=winners,
            correctness_source=winners_ref.path if winners_ref is not None else None,
        )

        stage_root = context.work_root / "runs" / run_id / "generate"
        artifact_root = stage_root / "catalog"
        generated_root = stage_root / "generated"
        emit_result = autotune_catalog.emit(
            manifest, source_root, artifact_root, generated_root=generated_root
        )

        tree_manifest = generated_tree.build_manifest(
            generated_root, compile_inputs=emit_result.compile_input_paths
        )

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
    extra_binary_names: tuple[str, ...] = (),
    source_provenance: provenance.SourceProvenance | None = None,
    project_revision: str = "",
    local_provenance_class: provenance.ProvenanceClass = "production",
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

    ``extra_binary_names`` (RE26): additional executables built alongside
    ``binary_relative_path`` from the same configure (e.g. test-backend-ops
    alongside the tune lane's llama-bench), published into the SAME runtime
    bundle as ``binary`` since they share its shared-library closure. The
    caller is responsible for including these names in ``cmake_targets`` too
    -- this parameter only controls bundle membership, not what gets built.
    """
    lane_inputs = lane_inputs or {}

    def run_build(inputs: tuple[ArtifactRef, ...]) -> tuple[ArtifactRef, ...]:
        # Computed unconditionally (a cheap Path join): the post-compile
        # re-verify below needs it typed as a plain Path rather than
        # Path | None, and for a non-generated build it is never read.
        generated_root = context.work_root / "runs" / run_id / "generate" / "generated"
        if has_generate_stage:
            by_kind = {ref.kind: ref for ref in inputs}
            if len(inputs) != 2 or set(by_kind) != {"manifest", "generated-tree"}:
                raise campaign_build.CampaignBuildError(
                    f"build worker expects exactly one 'manifest' and one "
                    f"'generated-tree' input from the generate stage, got "
                    f"{[ref.kind for ref in inputs]}"
                )
            tree_document = json.loads(
                by_kind["generated-tree"].path.read_text(encoding="utf-8")
            )

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
            generated_tree.verify_tree(generated_root, tree_document)
            compile_inputs_hash = tree_document["compile_inputs_hash"]
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
            # Empty (not None): the post-compile re-verify is gated on
            # has_generate_stage and never runs here; a plain dict keeps
            # the variable's type honest without an Optional dance.
            tree_document: dict[str, Any] = {}
            compile_inputs_hash = None

        build_dir = build_directory(context, source_slice_id, build_plan)
        build_dir.mkdir(parents=True, exist_ok=True)
        binary = build_dir / binary_relative_path

        def _resolve_binary(candidate: Path) -> Path:
            # First production run on a windows platform (local RD04
            # control lane, 2026-08-19): cmake/ninja target names are
            # extension-less ("llama-bench") but the produced image file
            # is "llama-bench.exe", so the plain path never exists on
            # win32. Every is_file() gate in this worker (reuse check,
            # post-build check) was therefore permanently false on a
            # windows platform: each lane run forced a full recompile and
            # then failed the post-build check even though the .exe sat
            # right next to it. Accept the .exe variant on Windows only;
            # linux paths are byte-for-byte unchanged.
            if not candidate.is_file() and os.name == "nt":
                exe_variant = candidate.with_name(candidate.name + ".exe")
                if exe_variant.is_file():
                    return exe_variant
            return candidate

        binary = _resolve_binary(binary)
        # RE26: extra executables live next to `binary` (same bin/ dir --
        # cmake places every target's output there), resolved once here so
        # every downstream resolve_runtime_artifacts() call site (reuse-hash
        # check, fresh-build metadata, publish loop) sees the same set.
        extra_binaries = tuple(binary.parent / name for name in extra_binary_names)
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
        # Keyed on the TARGET name (binary_relative_path), not the resolved
        # binary file name: on windows the resolved name carries .exe only
        # once the file exists, so keying on binary.name would make the
        # metadata file (and with it the whole reuse gate) move between the
        # first build and every later run. Linux: identical to before.
        metadata_path = build_dir / (
            f"bigcherry-build-metadata-{Path(binary_relative_path).name}.json"
        )

        metadata: dict[str, Any] | None = None
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
                # Local (not metadata) while validating: metadata is typed
                # Optional until a validated record exists, and validate_reuse
                # takes a plain dict -- binding it only after validation keeps
                # the 'None means no validated identity' invariant exact.
                cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                # Computed from whatever is CURRENTLY on disk next to the
                # cached binary, not trusted from metadata -- validate_reuse
                # compares this against the recorded runtime_bundle_hash, so
                # a library modified since the metadata was written (even
                # with the launcher itself untouched) is caught here.
                current_runtime_hash = None
                # A cached build_dir from before this lane requested
                # extra_binary_names has the main binary but not the
                # extras -- that is a genuine cache miss (this exact
                # cmake_targets set was never built here), not corruption,
                # so only compute a comparable hash when every expected
                # member is actually present.
                if binary.is_file() and all(p.is_file() for p in extra_binaries):
                    current_runtime_hash = runtime_bundle_hash(
                        {
                            artifact.name: binary_hash(artifact)
                            for artifact in resolve_runtime_artifacts(
                                binary, extra_binaries=extra_binaries)
                        }
                    )
                validate_reuse(
                    cached_metadata,
                    build_plan,
                    binary=binary,
                    expected_toolchain=expected_toolchain,
                    runtime_bundle_hash=current_runtime_hash,
                )
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
            # Only NOW does the record become `metadata` (the validated
            # identity) -- after validate_reuse() has proven it, not before.
            metadata = cached_metadata
            reused = (
                cached_metadata.get("generated_compile_inputs_hash")
                == compile_inputs_hash
            )

        if not reused:
            inventory_ref = lane_inputs.get("inventory")
            configure_args = campaign_build.cmake_configure_args(
                build,
                platform,
                source_root,
                build_dir,
                generated_root=generated_root,
                inventory=inventory_ref.path if inventory_ref is not None else None,
                c_compiler=platform.c_compiler,
                cxx_compiler=platform.cxx_compiler,
            )
            subprocess.run(configure_args, cwd=source_root, check=True)
            subprocess.run(
                campaign_build.cmake_build_args(build_dir, targets=cmake_targets),
                cwd=source_root,
                check=True,
            )

            binary = _resolve_binary(binary)
            if not binary.is_file():
                raise campaign_build.CampaignBuildError(
                    f"build did not produce the expected binary: {binary} "
                    f"(also probed {binary.with_name(binary.name + '.exe')})"
                )
            for extra in extra_binaries:
                if not extra.is_file():
                    raise campaign_build.CampaignBuildError(
                        f"build did not produce the expected extra binary: {extra}"
                    )
            if has_generate_stage:
                # Verify again, after compiling: a pass before configure
                # alone would still miss a modification that happened WHILE
                # the compiler was running, between the pre-configure check
                # and publication. Only meaningful when a compile actually
                # ran AND there was a generated tree to begin with.
                generated_tree.verify_tree(generated_root, tree_document)

            effective_configure = parse_effective_configure(
                build_dir / "CMakeCache.txt"
            )
            runtime_artifacts = {
                artifact.name: binary_hash(artifact)
                for artifact in resolve_runtime_artifacts(
                    binary, extra_binaries=extra_binaries)
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
                json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

        if metadata is None:
            # Unreachable in practice (a cache hit either validates and binds
            # metadata or raises; otherwise the fresh-build branch above ran),
            # but fail closed anyway: never publish a build artifact whose
            # identity record was not established on a path we can see.
            raise campaign_build.CampaignBuildError(
                f"build directory {build_dir} has no validated build metadata -- "
                "neither a cache hit passed reuse validation nor a fresh build "
                "completed; refusing to publish unattributed artifacts"
            )

        # RE25.2: typed provenance with the REAL build identity (the
        # effective_build_id validate_reuse() itself trusts) and the full
        # inherited source lineage -- replacing an empty project namespace
        # and a bare source_slice_id string. Reused builds carry the same
        # effective identity as a fresh one (metadata['build_id'] is set on
        # both paths), so the provenance is identical either way.
        # RE25.2 review fix: build must record the artifacts it actually
        # consumed -- generate's manifest/tree (the stage inputs, verified
        # against the store above) plus the lane inputs (e.g. inventory,
        # promoted-winners) -- otherwise the lineage chain terminates here:
        # generate records its own parents, but a build that republishes
        # with an empty parent set severs the chain exactly where release
        # provenance needs it most. Recording the IDs now does not pull
        # RE25.3's sticky taint forward; it just makes the taint have real
        # parents to walk when it lands.
        consumed: dict[str, ArtifactRef] = {ref.kind: ref for ref in inputs}
        consumed.update(lane_inputs)
        input_entries, parent_ids = provenance.lane_input_provenance(consumed)
        # RE25.3: the PROVENANCE CLASS taint needs the real parent documents,
        # not just their identity strings (the IDs above are recorded too, so
        # sticky class derivation and identity lineage agree). Imported-
        # legacy lane inputs (raw Path imports, downgraded legacy refs) now
        # actually taint build + runtime-bundle instead of being silently
        # re-stamped production here. Parents that don't parse as schema-v2
        # docs fail closed: a build whose parents' classes can't be verified
        # must not claim production.
        try:
            parent_docs = tuple(
                provenance.ProvenanceV2.from_document(ref.provenance)
                for ref in consumed.values()
            )
        except provenance.ProvenanceError as exc:
            raise campaign_build.CampaignBuildError(
                f"build input provenance is not a schema-v2 document: {exc}"
            ) from exc
        build_provenance = provenance.BuildProvenance(
            build_plan_id=build_plan.build_plan_id,
            effective_build_id=metadata["build_id"],
            binary_hash=binary_hash(binary),
            runtime_bundle_hash=metadata["runtime_bundle_hash"],
            targets=tuple(build_plan.targets),
            catalog_architectures=tuple(build_plan.catalog_architectures),
            inputs=input_entries,
        )
        doc = provenance.derive(
            parents=parent_docs,
            parent_artifact_ids=parent_ids,
            project_revision=project_revision,
            source=source_provenance
            if source_provenance is not None
            else provenance.SourceProvenance(source_slice_id=source_slice_id),
            build=build_provenance,
            workload=provenance.WorkloadProvenance(workload_id=workload_id),
            run_id=run_id,
            producer_stage="build",
            local_class=local_provenance_class,
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
        # RE25.2: descriptor-backed publication -- the binary artifact now
        # has a real artifact_id and persists a rehydratable identity,
        # not just bytes + a provenance dict on an in-memory ref.
        # publish_file_ref verifies after publishing; the digest check below
        # keeps the build-tree cross-check (store digest vs independently
        # hashed binary) that plain publish_file never provided by itself.
        binary_ref = store.publish_file_ref(
            binary_relative, binary, kind="binary", provenance=doc
        )
        if binary_ref.content_hash != binary_hash(binary):
            raise campaign_build.CampaignBuildError(
                "published binary digest differs from build-tree binary"
            )
        # ArtifactStore copies bytes only; it makes no promise about
        # executable mode, correctly -- most published artifacts are data
        # (JSON, manifests), not executables. A binary artifact is the one
        # kind that specifically needs its execute bit set here, by the
        # caller that knows it is a binary, not inside the general-purpose
        # store. Owner-execute only: this store is not (yet) a shared
        # multi-user artifact store, and mkstemp's default 0600 becoming
        # 0700 is the narrowest change that makes the binary runnable by
        # the same process/user that just published it.
        binary_ref.path.chmod(binary_ref.path.stat().st_mode | stat.S_IXUSR)
        refs.append(binary_ref)

        # RE07/RV48 audit fix: publish the FULL runtime .so closure too --
        # previously only the launcher (binary) reached ArtifactStore, so a
        # downstream consumer reading binary_ref alone had no immutable,
        # verified copy of the libraries it actually loads at runtime
        # (libggml-hip.so etc, where RE09 established the real HIP dispatch
        # logic lives). Every regular runtime file next to the binary is
        # published under the same content-addressed prefix, verified
        # individually, then described by one runtime-bundle manifest
        # artifact carrying the same identity fields validate_reuse()
        # already trusts, so a downstream consumer has one self-describing
        # reference instead of having to re-derive the bundle shape itself.
        bundle_members: dict[str, str] = {}
        for artifact in resolve_runtime_artifacts(binary, extra_binaries=extra_binaries):
            member_relative = f"{prefix}/{artifact.name}"
            member_digest = store.publish_file(member_relative, artifact)
            if not store.verify(member_relative, member_digest):
                raise campaign_build.CampaignBuildError(
                    f"published {member_relative} failed verification"
                )
            member_path = store.root / member_relative
            if artifact == binary or artifact in extra_binaries:
                # RE26: an extra binary (e.g. test-backend-ops) is itself
                # something a later stage runs, not a passive shared
                # library -- same reasoning as binary_ref's own chmod above.
                member_path.chmod(member_path.stat().st_mode | stat.S_IXUSR)
            bundle_members[artifact.name] = member_digest
        computed_runtime_bundle_hash = runtime_bundle_hash(bundle_members)
        bundle_manifest = {
            "entrypoint": binary.name,
            "members": bundle_members,
            "runtime_bundle_hash": computed_runtime_bundle_hash,
            "effective_build_id": metadata["build_id"],
            "generated_compile_inputs_hash": metadata.get(
                "generated_compile_inputs_hash"
            ),
            "toolchain": expected_toolchain,
        }
        # Content-addressed by a hash of the FULL manifest, not just
        # runtime_bundle_hash (member file bytes only): the same build_plan_id
        # can legitimately recompile to a different generated catalog on a
        # later run (compile_inputs_hash mismatch is a documented legitimate
        # cache-miss above, not tampering) while the compiled binary/library
        # bytes happen to stay identical -- runtime_bundle_hash alone would
        # not change, but the manifest's own generated_compile_inputs_hash
        # field would, and a path keyed on identity alone would collide
        # ArtifactStore's immutability check on that recompile even though
        # nothing is wrong.
        bundle_content_hash = ArtifactStore.digest(
            json.dumps(bundle_manifest, sort_keys=True, separators=(",", ":")).encode()
        )
        bundle_relative = f"{prefix}/runtime-bundle-{bundle_content_hash}.json"
        refs.append(
            store.publish_json_ref(
                bundle_relative, bundle_manifest, kind="runtime-bundle", provenance=doc
            )
        )

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
    source_provenance: provenance.SourceProvenance | None = None,
    project_revision: str = "",
    local_provenance_class: provenance.ProvenanceClass = "production",
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

        # RE07/RV48 follow-up (GPT review, 2026-08-17): binary_ref alone
        # only proves the launcher is unchanged -- it says nothing about
        # whether the .so files it will actually load at runtime (where
        # RE09 established the real HIP dispatch logic lives) are the
        # verified ones this build published, or a mutable build-tree copy
        # that has since changed. Require the runtime-bundle artifact too,
        # verify every one of its members against the store, and cross-
        # check the manifest's own entrypoint hash against binary_ref's --
        # only THEN is running binary_ref.path actually running the
        # verified bundle rather than merely a verified launcher.
        bundles = [ref for ref in inputs if ref.kind == "runtime-bundle"]
        if len(bundles) != 1:
            raise campaign_build.CampaignBuildError(
                f"runtime-smoke worker expects exactly one 'runtime-bundle' "
                f"input, found {len(bundles)}"
            )
        bundle_ref = bundles[0]
        if not store.verify(
            bundle_ref.path.resolve().relative_to(store.root), bundle_ref.content_hash
        ):
            raise campaign_build.CampaignBuildError(
                "runtime-bundle manifest failed store verification before smoke"
            )
        bundle_manifest = json.loads(bundle_ref.path.read_text(encoding="utf-8"))
        entrypoint_hash = bundle_manifest["members"].get(bundle_manifest["entrypoint"])
        if entrypoint_hash != binary_ref.content_hash:
            raise campaign_build.CampaignBuildError(
                "runtime-bundle manifest's entrypoint hash does not match "
                "the binary this smoke stage was given"
            )
        for member_name, member_hash in bundle_manifest["members"].items():
            member_relative = (
                bundle_ref.path.parent.relative_to(store.root) / member_name
            )
            if not store.verify(member_relative, member_hash):
                raise campaign_build.CampaignBuildError(
                    f"runtime-bundle member {member_name!r} failed store "
                    f"verification before smoke -- refusing to run against "
                    f"a tampered or missing runtime dependency"
                )

        argv = runtime_smoke.smoke_argv(binary_ref.path, spec)
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            env=environment,
        )
        if completed.returncode != 0:
            raise runtime_smoke.SmokeError(
                f"runtime smoke exited {completed.returncode}: {completed.stderr[-2000:]}"
            )
        rows = runtime_smoke.evaluate_smoke_result(completed.stdout)

        # RE25.2: the smoke result records its REAL parent artifacts (the
        # binary and runtime bundle it actually ran -- verified against the
        # store just above), not an anonymous build identity.
        # RE25.3: those same parents now also drive the provenance CLASS
        # taint -- a smoke run against a tainted (imported-legacy /
        # development) build is itself tainted, and never promotable.
        entries, parent_ids = provenance.lane_input_provenance(
            {
                "binary": binary_ref,
                "runtime-bundle": bundle_ref,
            }
        )
        try:
            parent_docs = (
                provenance.ProvenanceV2.from_document(binary_ref.provenance),
                provenance.ProvenanceV2.from_document(bundle_ref.provenance),
            )
        except provenance.ProvenanceError as exc:
            raise runtime_smoke.SmokeError(
                f"smoke parent provenance is not a schema-v2 document: {exc}"
            ) from exc
        doc = provenance.derive(
            parents=parent_docs,
            parent_artifact_ids=parent_ids,
            project_revision=project_revision,
            source=source_provenance
            if source_provenance is not None
            else provenance.SourceProvenance(source_slice_id=source_slice_id),
            build=provenance.BuildProvenance(
                build_plan_id=build_plan_id, inputs=entries
            ),
            workload=provenance.WorkloadProvenance(workload_id=workload_id),
            run_id=run_id,
            producer_stage="runtime-smoke",
            local_class=local_provenance_class,
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
        return (
            store.publish_json_ref(
                relative, result, kind="smoke-result", provenance=doc
            ),
        )

    return run_smoke

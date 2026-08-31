"""RE14 cutover: the actual executor work behind the new campaign services.

CampaignRun/PipelineService/CampaignGraph/ArtifactStore/BuildPlan/workspace
all existed, unit-tested, with zero production callers before RE14. This
module is the missing glue: it materialises an isolated source per
``config.Source`` (never the one shared mutable checkout the legacy
``cmd_build`` path uses), configures/builds against a content-addressed
build directory, and publishes outputs through :class:`ArtifactStore` with
an explicit content_hash-vs-actual-bytes verification step.

That verification step is deliberate, not incidental: ``PipelineService``
only checks provenance *field* compatibility on its ``ArtifactRef`` inputs
and outputs, and ``CampaignRun.execute`` stores whatever hashes an executor
returns without checking them against anything. Neither primitive verifies
that an artifact's bytes actually match its claimed hash -- that has to
happen here, in the executor, before an ArtifactRef claiming a given
content_hash is allowed to exist at all.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Callable

from . import source as campaign_source
from ..core import config as campaign_config
from ..build import builds
from ..source import identity as source_identity
from ..core.artifacts import ArtifactStore
from ..build.builds import BuildPlan
from ..core.context import ProjectContext
from ..source.identity import (
    SourceAttestation, SourceIdentityError, git_tree_oid, verify_source_attestation,
)
from ..source.workspace import (
    SourcePlan, UpstreamRepository, materialize, require_clean_bigcherry,
)

Runner = Callable[[list[str], Path], None]


class CampaignBuildError(RuntimeError):
    pass


#: Environment variables that can change what a build actually produces
#: without changing any requested CMake option -- PATH affects which tools
#: resolve at all, the rest affect where ROCm/HIP get found. Deliberately a
#: small fixed allowlist, not the whole ambient environment: hashing every
#: inherited variable would make build_plan_id sensitive to unrelated shell
#: state (terminal colors, unrelated app config) with no bearing on the
#: build, and would leak local machine details into a supposedly portable
#: identity.
_BUILD_RELEVANT_ENV_VARS = ("PATH", "ROCM_PATH", "HIP_PATH", "LD_LIBRARY_PATH")


def resolve_build_environment() -> tuple[tuple[str, str], ...]:
    """The subset of the current process environment relevant to
    ``BuildPlan.environment`` -- real values, not merely "whatever the
    caller happened to have set", so a build run under a different PATH/
    ROCm environment gets its own build_plan_id instead of silently
    colliding with one built under a different environment.
    """
    return tuple(
        (name, os.environ[name])
        for name in _BUILD_RELEVANT_ENV_VARS
        if name in os.environ
    )


def toolchain_request_for_platform(
    platform: campaign_config.Platform,
) -> tuple[tuple[str, str], ...]:
    """The requested toolchain identity for ``BuildPlan.toolchain_request``.

    These are literally the CMake controls that get passed on the command
    line (see ``cmake_configure_args``), not a separately-invented naming
    scheme -- so a BuildPlan's requested toolchain and the actual argv
    built from it stay traceable to the same names instead of two
    independent spellings of the same fact.

    Also folds in resolve_toolchain_versions() (real, content-level
    identity -- resolved compiler realpath, --version output, ROCm's own
    .info/version, cmake/ninja versions), not just the requested compiler
    PATH: this repo's linux-multi platform declares no c_compiler/
    cxx_compiler in recipes.toml at all, relying on whatever cc/c++ resolve
    to on PATH, and even an explicit path (e.g. /opt/rocm/llvm/bin/clang)
    is itself a symlink chain that an in-place ROCm upgrade repoints
    without changing. Since this is part of BuildPlan.toolchain_request,
    it is hashed into build_plan_id -- an in-place ROCm upgrade at the same
    path now gets its own build_directory() rather than colliding with (or
    being silently reused by) a build made under the old version, and
    pointing two BuildPlans at two different ROCm installs (e.g. via
    --c-compiler) produces two genuinely separate, side-by-side comparable
    build directories instead of one overwriting the other.
    """
    values: dict[str, str] = {"CMAKE_GENERATOR": "Ninja"}
    if platform.c_compiler:
        values["CMAKE_C_COMPILER"] = platform.c_compiler
    if platform.cxx_compiler:
        values["CMAKE_CXX_COMPILER"] = platform.cxx_compiler
    values.update(
        resolve_toolchain_versions(platform.c_compiler, platform.cxx_compiler)
    )
    return tuple(sorted(values.items()))


def resolve_toolchain_versions(
    c_compiler: str | None,
    cxx_compiler: str | None,
) -> dict[str, str]:
    """Real, content-level toolchain identity, not just a requested path.

    Resolves whichever compiler is actually in play (explicit override, or
    else whatever cc/c++ find on PATH -- resolving that explicitly rather
    than skipping fingerprinting whenever no override is configured is
    what lets a PATH-default build's cache still get invalidated by a
    real toolchain change), follows symlinks to the real file, and records
    --version output plus ROCm's own .info/version when the compiler lives
    under a ROCm install. Best-effort throughout: a probe that fails (tool
    missing, no ROCm layout, non-zero exit) is simply omitted, not an
    error -- this must not make BuildPlan construction fail on a machine
    without cmake/ninja/ROCm actually present yet (e.g. a dry planning
    call), and every value that IS resolved still participates in
    build_plan_id via the caller.
    """
    values: dict[str, str] = {}

    resolved_c = _resolve_compiler_path(c_compiler, ("cc", "clang", "gcc"))
    if resolved_c:
        values["c_compiler_realpath"] = str(resolved_c)
        version = _version_probe([str(resolved_c), "--version"])
        if version:
            values["c_compiler_version"] = version
        rocm_version = _find_rocm_version(resolved_c)
        if rocm_version:
            values["rocm_version"] = rocm_version

    resolved_cxx = _resolve_compiler_path(cxx_compiler, ("c++", "clang++", "g++"))
    if resolved_cxx:
        values["cxx_compiler_realpath"] = str(resolved_cxx)
        version = _version_probe([str(resolved_cxx), "--version"])
        if version:
            values["cxx_compiler_version"] = version

    cmake_version = _version_probe(["cmake", "--version"])
    if cmake_version:
        values["cmake_version"] = cmake_version
    ninja_version = _version_probe(["ninja", "--version"])
    if ninja_version:
        values["ninja_version"] = ninja_version
    return values


def _resolve_compiler_path(
    explicit: str | None, fallback_names: tuple[str, ...]
) -> Path | None:
    if explicit:
        candidate = Path(explicit)
        return candidate.resolve() if candidate.exists() else None
    for name in fallback_names:
        found = shutil.which(name)
        if found:
            return Path(found).resolve()
    return None


def _version_probe(argv: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            argv, capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    first_line = completed.stdout.splitlines()[0].strip() if completed.stdout else None
    return first_line or None


def _find_rocm_version(compiler_realpath: Path) -> str | None:
    """ROCm's own version file, at <rocm_root>/.info/version, where
    <rocm_root> is a few directories up from <rocm_root>/llvm/bin/clang.
    Walks upward rather than assuming a fixed depth -- robust to layout
    differences instead of a hardcoded ``.parents[2]`` -- but bounded, so a
    compiler that isn't part of a ROCm install at all doesn't walk all the
    way to the filesystem root looking for an unrelated .info/version.
    """
    for ancestor in list(compiler_realpath.parents)[:6]:
        candidate = ancestor / ".info" / "version"
        if candidate.is_file():
            try:
                return candidate.read_text(encoding="utf-8").strip()
            except OSError:
                return None
    return None


def _find_rocm_prefix(compiler_realpath: Path) -> Path | None:
    """The ROCm install root that CMake's find_package(hip) etc. need on
    CMAKE_PREFIX_PATH -- i.e. the ancestor of the resolved compiler that
    actually carries lib/cmake/hip/hip-config.cmake. Deliberately not a
    fixed-depth formula (e.g. two ``.parent``s): real ROCm layouts differ
    (confirmed against an actual install: the compiler can resolve to
    ``<root>/lib/llvm/bin/clang``, not the ``<root>/llvm/bin/clang`` a
    fixed-depth guess would assume), so this walks upward and checks for
    the actual marker file, the same pattern _find_rocm_version uses.
    """
    for ancestor in list(compiler_realpath.parents)[:6]:
        if (ancestor / "lib" / "cmake" / "hip" / "hip-config.cmake").is_file():
            return ancestor
    return None


def _default_runner(args: list[str], cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def _backend_configure_options(
    backend: str,
    platform: campaign_config.Platform,
) -> dict[str, str]:
    """Backend-specific cmake options that don't belong in a shared dict
    literal (RE30 phase 1: the split this project's Vulkan evaluation
    flagged as a prerequisite -- ``AMDGPU_TARGETS`` is HIP-specific and must
    not be emitted, even empty, for a non-HIP backend).

    Only ``"hip"`` (the default, unchanged) and ``"vulkan"`` (a stub -- no
    Vulkan patches or tuning subsystem exist yet, this just lets a Vulkan
    *stock* build configure) are known. Any other value is a caller bug.
    """
    if backend == "hip":
        return {"AMDGPU_TARGETS": ";".join(platform.targets)}
    if backend == "vulkan":
        # Stock lane only (RE30 phase 1): GGML_VULKAN=ON if the caller
        # hasn't already set it via build/platform options. No SDK/glslc
        # identity capture yet -- that needs a real Vulkan toolchain to
        # design against, deferred to a later phase.
        return {"GGML_VULKAN": "ON"}
    raise ValueError(f"unknown backend {backend!r} (expected 'hip' or 'vulkan')")


def cmake_configure_args(
    build: campaign_config.Build,
    platform: campaign_config.Platform,
    source_root: Path,
    build_dir: Path,
    *,
    generated_root: Path | None = None,
    inventory: Path | None = None,
    c_compiler: str | None = None,
    cxx_compiler: str | None = None,
    backend: str = "hip",
) -> list[str]:
    """The cmake configure argv for one v2 ``config.Build``/``config.Platform``.

    Mirrors ``__main__._cmake_configure_args`` semantics (same option
    precedence, same GGML_HIP_AUTOTUNE_* wiring) but reads the v2 config
    types (tuple-based options, tuple-based targets) instead of the legacy
    ``recipes.py`` dataclasses, since the new path never touches those.

    ``backend`` defaults to ``"hip"`` -- the only backend that existed
    before RE30 phase 1 -- so every existing caller is byte-for-byte
    unchanged. Passing ``backend="vulkan"`` swaps in the Vulkan stub options
    (see ``_backend_configure_options``) instead of ``AMDGPU_TARGETS``.
    ``campaign_workers.make_build_worker`` and ``campaign_lane.py``'s
    ``BuildPlan.cmake_options`` construction both resolve this from
    ``config.Source.backend`` (RE30's real ``vulkan-stock`` source uses it
    today, e.g. via ``bigcherry build --lane vulkan-stock:...``).
    """
    options: dict[str, str] = {
        "CMAKE_BUILD_TYPE": "Release",
        **dict(platform.options),
        **dict(build.options),
        **_backend_configure_options(backend, platform),
    }
    if build.variant_set:
        options["GGML_HIP_AUTOTUNE_VARIANT_SET"] = build.variant_set
        if generated_root is not None:
            options["GGML_HIP_AUTOTUNE_GENERATED_DIR"] = str(generated_root.resolve())
        if inventory is not None:
            options["GGML_HIP_AUTOTUNE_SIGNATURE_FILE"] = str(inventory.resolve())
    if c_compiler:
        options["CMAKE_C_COMPILER"] = c_compiler
    if cxx_compiler:
        options["CMAKE_CXX_COMPILER"] = cxx_compiler
    # Pointing --c-compiler at a specific ROCm install (to build/compare
    # against a different ROCm version) is not enough on its own: without
    # this, CMake's find_package(hip) etc. still search the DEFAULT
    # CMAKE_PREFIX_PATH, which could pull in a DIFFERENT ROCm version's
    # runtime/package-config files than the one the compiler came from --
    # silently mixing versions rather than genuinely isolating them.
    resolved_compiler = c_compiler or cxx_compiler
    if resolved_compiler and Path(resolved_compiler).exists():
        rocm_root = _find_rocm_prefix(Path(resolved_compiler).resolve())
        if rocm_root is not None:
            existing = options.get("CMAKE_PREFIX_PATH", "")
            prefix_path = rocm_root.as_posix()
            options["CMAKE_PREFIX_PATH"] = (
                f"{existing};{prefix_path}" if existing else prefix_path
            )
    return [
        "cmake",
        "-S",
        str(source_root),
        "-B",
        str(build_dir),
        "-G",
        "Ninja",
        *(f"-D{key}={value}" for key, value in sorted(options.items())),
    ]


def cmake_build_args(build_dir: Path, *, targets: tuple[str, ...] = ()) -> list[str]:
    args = ["cmake", "--build", str(build_dir), "-j"]
    if targets:
        args += ["--target", *targets]
    return args


def _source_metadata_path(destination: Path) -> Path:
    return destination.parent / f"{destination.name}.metadata.json"


def materialize_source(
    context: ProjectContext,
    plan: SourcePlan,
    *,
    allow_dirty_bigcherry: bool = False,
    verify_strict: bool = False,
) -> dict[str, Any]:
    """Materialise ``plan`` into an isolated, content-identified worktree.

    Idempotent by construction: the destination is keyed by
    ``campaign_source.materialization_plan_id()`` of the CONTENT-resolved
    identity (patch module content hashes + overlay file bytes, not just
    IDs/flags) -- RE04/RV48: an in-place edit to a patch module or overlay
    file under an unchanged canonical ID must not silently reuse the old
    materialisation. If that directory already carries metadata from a
    prior materialisation, it is reused rather than re-materialised -- but
    the cached destination TREE is re-hashed and checked against the
    stored ``source_tree_oid`` first (a hash-collision-only comparison
    would not be trusted silently, and neither would trusting a cached
    identity match without re-verifying the actual bytes on disk: a
    modified cached worktree must fail closed, not compile stale bytes).

    The dirty-BigCherry-tree check runs unconditionally here, BEFORE the
    cache-hit branch -- ``materialize()``'s own check only guards the
    fresh-creation path, which a cache hit never reaches.

    ``verify_strict`` (PA17/C2, adversarial-review follow-up): every check
    above proves the cached destination's live bytes are self-consistent
    with its own sidecar metadata -- none of them proves that metadata's
    OWN claimed composition actually PRODUCES that tree, i.e. a forged
    worktree paired with a fully self-consistent forged sidecar would
    still pass. When True, a cache hit additionally re-runs the exact same
    materialize() sequence into a throwaway scratch worktree and requires
    its tree OID to match, the only check not dependent on trusting
    anything previously written to disk by this module (mirrors HI82's
    ``_verify_by_rematerialization``). Off by default: with N campaign
    lanes sharing one cached source plan under the same plan_lock, this
    repeats a full worktree-add + overlay + whole patch composition + tree
    hash for every cache hit while the lock is held -- real GPU-campaign
    cost, not free, so it is opt-in for strict/release verification paths
    rather than every lane's default cache hit.
    """
    # Cheap fail-fast before even computing identity/waiting on the lock --
    # the authoritative check is the one repeated inside the lock below.
    require_clean_bigcherry(context, allow_dirty_bigcherry=allow_dirty_bigcherry)

    plan = campaign_source.resolve_materialization_inputs(context, plan)
    identity = campaign_source.resolve_materialization_identity(context, plan)
    plan_id = campaign_source.materialization_plan_id(identity)
    # PA12 (L6.2): two processes computing this same plan_id must not both
    # create/reuse/mutate the same destination worktree -- serialize the
    # whole inspect-cache -> materialize -> attest -> publish-metadata
    # sequence below per plan_id, not just the final write.
    with source_identity.plan_lock(context.work_root, plan_id):
        # RD100 (gpt-auto-agent review follow-up): the check above ran
        # BEFORE waiting on plan_lock -- a contended lock can mean a real,
        # possibly long wait (up to plan_lock's own timeout), during which
        # the shared BigCherry tree could become dirty. Re-check now that
        # the lock is actually held, immediately before any cache-hit
        # decision or fresh materialisation, so the property this check
        # exists to guarantee is still true at the moment it's relied on.
        require_clean_bigcherry(context, allow_dirty_bigcherry=allow_dirty_bigcherry)
        return _materialize_source_locked(
            context, plan, identity, plan_id,
            allow_dirty_bigcherry=allow_dirty_bigcherry, verify_strict=verify_strict,
        )


def _verify_by_rematerialization(
    context: ProjectContext, plan: SourcePlan, *,
    expected_tree_oid: str, allow_dirty_bigcherry: bool,
) -> None:
    """PA17/C2: deterministically re-derive the expected tree from scratch
    and compare, rather than trusting anything previously written to disk.

    Mirrors patch/source.py's HI82 ``_verify_by_rematerialization`` --
    the scratch directory never enters any persisted identity, and
    cleanup can never raise (unconditional rmtree + worktree prune,
    exactly like HI82's own finally block) regardless of how far
    materialize() got before failing.
    """
    scratch_dir = context.work_root / "sources" / f".verify-{uuid.uuid4().hex}"
    try:
        # Adversarial-review follow-up: an overlay that adds NEW files (the
        # whole point of an overlay) makes them untracked in the fresh
        # scratch worktree. materialize() already computed and returned the
        # exact allowed_untracked set for this plan (workspace.materialize's
        # own describe() call uses it) -- git_tree_oid() must be given that
        # SAME set here, or it raises on every legitimate overlay-added file
        # as an "unexpected" untracked file, breaking verify_strict for the
        # common (overlay-enabled) case instead of just the tampered one.
        scratch_metadata = materialize(
            context, plan, scratch_dir, allow_dirty_bigcherry=allow_dirty_bigcherry,
        )
        actual_tree_oid = git_tree_oid(
            scratch_dir,
            allowed_untracked=set(scratch_metadata.get("allowed_untracked", ())),
        )
    finally:
        # HI82's own proven-safe order: rmtree the directory FIRST (it may
        # be a real git worktree registration, a partial one, or not exist
        # at all if materialize() failed before add_detached_worktree even
        # ran -- all three are fine to rmtree), THEN `git worktree prune`
        # to clean up the now-dangling registration. Unlike
        # UpstreamRepository.remove_worktree() ("remove --force"), prune
        # never raises on a directory that's already gone, so this cleanup
        # can never itself fail regardless of how far materialize() got.
        shutil.rmtree(scratch_dir, ignore_errors=True)
        subprocess.run(
            ["git", "worktree", "prune"], cwd=context.upstream_repo, check=False,
            capture_output=True,
        )
    if actual_tree_oid != expected_tree_oid:
        raise CampaignBuildError(
            f"deterministic re-materialization mismatch for plan (expected "
            f"tree, freshly re-derived from base+composition+overlay, "
            f"{actual_tree_oid!r} != cached/verified tree {expected_tree_oid!r}) "
            f"-- the cached source directory's self-reported identity cannot "
            f"be trusted; refusing to reuse it"
        )


def _materialize_source_locked(
    context: ProjectContext,
    plan: SourcePlan,
    identity: dict[str, Any],
    plan_id: str,
    *,
    allow_dirty_bigcherry: bool,
    verify_strict: bool = False,
) -> dict[str, Any]:
    destination = context.work_root / "sources" / plan_id
    metadata_path = _source_metadata_path(destination)

    if destination.is_dir() and metadata_path.is_file():
        cached = json.loads(metadata_path.read_text(encoding="utf-8"))
        cached_plan = cached.get("plan", {})
        cached_patches = sorted(
            (item["patch_id"], item["content_hash"])
            for item in cached_plan.get("patches", [])
        )
        current_patches = sorted(
            (item["patch_id"], item["content_hash"]) for item in identity["patches"]
        )
        if (
            cached_plan.get("upstream_revision") != identity["upstream_revision"]
            or cached_plan.get("overlay_enabled") != identity["overlay_enabled"]
            or cached_patches != current_patches
            or cached_plan.get("required_state") != identity["required_state"]
            or cached.get("overlay_content_hash") != identity["overlay_content_hash"]
            or cached_plan.get("patch_set_id") != identity.get("patch_set_id")
            or cached_plan.get("classification") != identity.get("classification")
        ):
            raise CampaignBuildError(
                f"source directory {destination} exists with metadata for a "
                f"different plan than requested (plan id collision or stale "
                f"cache) -- refusing to reuse or overwrite it"
            )
        # Re-hash the cached destination tree itself: the plan_id/metadata
        # comparison above proves the REQUEST matches what was recorded,
        # not that the on-disk bytes still do. A worktree modified after
        # materialisation (by anything, not necessarily this code) must
        # fail closed rather than be silently compiled.
        try:
            verify_source_attestation(destination, SourceAttestation(
                upstream_revision=cached_plan["upstream_revision"],
                tree_oid=cached["source_tree_oid"],
                object_format=cached["git_object_format"],
                source_slice_id=cached["source_slice_id"],
                allowed_untracked=frozenset(cached.get("allowed_untracked", ())),
            ))
            actual_tree_oid = cached["source_tree_oid"]
        except (SourceIdentityError, KeyError) as exc:
            raise CampaignBuildError(
                f"cached source directory {destination} failed re-verification: {exc}"
            ) from exc
        if actual_tree_oid != cached.get("source_tree_oid"):
            raise CampaignBuildError(
                f"cached source directory {destination} has been modified since "
                f"materialisation (tree oid {actual_tree_oid!r} != recorded "
                f"{cached.get('source_tree_oid')!r}) -- refusing to reuse it"
            )
        # GPT-auto-agent review (RE03/RE04/RE05 comprehensive follow-up,
        # 2026-08-17): the checks above prove the REQUEST matches what was
        # recorded and that the worktree BYTES are unmodified -- neither
        # proves the persisted derived-identity FIELDS in the sibling
        # .metadata.json themselves haven't been directly edited (that file
        # is not covered by git_tree_oid() at all, since it lives beside
        # the worktree, not inside it). A forged source_slice_id/
        # source_plan_id/materialization_plan_id there would previously be
        # returned to the caller verbatim and trusted for the rest of
        # campaign execution. Re-derive each from the now-verified
        # (upstream_revision, tree_oid, object_format) and the current
        # request, and fail closed on any disagreement -- treating the
        # persisted record as an assertion to re-prove, not an authority.
        recomputed_source_slice_id = source_identity.source_slice_id(
            upstream_revision=identity["upstream_revision"],
            tree_oid=actual_tree_oid,
            object_format=cached["git_object_format"],
        )
        recomputed_source_plan_id = campaign_source.source_plan_id(plan)
        for label, recomputed, cached_value in (
            (
                "source_slice_id",
                recomputed_source_slice_id,
                cached.get("source_slice_id"),
            ),
            ("source_plan_id", recomputed_source_plan_id, cached.get("source_plan_id")),
            ("materialization_plan_id", plan_id, cached.get("materialization_plan_id")),
        ):
            if recomputed != cached_value:
                raise CampaignBuildError(
                    f"cached source directory {destination} has a persisted "
                    f"{label}={cached_value!r} that disagrees with the "
                    f"re-derived value {recomputed!r} -- refusing to trust "
                    f"tampered or corrupted source metadata"
                )
        if verify_strict:
            _verify_by_rematerialization(
                context, plan,
                expected_tree_oid=actual_tree_oid,
                allow_dirty_bigcherry=allow_dirty_bigcherry,
            )
        return cached

    if destination.exists():
        raise CampaignBuildError(
            f"source directory {destination} exists without matching "
            f"metadata -- refusing to materialise over it"
        )

    try:
        metadata = materialize(
            context, plan, destination, allow_dirty_bigcherry=allow_dirty_bigcherry
        )
    except BaseException:
        # RD100 (gpt-auto-agent review follow-up): materialize() can fail
        # partway through (worktree added, then an overlay hash mismatch or
        # patch application failure) and leave `destination` on disk with
        # no metadata file -- exactly the state the check above refuses to
        # materialise over. Without cleanup here, this plan_id is
        # permanently poisoned: every future attempt hits the same
        # "exists without matching metadata" error forever, with no
        # automatic recovery. Best-effort cleanup, never masking the
        # original failure -- a cleanup error must not hide why
        # materialisation actually failed.
        if destination.exists():
            try:
                if (destination / ".git").exists():
                    UpstreamRepository(context.upstream_repo).remove_worktree(destination)
                else:
                    shutil.rmtree(destination, ignore_errors=True)
            except Exception:
                pass
        raise
    record = dict(metadata)
    record["overlay_content_hash"] = identity["overlay_content_hash"]
    # RE05 (RV48 audit): the three source identities, all explicit in the
    # persisted record rather than two of them being implicit in facts the
    # reader has to already know (the destination directory's own name IS
    # materialization_plan_id, but that is not the same as it being a named
    # field a reader can find without knowing that convention). source_plan_id
    # answers "what request (IDs only) did we ask for"; materialization_plan_id
    # answers "what CONTENT-resolved request produced this destination";
    # source_tree_oid/source_slice_id (already in `metadata` from
    # source_identity.describe()) answer "what did materialising it actually
    # produce" and "BigCherry's durable content-domain identity" respectively.
    record["source_plan_id"] = campaign_source.source_plan_id(plan)
    record["materialization_plan_id"] = plan_id
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    # PA12: temp-file + fsync + os.replace(), shared with patch/source.py's
    # HI82 manifest writer rather than this call site's own plain
    # write_text() -- a crash mid-write must not leave truncated metadata
    # beside an otherwise-valid source worktree.
    # PA17 (C1): read_only=True, matching HI82's own manifest writer -- this
    # sidecar's fields are what the cache-hit path above compares re-derived
    # live facts against; leaving it editable is a much easier tamper vector
    # than mutating a git worktree convincingly. Defense-in-depth only (see
    # atomic_write_json's own docstring), not an independent trust boundary.
    source_identity.atomic_write_json(metadata_path, record, read_only=True)
    return record


def publish_build_outputs(
    store: ArtifactStore,
    *,
    source_slice_id: str,
    build_plan_id: str,
    manifest: dict[str, Any] | None = None,
    descriptor: dict[str, Any] | None = None,
    binary: Path | None = None,
    patches_doc: str | None = None,
) -> dict[str, str]:
    """Publish this build's identity artifacts, verifying every one.

    Every ``publish_*`` call is immediately followed by ``store.verify()``
    on the same relative path: ``ArtifactStore.publish_*`` computing a
    digest and a caller trusting that digest without re-reading the bytes
    back are two different levels of assurance, and this function exists
    specifically to close that gap for the campaign executor (see module
    docstring). A verify failure here means the store's own write path is
    broken, not a caller error -- it must never be swallowed.
    """
    prefix = f"builds/{source_slice_id}/{build_plan_id}"
    published: dict[str, str] = {}

    def _publish_json(name: str, value: dict[str, Any]) -> None:
        relative = f"{prefix}/{name}"
        digest = store.publish_json(relative, value)
        if not store.verify(relative, digest):
            raise CampaignBuildError(
                f"published artifact {relative} failed immediate verification"
            )
        published[relative] = digest

    if manifest is not None:
        _publish_json("manifest.json", manifest)
    if descriptor is not None:
        _publish_json("descriptor.json", descriptor)
    if patches_doc is not None:
        relative = f"{prefix}/patches.md"
        digest = store.publish_bytes(relative, patches_doc.encode("utf-8"))
        if not store.verify(relative, digest):
            raise CampaignBuildError(
                f"published artifact {relative} failed immediate verification"
            )
        published[relative] = digest
    if binary is not None:
        relative = f"{prefix}/{binary.name}"
        digest = store.publish_file(relative, binary)
        if not store.verify(relative, digest):
            raise CampaignBuildError(
                f"published artifact {relative} failed immediate verification"
            )
        published[relative] = digest

    return published


def execute_build_stage(
    context: ProjectContext,
    *,
    source_plan: SourcePlan,
    build_plan: BuildPlan,
    build: campaign_config.Build,
    platform: campaign_config.Platform,
    artifact_store: ArtifactStore,
    binary_relative_path: str,
    cmake_targets: tuple[str, ...] = (),
    generated_root: Path | None = None,
    inventory: Path | None = None,
    manifest: dict[str, Any] | None = None,
    descriptor: dict[str, Any] | None = None,
    c_compiler: str | None = None,
    cxx_compiler: str | None = None,
    allow_dirty_bigcherry: bool = False,
    runner: Runner = _default_runner,
) -> dict[str, str]:
    """One real campaign build stage: materialise, configure, build, publish.

    This is the executor CampaignRun/PipelineService are meant to call per
    stage -- unlike the legacy ``_build_one_recipe``, it never touches a
    shared checkout: ``materialize_source`` isolates the source by
    ``source_plan_id``, and ``builds.build_directory`` isolates the cmake
    build by ``(source_slice_id, build_plan_id)``. Compiling two different
    build plans against the same source, or the same build plan against two
    different sources, cannot collide on disk.

    Returns the published artifact relative-path -> content-hash mapping
    from :func:`publish_build_outputs`, already bytes-verified.
    """
    source_metadata = materialize_source(
        context, source_plan, allow_dirty_bigcherry=allow_dirty_bigcherry
    )
    source_slice_id = source_metadata.get("source_slice_id")
    if not isinstance(source_slice_id, str) or not source_slice_id:
        raise CampaignBuildError(
            "materialised source metadata is missing source_slice_id"
        )
    if source_slice_id != build_plan.source_slice_id:
        raise CampaignBuildError(
            f"materialised source_slice_id {source_slice_id!r} does not "
            f"match build_plan.source_slice_id {build_plan.source_slice_id!r}"
        )

    source_root = (
        context.work_root
        / "sources"
        / campaign_source.materialization_plan_id(
            campaign_source.resolve_materialization_identity(context, source_plan)
        )
    )
    build_dir = builds.build_directory(context, source_slice_id, build_plan)
    build_dir.mkdir(parents=True, exist_ok=True)

    configure_args = cmake_configure_args(
        build,
        platform,
        source_root,
        build_dir,
        generated_root=generated_root,
        inventory=inventory,
        c_compiler=c_compiler,
        cxx_compiler=cxx_compiler,
    )
    runner(configure_args, source_root)
    runner(cmake_build_args(build_dir, targets=cmake_targets), source_root)

    binary = build_dir / binary_relative_path
    if not binary.is_file():
        raise CampaignBuildError(
            f"build stage did not produce the expected binary: {binary}"
        )

    return publish_build_outputs(
        artifact_store,
        source_slice_id=source_slice_id,
        build_plan_id=build_plan.build_plan_id,
        manifest=manifest,
        descriptor=descriptor,
        binary=binary,
        patches_doc=_render_build_patch_doc_best_effort(
            context, source_plan.patch_ids, source_metadata,
        ),
    )


def _render_build_patch_doc_best_effort(
    context: ProjectContext, patch_ids: tuple[str, ...], source_metadata: dict[str, Any],
) -> str | None:
    """Merge this build's ACTUAL resolved patch_ids' SUMMARY.md into one doc.

    Uses source_plan.patch_ids (the exact content-identified selection this
    build materialised) rather than re-deriving a recipe name -- a campaign
    lane's patch set need not correspond to any named recipe at all. Every
    real build gets its own doc this way, not just pin-bump's target
    revision. Best-effort like the pin-bump equivalent: a doc failure must
    never fail a real build -- but WHAT failed is printed to stderr rather
    than silently discarded (gpt-dev-agent review, 2026-08-31).
    """
    try:
        from ..patch import docs as patch_docs

        bigcherry_revision = "unknown"
        rev_result = subprocess.run(
            ["git", "-C", str(context.project_root), "rev-parse", "HEAD"],
            capture_output=True, text=True,
        )
        if rev_result.returncode == 0:
            bigcherry_revision = rev_result.stdout.strip()

        pin_info = {
            "llama.cpp revision": str(source_metadata.get("upstream_revision", "unknown")),
            "bigcherry revision": bigcherry_revision,
            "source_slice_id": str(source_metadata.get("source_slice_id", "unknown")),
        }
        return patch_docs.render_patch_selection_doc(
            patch_ids=patch_ids, pin_info=pin_info,
            selection_label=f"campaign build ({len(patch_ids)} patch(es) resolved)",
            patches_dir=context.patches_root,
        )
    except Exception as exc:  # noqa: BLE001 -- best-effort convenience, never fatal
        print(
            f"campaign build: patches.md was not rendered ({len(patch_ids)} "
            f"patch_ids): {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return None

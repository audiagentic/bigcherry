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
import subprocess
from pathlib import Path
from typing import Any, Callable

from . import builds, campaign_source, config as campaign_config
from .artifacts import ArtifactStore
from .builds import BuildPlan
from .context import ProjectContext
from .workspace import SourcePlan, materialize

Runner = Callable[[list[str], Path], None]


class CampaignBuildError(RuntimeError):
    pass


def _default_runner(args: list[str], cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True)


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
) -> list[str]:
    """The cmake configure argv for one v2 ``config.Build``/``config.Platform``.

    Mirrors ``__main__._cmake_configure_args`` semantics (same option
    precedence, same GGML_HIP_AUTOTUNE_* wiring) but reads the v2 config
    types (tuple-based options, tuple-based targets) instead of the legacy
    ``recipes.py`` dataclasses, since the new path never touches those.
    """
    options: dict[str, str] = {
        "CMAKE_BUILD_TYPE": "Release",
        **dict(platform.options),
        **dict(build.options),
        "AMDGPU_TARGETS": ";".join(platform.targets),
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
    return [
        "cmake", "-S", str(source_root), "-B", str(build_dir), "-G", "Ninja",
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
) -> dict[str, Any]:
    """Materialise ``plan`` into an isolated, content-identified worktree.

    Idempotent by construction: the destination is keyed by
    ``campaign_source.source_plan_id(plan)``, not by any caller-chosen name,
    so two calls with an equal plan always target the same directory. If
    that directory already carries metadata from a prior materialisation of
    the SAME plan, it is reused rather than re-materialised -- but the
    stored plan is compared field-by-field first; a hash collision alone
    would not be trusted silently, since a mismatch on read is exactly the
    kind of fail-closed check this project's provenance rules require.
    """
    plan_id = campaign_source.source_plan_id(plan)
    destination = context.work_root / "sources" / plan_id
    metadata_path = _source_metadata_path(destination)

    if destination.is_dir() and metadata_path.is_file():
        cached = json.loads(metadata_path.read_text(encoding="utf-8"))
        cached_plan = cached.get("plan", {})
        if (cached_plan.get("upstream_revision") != plan.upstream_revision or
                cached_plan.get("overlay_enabled") != plan.overlay_enabled or
                sorted(cached_plan.get("patch_ids", [])) != sorted(plan.patch_ids) or
                cached_plan.get("required_state") != plan.required_state):
            raise CampaignBuildError(
                f"source directory {destination} exists with metadata for a "
                f"different plan than requested (plan id collision or stale "
                f"cache) -- refusing to reuse or overwrite it"
            )
        return cached

    if destination.exists():
        raise CampaignBuildError(
            f"source directory {destination} exists without matching "
            f"metadata -- refusing to materialise over it"
        )

    metadata = materialize(
        context, plan, destination, allow_dirty_bigcherry=allow_dirty_bigcherry
    )
    record = dict(metadata)
    record["plan"] = {
        "upstream_revision": plan.upstream_revision,
        "overlay_enabled": plan.overlay_enabled,
        "patch_ids": list(plan.patch_ids),
        "required_state": plan.required_state,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def publish_build_outputs(
    store: ArtifactStore,
    *,
    source_slice_id: str,
    build_plan_id: str,
    manifest: dict[str, Any] | None = None,
    descriptor: dict[str, Any] | None = None,
    binary: Path | None = None,
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

    source_root = context.work_root / "sources" / campaign_source.source_plan_id(source_plan)
    build_dir = builds.build_directory(context, source_slice_id, build_plan)
    build_dir.mkdir(parents=True, exist_ok=True)

    configure_args = cmake_configure_args(
        build, platform, source_root, build_dir,
        generated_root=generated_root, inventory=inventory,
        c_compiler=c_compiler, cxx_compiler=cxx_compiler,
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
    )

"""Common provenance-v2 construction and namespace compatibility checks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal


class ProvenanceError(ValueError):
    pass


SCHEMA_VERSION = 2
REQUIRED_NAMESPACES = ("project", "source", "build", "workload", "campaign")


# ---------------------------------------------------------------- RE25 typed layer
#
# GPT-auto-agent implementation plan (2026-08-17), RE25.1: a typed, strictly
# validated construction/parsing layer over the SAME dict document shape
# make()/validate() already produce and consume -- ProvenanceV2.document()
# returns exactly that dict shape, so it is a drop-in value for
# ArtifactRef.provenance (still typed dict[str, object] everywhere else in
# the codebase; migrating every call site to construct/consume ProvenanceV2
# directly is RE25.2/RE25.3+RE10 work, not done here). This layer adds real
# value on its own: strict from_document() parsing, sticky taint
# propagation, and kind-specific promotion contracts, all usable today
# without any other call site changing.

ProvenanceClass = Literal["production", "imported-legacy", "development", "diagnostic"]
_PROVENANCE_CLASSES = frozenset(
    ("production", "imported-legacy", "development", "diagnostic")
)


def _require_str_or_none(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or isinstance(value, bool):
        raise ProvenanceError(
            f"provenance field {field_name!r} must be a string or null"
        )
    return value


def _require_str_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) and not isinstance(item, bool) for item in value
    ):
        raise ProvenanceError(
            f"provenance field {field_name!r} must be a list of strings"
        )
    return tuple(value)


@dataclass(frozen=True)
class PatchModuleProvenance:
    patch_id: str
    content_hash: str

    def document(self) -> dict[str, object]:
        return {"patch_id": self.patch_id, "content_hash": self.content_hash}

    @classmethod
    def from_document(cls, document: object) -> PatchModuleProvenance:
        if not isinstance(document, dict):
            raise ProvenanceError("patch module provenance entry must be an object")
        patch_id = document.get("patch_id")
        content_hash = document.get("content_hash")
        if not isinstance(patch_id, str) or not isinstance(content_hash, str):
            raise ProvenanceError(
                "patch module provenance entry requires patch_id/content_hash strings"
            )
        return cls(patch_id=patch_id, content_hash=content_hash)


@dataclass(frozen=True)
class BuildInputProvenance:
    name: str
    artifact_id: str
    content_hash: str

    def document(self) -> dict[str, object]:
        return {
            "name": self.name,
            "artifact_id": self.artifact_id,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_document(cls, document: object) -> BuildInputProvenance:
        if not isinstance(document, dict):
            raise ProvenanceError("build input provenance entry must be an object")
        name = document.get("name")
        artifact_id = document.get("artifact_id")
        content_hash = document.get("content_hash")
        # Explicit per-field isinstance (not all(isinstance(...))): behaviour
        # is identical, but pyright cannot narrow the three locals through
        # the generator form and would reject the constructor call.
        if (
            not isinstance(name, str)
            or not isinstance(artifact_id, str)
            or not isinstance(content_hash, str)
        ):
            raise ProvenanceError(
                "build input provenance entry requires name/artifact_id/content_hash strings"
            )
        return cls(name=name, artifact_id=artifact_id, content_hash=content_hash)


@dataclass(frozen=True)
class ProjectProvenance:
    provenance_class: ProvenanceClass
    bigcherry_revision: str | None = None

    def document(self) -> dict[str, object]:
        return {
            "provenance_class": self.provenance_class,
            "bigcherry_revision": self.bigcherry_revision,
        }

    @classmethod
    def from_document(cls, document: object) -> ProjectProvenance:
        if not isinstance(document, dict):
            raise ProvenanceError("project provenance must be an object")
        provenance_class = document.get("provenance_class", "production")
        if provenance_class not in _PROVENANCE_CLASSES:
            raise ProvenanceError(
                f"unknown project.provenance_class: {provenance_class!r}"
            )
        return cls(
            provenance_class=provenance_class,
            bigcherry_revision=_require_str_or_none(
                document.get("bigcherry_revision"), "project.bigcherry_revision"
            ),
        )


@dataclass(frozen=True)
class SourceProvenance:
    upstream_revision: str | None = None
    source_plan_id: str | None = None
    materialization_plan_id: str | None = None
    source_tree_oid: str | None = None
    source_slice_id: str | None = None
    git_object_format: str | None = None
    overlay_content_hash: str | None = None
    patch_set_id: str | None = None
    patch_classification: str | None = None
    patch_modules: tuple[PatchModuleProvenance, ...] = ()

    def document(self) -> dict[str, object]:
        return {
            "upstream_revision": self.upstream_revision,
            "source_plan_id": self.source_plan_id,
            "materialization_plan_id": self.materialization_plan_id,
            "source_tree_oid": self.source_tree_oid,
            "source_slice_id": self.source_slice_id,
            "git_object_format": self.git_object_format,
            "overlay_content_hash": self.overlay_content_hash,
            "patch_set_id": self.patch_set_id,
            "patch_classification": self.patch_classification,
            "patch_modules": [module.document() for module in self.patch_modules],
        }

    @classmethod
    def from_document(cls, document: object) -> SourceProvenance:
        if not isinstance(document, dict):
            raise ProvenanceError("source provenance must be an object")
        raw_modules = document.get("patch_modules") or []
        if not isinstance(raw_modules, list):
            raise ProvenanceError("source.patch_modules must be a list")
        return cls(
            upstream_revision=_require_str_or_none(
                document.get("upstream_revision"), "source.upstream_revision"
            ),
            source_plan_id=_require_str_or_none(
                document.get("source_plan_id"), "source.source_plan_id"
            ),
            materialization_plan_id=_require_str_or_none(
                document.get("materialization_plan_id"),
                "source.materialization_plan_id",
            ),
            source_tree_oid=_require_str_or_none(
                document.get("source_tree_oid"), "source.source_tree_oid"
            ),
            source_slice_id=_require_str_or_none(
                document.get("source_slice_id"), "source.source_slice_id"
            ),
            git_object_format=_require_str_or_none(
                document.get("git_object_format"), "source.git_object_format"
            ),
            overlay_content_hash=_require_str_or_none(
                document.get("overlay_content_hash"), "source.overlay_content_hash"
            ),
            patch_set_id=_require_str_or_none(
                document.get("patch_set_id"), "source.patch_set_id"
            ),
            patch_classification=_require_str_or_none(
                document.get("patch_classification"), "source.patch_classification"
            ),
            patch_modules=tuple(
                PatchModuleProvenance.from_document(item) for item in raw_modules
            ),
        )


@dataclass(frozen=True)
class BuildProvenance:
    build_plan_id: str | None = None
    effective_build_id: str | None = None
    binary_hash: str | None = None
    runtime_bundle_hash: str | None = None
    targets: tuple[str, ...] = ()
    catalog_architectures: tuple[str, ...] = ()
    inputs: tuple[BuildInputProvenance, ...] = ()

    def document(self) -> dict[str, object]:
        return {
            "build_plan_id": self.build_plan_id,
            "effective_build_id": self.effective_build_id,
            "binary_hash": self.binary_hash,
            "runtime_bundle_hash": self.runtime_bundle_hash,
            "targets": list(self.targets),
            "catalog_architectures": list(self.catalog_architectures),
            "inputs": [item.document() for item in self.inputs],
        }

    @classmethod
    def from_document(cls, document: object) -> BuildProvenance:
        if not isinstance(document, dict):
            raise ProvenanceError("build provenance must be an object")
        raw_inputs = document.get("inputs") or []
        if not isinstance(raw_inputs, list):
            raise ProvenanceError("build.inputs must be a list")
        return cls(
            build_plan_id=_require_str_or_none(
                document.get("build_plan_id"), "build.build_plan_id"
            ),
            effective_build_id=_require_str_or_none(
                document.get("effective_build_id"), "build.effective_build_id"
            ),
            binary_hash=_require_str_or_none(
                document.get("binary_hash"), "build.binary_hash"
            ),
            runtime_bundle_hash=_require_str_or_none(
                document.get("runtime_bundle_hash"), "build.runtime_bundle_hash"
            ),
            targets=_require_str_tuple(document.get("targets"), "build.targets"),
            catalog_architectures=_require_str_tuple(
                document.get("catalog_architectures"), "build.catalog_architectures"
            ),
            inputs=tuple(
                BuildInputProvenance.from_document(item) for item in raw_inputs
            ),
        )


@dataclass(frozen=True)
class WorkloadProvenance:
    workload_id: str | None = None
    workload_spec_id: str | None = None

    def document(self) -> dict[str, object]:
        return {
            "workload_id": self.workload_id,
            "workload_spec_id": self.workload_spec_id,
        }

    @classmethod
    def from_document(cls, document: object) -> WorkloadProvenance:
        if not isinstance(document, dict):
            raise ProvenanceError("workload provenance must be an object")
        return cls(
            workload_id=_require_str_or_none(
                document.get("workload_id"), "workload.workload_id"
            ),
            workload_spec_id=_require_str_or_none(
                document.get("workload_spec_id"), "workload.workload_spec_id"
            ),
        )


@dataclass(frozen=True)
class CampaignProvenance:
    run_id: str | None = None
    campaign_plan_id: str | None = None
    producer_stage: str | None = None
    producer_artifact_ids: tuple[str, ...] = ()
    comparison_plan_id: str | None = None

    def document(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "campaign_plan_id": self.campaign_plan_id,
            "producer_stage": self.producer_stage,
            "producer_artifact_ids": list(self.producer_artifact_ids),
            "comparison_plan_id": self.comparison_plan_id,
        }

    @classmethod
    def from_document(cls, document: object) -> CampaignProvenance:
        if not isinstance(document, dict):
            raise ProvenanceError("campaign provenance must be an object")
        return cls(
            run_id=_require_str_or_none(document.get("run_id"), "campaign.run_id"),
            campaign_plan_id=_require_str_or_none(
                document.get("campaign_plan_id"), "campaign.campaign_plan_id"
            ),
            producer_stage=_require_str_or_none(
                document.get("producer_stage"), "campaign.producer_stage"
            ),
            producer_artifact_ids=_require_str_tuple(
                document.get("producer_artifact_ids"), "campaign.producer_artifact_ids"
            ),
            comparison_plan_id=_require_str_or_none(
                document.get("comparison_plan_id"), "campaign.comparison_plan_id"
            ),
        )


@dataclass(frozen=True)
class ProvenanceV2:
    schema_version: int
    project: ProjectProvenance
    source: SourceProvenance
    build: BuildProvenance
    workload: WorkloadProvenance
    campaign: CampaignProvenance

    def document(self) -> dict[str, object]:
        """Exactly the dict shape provenance.make()/validate() already
        produce/consume -- usable as an existing ArtifactRef.provenance
        value with no other call site changes required."""
        return {
            "schema_version": self.schema_version,
            "project": self.project.document(),
            "source": self.source.document(),
            "build": self.build.document(),
            "workload": self.workload.document(),
            "campaign": self.campaign.document(),
        }

    @classmethod
    def from_document(cls, document: object) -> ProvenanceV2:
        if (
            not isinstance(document, dict)
            or document.get("schema_version") != SCHEMA_VERSION
        ):
            raise ProvenanceError("only complete provenance schema v2 is promotable")
        missing = [
            name
            for name in REQUIRED_NAMESPACES
            if not isinstance(document.get(name), dict)
        ]
        if missing:
            raise ProvenanceError(
                "missing provenance namespace(s): " + ", ".join(missing)
            )
        return cls(
            schema_version=SCHEMA_VERSION,
            project=ProjectProvenance.from_document(document["project"]),
            source=SourceProvenance.from_document(document["source"]),
            build=BuildProvenance.from_document(document["build"]),
            workload=WorkloadProvenance.from_document(document["workload"]),
            campaign=CampaignProvenance.from_document(document["campaign"]),
        )


def derived_provenance_class(
    parents: tuple[ProvenanceV2, ...],
    *,
    local_class: ProvenanceClass = "production",
) -> ProvenanceClass:
    """Sticky taint: an artifact derived from a non-production parent
    inherits that class even when the current stage's own work is
    otherwise ordinary production work. This is what makes it safe to keep
    accepting raw-Path cross-run inputs (the real record -> tune/replay
    workflow genuinely needs them today) while guaranteeing they can never
    silently become promotable release evidence -- taint propagates
    forward through every derived artifact instead of being lost at the
    first build step."""
    if local_class != "production":
        return local_class
    for candidate in ("imported-legacy", "development", "diagnostic"):
        if any(parent.project.provenance_class == candidate for parent in parents):
            return candidate
    return "production"


def derive(
    *,
    parents: tuple[ProvenanceV2, ...],
    parent_artifact_ids: tuple[str, ...],
    project_revision: str,
    source: SourceProvenance,
    build: BuildProvenance,
    workload: WorkloadProvenance,
    run_id: str,
    producer_stage: str,
    campaign_plan_id: str | None = None,
    comparison_plan_id: str | None = None,
    local_class: ProvenanceClass = "production",
) -> ProvenanceV2:
    return ProvenanceV2(
        schema_version=SCHEMA_VERSION,
        project=ProjectProvenance(
            provenance_class=derived_provenance_class(parents, local_class=local_class),
            bigcherry_revision=project_revision,
        ),
        source=source,
        build=build,
        workload=workload,
        campaign=CampaignProvenance(
            run_id=run_id,
            campaign_plan_id=campaign_plan_id,
            producer_stage=producer_stage,
            producer_artifact_ids=tuple(sorted(parent_artifact_ids)),
            comparison_plan_id=comparison_plan_id,
        ),
    )


# Minimum non-empty fields a PRODUCTION artifact of each kind must carry
# before require_promotable() will accept it -- dotted paths into the
# document() shape. Deliberately does not require source.patch_modules to
# be nonempty: an upstream/no-patch composition or an empty
# validated-enhancements set legitimately has none.
_KIND_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "source-metadata": (
        "project.bigcherry_revision",
        "source.upstream_revision",
        "source.source_plan_id",
        "source.materialization_plan_id",
        "source.source_tree_oid",
        "source.source_slice_id",
        "source.git_object_format",
        "source.patch_set_id",
        "campaign.run_id",
        "campaign.producer_stage",
    ),
    "manifest": (
        "source.source_plan_id",
        "source.materialization_plan_id",
        "source.source_slice_id",
        "source.patch_set_id",
        "build.build_plan_id",
        "campaign.run_id",
    ),
    "generated-tree": (
        "source.source_slice_id",
        "build.build_plan_id",
        "campaign.run_id",
    ),
    "binary": (
        "source.source_plan_id",
        "source.materialization_plan_id",
        "source.source_slice_id",
        "source.patch_set_id",
        "build.build_plan_id",
        "build.effective_build_id",
        "build.binary_hash",
        "campaign.run_id",
    ),
    "runtime-bundle": (
        "source.source_plan_id",
        "source.materialization_plan_id",
        "source.source_slice_id",
        "source.patch_set_id",
        "build.build_plan_id",
        "build.effective_build_id",
        "build.binary_hash",
        "build.runtime_bundle_hash",
        "campaign.run_id",
    ),
    "record-jsonl": (
        "source.source_slice_id",
        "build.build_plan_id",
        "build.effective_build_id",
        "workload.workload_spec_id",
        "campaign.run_id",
    ),
    "inventory": (
        "source.source_slice_id",
        "build.build_plan_id",
        "build.effective_build_id",
        "workload.workload_id",
        "campaign.run_id",
    ),
    "dispatch-db": (
        "source.source_slice_id",
        "build.build_plan_id",
        "build.effective_build_id",
        "workload.workload_id",
        "campaign.run_id",
    ),
    "tuning-measurements": (
        "source.source_slice_id",
        "build.build_plan_id",
        "build.effective_build_id",
        "workload.workload_id",
        "campaign.run_id",
    ),
    "promoted-winners": (
        "source.source_slice_id",
        "build.build_plan_id",
        "build.effective_build_id",
        "workload.workload_id",
        "campaign.run_id",
        "campaign.producer_stage",
    ),
    "replay-cache": (
        "source.source_slice_id",
        "build.build_plan_id",
        "build.effective_build_id",
        "workload.workload_id",
        "campaign.run_id",
    ),
    "comparison-report": (
        "campaign.campaign_plan_id",
        "campaign.run_id",
        "campaign.comparison_plan_id",
        "campaign.producer_stage",
    ),
}


def _get_dotted(document: dict[str, object], dotted: str) -> object:
    namespace, _, field_name = dotted.partition(".")
    namespace_document = document.get(namespace)
    if not isinstance(namespace_document, dict):
        return None
    return namespace_document.get(field_name)


def validate_for_kind(document: object, *, kind: str) -> ProvenanceV2:
    """Strict schema validation (via ProvenanceV2.from_document) PLUS this
    artifact kind's minimum required-field contract -- provenance.validate()
    alone only proves the five namespaces exist as (possibly entirely
    empty) dicts, which the audit that created this project's provenance
    model itself called 'a useful early-stage shape but not complete
    production provenance'. Applies the kind's required-field contract
    regardless of provenance_class -- an imported-legacy artifact is
    exempt from having a REAL identity (it never claimed one), so callers
    that only care about promotable evidence should use
    require_promotable() instead, which additionally demands
    provenance_class == 'production'."""
    parsed = ProvenanceV2.from_document(document)
    if parsed.project.provenance_class != "production":
        return parsed
    required = _KIND_REQUIRED_FIELDS.get(kind)
    if required is None:
        raise ProvenanceError(
            f"no provenance contract registered for artifact kind {kind!r}"
        )
    doc = parsed.document()
    missing = [dotted for dotted in required if not _get_dotted(doc, dotted)]
    if missing:
        raise ProvenanceError(
            f"production {kind!r} artifact is missing required provenance field(s): {', '.join(missing)}"
        )
    return parsed


def require_promotable(document: object, *, kind: str) -> ProvenanceV2:
    parsed = validate_for_kind(document, kind=kind)
    if parsed.project.provenance_class != "production":
        raise ProvenanceError(
            f"{kind!r} artifact is not promotable: provenance_class={parsed.project.provenance_class!r}"
        )
    return parsed


def lane_input_provenance(
    parents: Mapping[str, object],
) -> tuple[tuple[BuildInputProvenance, ...], tuple[str, ...]]:
    """RE25.2: the (build.inputs, campaign.producer_artifact_ids) pair for a
    stage given its parent artifacts as a {name: ArtifactRef} mapping
    (or {(kind): ref} for same-kind multi-sets -- name is just the label).

    Pre-descriptor (legacy) ArtifactRefs carry no artifact_id yet, so their
    content_hash stands in as the identity; RE25.3 gives lane inputs real
    descriptors and this fallback disappears. Sorting makes both results
    deterministic regardless of dict insertion order.
    """
    names = tuple(sorted(parents))
    entries = tuple(
        BuildInputProvenance(
            name=name,
            artifact_id=_parent_identity(parent_ref),
            content_hash=_parent_content_hash(parent_ref),
        )
        for name, parent_ref in ((name, parents[name]) for name in names)
    )
    ids = tuple(_parent_identity(parents[name]) for name in names)
    return entries, tuple(sorted(ids))


def _parent_identity(ref: object) -> str:
    artifact_id = getattr(ref, "artifact_id", "")
    return (
        artifact_id
        if isinstance(artifact_id, str) and artifact_id
        else _parent_content_hash(ref)
    )


def _parent_content_hash(ref: object) -> str:
    content_hash = getattr(ref, "content_hash", "")
    if not isinstance(content_hash, str):
        raise ProvenanceError(
            f"parent artifact {ref!r} has no usable content_hash identity"
        )
    return content_hash


def make(
    *,
    project: dict[str, Any],
    source: dict[str, Any],
    build: dict[str, Any],
    workload: dict[str, Any],
    campaign: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "project": dict(project),
        "source": dict(source),
        "build": dict(build),
        "workload": dict(workload),
        "campaign": dict(campaign),
    }


def validate(document: object) -> dict[str, Any]:
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != SCHEMA_VERSION
    ):
        raise ProvenanceError("only complete provenance schema v2 is promotable")
    missing = [
        name for name in REQUIRED_NAMESPACES if not isinstance(document.get(name), dict)
    ]
    if missing:
        raise ProvenanceError("missing provenance namespace(s): " + ", ".join(missing))
    return document


def require_compatible(document: object, **expected: str) -> dict[str, Any]:
    value = validate(document)
    for dotted, wanted in expected.items():
        namespace, _, field = dotted.partition(".")
        if namespace not in REQUIRED_NAMESPACES or not field:
            raise ProvenanceError(f"invalid expected provenance field {dotted!r}")
        actual = value[namespace].get(field)
        if actual != wanted:
            raise ProvenanceError(
                f"provenance mismatch for {dotted}: expected {wanted!r}, got {actual!r}"
            )
    return value

"""Candidate manifest schema and validation (HI03).

The manifest is the contract between the generator and everything downstream:
the runtime registry, the database, the replay cache, and the tests. Validating
it here — rather than discovering a malformed record when the HIP compiler
chokes on generated code — is the cheapest place to catch a generator bug.

Validation is hand-written rather than jsonschema-based so the package has no
runtime dependencies; the schema is small and closed enough that this costs
little and the error messages are better for it.
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass
from typing import Any

# Standards 1: a kernel family is a major algorithmic path.
FAMILIES = ("mmvq", "mmq", "mmvf", "mmf", "blas")

# HI17 BLAS-1 starts with the resolved native path.  These values describe a
# plan, not an observation: the latter uses the exact native API/provider
# fields captured by HI57.  Keeping the two namespaces separate prevents a
# telemetry value from becoming a candidate identity by accident.
BLAS_OPERAND_TYPES = ("native", "f32", "f16", "bf16")
BLAS_ACCUMULATION_TYPES = ("native", "f16", "f32")
BLAS_OUTPUT_TYPES = ("native", "f16", "bf16", "f32")
BLAS_SOURCE_CONVERSIONS = ("none", "contiguous", "non_contiguous")
BLAS_OUTPUT_CONVERSIONS = ("none", "temporary_to_f32")
BLAS_NUMERICAL_CLASSES = (
    "exact_baseline",
    "equivalent_within_backend_tolerance",
    "reduced_precision",
)

# ggml_prec is an upstream enum: GGML_PREC_DEFAULT == 0 and
# GGML_PREC_F32 == 10.  The schema deliberately stores this as an integer at
# the boundary so it can validate catalog plans without importing C headers.
GGML_PREC_F32 = 10

BLAS_PLAN_FIELDS = (
    "operand_type",
    "accumulation_type",
    "output_type",
    "source_a_conversion",
    "source_b_conversion",
    "output_conversion",
    "numerical_class",
)

BLAS_SIGNATURE_FIELDS = (
    "src0_type",
    "src1_type",
    "dst_type",
    "prec",
    "has_ids",
    "source_a_contiguous",
    "source_b_contiguous",
    "batched",
)


@dataclass(frozen=True)
class BlasPlanResolution:
    """Deterministic offline result for one plan/signature eligibility check."""

    plan: dict[str, str] | None
    rejection_reason: str | None = None


def validate_blas_plan(plan: dict[str, Any], where: str, *, prec: int | None = None) -> None:
    """Validate a structured BLAS candidate plan before catalog emission.

    The native/forced-native slice intentionally has no provider or API
    choice.  Those are effective-call observations today and become candidate
    dimensions only after a runtime seam proves they can be applied.

    ``prec`` is optional because catalog validation has no operation signature.
    When supplied, it applies the correctness gate before a plan can enter a
    strict-precision measurement set.
    """
    if not isinstance(plan, dict):
        raise SchemaError(f"{where}: BLAS plan must be an object")

    missing = [field for field in BLAS_PLAN_FIELDS if field not in plan]
    if missing:
        raise SchemaError(
            f"{where}: BLAS plan is missing {', '.join(repr(field) for field in missing)}")
    unexpected = sorted(set(plan) - set(BLAS_PLAN_FIELDS))
    if unexpected:
        raise SchemaError(
            f"{where}: BLAS plan has unexpected fields {', '.join(unexpected)}")

    allowed = {
        "operand_type": BLAS_OPERAND_TYPES,
        "accumulation_type": BLAS_ACCUMULATION_TYPES,
        "output_type": BLAS_OUTPUT_TYPES,
        "source_a_conversion": BLAS_SOURCE_CONVERSIONS,
        "source_b_conversion": BLAS_SOURCE_CONVERSIONS,
        "output_conversion": BLAS_OUTPUT_CONVERSIONS,
        "numerical_class": BLAS_NUMERICAL_CLASSES,
    }
    for field, values in allowed.items():
        value = plan[field]
        if not isinstance(value, str) or value not in values:
            raise SchemaError(
                f"{where}: BLAS plan {field} must be one of {values}, got {value!r}")

    if prec is not None:
        if isinstance(prec, bool) or not isinstance(prec, int):
            raise SchemaError(f"{where}: BLAS precision must be an integer")
        if prec == GGML_PREC_F32 and plan["numerical_class"] == "reduced_precision":
            raise SchemaError(
                f"{where}: reduced_precision BLAS plan is not eligible for "
                "GGML_PREC_F32")


def blas_plan_name(mode: str, plan: dict[str, str], version: int) -> str:
    """Return the durable name for a resolved BLAS plan.

    Each plan field is present in the name so a stored winner cannot silently
    refer to a different configuration when the catalog evolves.
    """
    if not isinstance(mode, str) or not mode:
        raise SchemaError("BLAS plan mode must be a non-empty string")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise SchemaError("BLAS plan version must be a positive integer")
    validate_blas_plan(plan, "BLAS plan")
    tokens = [
        f"operand-{plan['operand_type']}",
        f"accumulation-{plan['accumulation_type']}",
        f"output-{plan['output_type']}",
        f"source-a-{plan['source_a_conversion']}",
        f"source-b-{plan['source_b_conversion']}",
        f"output-conversion-{plan['output_conversion']}",
        f"numerical-{plan['numerical_class']}",
    ]
    return f"blas:{mode}:" + ":".join(tokens) + f":v{version}"


def resolve_blas_plan(plan: dict[str, Any], signature: dict[str, Any],
                      where: str = "BLAS plan") -> BlasPlanResolution:
    """Resolve statically provable BLAS eligibility without choosing a backend.

    This is deliberately smaller than the eventual HIP call resolver.  It
    only handles constraints visible in the dispatch signature: precision,
    MUL_MAT_ID, source layout, and whether an explicit output conversion is
    structurally meaningful.  Provider, effective API, and HIP datatype
    selection remain runtime observations until the HI17 apply/execute seam is
    implemented.
    """
    if not isinstance(signature, dict):
        raise SchemaError(f"{where}: BLAS signature context must be an object")
    missing = [field for field in BLAS_SIGNATURE_FIELDS if field not in signature]
    if missing:
        raise SchemaError(
            f"{where}: BLAS signature context is missing "
            f"{', '.join(repr(field) for field in missing)}")

    for field in ("src0_type", "src1_type", "dst_type"):
        if not isinstance(signature[field], str) or not signature[field].strip():
            raise SchemaError(f"{where}: signature {field} must be a non-empty string")
    if isinstance(signature["prec"], bool) or not isinstance(signature["prec"], int):
        raise SchemaError(f"{where}: signature prec must be an integer")
    for field in ("has_ids", "source_a_contiguous", "source_b_contiguous", "batched"):
        if not isinstance(signature[field], bool):
            raise SchemaError(f"{where}: signature {field} must be boolean")

    try:
        validate_blas_plan(plan, f"{where}.plan")
    except SchemaError:
        return BlasPlanResolution(None, "invalid_plan")

    normalized = {field: plan[field] for field in BLAS_PLAN_FIELDS}
    if (signature["prec"] == GGML_PREC_F32 and
            normalized["numerical_class"] == "reduced_precision"):
        return BlasPlanResolution(None, "strict_precision_rejects_reduced_precision")
    if signature["has_ids"]:
        return BlasPlanResolution(None, "mul_mat_id_unsupported")

    source_routes = (
        ("source_a_conversion", "source_a_contiguous", "src0_type"),
        ("source_b_conversion", "source_b_contiguous", "src1_type"),
    )
    for route_field, layout_field, type_field in source_routes:
        route = normalized[route_field]
        contiguous = signature[layout_field]
        if route == "contiguous" and not contiguous:
            return BlasPlanResolution(None, f"{route_field}_requires_contiguous_layout")
        if route == "non_contiguous" and contiguous:
            return BlasPlanResolution(None, f"{route_field}_requires_strided_layout")

        # `none` is only a no-op for an explicitly typed operand.  The native
        # plan intentionally defers the actual source type to the runtime.
        if (route == "none" and normalized["operand_type"] != "native" and
                signature[type_field] != normalized["operand_type"]):
            return BlasPlanResolution(None, f"{route_field}_requires_conversion")

    output_type = normalized["output_type"]
    destination = signature["dst_type"]
    output_route = normalized["output_conversion"]
    if output_route == "none" and output_type != "native" and destination != output_type:
        return BlasPlanResolution(None, "output_conversion_required")
    if output_route == "temporary_to_f32" and (destination != "f32" or
                                                output_type not in ("f16", "bf16")):
        return BlasPlanResolution(None, "temporary_to_f32_requires_f16_or_bf16_output")

    # Batch structure is intentionally accepted here.  The native BLAS path
    # chooses among its APIs at runtime; rejecting a batched shape offline
    # would accidentally turn an API observation into candidate identity.
    return BlasPlanResolution(normalized)

# Standards 2.3. Exactly one per candidate.
SOURCE_CLASSES = (
    "native_wrapper",
    "existing_runtime",
    "existing_alternative",
    "new_generated_variant",
    "vendor_auto",
    "vendor_explicit",
)

# Standards 2.6. The index of an entry *is* its architecture_code, and therefore
# its bit position in the uint64 architecture_mask.
#
# APPEND ONLY. Inserting an entry renumbers every one after it, which silently
# reinterprets every architecture_mask already written to a database or replay
# cache -- candidates would appear to support hardware they were never measured
# on. New AMD parts go on the end, in whatever order they arrive.
#
# This list is the single source of truth for the enumeration: the catalog
# generator emits the matching C++ enum into hip-autotune-arch.h, so the two
# languages cannot drift apart (standards 2.5).
#
# 64 codes fit in the mask; 27 are used, leaving room for a decade of parts.
ARCHITECTURES = (
    "unknown",   # 0  -- matches nothing; an unrecognised GPU falls back to native
    # GCN / Vega, wave size 64
    "gfx803",    # 1  Tonga, Fiji, Polaris
    "gfx900",    # 2  Vega 56/64
    "gfx906",    # 3  Vega 20, MI50, Radeon VII
    # CDNA, wave size 64
    "gfx908",    # 4  CDNA1, MI100
    "gfx90a",    # 5  CDNA2, MI210/MI250
    "gfx942",    # 6  CDNA3, MI300
    "gfx950",    # 7  CDNA4, MI350X/MI355X
    # RDNA1, wave size 32
    "gfx1010",   # 8  RX 5700
    "gfx1011",   # 9
    "gfx1012",   # 10 RX 5500
    # RDNA2
    "gfx1030",   # 11 RX 6800/6900
    "gfx1031",   # 12 RX 6700
    "gfx1032",   # 13 RX 6600
    "gfx1034",   # 14
    "gfx1035",   # 15
    "gfx1036",   # 16
    # RDNA3
    "gfx1100",   # 17 RX 7900 XTX/XT/GRE
    "gfx1101",   # 18 RX 7800/7700
    "gfx1102",   # 19 RX 7600
    "gfx1103",   # 20 Phoenix APU
    # RDNA3.5
    "gfx1150",   # 21 Strix Point
    "gfx1151",   # 22 Strix Halo / AI Max 395
    "gfx1152",   # 23
    "gfx1153",   # 24
    # RDNA4
    "gfx1200",   # 25 RX 9060
    "gfx1201",   # 26 RX 9070, Radeon AI PRO R9700
)

# Which upstream MMQ config table each architecture resolves to. Mirrors the
# fallthrough in ggml_cuda_mmq_get_config: CDNA, then RDNA4, then RDNA3.5, then
# RDNA3, and everything older lands on the RDNA2 table.
ARCHITECTURE_FAMILY = {
    "gfx803": "rdna2", "gfx900": "rdna2", "gfx906": "rdna2",
    "gfx908": "cdna", "gfx90a": "cdna", "gfx942": "cdna", "gfx950": "cdna",
    "gfx1010": "rdna2", "gfx1011": "rdna2", "gfx1012": "rdna2",
    "gfx1030": "rdna2", "gfx1031": "rdna2", "gfx1032": "rdna2",
    "gfx1034": "rdna2", "gfx1035": "rdna2", "gfx1036": "rdna2",
    "gfx1100": "rdna3", "gfx1101": "rdna3", "gfx1102": "rdna3",
    "gfx1103": "rdna3",
    "gfx1150": "rdna3-5", "gfx1151": "rdna3-5", "gfx1152": "rdna3-5",
    "gfx1153": "rdna3-5",
    "gfx1200": "rdna4", "gfx1201": "rdna4",
}

# Matrix-core capability, mirroring vendors/hip.h and common.cuh:
#
#   AMD_MFMA_AVAILABLE  <- CDNA (gfx908/90a/942/950)
#   AMD_WMMA_AVAILABLE  <- RDNA4 || RDNA3, and RDNA3 is `__GFX11__`, which
#                          covers RDNA3.5 (gfx115x) as well as gfx110x
#
# This exists because `architecture_mask` was previously built from whichever
# architectures the catalog happened to enumerate, so it encoded "which GPUs
# does this build know about" rather than "which GPUs can run this candidate".
# See review RV06.
MFMA_ARCHITECTURES = frozenset(
    a for a, f in ARCHITECTURE_FAMILY.items() if f == "cdna")
WMMA_ARCHITECTURES = frozenset(
    a for a, f in ARCHITECTURE_FAMILY.items()
    if f in ("rdna3", "rdna3-5", "rdna4"))


def mmf_architectures(source_type: str, architectures: list[str]) -> list[str]:
    """Architectures on which MMF implements `source_type` at all.

    From the type switch at the end of ``ggml_cuda_should_use_mmf``: F32 needs
    MFMA (or NVIDIA Ampere, which a HIP overlay never sees), F16 and BF16 need
    WMMA or MFMA. So every `mmf:f32` candidate is CDNA-only, and on an RDNA
    build there is nothing for it to run on.

    Deliberately coarse. That function also excludes BF16 on CDNA3 and F16/BF16
    on CDNA1/CDNA2, and it is still the precise authority at dispatch time
    (`ggml_hip_mmf_can_execute` calls it). This narrows the mask only where a
    candidate can *never* run, because a mask that is too narrow silently drops
    a workable candidate, whereas one that is slightly too wide merely defers
    to the runtime check that already exists.
    """
    if source_type == "f32":
        allowed = MFMA_ARCHITECTURES
    else:
        allowed = MFMA_ARCHITECTURES | WMMA_ARCHITECTURES
    return [a for a in architectures if a in allowed]


# Named groups for the --arch flag, so a build does not have to spell out
# twenty-six targets.
ARCHITECTURE_GROUPS = {
    "all":     [a for a in ARCHITECTURES if a != "unknown"],
    "rdna":    [a for a in ARCHITECTURES if a.startswith("gfx10") or a.startswith("gfx11") or a.startswith("gfx12")],
    "rdna3":   ["gfx1100", "gfx1101", "gfx1102", "gfx1103"],
    "rdna3.5": ["gfx1150", "gfx1151", "gfx1152", "gfx1153"],
    "rdna4":   ["gfx1200", "gfx1201"],
    "cdna":    ["gfx908", "gfx90a", "gfx942", "gfx950"],
    "gcn":     ["gfx803", "gfx900", "gfx906"],
}


def resolve_architectures(spec: str) -> list[str]:
    """Expand a comma-separated --arch spec, honouring group names.

    Duplicates are collapsed and the result is ordered by architecture code, so
    two spellings of the same target set produce the same manifest hash.
    """
    seen: set[str] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if token in ARCHITECTURE_GROUPS:
            seen.update(ARCHITECTURE_GROUPS[token])
        else:
            architecture_code(token)  # validates, with a useful message
            seen.add(token)
    return sorted(seen, key=ARCHITECTURES.index)

VARIANT_SETS = ("inventory", "workload-max", "full-max", "replay-full", "replay-slim")

REQUIRED_CANDIDATE_FIELDS = {
    "stable_name": str,
    "family": str,
    "source_class": str,
    "implementation_version": int,
    "architectures": list,
    "graph_safe": bool,
    "deterministic": bool,
    "config": dict,
}

PRODUCTION_VARIANT_SETS = frozenset(("replay-full", "replay-slim"))


def _require_evidence_object(candidate: dict[str, Any], field: str,
                             where: str) -> dict[str, Any]:
    value = candidate.get(field)
    if not isinstance(value, dict):
        raise SchemaError(
            f"{where}: experimental fused candidate requires explicit {field}")
    if value.get("status") != "validated":
        raise SchemaError(
            f"{where}: {field}.status must be 'validated' for production")
    return value


def validate_production_candidate_safety(candidate: dict[str, Any],
                                         where: str) -> None:
    """Gate experimental fused candidates at the offline catalog boundary.

    HI26 planning describes fused/activation-reuse work, but no runtime launch
    path is allowed to treat an experimental candidate as production-ready by
    accident.  The marker is deliberately explicit; ordinary generated
    candidates retain the existing schema and behaviour.
    """
    if not (candidate.get("experimental_fused") is True or
            candidate.get("experimental") is True and
            candidate.get("fusion") is not None):
        return

    graph = _require_evidence_object(candidate, "graph_safety_evidence", where)
    if graph.get("capture_observed") is not True:
        raise SchemaError(
            f"{where}: graph_safety_evidence.capture_observed must be true")
    correctness = _require_evidence_object(candidate, "correctness_evidence", where)
    if not correctness.get("reference") or not correctness.get("comparison"):
        raise SchemaError(
            f"{where}: correctness_evidence needs reference and comparison")
    workspace = _require_evidence_object(candidate, "workspace_evidence", where)
    peak = workspace.get("peak_bytes")
    if isinstance(peak, bool) or not isinstance(peak, int) or peak < 0:
        raise SchemaError(
            f"{where}: workspace_evidence.peak_bytes must be a non-negative int")
    provenance = _require_evidence_object(candidate, "provenance_evidence", where)
    if not isinstance(provenance.get("source_revision"), str) or not provenance["source_revision"]:
        raise SchemaError(
            f"{where}: provenance_evidence.source_revision is required")
    refs = provenance.get("evidence_references")
    if (not isinstance(refs, list) or not refs or
            any(not isinstance(ref, str) or not ref for ref in refs) or
            len(set(refs)) != len(refs)):
        raise SchemaError(
            f"{where}: provenance_evidence.evidence_references must be unique")


class SchemaError(ValueError):
    """The manifest does not satisfy the candidate schema."""


_SHA256_RE = r"^[0-9a-fA-F]{64}$"


def _require_evidence_record(config: dict[str, Any], field: str,
                             where: str, *, require_digest: bool = True) -> dict[str, Any]:
    value = config.get(field)
    if not isinstance(value, dict):
        raise SchemaError(
            f"{where}: custom kernel requires a {field} evidence record")
    path = value.get("path")
    if not isinstance(path, str) or not path.strip():
        raise SchemaError(f"{where}: {field}.path must be a non-empty path")
    if require_digest:
        digest = value.get("sha256")
        if not isinstance(digest, str) or re.fullmatch(_SHA256_RE, digest) is None:
            raise SchemaError(
                f"{where}: {field}.sha256 must be a 64-character hex digest")
    return value


def validate_custom_candidate_enablement(candidate: dict[str, Any], where: str) -> None:
    """Fail closed when an explicitly enabled custom kernel lacks proof.

    The normal catalog contains existing/native alternatives and is unchanged
    by this contract.  A future custom MMQ/MMVQ entry opts in with
    ``config.custom_kernel: true`` and must carry immutable evidence links for
    architecture coverage, identity, correctness, compiler resources, and
    benchmark performance before the registry can be rendered.
    """
    config = candidate["config"]
    if config.get("custom_kernel") is not True:
        return

    if candidate["source_class"] != "new_generated_variant":
        raise SchemaError(
            f"{where}: custom kernels must use source_class "
            "'new_generated_variant'")
    if candidate["family"] not in ("mmq", "mmvq"):
        raise SchemaError(
            f"{where}: HI25 custom kernels must belong to the MMQ or MMVQ family")
    architectures = candidate["architectures"]
    mask = candidate.get("architecture_mask")
    expected_mask = architecture_mask(architectures)
    if isinstance(mask, bool) or not isinstance(mask, int) or mask != expected_mask:
        raise SchemaError(
            f"{where}: custom kernel architecture_mask must equal its "
            "architecture set")

    if config.get("candidate_identity") != candidate["stable_name"]:
        raise SchemaError(
            f"{where}: custom kernel candidate_identity must equal stable_name")

    reference = _require_evidence_record(
        config, "correctness_reference", where, require_digest=False)
    reference_candidate = reference.get("candidate")
    if not isinstance(reference_candidate, str) or not reference_candidate.strip():
        raise SchemaError(
            f"{where}: correctness_reference.candidate must identify a reference")
    if reference_candidate == candidate["stable_name"]:
        raise SchemaError(
            f"{where}: correctness reference must be distinct from the custom kernel")

    resource = _require_evidence_record(config, "resource_report", where)
    resource_architectures = resource.get("architectures")
    if not isinstance(resource_architectures, list) or not resource_architectures:
        raise SchemaError(
            f"{where}: resource_report.architectures must be a non-empty list")
    try:
        resource_mask = architecture_mask(resource_architectures)
    except (SchemaError, TypeError) as exc:
        raise SchemaError(f"{where}: resource_report has invalid architectures") from exc
    if resource_mask != expected_mask:
        raise SchemaError(
            f"{where}: resource report architecture coverage must match the custom kernel")

    benchmark = _require_evidence_record(config, "benchmark_evidence", where)
    if benchmark.get("status") != "passed":
        raise SchemaError(
            f"{where}: benchmark_evidence.status must be 'passed'")
    metric = benchmark.get("metric")
    if not isinstance(metric, str) or not metric.strip():
        raise SchemaError(
            f"{where}: benchmark_evidence.metric must name a measured metric")
    baseline = benchmark.get("baseline")
    if not isinstance(baseline, str) or not baseline.strip():
        raise SchemaError(
            f"{where}: benchmark_evidence.baseline must identify the reference")


def _validate_correctness_reference(candidate: dict[str, Any], manifest: dict[str, Any],
                                    where: str) -> None:
    """Require namespace-bound correctness evidence for production variants.

    Native wrappers are the diagnostic reference themselves.  Every other
    candidate needs an immutable reference artifact, explicit error metrics
    and tolerances, and evidence captured in the exact signature/build
    namespace represented by the manifest.
    """
    if candidate["source_class"] == "native_wrapper":
        return

    evidence = candidate.get("correctness_evidence")
    if not isinstance(evidence, dict):
        evidence = candidate.get("config", {}).get("correctness_reference")
    if not isinstance(evidence, dict):
        raise SchemaError(
            f"{where}: non-native candidate requires correctness evidence")

    path = evidence.get("reference_path", evidence.get("path"))
    if not isinstance(path, str) or not path.strip():
        raise SchemaError(
            f"{where}: correctness evidence requires a reference path")

    metrics = evidence.get("error_metrics", evidence.get("metrics"))
    if not isinstance(metrics, dict) or not metrics:
        raise SchemaError(
            f"{where}: correctness evidence requires error metrics")
    tolerances = evidence.get("tolerances", evidence.get("tolerance"))
    if not isinstance(tolerances, dict) or not tolerances:
        raise SchemaError(
            f"{where}: correctness evidence requires tolerances")
    for label, values in (("error metrics", metrics), ("tolerances", tolerances)):
        for name, value in values.items():
            if (not isinstance(name, str) or not name.strip() or
                    isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(value) or value < 0):
                raise SchemaError(
                    f"{where}: correctness {label} must contain non-negative finite numbers")

    signature = evidence.get("signature_namespace")
    if not isinstance(signature, dict):
        raise SchemaError(
            f"{where}: correctness evidence requires signature_namespace")
    for field in ("signature_schema_version", "hardware_schema_version"):
        if signature.get(field) != manifest.get(field):
            raise SchemaError(
                f"{where}: correctness signature_namespace does not match manifest")

    build = evidence.get("build_namespace")
    if not isinstance(build, dict):
        raise SchemaError(
            f"{where}: correctness evidence requires build_namespace")
    for field in ("source_revision", "manifest_hash", "build_descriptor_hash"):
        expected = manifest.get(field)
        if expected is not None and build.get(field) != expected:
            raise SchemaError(
                f"{where}: correctness build_namespace does not match manifest")


def architecture_code(name: str) -> int:
    try:
        return ARCHITECTURES.index(name)
    except ValueError as exc:
        raise SchemaError(
            f"unknown architecture {name!r}; add it to ARCHITECTURES "
            f"(append only -- inserting would renumber existing masks)") from exc


def architecture_mask(names: list[str]) -> int:
    mask = 0
    for name in names:
        mask |= 1 << architecture_code(name)
    return mask


def validate_candidate(candidate: dict[str, Any], where: str) -> None:
    for field, expected in REQUIRED_CANDIDATE_FIELDS.items():
        if field not in candidate:
            raise SchemaError(f"{where}: missing required field {field!r}")
        # bool is a subclass of int, so an int field must reject True.
        value = candidate[field]
        if expected is int and isinstance(value, bool):
            raise SchemaError(f"{where}: field {field!r} must be an int, got bool")
        if not isinstance(value, expected):
            raise SchemaError(
                f"{where}: field {field!r} must be {expected.__name__}, "
                f"got {type(value).__name__}")

    if candidate["family"] not in FAMILIES:
        raise SchemaError(
            f"{where}: family {candidate['family']!r} is not one of {FAMILIES}")
    if candidate["source_class"] not in SOURCE_CLASSES:
        raise SchemaError(
            f"{where}: source_class {candidate['source_class']!r} is not one "
            f"of {SOURCE_CLASSES}")
    if not candidate["architectures"]:
        raise SchemaError(f"{where}: architectures must not be empty")
    for arch in candidate["architectures"]:
        architecture_code(arch)  # raises with a useful message

    # Standards 2.1: the vN suffix of the stable name is the implementation
    # version, and it is what makes a stored winner from an older build with
    # changed behaviour distinguishable rather than silently reused.
    suffix = candidate["stable_name"].rsplit(":", 1)[-1]
    if not suffix.startswith("v") or not suffix[1:].isdigit():
        raise SchemaError(
            f"{where}: stable_name {candidate['stable_name']!r} must end in a "
            f"version suffix like ':v1'")
    if int(suffix[1:]) != candidate["implementation_version"]:
        raise SchemaError(
            f"{where}: stable_name suffix {suffix!r} disagrees with "
            f"implementation_version {candidate['implementation_version']}")

    validate_custom_candidate_enablement(candidate, where)
    if candidate["family"] == "blas" and "blas_plan" in candidate["config"]:
        plan = candidate["config"]["blas_plan"]
        validate_blas_plan(plan, f"{where}.config.blas_plan")

        # A structured BLAS plan is a candidate identity, not merely metadata.
        # Recompute the durable name at validation time so a malformed manifest
        # cannot advertise one plan while carrying another.
        expected_name = blas_plan_name(
            candidate["config"].get("mode"),
            plan,
            candidate["implementation_version"],
        )
        if candidate["stable_name"] != expected_name:
            raise SchemaError(
                f"{where}: stable_name {candidate['stable_name']!r} disagrees "
                f"with resolved BLAS plan identity {expected_name!r}")


def validate_manifest(manifest: dict[str, Any]) -> None:
    for field in ("artifact_version", "variant_set", "source_revision",
                  "architectures", "candidates"):
        if field not in manifest:
            raise SchemaError(f"manifest: missing required field {field!r}")

    if manifest["variant_set"] not in VARIANT_SETS:
        raise SchemaError(
            f"manifest: variant_set {manifest['variant_set']!r} is not one of "
            f"{VARIANT_SETS}")

    seen: dict[str, int] = {}
    natives: dict[str, int] = {}
    for index, candidate in enumerate(manifest["candidates"]):
        where = f"candidate[{index}]"
        validate_candidate(candidate, where)
        if manifest["variant_set"] in PRODUCTION_VARIANT_SETS:
            validate_production_candidate_safety(candidate, where)
            _validate_correctness_reference(candidate, manifest, where)

        name = candidate["stable_name"]
        if name in seen:
            raise SchemaError(
                f"{where}: duplicate stable_name {name!r} (first seen at "
                f"candidate[{seen[name]}]). Stable names are database "
                f"identities; two candidates sharing one would merge their "
                f"measurements.")
        seen[name] = index

        if candidate["source_class"] == "native_wrapper":
            natives[candidate["family"]] = natives.get(candidate["family"], 0) + 1

    # Standards 7.3: the native candidate must be measurable for every family
    # that has candidates at all, because it is the correctness reference and
    # the replacement baseline.
    families_present = {c["family"] for c in manifest["candidates"]}
    for family in families_present:
        if family not in natives:
            raise SchemaError(
                f"manifest: family {family!r} has candidates but no "
                f"native_wrapper. Without one there is no correctness "
                f"reference and no baseline to beat.")

    supported = manifest.get("supported_coverage")
    if supported is not None:
        validate_supported_coverage(supported)


def validate_supported_coverage(report: dict[str, Any]) -> None:
    """Validate optional supported-vs-observed candidate coverage evidence."""
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise SchemaError("supported_coverage must use schema_version 1")
    observed = report.get("observed_types")
    supported = report.get("supported_types")
    by_type = report.get("by_type")
    for name, value in (("observed_types", observed),
                        ("supported_types", supported)):
        if (not isinstance(value, list)
                or any(not isinstance(item, str) or not item for item in value)
                or value != sorted(set(value))):
            raise SchemaError(
                f"supported_coverage.{name} must be a sorted unique list")
    if (not isinstance(by_type, dict)
            or set(by_type) != set(observed) | set(supported)):
        raise SchemaError(
            "supported_coverage.by_type must cover observed and supported types")

    for type_name, row in by_type.items():
        if not isinstance(row, dict):
            raise SchemaError(f"supported_coverage.by_type.{type_name} is invalid")
        if (not isinstance(row.get("observed"), bool)
                or not isinstance(row.get("supported"), bool)):
            raise SchemaError(
                f"supported_coverage.by_type.{type_name} needs observed/supported booleans")
        if (row["observed"] != (type_name in observed)
                or row["supported"] != (type_name in supported)):
            raise SchemaError(
                f"supported_coverage.by_type.{type_name} contradicts its type sets")
        for field in ("candidate_count", "native_count", "alternative_count"):
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SchemaError(
                    f"supported_coverage.by_type.{type_name}.{field} must be non-negative int")
        if row["candidate_count"] != row["native_count"] + row["alternative_count"]:
            raise SchemaError(
                f"supported_coverage.by_type.{type_name} candidate counts disagree")
        families = row.get("alternative_families")
        if (not isinstance(families, list)
                or any(not isinstance(family, str) or not family for family in families)
                or families != sorted(set(families))):
            raise SchemaError(
                f"supported_coverage.by_type.{type_name}.alternative_families is invalid")
        architectures = row.get("architectures")
        if (not isinstance(architectures, list)
                or any(arch not in ARCHITECTURES for arch in architectures)
                or architectures != sorted(set(architectures))):
            raise SchemaError(
                f"supported_coverage.by_type.{type_name}.architectures is invalid")
        by_architecture = row.get("by_architecture")
        if (not isinstance(by_architecture, dict)
                or set(by_architecture) != set(architectures)):
            raise SchemaError(
                f"supported_coverage.by_type.{type_name}.by_architecture is incomplete")
        if not row["supported"]:
            reason = row.get("zero_alternative_reason")
            if (row["alternative_count"] != 0
                    or not isinstance(reason, str) or not reason):
                raise SchemaError(
                    f"unsupported type {type_name!r} lacks an explicit zero-alternative reason")


def validate_and_summarise(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate, then return per-family and per-source-class counts."""
    validate_manifest(manifest)
    by_family: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for candidate in manifest["candidates"]:
        by_family[candidate["family"]] = by_family.get(candidate["family"], 0) + 1
        source = candidate["source_class"]
        by_source[source] = by_source.get(source, 0) + 1
    return {
        "total": len(manifest["candidates"]),
        "by_family": dict(sorted(by_family.items())),
        "by_source_class": dict(sorted(by_source.items())),
    }

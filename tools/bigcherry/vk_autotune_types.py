"""RE30 phase 2: Vulkan autotune signature/hardware identity types.

Mirrors the HIP identity model (hip-autotune-signature.h/.cpp on the C++
side, autotune_schema.py's dict-validator style on the Python side) rather
than inventing a divergent shape, per RE30's own design notes. Digests use
distinct personalisation strings (``llama-vk-tune``/``llama-vk-hw``)
so a Vulkan digest and a HIP digest built from coincidentally identical
bytes can never collide -- same reasoning as HIP's
``GGML_HIP_PERSON_SIGNATURE``/``GGML_HIP_PERSON_HARDWARE`` (see
hip-autotune-signature.h).

This module is data-model only: validation, canonical JSON, and digesting.
No Vulkan patch, dispatch hook, or measurement writer exists yet -- nothing
here is wired into a build (RE30 is pre-implementation past this
scaffolding). It exists so the identity shape backing the new
``vk_hardware``/``vk_signature``/... tables in sql/dispatch-db.sql is
settled and testable before any Vulkan record/tune/replay code is written
against it.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

DIGEST_BYTES = 16
PERSON_VK_SIGNATURE = b"llama-vk-tune"
PERSON_VK_HARDWARE = b"llama-vk-hw"

VK_SIGNATURE_SCHEMA_VERSION = 1
VK_HARDWARE_SCHEMA_VERSION = 1

# Mirrors vk_candidate.family's CHECK constraint in sql/dispatch-db.sql --
# grows in step with that CHECK as Vulkan families are added.
VK_FAMILIES = ("mul_mat", "mul_mat_id")

VK_LAYOUTS = ("row_major", "col_major", "coopmat")

VK_SOURCE_CLASSES = (
    "native_wrapper",
    "existing_runtime",
    "existing_alternative",
    "new_generated_variant",
    "vendor_auto",
    "vendor_explicit",
)


class VkSchemaError(ValueError):
    """A Vulkan signature/hardware dict failed structural validation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VkSchemaError(message)


def _require_str(value: Any, field: str) -> str:
    _require(isinstance(value, str) and value != "", f"{field} must be a non-empty string")
    return value


def _require_int(value: Any, field: str, *, minimum: int | None = None) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{field} must be an int")
    if minimum is not None:
        _require(value >= minimum, f"{field} must be >= {minimum}")
    return value


def blake2b_digest(data: bytes, *, person: bytes) -> bytes:
    """The project's content hash, personalised per identity namespace.

    Same construction as replay_cache.blake2b_digest (blake2b, 16 bytes,
    personalised) -- distinct `person` values are what keep a Vulkan digest
    and a HIP digest over coincidentally identical canonical JSON from
    colliding into one key (mirrors hip-autotune-signature.h's comment on
    personalisation prefixes).
    """
    return hashlib.blake2b(data, digest_size=DIGEST_BYTES, person=person).digest()


# --------------------------------------------------------------- hardware


def validate_vk_hardware(hardware: dict[str, Any], *, where: str = "vk_hardware") -> dict[str, Any]:
    """Structural validation for a Vulkan hardware-identity dict.

    Field list mirrors RE30 detailed_solution's "Identity and persistence"
    section: vendor/device ID plus stable device class, driver and Vulkan
    API version, subgroup properties, extensions/features, limits, and a
    glslc/SPIR-V/source fingerprint. Contains no device ordinal (same
    standards-10 sharing rule HIP's hardware key follows: two identical GPUs
    must produce the same key so they can share a winner).
    """
    _require(isinstance(hardware, dict), f"{where}: must be a dict")
    _require_int(hardware.get("vendor_id"), f"{where}.vendor_id", minimum=0)
    _require_int(hardware.get("device_id"), f"{where}.device_id", minimum=0)
    _require_str(hardware.get("device_class"), f"{where}.device_class")
    _require_str(hardware.get("driver_version"), f"{where}.driver_version")
    _require_str(hardware.get("api_version"), f"{where}.api_version")
    _require_int(hardware.get("subgroup_size"), f"{where}.subgroup_size", minimum=1)
    _require_int(hardware.get("subgroup_ops_mask"), f"{where}.subgroup_ops_mask", minimum=0)

    extensions = hardware.get("extensions")
    _require(isinstance(extensions, list) and all(isinstance(e, str) for e in extensions),
             f"{where}.extensions must be a list of strings")

    limits = hardware.get("limits")
    _require(isinstance(limits, dict), f"{where}.limits must be a dict")

    shader_toolchain_digest = hardware.get("shader_toolchain_digest")
    _require(isinstance(shader_toolchain_digest, str)
             and len(shader_toolchain_digest) == DIGEST_BYTES * 2,
             f"{where}.shader_toolchain_digest must be a {DIGEST_BYTES * 2}-hex-char string")

    return hardware


def vk_hardware_canonical_json(hardware: dict[str, Any]) -> str:
    """Deterministic, sorted-key, no-whitespace JSON -- mirrors
    ggml_hip_hardware_json's contract so a stored key can be explained
    without re-deriving it."""
    validate_vk_hardware(hardware)
    canonical = {
        "schema_version": VK_HARDWARE_SCHEMA_VERSION,
        "vendor_id": hardware["vendor_id"],
        "device_id": hardware["device_id"],
        "device_class": hardware["device_class"],
        "driver_version": hardware["driver_version"],
        "api_version": hardware["api_version"],
        "subgroup_size": hardware["subgroup_size"],
        "subgroup_ops_mask": hardware["subgroup_ops_mask"],
        "extensions": sorted(hardware["extensions"]),
        "limits": hardware["limits"],
        "shader_toolchain_digest": hardware["shader_toolchain_digest"].lower(),
    }
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"))


def vk_hardware_digest(hardware: dict[str, Any]) -> bytes:
    return blake2b_digest(vk_hardware_canonical_json(hardware).encode("utf-8"), person=PERSON_VK_HARDWARE)


# -------------------------------------------------------------- signature


def validate_vk_signature(signature: dict[str, Any], *, where: str = "vk_signature") -> dict[str, Any]:
    """Structural validation for a Vulkan signature dict.

    Field list mirrors RE30 detailed_solution's signature description:
    operation, all dimensions/strides/types, output/accumulation precision,
    layout/alignment, batching, conversion/quantisation path, split-K and
    fusion/graph conditions. Diagnostic identity (model name, layer index,
    pointers, wall clock) is deliberately absent -- same standards-5.1 rule
    HIP's signature follows: that travels as separate observation metadata,
    never into the digest.
    """
    _require(isinstance(signature, dict), f"{where}: must be a dict")
    _require_str(signature.get("op"), f"{where}.op")
    _require_str(signature.get("src0_type"), f"{where}.src0_type")
    _require_str(signature.get("src1_type"), f"{where}.src1_type")
    _require_str(signature.get("dst_type"), f"{where}.dst_type")
    _require_str(signature.get("output_precision"), f"{where}.output_precision")
    _require_str(signature.get("accumulation_precision"), f"{where}.accumulation_precision")
    _require_int(signature.get("m"), f"{where}.m", minimum=0)
    _require_int(signature.get("n"), f"{where}.n", minimum=0)
    _require_int(signature.get("k"), f"{where}.k", minimum=0)

    layout = signature.get("layout")
    _require(layout in VK_LAYOUTS, f"{where}.layout must be one of {VK_LAYOUTS}")

    _require_int(signature.get("alignment_class"), f"{where}.alignment_class", minimum=0)
    _require_str(signature.get("batching", "none"), f"{where}.batching")
    _require_str(signature.get("conversion_route", "none"), f"{where}.conversion_route")
    _require_int(signature.get("split_k", 0), f"{where}.split_k", minimum=0)
    _require_str(signature.get("fusion", "none"), f"{where}.fusion")

    return signature


def vk_signature_canonical_json(signature: dict[str, Any], *, include_refinements: bool = True) -> str:
    """Deterministic, sorted-key, no-whitespace JSON -- mirrors
    ggml_hip_signature_json's contract. ``include_refinements=False``
    produces the base-digest input, same base/refined split as HIP's
    lookup-falls-back-to-base rule (standards 5.5)."""
    validate_vk_signature(signature)
    canonical = {
        "schema_version": VK_SIGNATURE_SCHEMA_VERSION,
        "op": signature["op"],
        "src0_type": signature["src0_type"],
        "src1_type": signature["src1_type"],
        "dst_type": signature["dst_type"],
        "output_precision": signature["output_precision"],
        "accumulation_precision": signature["accumulation_precision"],
        "m": signature["m"],
        "n": signature["n"],
        "k": signature["k"],
        "layout": signature["layout"],
        "conversion_route": signature.get("conversion_route", "none"),
        "split_k": signature.get("split_k", 0),
        "fusion": signature.get("fusion", "none"),
    }
    if include_refinements:
        canonical["alignment_class"] = signature["alignment_class"]
        canonical["batching"] = signature.get("batching", "none")
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"))


def vk_signature_digest(signature: dict[str, Any]) -> bytes:
    return blake2b_digest(
        vk_signature_canonical_json(signature, include_refinements=True).encode("utf-8"),
        person=PERSON_VK_SIGNATURE,
    )


def vk_signature_base_digest(signature: dict[str, Any]) -> bytes:
    """Refinements stripped -- lookup goes refined -> base -> native, same
    fallback chain as HIP's signature_base_digest (standards 5.5)."""
    return blake2b_digest(
        vk_signature_canonical_json(signature, include_refinements=False).encode("utf-8"),
        person=PERSON_VK_SIGNATURE,
    )


def validate_vk_candidate(candidate: dict[str, Any], *, where: str = "vk_candidate") -> dict[str, Any]:
    """Structural validation for a Vulkan candidate dict -- mirrors
    ggml_hip_candidate_descriptor's identity-bearing fields (stable_name,
    family, source_class) plus RE30's pipeline-recipe requirement:
    ``pipeline_stage_count``/``shader_module_digests_json`` say how many
    command-buffer stages the recipe comprises and what each compiles from,
    so a candidate can never be a bare terminal-kernel dispatch."""
    _require(isinstance(candidate, dict), f"{where}: must be a dict")
    _require_str(candidate.get("stable_name"), f"{where}.stable_name")

    family = candidate.get("family")
    _require(family in VK_FAMILIES, f"{where}.family must be one of {VK_FAMILIES}")

    source_class = candidate.get("source_class")
    _require(source_class in VK_SOURCE_CLASSES, f"{where}.source_class must be one of {VK_SOURCE_CLASSES}")

    stage_count = _require_int(candidate.get("pipeline_stage_count", 1), f"{where}.pipeline_stage_count", minimum=1)
    shader_digests = candidate.get("shader_module_digests")
    _require(isinstance(shader_digests, list) and len(shader_digests) == stage_count,
             f"{where}.shader_module_digests must list exactly pipeline_stage_count digests")
    for digest in shader_digests:
        _require(isinstance(digest, str) and len(digest) == DIGEST_BYTES * 2,
                 f"{where}.shader_module_digests entries must be {DIGEST_BYTES * 2}-hex-char strings")

    return candidate

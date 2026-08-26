"""HIP producer capability registry and source-owned declaration reader
(HI121 M1b).

A capability bit means "this producer's compiled code knew how to correctly
evaluate semantic question X" -- e.g. "this producer's ggml_hip_make_
signature() correctly evaluates fusion->x_bias presence". This is a distinct
axis from a signature's own CONTENT flags (ggml_hip_signature_flag in
hip-autotune-types.h), which describe what a dispatch actually HAS.

Authority for what a given compiled binary's capabilities are is the
EXPLICIT, source-owned #define declaration in hip-autotune-types.h
(GGML_HIP_PRODUCER_CAPABILITIES_LO/HI), read directly via
load_declared_producer_capabilities() below -- never inferred by pattern-
matching any other C++ code's behavior (e.g. scanning ggml_hip_make_
signature() for its real evaluator assignments and inferring capabilities
from what's found). HI121's round-8 review rejected that approach: a
semantically-neutral refactor (renaming a variable, restructuring an
if-statement, moving an assignment to a helper) could leave matched text
intact while changing what the code actually does, silently granting a
capability that isn't really true -- a false positive, which is unacceptable
since a capability bit is a permission grant for measurement reuse. An
explicit declaration doesn't have this failure mode: a refactor that doesn't
touch the #define doesn't change what's declared, and a genuine capability
change requires deliberately editing the declaration (matching the append-
only, never-redefine-a-bit governance below).

HIP capability IDs are append-only and never reused or renamed once
allocated. If an evaluator is later found to have had a bug, the fix is to
allocate a NEW capability id (e.g. a hypothetical FUSION_X_BIAS_PRESENCE_V2)
and update the relevant applicability rule (signature_capabilities.py) to
require it -- FUSION_X_BIAS_PRESENCE_V1 itself is never redefined, so old
measurements that only ever claimed V1 correctly stop qualifying once V2 is
required, rather than being silently treated as still-correct.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from enum import IntEnum
from pathlib import Path

from .capabilities import CapabilityMask128, CapabilityMaskError

_TYPES_H_RELATIVE = Path("ggml/src/ggml-cuda/hip-autotune-types.h")

_DECLARE_RE = re.compile(
    r"#define\s+GGML_HIP_PRODUCER_CAPABILITIES_(LO|HI)\s+UINT64_C\(\s*(0[xX][0-9a-fA-F]+)\s*\)"
)


class HipCapabilityError(ValueError):
    """The HIP producer capability declaration or registry is invalid."""


class HipCapability(IntEnum):
    """Append-only HIP capability IDs. Never reuse or rename an entry --
    see this module's own docstring for the versioned-correction procedure."""

    CORE_SIGNATURE_V1 = 0
    FUSION_X_BIAS_PRESENCE_V1 = 1
    FUSION_GATE_BIAS_PRESENCE_V1 = 2
    FUSION_X_SCALE_PRESENCE_V1 = 3
    FUSION_GATE_SCALE_PRESENCE_V1 = 4


def hip_capability_mask(capabilities: Iterable[HipCapability]) -> CapabilityMask128:
    """Build a mask from immutable HIP capability IDs."""
    value = 0
    for capability in capabilities:
        if not isinstance(capability, HipCapability):
            raise HipCapabilityError(f"expected a HipCapability, got {capability!r}")
        value |= 1 << int(capability)
    return CapabilityMask128(value)


def known_hip_capability_mask() -> CapabilityMask128:
    """Return every HIP capability bit this tooling currently understands."""
    return hip_capability_mask(HipCapability)


def load_declared_producer_capabilities(root: Path) -> CapabilityMask128:
    """Read the explicit HIP producer capability declaration from source.

    Reads `<root>/ggml/src/ggml-cuda/hip-autotune-types.h` and parses the
    real GGML_HIP_PRODUCER_CAPABILITIES_LO/HI #define pair -- the ONLY
    authority for what a materialized source tree's compiled binary will
    claim (see this module's own docstring for why this is a declaration
    read, never a behavioral inference).

    Raises HipCapabilityError if the file is missing, either #define is
    missing or duplicated, either value doesn't fit a uint64, or the parsed
    mask contains any bit not in known_hip_capability_mask() (an unknown bit
    in a materialized tree this tooling doesn't understand yet must fail
    closed, not be silently accepted or dropped).
    """
    types_h = Path(root) / _TYPES_H_RELATIVE
    if not types_h.is_file():
        raise HipCapabilityError(f"hip-autotune-types.h not found under root {root!r} (looked at {types_h})")
    text = types_h.read_text(encoding="utf-8")

    found: dict[str, int] = {}
    for match in _DECLARE_RE.finditer(text):
        half, hex_value = match.group(1), match.group(2)
        if half in found:
            raise HipCapabilityError(
                f"GGML_HIP_PRODUCER_CAPABILITIES_{half} is declared more than once in {types_h}"
            )
        value = int(hex_value, 16)
        if value < 0 or value > (1 << 64) - 1:
            raise HipCapabilityError(
                f"GGML_HIP_PRODUCER_CAPABILITIES_{half}={hex_value!r} does not fit a uint64"
            )
        found[half] = value

    missing = {"LO", "HI"} - found.keys()
    if missing:
        raise HipCapabilityError(
            f"GGML_HIP_PRODUCER_CAPABILITIES_{'/'.join(sorted(missing))} not found (or malformed) in {types_h}"
        )

    try:
        mask = CapabilityMask128.from_words(lo=found["LO"], hi=found["HI"])
    except CapabilityMaskError as exc:
        raise HipCapabilityError(f"malformed producer capability declaration in {types_h}: {exc}") from exc

    known = known_hip_capability_mask()
    unknown_bits = mask.value & ~known.value
    if unknown_bits:
        raise HipCapabilityError(
            f"{types_h} declares producer capability bits {unknown_bits:#x} this tooling does not "
            f"recognize (known bits: {known.to_hex()}) -- update hip_capabilities.HipCapability first"
        )
    return mask

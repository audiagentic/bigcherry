"""HI121 M3: required_capabilities() -- the fail-closed applicability rule
that decides what a producer must have KNOWN (not just what a signature's
CONTENT flags currently show) before a measurement for that signature can be
trusted as reusable.

This is the central distinction the whole HI121 redesign exists for:

    flag == 0, producer HAS the relevant capability  -> known absent
    flag == 0, producer LACKS the relevant capability -> unknown

Getting the applicability rule below backwards (checking "is this content
flag currently set" instead of "could this semantic question apply to this
signature's class") silently recreates that exact ambiguity one level up --
see docs/planning/active/hip-autotune/HI121.md's round-8/9 findings for the
concrete RD17 counterexample that first exposed this.

Deliberately NOT a giant (op, fusion, glu_op) -> mask Cartesian dict: the
applicability rules below are structured as an explicit allowlist per op
class, each with its own real preconditions, so a genuinely new/unaudited
op or domain fails closed (UnsupportedSignatureDomain) rather than an
absent dict entry defaulting to something permissive.

Scope note (deliberate, not an oversight): this module does NOT reimplement
the real C++ canonical-signature blake2b digest algorithm in Python. This
session's own established practice (see hi80_generate_correctness_evidence.
py's _observed_signature_hex, and its own docstring on why) is that a
signature digest is only ever trusted when it comes from the real, compiled
C++ hashing code itself -- reimplementing that byte-for-byte in Python here
would be new, untested surface for exactly the kind of subtle mismatch this
whole safety model exists to prevent. Round 9's design doc proposed a
Python-side canonical_signature_digest() recheck; this module intentionally
omits it for that reason. A caller that needs proof a signature's recorded
digest matches its own canonical content should use the C++ record-mode
comparison pattern instead (see hi80_generate_correctness_evidence.py).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import dispatch_abi
from . import hip_capabilities as hc
from . import signature_mapping
from .capabilities import CapabilityMask128

# Real ggml_hip_signature_flag bit values (hip-autotune-types.h) -- mirrors
# signature_mapping.py's own private constants (kept independent rather than
# imported, since that module marks them private/internal to its own
# MUL_MAT_ID/GLU mapping logic).
_SIG_SRC0_CONTIGUOUS = 1 << 0
_SIG_SRC1_CONTIGUOUS = 1 << 1
_SIG_DST_CONTIGUOUS = 1 << 2
_SIG_HAS_IDS = 1 << 3
_SIG_BROADCAST_CH = 1 << 4
_SIG_BROADCAST_SMP = 1 << 5
_SIG_BAD_PADDING = 1 << 6
_SIG_FUSION_X_BIAS = 1 << 7
_SIG_FUSION_GATE_BIAS = 1 << 8
_SIG_FUSION_X_SCALE = 1 << 9
_SIG_FUSION_GATE_SCALE = 1 << 10
_HI118_AUX_FLAGS = _SIG_FUSION_X_BIAS | _SIG_FUSION_GATE_BIAS | _SIG_FUSION_X_SCALE | _SIG_FUSION_GATE_SCALE

# Every flags bit ggml_hip_signature_flag (hip-autotune-types.h) currently
# defines. A bit outside this mask is a semantic distinction this module has
# never been audited against -- round 9 explicitly requires that case fail
# closed rather than silently returning CORE_SIGNATURE_V1 for a signature
# that may carry meaning this rule set doesn't know to check.
_KNOWN_FLAGS_MASK = (
    _SIG_SRC0_CONTIGUOUS | _SIG_SRC1_CONTIGUOUS | _SIG_DST_CONTIGUOUS | _SIG_HAS_IDS
    | _SIG_BROADCAST_CH | _SIG_BROADCAST_SMP | _SIG_BAD_PADDING | _HI118_AUX_FLAGS
)

_FUSION_KIND_NONE = 0  # GGML_HIP_FUSION_NONE
_FUSION_KIND_GATE = 2  # GGML_HIP_FUSION_GATE
_FUSION_KIND_GATE_BIAS = 3  # GGML_HIP_FUSION_GATE_BIAS
_FUSABLE_GLU_OPS = {1, 2, 3}  # GEGLU, SWIGLU, SWIGLU_OAI (ggml.h's real enum order)


class UnsupportedSignatureDomain(ValueError):
    """This signature's op/fusion/glu_op combination has no audited
    capability-applicability rule -- fails closed rather than guessing."""


def _require_int(signature: Mapping[str, Any], key: str) -> int:
    value = signature.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise UnsupportedSignatureDomain(f"signature field {key!r} must be an int, got {value!r}")
    return value


def hip_required_capabilities(
    signature: Mapping[str, Any], *, vendor_root: Path,
) -> CapabilityMask128:
    """Return the producer capabilities required to trust this HIP
    signature's content as authoritative, or raise UnsupportedSignatureDomain
    if this signature's op/fusion/glu_op combination has no audited rule.

    Fails closed on anything not explicitly recognized below -- an
    unsupported/unaudited signature must be reported as needing a rerun, not
    silently treated as compatible with whatever the caller happens to have.
    """
    schema_version = _require_int(signature, "schema_version")
    if schema_version != dispatch_abi.SIGNATURE_IDENTITY_EPOCH:
        raise UnsupportedSignatureDomain(
            f"signature schema_version={schema_version!r} is not the current identity epoch "
            f"({dispatch_abi.SIGNATURE_IDENTITY_EPOCH!r}) -- cannot be trusted under today's "
            f"capability applicability rules"
        )

    op_id = _require_int(signature, "op")
    op_names = signature_mapping.load_ggml_op_names(vendor_root)
    op_name = op_names.get(op_id)
    if op_name is None:
        raise UnsupportedSignatureDomain(f"unknown ggml_op id {op_id!r} -- not present in the parsed enum/name tables")

    flags = _require_int(signature, "flags")
    fusion = _require_int(signature, "fusion")
    glu_op = _require_int(signature, "glu_op")
    if flags < 0 or (flags & ~_KNOWN_FLAGS_MASK) != 0:
        raise UnsupportedSignatureDomain(
            f"signature flags={flags!r} sets bit(s) outside the known flags mask "
            f"({_KNOWN_FLAGS_MASK:#x}) -- this signature may carry a semantic distinction "
            f"this rule set has never been audited against; refusing to guess"
        )
    has_ids = bool(flags & _SIG_HAS_IDS)
    has_aux = bool(flags & _HI118_AUX_FLAGS)

    if op_name in ("MUL_MAT", "MUL_MAT_ID"):
        # Defensive per round 8: this combination is not currently reachable
        # (ggml_hip_fusion_kind() cannot return NONE while glu_op is set --
        # verified against real source), but a future refactor could
        # decouple them, so require it explicitly rather than assume it.
        if fusion != _FUSION_KIND_NONE or glu_op != 0:
            raise UnsupportedSignatureDomain(
                f"{op_name} signature has fusion={fusion!r}/glu_op={glu_op!r} -- "
                f"a plain (unfused) {op_name} must have neither set"
            )
        if has_aux:
            raise UnsupportedSignatureDomain(
                f"{op_name} signature has HI118 bias/scale flag bits set ({flags:#x}) -- "
                f"not a plain unfused dispatch, no applicability rule for this combination"
            )
        expected_has_ids = op_name == "MUL_MAT_ID"
        if has_ids != expected_has_ids:
            raise UnsupportedSignatureDomain(
                f"{op_name} signature has HAS_IDS={has_ids!r}, expected {expected_has_ids!r}"
            )
        return hc.hip_capability_mask([hc.HipCapability.CORE_SIGNATURE_V1])

    if op_name == "GLU":
        if not has_ids:
            raise UnsupportedSignatureDomain(
                "GLU signature does not have GGML_HIP_SIG_HAS_IDS set -- only the MoE-routed "
                "(MUL_MAT_ID-based) fused GLU case has an audited rule; a non-routed/dense GLU "
                "fusion needs its own sibling rule, not yet written"
            )
        # ggml_hip_fusion_kind() (hip-autotune-signature.cpp) classifies a
        # GATE fusion that ALSO has a real x_bias/gate_bias tensor as
        # GGML_HIP_FUSION_GATE_BIAS (3), not GATE (2) -- confirmed against
        # real source. A real biased GLU dispatch therefore never has
        # fusion==GATE; checking only for GATE here would fail closed on
        # every real biased dispatch forever, even after a producer declares
        # the bias-presence capabilities specifically to support them. Both
        # real representations are accepted; the required-capabilities
        # result is identical either way (all four presence capabilities),
        # since a GATE signature's biases are unknown-not-necessarily-absent
        # exactly the same way a GATE_BIAS signature's scales are.
        if fusion not in (_FUSION_KIND_GATE, _FUSION_KIND_GATE_BIAS):
            raise UnsupportedSignatureDomain(
                f"GLU signature fusion={fusion!r} is neither GGML_HIP_FUSION_GATE "
                f"({_FUSION_KIND_GATE}) nor GGML_HIP_FUSION_GATE_BIAS ({_FUSION_KIND_GATE_BIAS}) -- "
                f"no audited rule for this fusion kind"
            )
        # HI121 review follow-up: ggml_hip_fusion_kind() classifies GATE_BIAS
        # (verified against real source) precisely when at least one of
        # fusion->x_bias/gate_bias is non-null, and the X_BIAS/GATE_BIAS
        # content flags are set from those SAME pointers -- so GATE_BIAS
        # with NEITHER bias flag set, or GATE with EITHER bias flag set, is
        # a self-contradictory signature no real dispatch could produce.
        # Accepting either as "well-formed, just needs all 4 capabilities"
        # (as an earlier version of this fixture/rule did) admits a second
        # impossible state instead of fixing the first one.
        has_any_bias_flag = bool(flags & (_SIG_FUSION_X_BIAS | _SIG_FUSION_GATE_BIAS))
        if fusion == _FUSION_KIND_GATE_BIAS and not has_any_bias_flag:
            raise UnsupportedSignatureDomain(
                "GLU signature has fusion=GATE_BIAS but neither X_BIAS nor GATE_BIAS content "
                "flag is set -- ggml_hip_fusion_kind() cannot classify GATE_BIAS without a "
                "real bias tensor present; this combination cannot come from a real dispatch"
            )
        if fusion == _FUSION_KIND_GATE and has_any_bias_flag:
            raise UnsupportedSignatureDomain(
                "GLU signature has fusion=GATE but a bias content flag is set -- "
                "ggml_hip_fusion_kind() would have classified a real bias-bearing dispatch as "
                "GATE_BIAS, not GATE; this combination cannot come from a real dispatch"
            )
        if glu_op not in _FUSABLE_GLU_OPS:
            raise UnsupportedSignatureDomain(
                f"GLU signature glu_op={glu_op!r} is not one of GEGLU/SWIGLU/SWIGLU_OAI "
                f"({sorted(_FUSABLE_GLU_OPS)}) -- ggml-cuda's own fusion detector never "
                f"fuses this glu_op, so no real fused dispatch could have produced this signature"
            )
        # HI120 gate: RD17's x_scale_channel_dst mode and RD12's dst_gate
        # domain are not yet fully represented in signature content -- until
        # that lands, ANY signature that could be in one of those domains
        # must fail closed rather than let a capability check paper over a
        # content-identity gap no amount of correct capability-gating can
        # fix (see HI121's round-3 finding). Presence of the X_SCALE/
        # GATE_SCALE content bits is exactly the case HI108's own real
        # production dispatches never exercise, so this costs no real
        # coverage today.
        if flags & (_SIG_FUSION_X_SCALE | _SIG_FUSION_GATE_SCALE):
            raise UnsupportedSignatureDomain(
                "GLU signature has an x_scale/gate_scale content flag set -- RD17's "
                "x_scale_channel_dst mode is not yet fully represented in signature content "
                "(HI120, not yet landed); refusing to certify compatibility for this domain"
            )
        # The central HI121 distinction: require ALL FOUR presence
        # capabilities regardless of whether the corresponding content bits
        # are currently 0 or 1 -- a zero content bit still needs
        # authoritative producer knowledge that it was actually evaluated,
        # not just that this particular dispatch happens not to need it.
        return hc.hip_capability_mask([
            hc.HipCapability.CORE_SIGNATURE_V1,
            hc.HipCapability.FUSION_X_BIAS_PRESENCE_V1,
            hc.HipCapability.FUSION_GATE_BIAS_PRESENCE_V1,
            hc.HipCapability.FUSION_X_SCALE_PRESENCE_V1,
            hc.HipCapability.FUSION_GATE_SCALE_PRESENCE_V1,
        ])

    raise UnsupportedSignatureDomain(
        f"op={op_name!r} has no audited capability-applicability rule -- refusing to guess"
    )

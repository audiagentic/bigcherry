"""HI18: SPLIT_REDUCE correctness comparison (RCCL vs META vs AUTO against a
CPU-double oracle).

Architecture mirrors tools/bigcherry/correctness_evidence.py (HI67): a native
probe reports raw, machine-readable execution facts; this module owns every
policy decision -- corpus generation, digest-based input-identity proof,
numerical correctness, and fail-closed rejection. It does not itself invoke
the native probe; callers pass in a `ProviderRun` per arm (from a real
`test-hip-reduce` invocation, or a synthetic stand-in in tests).

Corpus (design per GPT HI18 review, 2026-08-22, verified against
patches/0830_split_reduce_telemetry/patch.py and src/ggml/src/ggml-cuda/hip-
autotune-reduce-telemetry.h before implementation):

    case-dir/
        case.json      -- generator_version, seed, pattern, reduction
                           signature, per-rank input digests
        rank-0.f32      -- frozen little-endian F32 bytes for device 0
        rank-1.f32      -- frozen little-endian F32 bytes for device 1
        ...

Generated once with a versioned seeded PRNG, then given byte-for-byte
identical to every provider arm -- never regenerated per arm. That is what
makes "RCCL vs META vs AUTO" a valid comparison instead of three
independently-executed things that happen to share a seed.

Correctness gate: NOT an arbitrary fixed NMSE threshold (that is HI67's
matmul contract, not this one). Reduction has a clean analytical model: with
D participating devices and F32 unit roundoff u = 2^-24,

    gamma = ((D - 1) * u) / (1 - (D - 1) * u)
    allowed_abs_error[i] = gamma * sum_abs[i] + D * FLT_MIN

and the hard elementwise gate is abs(output[i] - reference[i]) <=
allowed_abs_error[i] for every element, computed against a CPU reference
accumulated in double. NMSE/RMSE/max-abs are recorded for diagnostics only
-- an aggregate metric can hide isolated bad values the elementwise check
would catch.

Cross-device agreement (every participant of an allreduce must receive the
same result) gets the same treatment: a hard elementwise bound between each
pair of participating devices' outputs, not a bitwise-equality requirement
-- bitwise equality is recorded (`cross_device_digest_equal`) as stronger-
than-required evidence when it holds, but is not baked into the general
contract because a >2-device topology can legitimately use a different
reduction tree with different rounding.

Provenance gate: a numerically correct output is not RCCL evidence if the
run actually fell through to META. `PROVENANCE_GATES` requires the RCCL arm
to show requested_provider=effective_provider="rccl", provider_succeeded,
handoff="none", fallback_depth=0; the META arm to show
requested_provider="meta". AUTO has no gate -- it records whatever
production actually chose, which is itself a finding if it isn't RCCL.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

CONTRACT_VERSION = "hi18-reduce-correctness-v1"
GENERATOR_VERSION = "hi18-f32-v1"

# F32 unit roundoff (2^-24) and smallest positive normal F32 (2^-126), used
# by the analytical summation error bound -- not numpy.finfo so this module
# has no hard runtime dependency on numpy.
F32_ULP = 2.0 ** -24
FLT_MIN = 2.0 ** -126

# GP05: BF16 unit roundoff (7 explicit mantissa bits + implicit leading one
# => 2^-8), used to extend the analytical bound for the real BF16-wire-
# compression regime upstream's RCCL provider silently enters at large
# element counts (see NCCL_BF16_WIRE_THRESHOLD_BY_DEVICE_COUNT below). This
# is NOT a hypothetical -- ggml_backend_cuda_comm_allreduce_nccl() really
# does compress each device's F32 operand to BF16 before the collective and
# convert back, as a deliberate bandwidth/precision tradeoff, not a bug.
BF16_ULP = 2.0 ** -8

# GP05: upstream's real hardcoded element-count thresholds (ggml-cuda.cu
# ~line 1053-1128) above which ggml_backend_cuda_comm_allreduce_nccl()
# switches from exact F32 to BF16-wire-compressed transport, keyed by
# participant count. Verified via real hardware threshold bisection
# (PASS at ne=30720, FAIL at ne=35840, bracketing the <=2-device 32768
# constant precisely) plus source read. Do not invent additional entries
# speculatively -- these three are the only ones confirmed against the
# pinned source; a device_count this table doesn't cover is a real gap to
# re-verify against source, not to guess at.
NCCL_BF16_WIRE_THRESHOLD_BY_DEVICE_COUNT: dict[int, int] = {
    1: 32768,
    2: 32768,
    3: 131072,
    4: 262144,
}


def nccl_bf16_wire_threshold(device_count: int) -> int:
    """The real upstream element-count threshold at/above which the rccl
    provider switches to BF16-wire-compressed transport for this many
    participants. Devices counts >=4 all share the same >=4 threshold per
    the pinned source (no further per-count scaling beyond 4)."""
    if device_count >= 4:
        return NCCL_BF16_WIRE_THRESHOLD_BY_DEVICE_COUNT[4]
    if device_count not in NCCL_BF16_WIRE_THRESHOLD_BY_DEVICE_COUNT:
        raise CorrectnessError(
            f"device_count={device_count} has no confirmed BF16-wire-"
            "compression threshold -- re-verify against the pinned "
            "ggml-cuda.cu source before adding it, do not guess"
        )
    return NCCL_BF16_WIRE_THRESHOLD_BY_DEVICE_COUNT[device_count]


def provider_uses_bf16_wire(provider: str, element_count: int, device_count: int) -> bool:
    """Whether THIS specific provider run would have gone through
    upstream's real BF16-wire-compression path for this case shape. Only
    the rccl provider does this (confirmed via source read) -- meta and
    auto (when auto resolves to something other than rccl) stay exact F32,
    so a shared per-case bound would either be too strict for rccl's real
    large-signature behavior or too loose for meta's -- this must be
    evaluated per provider run, not once per case."""
    return provider == "rccl" and element_count >= nccl_bf16_wire_threshold(device_count)

PATTERNS = (
    "ordinary_signed",
    "all_positive",
    "alternating_signs",
    "cancellation",
    "mixed_magnitude",
    "sparse_zeros",
)


class CorrectnessError(RuntimeError):
    """A case, comparison, or aggregate failed closed -- never a partial or
    unproven verdict."""


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _to_f32(value: float) -> float:
    """Round a Python (double) float to its nearest F32 representable
    value, so every downstream byte is the actual value that will be
    written to disk and fed to every provider arm."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def to_f32_bytes(values: list[float]) -> bytes:
    return b"".join(struct.pack("<f", v) for v in values)


def from_f32_bytes(data: bytes) -> list[float]:
    n = len(data) // 4
    if n * 4 != len(data):
        raise CorrectnessError(f"byte length {len(data)} is not a multiple of 4 (not valid F32 data)")
    return list(struct.unpack(f"<{n}f", data))


# ---------------------------------------------------------------------------
# Corpus generation
# ---------------------------------------------------------------------------

def generate_case(seed: int, pattern: str, element_count: int, device_count: int) -> list[list[float]]:
    """Deterministic per-device F32 values for one synthetic case. Pure
    function of (seed, pattern, element_count, device_count) -- same inputs
    always produce the same floats, so a corpus written once and replayed
    later reproduces exactly. Values are rounded to F32 immediately so the
    generator's own arithmetic never leaks double-only precision into what
    is nominally frozen F32 evidence.

    Uses only integer arithmetic to seed random.Random -- never str hash(),
    which is randomized per-process unless PYTHONHASHSEED is fixed."""
    if pattern not in PATTERNS:
        raise CorrectnessError(f"unknown pattern {pattern!r}, expected one of {PATTERNS}")
    if element_count <= 0:
        raise CorrectnessError(f"element_count must be positive, got {element_count}")
    if device_count < 2:
        raise CorrectnessError(f"device_count must be >= 2 for a reduction, got {device_count}")

    import random
    pattern_index = PATTERNS.index(pattern)
    rng = random.Random(seed * 104_729 + pattern_index * 7_919)

    devices: list[list[float]] = [[] for _ in range(device_count)]
    for i in range(element_count):
        if pattern == "ordinary_signed":
            for d in range(device_count):
                devices[d].append(_to_f32(rng.uniform(-1.0, 1.0)))
        elif pattern == "all_positive":
            for d in range(device_count):
                devices[d].append(_to_f32(rng.uniform(0.0, 1.0)))
        elif pattern == "alternating_signs":
            sign = 1.0 if i % 2 == 0 else -1.0
            for d in range(device_count):
                devices[d].append(_to_f32(sign * rng.uniform(0.0, 1.0)))
        elif pattern == "cancellation":
            v0 = _to_f32(rng.uniform(0.1, 1.0))
            devices[0].append(v0)
            for d in range(1, device_count):
                residual = rng.uniform(-1e-3, 1e-3)
                devices[d].append(_to_f32(-v0 + residual))
        elif pattern == "mixed_magnitude":
            for d in range(device_count):
                scale = 2.0 ** (d * 10)
                devices[d].append(_to_f32(rng.uniform(-1.0, 1.0) * scale))
        elif pattern == "sparse_zeros":
            nonzero = (i % 7 == 0)
            for d in range(device_count):
                devices[d].append(_to_f32(rng.uniform(-1.0, 1.0)) if nonzero else 0.0)
    return devices


def make_reduction_signature_key(
    *, element_type: str, element_count: int,
    slice_shape: tuple[int, int, int, int], topology_key: str,
) -> str:
    """The exact canonical key format tools/bigcherry/telemetry.py derives
    from real production reduction telemetry -- verified against its
    signature_key construction before use here, so a case's declared
    reduction_signature_key and a probe's observed runtime signature are
    provably comparable, not merely similarly-shaped strings."""
    return (
        f"split_reduce:v1:{element_type}:{element_count}:"
        f"{','.join(str(v) for v in slice_shape)}:{topology_key}"
    )


@dataclass(frozen=True)
class CaseManifest:
    case_id: str
    seed: int
    pattern: str
    generator_version: str
    element_count: int
    device_count: int
    reduction_signature_key: str
    input_digests: tuple[str, ...]
    case_dir: Path


def write_case(
    case_dir: Path, *, case_id: str, seed: int, pattern: str,
    reduction_signature_key: str, topology_key: str, peer_access: str,
    devices: list[list[float]], slice_shape: tuple[int, int, int, int] | None = None,
) -> CaseManifest:
    """slice_shape is the real production reduction signature's 4D tensor
    shape (element_count, slice_shape[0..3] with slice_shape's product ==
    element_count) -- required, not optional, because a case's evidence is
    only comparable to a real recorded reduction_signature_key if it
    reproduces that exact shape, not merely the same total element count
    (schema v2 fix; verified against tools/bigcherry/telemetry.py's key
    format, which joins slice_shape into the key, not just element_count)."""
    device_count = len(devices)
    element_count = len(devices[0])
    if any(len(v) != element_count for v in devices):
        raise CorrectnessError(f"case {case_id}: device value arrays have inconsistent length")
    if slice_shape is None:
        raise CorrectnessError(
            f"case {case_id}: slice_shape is required -- a case's evidence is only "
            "comparable to a real recorded reduction_signature_key if it reproduces "
            "that signature's exact 4D shape, not merely its total element count"
        )
    if len(slice_shape) != 4:
        raise CorrectnessError(
            f"case {case_id}: slice_shape must have exactly 4 entries, got "
            f"{len(slice_shape)}: {slice_shape}"
        )
    if any(dim <= 0 for dim in slice_shape):
        raise CorrectnessError(
            f"case {case_id}: slice_shape entries must all be positive, got {slice_shape}"
        )
    shape_product = slice_shape[0] * slice_shape[1] * slice_shape[2] * slice_shape[3]
    if shape_product != element_count:
        raise CorrectnessError(
            f"case {case_id}: slice_shape {slice_shape} has product {shape_product}, "
            f"expected element_count {element_count}"
        )
    expected_key = make_reduction_signature_key(
        element_type="f32", element_count=element_count,
        slice_shape=slice_shape, topology_key=topology_key,
    )
    if reduction_signature_key != expected_key:
        raise CorrectnessError(
            f"case {case_id}: reduction_signature_key {reduction_signature_key!r} does not "
            f"match the canonical key derived from this case's own element_count/slice_shape/"
            f"topology_key: {expected_key!r} -- a caller could otherwise write a case whose "
            "declared identity disagrees with what it actually reproduces"
        )

    case_dir.mkdir(parents=True, exist_ok=True)
    digests = []
    for d, values in enumerate(devices):
        data = to_f32_bytes(values)
        (case_dir / f"rank-{d}.f32").write_bytes(data)
        digests.append(sha256_hex(data))

    manifest = {
        "schema_version": 2,
        "case_id": case_id,
        "seed": seed,
        "pattern": pattern,
        "generator_version": GENERATOR_VERSION,
        "element_count": element_count,
        "slice_shape": list(slice_shape),
        "device_count": device_count,
        "reduction_signature_key": reduction_signature_key,
        "topology_key": topology_key,
        "peer_access": peer_access,
        "input_digests": digests,
    }
    (case_dir / "case.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return CaseManifest(
        case_id=case_id, seed=seed, pattern=pattern, generator_version=GENERATOR_VERSION,
        element_count=element_count, device_count=device_count,
        reduction_signature_key=reduction_signature_key,
        input_digests=tuple(digests), case_dir=case_dir,
    )


def load_case(case_dir: Path) -> tuple[dict, list[bytes]]:
    """Load a previously written case, verifying every rank file still
    matches its recorded digest -- corruption or an edited file after
    generation must fail closed, not silently feed altered bytes to a
    provider arm."""
    manifest = json.loads((case_dir / "case.json").read_text())
    device_count = manifest["device_count"]
    ranks = []
    for d in range(device_count):
        data = (case_dir / f"rank-{d}.f32").read_bytes()
        digest = sha256_hex(data)
        if digest != manifest["input_digests"][d]:
            raise CorrectnessError(
                f"case {manifest['case_id']}: rank-{d}.f32 digest mismatch "
                f"(expected {manifest['input_digests'][d]}, got {digest}) -- "
                "file corrupted or modified since generation"
            )
        ranks.append(data)
    return manifest, ranks


# ---------------------------------------------------------------------------
# CPU-double oracle and analytical error bound
# ---------------------------------------------------------------------------

def cpu_reference(device_values: list[list[float]]) -> tuple[list[float], list[float]]:
    """Reference[i] = sum_d double(x[d][i]), accumulated in double so this
    is independent reference arithmetic rather than a recreation of any one
    GPU implementation. Also returns sum_abs[i], the input to the
    analytical error bound."""
    device_count = len(device_values)
    element_count = len(device_values[0])
    if any(len(v) != element_count for v in device_values):
        raise CorrectnessError("device value arrays have inconsistent length")

    reference = [0.0] * element_count
    sum_abs = [0.0] * element_count
    for d in range(device_count):
        vals = device_values[d]
        for i in range(element_count):
            v = vals[i]
            reference[i] += v
            sum_abs[i] += abs(v)
    return reference, sum_abs


def analytical_error_bound(
    sum_abs: list[float], device_count: int, *, bf16_wire: bool = False,
) -> list[float]:
    """Elementwise summation error bound (standard floating-point
    analysis): gamma_n = n*u / (1 - n*u) for n = device_count - 1 pairwise
    additions, plus a D*FLT_MIN floor for flush-to-zero/subnormal coverage
    that does not create a meaningful tolerance for ordinary values.

    GP05 (gpt-dev-agent review, 2026-09-02, req_8714cf1798af40a4): when
    `bf16_wire` is set, this must account for TWO distinct BF16 error
    sources, not one. The first version of this fix only widened by a
    flat BF16_ULP * sum_abs[i], covering just the initial operand
    quantization to BF16 before the collective -- it MISSED that RCCL's
    real BF16 reduction also casts the running sum back to BF16 after
    EVERY addition (`hip_bfloat16((float)x + (float)y)`), so there are up
    to (device_count - 1) further BF16-rounded additions on top of the
    initial quantization, not just one. The corrected composition:

        u_in = BF16_ULP                          (initial quantization)
        u_step = (1 + F32_ULP) * (1 + BF16_ULP) - 1   (each BF16-rounded add)
        gamma_step = n * u_step / (1 - n * u_step), n = device_count - 1
        bound = [u_in + (1 + u_in) * gamma_step] * sum_abs + device_count * FLT_MIN

    This bound is WIDER than the original flat-BF16_ULP version (which was
    too tight, not too loose) -- a case that passed under the old bound
    still passes under this one; a case that failed under the old bound
    may have been a false negative and must be re-evaluated, never taken
    as proof RCCL itself is wrong without re-checking against this
    corrected bound first.

    Never applied unconditionally: a provider run that stayed exact F32
    (meta, or rccl below the real wire threshold) gets NO artificial
    slack -- see provider_uses_bf16_wire()."""
    if device_count < 1:
        raise CorrectnessError(f"device_count must be >= 1, got {device_count}")
    n = device_count - 1
    denom = 1.0 - n * F32_ULP
    if denom <= 0.0:
        raise CorrectnessError(
            f"device_count={device_count} is too large for the F32 summation "
            "bound to be meaningful (n * unit_roundoff >= 1)"
        )
    gamma = (n * F32_ULP) / denom
    if not bf16_wire:
        return [gamma * s + device_count * FLT_MIN for s in sum_abs]

    u_in = BF16_ULP
    u_step = (1.0 + F32_ULP) * (1.0 + BF16_ULP) - 1.0
    step_denom = 1.0 - n * u_step
    if step_denom <= 0.0:
        raise CorrectnessError(
            f"device_count={device_count} is too large for the BF16-wire "
            "summation bound to be meaningful (n * per-step unit_roundoff >= 1)"
        )
    gamma_step = (n * u_step) / step_denom
    coefficient = u_in + (1.0 + u_in) * gamma_step
    return [coefficient * s + device_count * FLT_MIN for s in sum_abs]


# ---------------------------------------------------------------------------
# Provider execution facts and evaluation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProviderRun:
    """Raw, machine-readable facts about one provider's execution of one
    case, as reported by the native test-hip-reduce probe. This module owns
    all evidence policy; ProviderRun carries only facts the probe reported,
    never a verdict."""
    provider: str                     # "auto" | "rccl" | "meta"
    requested_provider: str
    effective_provider: str
    provider_succeeded: bool
    handoff: str
    fallback_depth: int
    completion_synchronized: bool
    devices: tuple[int, ...]
    input_digests: tuple[str, ...]    # what the probe says it actually fed in, per device
    outputs: tuple[bytes, ...]        # raw F32 bytes per participating device, same order as `devices`


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    provider: str
    valid: bool
    correct: bool
    reason: str | None
    nmse_vs_reference: float | None
    max_abs_vs_reference: float | None
    rmse_vs_reference: float | None
    cross_device_nmse: float | None
    cross_device_max_abs: float | None
    cross_device_digest_equal: bool | None
    output_digests: tuple[str, ...]
    requested_provider: str
    effective_provider: str
    provider_succeeded: bool
    handoff: str
    fallback_depth: int
    completion_synchronized: bool


def _reject(case_id: str, run: ProviderRun, reason: str) -> CaseResult:
    return CaseResult(
        case_id=case_id, provider=run.provider, valid=False, correct=False, reason=reason,
        nmse_vs_reference=None, max_abs_vs_reference=None, rmse_vs_reference=None,
        cross_device_nmse=None, cross_device_max_abs=None, cross_device_digest_equal=None,
        output_digests=tuple(sha256_hex(o) for o in run.outputs),
        requested_provider=run.requested_provider, effective_provider=run.effective_provider,
        provider_succeeded=run.provider_succeeded, handoff=run.handoff,
        fallback_depth=run.fallback_depth, completion_synchronized=run.completion_synchronized,
    )


# For the RCCL candidate to count as (nccl, none) evidence, require exactly
# this path -- no exceptions. An RCCL failure followed by a correct META
# result is valid fallback-behavior evidence, but zero evidence about RCCL
# correctness. META has no "effective" requirement beyond having been
# requested: what it actually resolves to is itself part of what's observed.
PROVENANCE_GATES: dict[str, dict[str, object]] = {
    "rccl": {
        "requested_provider": "rccl",
        "effective_provider": "rccl",
        "provider_succeeded": True,
        "handoff": "none",
        "fallback_depth": 0,
    },
    "meta": {
        "requested_provider": "meta",
    },
}


def load_probe_run(result_path: Path, *, manifest: dict, provider: str) -> ProviderRun:
    """Strict ingestion of one test-hip-reduce result JSON into a
    ProviderRun, closing the seam GPT's review flagged: the native probe
    proves far more (probe_valid, an exact observed runtime signature, per-
    output digests) than the original bare ProviderRun construction used --
    a caller could silently ignore probe_valid or the process exit code and
    still hand evaluate_provider_run() an apparently valid object. This is
    the only sanctioned way to build a ProviderRun from a real probe
    invocation; it fails closed on every provenance/identity mismatch
    rather than deferring that check to evaluate_provider_run(), which
    trusts whatever ProviderRun it is given."""
    obj = json.loads(result_path.read_text(encoding="utf-8"))

    if obj.get("schema_version") != 1:
        raise CorrectnessError(f"{result_path}: unsupported test-hip-reduce result schema")
    if obj.get("case_id") != manifest["case_id"]:
        raise CorrectnessError(
            f"{result_path}: probe case_id {obj.get('case_id')!r} does not match "
            f"manifest {manifest['case_id']!r}"
        )
    if obj.get("plan") != provider:
        raise CorrectnessError(f"{result_path}: probe plan {obj.get('plan')!r} does not match arm {provider!r}")
    if obj.get("probe_valid") is not True:
        raise CorrectnessError(f"{result_path}: native probe did not establish a valid reduction (probe_valid != true)")
    if obj.get("completion_synchronized") is not True:
        raise CorrectnessError(f"{result_path}: native probe was not completion-synchronized")
    if obj.get("reduction_signature_matches_case") is not True:
        raise CorrectnessError(f"{result_path}: runtime reduction signature did not match the case")
    if obj.get("reduction_signature_key") != manifest["reduction_signature_key"]:
        raise CorrectnessError(
            f"{result_path}: native runtime reduction_signature_key "
            f"{obj.get('reduction_signature_key')!r} does not match manifest "
            f"{manifest['reduction_signature_key']!r}"
        )

    sig = obj.get("reduction_signature")
    if not isinstance(sig, dict):
        raise CorrectnessError(f"{result_path}: probe result has no reduction_signature")
    if sig.get("slice_shape") != list(manifest["slice_shape"]):
        raise CorrectnessError(f"{result_path}: probe runtime slice_shape does not match manifest")
    if sig.get("topology_key") != manifest["topology_key"]:
        raise CorrectnessError(f"{result_path}: probe runtime topology_key does not match manifest")
    if sig.get("peer_access") != manifest["peer_access"]:
        raise CorrectnessError(f"{result_path}: probe runtime peer_access does not match manifest")
    if sig.get("element_type") != "f32":
        raise CorrectnessError(f"{result_path}: probe runtime element_type is not f32")

    outputs_obj = obj.get("outputs")
    if not isinstance(outputs_obj, list):
        raise CorrectnessError(f"{result_path}: probe outputs must be an array")
    devices = tuple(obj["devices"])
    if len(outputs_obj) != manifest["device_count"]:
        raise CorrectnessError(
            f"{result_path}: probe output count {len(outputs_obj)} does not match "
            f"manifest device_count {manifest['device_count']}"
        )

    outputs: list[bytes] = []
    for rank, output_obj in enumerate(outputs_obj):
        if output_obj.get("device") != devices[rank]:
            raise CorrectnessError(f"{result_path}: output rank {rank} device does not match devices[{rank}]")
        data = Path(output_obj["path"]).read_bytes()
        if len(data) != manifest["element_count"] * 4:
            raise CorrectnessError(
                f"{result_path}: output rank {rank} is {len(data)} bytes, expected "
                f"{manifest['element_count'] * 4}"
            )
        digest = sha256_hex(data)
        if digest != output_obj.get("sha256"):
            raise CorrectnessError(f"{result_path}: output rank {rank} digest mismatch")
        outputs.append(data)

    return ProviderRun(
        provider=provider,
        requested_provider=obj["requested_provider"],
        effective_provider=obj["effective_provider"],
        provider_succeeded=obj["provider_succeeded"],
        handoff=obj["handoff"],
        fallback_depth=obj["fallback_depth"],
        completion_synchronized=True,
        devices=devices,
        input_digests=tuple(obj["input_digests"]),
        outputs=tuple(outputs),
    )


def evaluate_provider_run(
    manifest: dict,
    device_values: list[list[float]],
    reference: list[float],
    sum_abs: list[float],
    run: ProviderRun,
) -> CaseResult:
    """Evaluate one provider's execution against the CPU-double oracle and
    that provider's provenance gate. Fails closed on every condition HI18's
    design calls out: input-digest disagreement, a missing participant,
    non-finite output, a provenance-gate miss, an elementwise bound
    violation, or a cross-device disagreement -- never averaged away."""
    case_id = manifest["case_id"]
    device_count = manifest["device_count"]
    element_count = manifest["element_count"]

    if tuple(run.input_digests) != tuple(manifest["input_digests"]):
        return _reject(
            case_id, run,
            f"input digest mismatch: probe reported {run.input_digests}, "
            f"case manifest says {tuple(manifest['input_digests'])} -- this arm "
            "did not consume the same operands as the other arms, so its "
            "output is not comparable evidence"
        )
    if not run.completion_synchronized:
        return _reject(case_id, run, "completion not synchronized on every participating device")
    if len(run.devices) != device_count or len(run.outputs) != device_count:
        return _reject(
            case_id, run,
            f"missing participant output (expected {device_count} devices, "
            f"got {len(run.outputs)} outputs for {len(run.devices)} devices)"
        )
    if len(set(run.devices)) != device_count:
        return _reject(
            case_id, run,
            f"duplicate participating devices: {run.devices} -- two logical "
            "participants backed by the same physical device is not evidence "
            "about a real multi-device reduction"
        )

    gate = PROVENANCE_GATES.get(run.provider)
    if gate is not None:
        for field_name, expected in gate.items():
            actual = getattr(run, field_name)
            if actual != expected:
                return _reject(
                    case_id, run,
                    f"provenance gate failed for provider={run.provider!r}: "
                    f"{field_name}={actual!r}, required {expected!r}"
                )

    # GP05: the analytical bound is computed PER RUN, keyed off what this
    # specific provider run actually did (run.effective_provider, the real
    # servicing provider, not merely what was requested) -- rccl above the
    # real wire threshold gets the BF16-widened bound; every other run
    # (meta, or rccl below threshold) gets the plain exact-F32 bound with
    # zero artificial slack.
    bf16_wire = provider_uses_bf16_wire(run.effective_provider, element_count, device_count)
    allowed_abs_error = analytical_error_bound(sum_abs, device_count, bf16_wire=bf16_wire)

    outputs: list[list[float]] = []
    for data in run.outputs:
        if len(data) != element_count * 4:
            return _reject(
                case_id, run,
                f"output byte length mismatch (expected {element_count * 4} "
                f"bytes for {element_count} F32 elements, got {len(data)})"
            )
        outputs.append(from_f32_bytes(data))

    for out in outputs:
        for v in out:
            if math.isnan(v) or math.isinf(v):
                return _reject(case_id, run, "non-finite value in provider output for a finite-input case")

    sq_err_sum = 0.0
    sq_ref_sum = 0.0
    max_abs_err = 0.0
    for out in outputs:
        for i in range(element_count):
            err = abs(out[i] - reference[i])
            if err > allowed_abs_error[i]:
                return _reject(
                    case_id, run,
                    f"element {i} exceeds the analytical F32 summation bound: "
                    f"abs({out[i]!r} - {reference[i]!r}) = {err!r} > {allowed_abs_error[i]!r}"
                )
            sq_err_sum += err * err
            sq_ref_sum += reference[i] * reference[i]
            max_abs_err = max(max_abs_err, err)

    nmse = (sq_err_sum / sq_ref_sum) if sq_ref_sum > 0.0 else sq_err_sum
    rmse = math.sqrt(sq_err_sum / (element_count * device_count))

    cross_max_abs = 0.0
    cross_sq_err_sum = 0.0
    pair_count = 0
    for a in range(device_count):
        for b in range(a + 1, device_count):
            pair_count += 1
            for i in range(element_count):
                diff = abs(outputs[a][i] - outputs[b][i])
                bound = allowed_abs_error[i] + allowed_abs_error[i]
                if diff > bound:
                    return _reject(
                        case_id, run,
                        f"cross-device disagreement at element {i} between "
                        f"device {run.devices[a]} and device {run.devices[b]}: "
                        f"{diff!r} > {bound!r}"
                    )
                cross_max_abs = max(cross_max_abs, diff)
                cross_sq_err_sum += diff * diff

    cross_nmse = (cross_sq_err_sum / (pair_count * element_count)) if pair_count > 0 else 0.0
    output_digests = tuple(sha256_hex(data) for data in run.outputs)
    digest_equal = len(set(output_digests)) == 1

    return CaseResult(
        case_id=case_id, provider=run.provider, valid=True, correct=True, reason=None,
        nmse_vs_reference=nmse, max_abs_vs_reference=max_abs_err, rmse_vs_reference=rmse,
        cross_device_nmse=cross_nmse, cross_device_max_abs=cross_max_abs,
        cross_device_digest_equal=digest_equal,
        output_digests=output_digests,
        requested_provider=run.requested_provider, effective_provider=run.effective_provider,
        provider_succeeded=run.provider_succeeded, handoff=run.handoff,
        fallback_depth=run.fallback_depth, completion_synchronized=run.completion_synchronized,
    )


def evaluate_case(
    manifest: dict, device_values: list[list[float]], runs: dict[str, ProviderRun],
) -> list[CaseResult]:
    """Evaluate every provider arm for one case against a single CPU-double
    reference. GP05: the analytical error bound is NOT shared across arms
    -- each provider run gets its own bound from evaluate_provider_run,
    widened for BF16-wire-compression only when that specific run actually
    went through it (rccl, at/above the real element-count threshold).
    sum_abs (the input to every arm's bound) is computed once here since it
    depends only on the frozen input values, not on which provider ran."""
    element_count = manifest["element_count"]
    if any(len(v) != element_count for v in device_values):
        raise CorrectnessError(
            f"case {manifest['case_id']}: device value arrays disagree with manifest element_count"
        )
    reference, sum_abs = cpu_reference(device_values)
    return [evaluate_provider_run(manifest, device_values, reference, sum_abs, run) for run in runs.values()]


# ---------------------------------------------------------------------------
# Aggregation and JSONL evidence
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AggregateResult:
    reduction_signature_key: str
    provider: str
    case_count: int
    all_valid: bool
    all_correct: bool
    worst_nmse: float | None
    worst_max_abs: float | None
    worst_cross_device_max_abs: float | None
    failing_case_ids: tuple[str, ...]


def aggregate_case_results(
    reduction_signature_key: str, provider: str, results: list[CaseResult],
) -> AggregateResult:
    """Worst-of-corpus reduction, matching HI67's 'use the worst result, not
    an average' philosophy: a single failing case fails the whole
    signature/provider aggregate, and the recorded diagnostics are the
    worst observed across every case, not a mean that could hide an
    outlier."""
    if not results:
        raise CorrectnessError(
            f"no case results for signature={reduction_signature_key!r} provider={provider!r}"
        )
    failing = tuple(r.case_id for r in results if not r.valid or not r.correct)
    valid_results = [r for r in results if r.valid and r.correct]
    return AggregateResult(
        reduction_signature_key=reduction_signature_key, provider=provider,
        case_count=len(results),
        all_valid=all(r.valid for r in results),
        all_correct=len(failing) == 0,
        worst_nmse=max((r.nmse_vs_reference for r in valid_results), default=None),
        worst_max_abs=max((r.max_abs_vs_reference for r in valid_results), default=None),
        worst_cross_device_max_abs=max((r.cross_device_max_abs for r in valid_results), default=None),
        failing_case_ids=failing,
    )


def case_result_to_row(
    result: CaseResult, *, source_revision: str, manifest_hash: str,
    reduction_signature_key: str, topology_key: str, peer_access: str,
    element_count: int, seed: int, contract_version: str = CONTRACT_VERSION,
) -> dict:
    return {
        "schema_version": 1,
        "contract_version": contract_version,
        "case_id": result.case_id,
        "seed": seed,
        "provider": result.provider,
        "source_revision": source_revision,
        "manifest_hash": manifest_hash,
        "element_type": "f32",
        "element_count": element_count,
        "reduction_signature_key": reduction_signature_key,
        "topology_key": topology_key,
        "peer_access": peer_access,
        "requested_provider": result.requested_provider,
        "effective_provider": result.effective_provider,
        "provider_succeeded": result.provider_succeeded,
        "handoff": result.handoff,
        "fallback_depth": result.fallback_depth,
        "completion_synchronized": result.completion_synchronized,
        "output_digests": list(result.output_digests),
        "nmse_vs_reference": result.nmse_vs_reference,
        "max_abs_vs_reference": result.max_abs_vs_reference,
        "rmse_vs_reference": result.rmse_vs_reference,
        "cross_device_nmse": result.cross_device_nmse,
        "cross_device_max_abs": result.cross_device_max_abs,
        "cross_device_digest_equal": result.cross_device_digest_equal,
        "valid": result.valid,
        "correct": result.correct,
        "reason": result.reason,
    }


def write_reduce_correctness_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")

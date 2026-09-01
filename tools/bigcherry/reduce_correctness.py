"""HI18/HI142 P2.6: correctness-evidence policy layer for the test-hip-reduce
probe (patches/1224, src/tests/test-hip-reduce.cpp).

The C++ probe is deliberately a fact-reporter only -- its own header
comment (lines 28-31) states: "This is a Python-facing FACT reporter, not
a correctness verdict: see tools/bigcherry/reduce_correctness.py, which
owns every pass/fail decision." This module is that owner. Verified
directly against the real, current probe source (not inherited from any
summary) before writing this:

- case.json requires: device_count, element_count, slice_shape (4 entries,
  product == element_count), input_digests (one sha256 hex per rank,
  matched against rank-N.f32 file contents), reduction_signature_key,
  topology_key, peer_access, case_id.
- rank-N.f32: raw little-endian F32 bytes, element_count*4 bytes each.
- reduction_signature_key format (test-hip-reduce.cpp's own
  make_reduction_signature_key(), lines 137-145):
  "split_reduce:v1:<element_type>:<element_count>:<s0>,<s1>,<s2>,<s3>:<topology_key>"
- topology_key/peer_access format (src/ggml/src/ggml-cuda/
  hip-autotune-reduce-telemetry.cpp's observe_topology()): topology_key =
  "n<device_count>:peer<flattened DxD hipDeviceCanAccessPeer matrix,
  '1' for i==j or real peer access, '0' otherwise>"; peer_access =
  "complete" (all pairs true) | "partial" (some false) | "unknown" (a
  query failed). Confirmed on real Brutus hardware (direct
  hipDeviceCanAccessPeer check, all cross-device pairs return false): for
  {0,2} this is topology_key="n2:peer1001", peer_access="partial"; for
  {0,1,2} it is topology_key="n3:peer100010001", peer_access="partial".
- Probe --out JSON fields consumed here: probe_valid,
  reduction_signature_matches_case, requested_provider,
  effective_provider, handoff, fallback_depth, outputs (per-rank
  {device, byte_count, sha256, path}).

Independent of tuning/catalog.py, promotion, replay, and the dispatch
ABI -- this is qualification/correctness evidence, not a tuning
candidate source.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = 1

# 2^-24: F32 unit roundoff. Standard analytical bound for a summation of D
# terms with round-to-nearest F32 arithmetic (Higham's gamma_n bound).
_F32_UNIT_ROUNDOFF = 2.0**-24
_FLT_MIN = 1.1754943508222875e-38


def make_reduction_signature_key(
    element_type: str, element_count: int, slice_shape: tuple[int, int, int, int],
    topology_key: str,
) -> str:
    """Exact format from test-hip-reduce.cpp's make_reduction_signature_key()
    (lines 137-145) -- must match byte-for-byte or the probe's own
    reduction_signature_matches_case check will (correctly) fail."""
    s0, s1, s2, s3 = slice_shape
    return f"split_reduce:v1:{element_type}:{element_count}:{s0},{s1},{s2},{s3}:{topology_key}"


def compute_topology_key(device_count: int, peer_access_matrix: list[list[bool]]) -> tuple[str, str]:
    """Exact format from observe_topology() in hip-autotune-reduce-telemetry.cpp.

    peer_access_matrix[i][j] must be the real hipDeviceCanAccessPeer(i, j)
    result for i != j (the caller must supply real hardware evidence, not
    an assumption -- this function does not silently assume any topology).
    Diagonal (i == j) is always treated as accessible, matching the real
    C++ derivation's `i == j ? 1 : ...`.
    """
    if len(peer_access_matrix) != device_count or any(
        len(row) != device_count for row in peer_access_matrix
    ):
        raise ValueError(
            f"peer_access_matrix must be {device_count}x{device_count}, "
            f"got {len(peer_access_matrix)} rows"
        )

    bits: list[str] = []
    complete = True
    for i in range(device_count):
        for j in range(device_count):
            can = True if i == j else bool(peer_access_matrix[i][j])
            bits.append("1" if can else "0")
            if not can:
                complete = False
    key = f"n{device_count}:peer{''.join(bits)}"
    peer_access = "complete" if complete else "partial"
    return key, peer_access


def f32_error_bound(participant_count: int, sum_abs: float) -> float:
    """Analytical F32 summation error bound: gamma_{D-1} * sum(|x_i|) + D*FLT_MIN.

    gamma_n = n*u / (1 - n*u) for n round-to-nearest F32 additions (a
    D-term sum has D-1 additions). The D*FLT_MIN term accounts for
    denormal/underflow slack, matching HI67/HI18's existing convention."""
    n = participant_count - 1
    gamma = (n * _F32_UNIT_ROUNDOFF) / (1.0 - n * _F32_UNIT_ROUNDOFF)
    return gamma * sum_abs + participant_count * _FLT_MIN


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pack_f32(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


class MalformedOutputError(Exception):
    """Raised by _unpack_f32 on a byte length that isn't a whole number of
    F32 elements -- caught by evaluate_probe_result and converted to a
    clean EvaluationResult(False, ...) rather than an uncaught exception
    (gpt-flagged real gap: a fail-closed verdict owner must not let a
    malformed file escape as a raised exception)."""


def _unpack_f32(data: bytes) -> list[float]:
    if len(data) % 4 != 0:
        raise MalformedOutputError(f"byte length {len(data)} is not a multiple of 4 (not valid F32 data)")
    n = len(data) // 4
    return list(struct.unpack(f"<{n}f", data))


def write_case(
    case_dir: Path, *, case_id: str, rank_values: list[list[float]],
    slice_shape: tuple[int, int, int, int], topology_key: str, peer_access: str,
    element_type: str = "f32",
) -> None:
    """Generate case.json + rank-N.f32 for the real test-hip-reduce probe.

    rank_values: one list of floats per participating rank, all the same
    length == element_count == product(slice_shape). These are frozen to
    real F32 bytes on disk -- the written files ARE the evidence subject
    (never regenerated per evaluation), matching HI67/HI18's existing
    corpus-freezing discipline.
    """
    device_count = len(rank_values)
    if not (2 <= device_count <= 4):
        raise ValueError(f"test-hip-reduce supports 2-4 devices, got {device_count}")
    if element_type != "f32":
        raise ValueError(f"only f32 is supported (probe hard-requires it), got {element_type!r}")
    if not rank_values[0]:
        raise ValueError("rank_values must be non-empty (element_count must be >= 1)")

    element_count = len(rank_values[0])
    for r, values in enumerate(rank_values):
        if len(values) != element_count:
            raise ValueError(f"rank {r} has {len(values)} elements, expected {element_count}")

    # Real gap found during HI18's real-production-signature corpus slice
    # (2026-09-01, GPT review): slice_shape's tuple[int,int,int,int] type
    # hint is not runtime-enforced -- only the product check below ran, so
    # a wrong-length or non-positive slice_shape with a coincidentally
    # matching product would silently write a malformed case.json (the
    # probe's own reduction_signature_key format is hard-coded to exactly
    # 4 dimensions -- see make_reduction_signature_key()).
    if len(slice_shape) != 4:
        raise ValueError(f"slice_shape must have exactly 4 entries, got {len(slice_shape)}: {slice_shape}")
    if any(dim <= 0 for dim in slice_shape):
        raise ValueError(f"slice_shape entries must all be positive, got {slice_shape}")

    shape_product = slice_shape[0] * slice_shape[1] * slice_shape[2] * slice_shape[3]
    if shape_product != element_count:
        raise ValueError(
            f"slice_shape product {shape_product} does not match element_count {element_count}"
        )

    case_dir.mkdir(parents=True, exist_ok=True)

    input_digests: list[str] = []
    for r, values in enumerate(rank_values):
        raw = _pack_f32(values)
        digest = _sha256_hex(raw)
        (case_dir / f"rank-{r}.f32").write_bytes(raw)
        input_digests.append(digest)

    reduction_signature_key = make_reduction_signature_key(
        element_type, element_count, slice_shape, topology_key,
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "device_count": device_count,
        "element_count": element_count,
        "slice_shape": list(slice_shape),
        "input_digests": input_digests,
        "reduction_signature_key": reduction_signature_key,
        "topology_key": topology_key,
        "peer_access": peer_access,
    }
    (case_dir / "case.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def read_rank_values(case_dir: Path, device_count: int) -> list[list[float]]:
    """Read back the frozen rank-N.f32 files a case_dir was generated with."""
    return [_unpack_f32((case_dir / f"rank-{r}.f32").read_bytes()) for r in range(device_count)]


def independent_expected_sum(rank_values: list[list[float]]) -> list[float]:
    """CPU-double reference sum, independent of any GPU implementation.
    Python floats are IEEE-754 doubles, so this already computes in F64 --
    no numpy/GPU dependency needed for the oracle. Uses math.fsum() (an
    exact/robust summation, not naive sequential sum()) so the reference
    itself never accumulates rounding error across many terms -- gpt-
    flagged real gap: ordinary sum() is not an exact oracle even in F64."""
    device_count = len(rank_values)
    element_count = len(rank_values[0])
    return [
        math.fsum(rank_values[r][i] for r in range(device_count))
        for i in range(element_count)
    ]


def sum_abs_per_element(rank_values: list[list[float]]) -> list[float]:
    device_count = len(rank_values)
    element_count = len(rank_values[0])
    return [
        sum(abs(rank_values[r][i]) for r in range(device_count))
        for i in range(element_count)
    ]


@dataclass(frozen=True)
class EvaluationResult:
    valid: bool
    reason: str
    max_abs_error: float = 0.0
    worst_element_index: int = -1


_PROVENANCE_GATES = {
    "rccl": lambda r: (
        r.get("requested_provider") == "rccl"
        and r.get("effective_provider") == "rccl"
        and r.get("handoff") == "none"
        and r.get("fallback_depth") == 0
    ),
    # Strengthened per gpt review: requested_provider=="meta" alone was too
    # weak -- an explicit-meta request that actually fell through some
    # other handoff/fallback path would still have passed. Require the
    # same full-provenance shape as rccl, just for the meta provider.
    "meta": lambda r: (
        r.get("requested_provider") == "meta"
        and r.get("effective_provider") == "meta"
        and r.get("handoff") == "none"
        and r.get("fallback_depth") == 0
    ),
    # auto records whatever production actually chose, but must still
    # prove the CALL was actually requested as auto -- otherwise evidence
    # from a differently-requested plan could silently pass an auto
    # evaluation (gpt-flagged real gap).
    "auto": lambda r: r.get("requested_provider") == "auto",
}


def evaluate_probe_result(
    result_json: dict, *, case_dir: Path, out_dir: Path, out_stem: str, plan: str,
) -> EvaluationResult:
    """The single owner of pass/fail for one test-hip-reduce invocation.

    Reads the probe's own --out JSON (already loaded by the caller as
    result_json) plus the case's frozen rank-N.f32 inputs and the probe's
    written <out_stem>-rank-N.f32 per-device outputs, and renders a
    verdict. Never trusts the probe's own claims about correctness --
    the probe only reports structural facts (signature match, provenance,
    synchronization); this function independently recomputes the expected
    result and compares.
    """
    if plan not in _PROVENANCE_GATES:
        raise ValueError(f"unknown plan: {plan!r}")

    if not result_json.get("probe_valid"):
        return EvaluationResult(False, "probe_valid is false (structural failure -- see probe's own fields)")

    if not result_json.get("reduction_signature_matches_case"):
        return EvaluationResult(
            False,
            "reduction_signature_matches_case is false -- observed signature "
            f"{result_json.get('reduction_signature')} vs case's declared "
            f"{result_json.get('expected_reduction_signature_key')}",
        )

    if not _PROVENANCE_GATES[plan](result_json):
        return EvaluationResult(
            False,
            f"provenance gate for plan={plan!r} failed: requested="
            f"{result_json.get('requested_provider')} effective="
            f"{result_json.get('effective_provider')} handoff="
            f"{result_json.get('handoff')} fallback_depth="
            f"{result_json.get('fallback_depth')}",
        )

    # Independently load and cross-validate against the real case.json --
    # gpt-flagged real gap: previously every check below trusted
    # result_json's own self-reported device_count/signature fields
    # rather than the case manifest itself, so edited result metadata (or
    # a case mutated after the probe ran) could not be caught.
    try:
        case_manifest = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return EvaluationResult(False, f"cannot read/parse case.json in {case_dir}: {exc}")

    manifest_device_count = case_manifest.get("device_count")
    if result_json.get("device_count") != manifest_device_count:
        return EvaluationResult(
            False,
            f"result device_count={result_json.get('device_count')} does not match "
            f"case.json's device_count={manifest_device_count}",
        )
    manifest_key = case_manifest.get("reduction_signature_key")
    if result_json.get("expected_reduction_signature_key") != manifest_key:
        return EvaluationResult(
            False,
            "result's expected_reduction_signature_key does not match case.json's "
            f"own reduction_signature_key: {result_json.get('expected_reduction_signature_key')!r} "
            f"vs {manifest_key!r}",
        )

    device_count = manifest_device_count
    rank_values = read_rank_values(case_dir, device_count)

    # Reject non-finite inputs before computing anything -- a NaN/Inf
    # anywhere makes every downstream comparison meaningless (a NaN
    # "exceeds bound" check is always False, so a corrupted input could
    # otherwise silently pass). gpt-flagged real gap.
    for r, values in enumerate(rank_values):
        for i, v in enumerate(values):
            if not math.isfinite(v):
                return EvaluationResult(False, f"case input rank {r} element {i} is non-finite: {v!r}")

    expected = independent_expected_sum(rank_values)
    sum_abs = sum_abs_per_element(rank_values)
    for i, v in enumerate(expected):
        if not math.isfinite(v):
            return EvaluationResult(False, f"independent expected sum at element {i} is non-finite: {v!r}")

    outputs = result_json.get("outputs", [])
    if len(outputs) != device_count:
        return EvaluationResult(False, f"expected {device_count} output entries, got {len(outputs)}")

    # Exact participant coverage: ranks 0..D-1, each present exactly once
    # -- not merely "device_count entries" (gpt-flagged: a duplicate rank
    # with a missing one would still pass a bare length check).
    seen_devices = sorted(entry["device"] for entry in outputs)
    if seen_devices != list(range(device_count)):
        return EvaluationResult(
            False, f"expected output devices exactly {list(range(device_count))}, got {seen_devices}",
        )

    # Output identity must be bound to a real, unique per-rank file --
    # gpt-flagged real gap: without this, multiple output entries could
    # reference the SAME path (not actually independent per-rank
    # evidence) and out_dir/out_stem were otherwise unused/unchecked.
    seen_paths: set[str] = set()
    for entry in outputs:
        expected_prefix = str(out_dir / f"{out_stem}-rank-{entry['device']}")
        if not str(entry["path"]).startswith(expected_prefix):
            return EvaluationResult(
                False,
                f"device {entry['device']} output path {entry['path']!r} does not match the "
                f"expected per-rank naming convention {expected_prefix!r}",
            )
        if entry["path"] in seen_paths:
            return EvaluationResult(False, f"duplicate output path across ranks: {entry['path']!r}")
        seen_paths.add(entry["path"])

    element_count = len(expected)
    per_rank_outputs: dict[int, list[float]] = {}
    for entry in outputs:
        rank_path = Path(entry["path"])
        raw = rank_path.read_bytes()
        digest = _sha256_hex(raw)
        if digest != entry["sha256"]:
            return EvaluationResult(
                False, f"output file {rank_path} digest mismatch (tampered or corrupted since probe wrote it)",
            )
        try:
            values = _unpack_f32(raw)
        except MalformedOutputError as exc:
            return EvaluationResult(False, f"device {entry['device']} output file malformed: {exc}")
        if len(values) != element_count:
            return EvaluationResult(
                False,
                f"device {entry['device']} output has {len(values)} elements, expected {element_count}",
            )
        for i, v in enumerate(values):
            if not math.isfinite(v):
                return EvaluationResult(
                    False, f"device {entry['device']} output element {i} is non-finite: {v!r}",
                )
        per_rank_outputs[entry["device"]] = values

    worst_abs_error = 0.0
    worst_index = -1
    bounds = [f32_error_bound(device_count, sabs) for sabs in sum_abs]
    for device, output in per_rank_outputs.items():
        for i, (actual, exp, bound) in enumerate(zip(output, expected, bounds)):
            err = abs(actual - exp)
            if err > worst_abs_error:
                worst_abs_error = err
                worst_index = i
            if err > bound:
                return EvaluationResult(
                    False,
                    f"device {device} element {i}: |actual-expected|={err:.6e} exceeds "
                    f"analytical F32 bound {bound:.6e} (expected={exp!r}, actual={actual!r})",
                    max_abs_error=worst_abs_error,
                    worst_element_index=worst_index,
                )

    # Real pairwise cross-device agreement (not merely a length check).
    # Per-rank bounds each give |r_i - ref| <= B, so by the triangle
    # inequality |r_i - r_j| <= 2B follows -- NOT <= B. gpt correctly
    # flagged the previous comment's "redundant" claim as too strong; this
    # check uses the honestly-derived 2B bound rather than asserting
    # redundancy that doesn't hold.
    devices = sorted(per_rank_outputs)
    reference_device = devices[0]
    reference = per_rank_outputs[reference_device]
    for device in devices[1:]:
        output = per_rank_outputs[device]
        for i, (a, b, bound) in enumerate(zip(reference, output, bounds)):
            pair_err = abs(a - b)
            if pair_err > 2.0 * bound:
                return EvaluationResult(
                    False,
                    f"cross-device disagreement: device {reference_device} vs {device} at element {i}: "
                    f"|{a!r}-{b!r}|={pair_err:.6e} exceeds 2x analytical bound {2.0 * bound:.6e}",
                )

    return EvaluationResult(True, "clean: signature matched, provenance gate passed, all ranks within analytical F32 bound", max_abs_error=worst_abs_error, worst_element_index=worst_index)

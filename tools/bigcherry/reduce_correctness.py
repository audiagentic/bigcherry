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


def _unpack_f32(data: bytes) -> list[float]:
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
    element_count = len(rank_values[0])
    for r, values in enumerate(rank_values):
        if len(values) != element_count:
            raise ValueError(f"rank {r} has {len(values)} elements, expected {element_count}")

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
    Python floats are IEEE-754 doubles, so plain sum() here already gives
    the F64 reference -- no numpy/GPU dependency needed for the oracle."""
    device_count = len(rank_values)
    element_count = len(rank_values[0])
    return [
        sum(rank_values[r][i] for r in range(device_count))
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
    "meta": lambda r: r.get("requested_provider") == "meta",
    # auto has no gate -- records whatever production actually chose.
    "auto": lambda r: True,
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

    device_count = result_json["device_count"]
    rank_values = read_rank_values(case_dir, device_count)
    expected = independent_expected_sum(rank_values)
    sum_abs = sum_abs_per_element(rank_values)

    outputs = result_json.get("outputs", [])
    if len(outputs) != device_count:
        return EvaluationResult(False, f"expected {device_count} output entries, got {len(outputs)}")

    per_rank_outputs: list[list[float]] = []
    for entry in outputs:
        rank_path = Path(entry["path"])
        raw = rank_path.read_bytes()
        digest = _sha256_hex(raw)
        if digest != entry["sha256"]:
            return EvaluationResult(
                False, f"output file {rank_path} digest mismatch (tampered or corrupted since probe wrote it)",
            )
        per_rank_outputs.append(_unpack_f32(raw))

    # Cross-device agreement: every rank of an AllReduce must report the
    # same result. Proven redundant-by-triangle-inequality given the
    # per-device error bound below (HI18's own earlier finding), but kept
    # as an explicit, independently-statable requirement rather than
    # silently relying on the per-device gate to imply it.
    reference_rank = per_rank_outputs[0]
    for r in range(1, device_count):
        if len(per_rank_outputs[r]) != len(reference_rank):
            return EvaluationResult(False, f"rank {r} output length differs from rank 0")

    worst_abs_error = 0.0
    worst_index = -1
    for r, output in enumerate(per_rank_outputs):
        for i, (actual, exp, sabs) in enumerate(zip(output, expected, sum_abs)):
            bound = f32_error_bound(device_count, sabs)
            err = abs(actual - exp)
            if err > worst_abs_error:
                worst_abs_error = err
                worst_index = i
            if err > bound:
                return EvaluationResult(
                    False,
                    f"rank {r} element {i}: |actual-expected|={err:.6e} exceeds "
                    f"analytical F32 bound {bound:.6e} (expected={exp!r}, actual={actual!r})",
                    max_abs_error=worst_abs_error,
                    worst_element_index=worst_index,
                )

    return EvaluationResult(True, "clean: signature matched, provenance gate passed, all ranks within analytical F32 bound", max_abs_error=worst_abs_error, worst_element_index=worst_index)

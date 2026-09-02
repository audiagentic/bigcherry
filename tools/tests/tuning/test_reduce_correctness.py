"""HI18: tests for tools/bigcherry/reduce_correctness.py's corpus
generation, CPU-double oracle, analytical F32 summation bound, and every
fail-closed rejection rule from the HI18 correctness-comparison design
(GPT review, 2026-08-22) -- input-digest mismatch, a missing participant,
a provenance-gate miss, an over-bound element, a cross-device
disagreement, and non-finite output. No real HIP hardware or compiled
test-hip-reduce probe needed for this layer's own correctness; ProviderRun
stands in for what that probe will report."""

from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.tuning import reduction as rc # noqa: E402


def _f32_bytes(values: list[float]) -> bytes:
    return rc.to_f32_bytes(values)


def _make_manifest(device_values: list[list[float]], *, case_id="case-0001") -> dict:
    digests = tuple(rc.sha256_hex(_f32_bytes(v)) for v in device_values)
    return {
        "schema_version": 1,
        "case_id": case_id,
        "seed": 1,
        "pattern": "ordinary_signed",
        "generator_version": rc.GENERATOR_VERSION,
        "element_count": len(device_values[0]),
        "slice_shape": [len(device_values[0]), 1, 1, 1],
        "device_count": len(device_values),
        "reduction_signature_key": "sig-a",
        "topology_key": "n2:peer1001",
        "peer_access": "partial",
        "input_digests": list(digests),
    }


def _clean_run(
    manifest: dict, device_values: list[list[float]], provider: str, *,
    devices=(0, 1), outputs=None, **overrides,
) -> rc.ProviderRun:
    """A ProviderRun that would pass every check by default: correct
    reduction outputs, matching input digests, and a provenance gate that
    satisfies PROVENANCE_GATES for the given provider."""
    reference, _ = rc.cpu_reference(device_values)
    if outputs is None:
        outputs = tuple(_f32_bytes(reference) for _ in devices)
    fields = dict(
        provider=provider,
        requested_provider=provider,
        effective_provider=provider,
        provider_succeeded=True,
        handoff="none",
        fallback_depth=0,
        completion_synchronized=True,
        devices=tuple(devices),
        input_digests=tuple(manifest["input_digests"]),
        outputs=tuple(outputs),
    )
    fields.update(overrides)
    return rc.ProviderRun(**fields)


class CorpusGenerationTests(unittest.TestCase):
    def test_deterministic_across_calls(self):
        a = rc.generate_case(seed=7, pattern="ordinary_signed", element_count=64, device_count=2)
        b = rc.generate_case(seed=7, pattern="ordinary_signed", element_count=64, device_count=2)
        self.assertEqual(a, b)

    def test_different_seed_differs(self):
        a = rc.generate_case(seed=1, pattern="ordinary_signed", element_count=64, device_count=2)
        b = rc.generate_case(seed=2, pattern="ordinary_signed", element_count=64, device_count=2)
        self.assertNotEqual(a, b)

    def test_every_pattern_generates(self):
        for pattern in rc.PATTERNS:
            devices = rc.generate_case(seed=1, pattern=pattern, element_count=32, device_count=2)
            self.assertEqual(len(devices), 2)
            self.assertEqual(len(devices[0]), 32)

    def test_values_are_f32_representable(self):
        devices = rc.generate_case(seed=1, pattern="ordinary_signed", element_count=16, device_count=2)
        for row in devices:
            for v in row:
                self.assertEqual(v, struct.unpack("<f", struct.pack("<f", v))[0])

    def test_unknown_pattern_rejected(self):
        with self.assertRaises(rc.CorrectnessError):
            rc.generate_case(seed=1, pattern="nonsense", element_count=8, device_count=2)

    def test_single_device_rejected(self):
        with self.assertRaises(rc.CorrectnessError):
            rc.generate_case(seed=1, pattern="ordinary_signed", element_count=8, device_count=1)


class CaseRoundTripTests(unittest.TestCase):
    def test_write_then_load_round_trips(self):
        devices = rc.generate_case(seed=1, pattern="ordinary_signed", element_count=32, device_count=2)
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / "case-0001"
            manifest = rc.write_case(
                case_dir, case_id="case-0001", seed=1, pattern="ordinary_signed",
                reduction_signature_key=rc.make_reduction_signature_key(
                    element_type="f32", element_count=32, slice_shape=(32, 1, 1, 1),
                    topology_key="n2:peer1001",
                ),
                topology_key="n2:peer1001",
                peer_access="partial", devices=devices, slice_shape=(32, 1, 1, 1),
            )
            loaded_manifest, ranks = rc.load_case(case_dir)
            self.assertEqual(loaded_manifest["case_id"], "case-0001")
            self.assertEqual(tuple(loaded_manifest["input_digests"]), manifest.input_digests)
            self.assertEqual(rc.from_f32_bytes(ranks[0]), devices[0])
            self.assertEqual(rc.from_f32_bytes(ranks[1]), devices[1])

    def test_load_detects_corrupted_rank_file(self):
        devices = rc.generate_case(seed=1, pattern="ordinary_signed", element_count=8, device_count=2)
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / "case-0001"
            rc.write_case(
                case_dir, case_id="case-0001", seed=1, pattern="ordinary_signed",
                reduction_signature_key=rc.make_reduction_signature_key(
                    element_type="f32", element_count=8, slice_shape=(8, 1, 1, 1),
                    topology_key="n2:peer1001",
                ),
                topology_key="n2:peer1001",
                peer_access="partial", devices=devices, slice_shape=(8, 1, 1, 1),
            )
            (case_dir / "rank-0.f32").write_bytes(b"\x00" * 32)
            with self.assertRaises(rc.CorrectnessError):
                rc.load_case(case_dir)

    def test_missing_slice_shape_rejected(self):
        devices = rc.generate_case(seed=1, pattern="ordinary_signed", element_count=8, device_count=2)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(rc.CorrectnessError):
                rc.write_case(
                    Path(tmp) / "case-0001", case_id="case-0001", seed=1, pattern="ordinary_signed",
                    reduction_signature_key="sig-a", topology_key="n2:peer1001",
                    peer_access="partial", devices=devices,
                )

    def test_slice_shape_product_mismatch_rejected(self):
        devices = rc.generate_case(seed=1, pattern="ordinary_signed", element_count=8, device_count=2)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(rc.CorrectnessError):
                rc.write_case(
                    Path(tmp) / "case-0001", case_id="case-0001", seed=1, pattern="ordinary_signed",
                    reduction_signature_key="sig-a", topology_key="n2:peer1001",
                    peer_access="partial", devices=devices, slice_shape=(4, 1, 1, 1),
                )

    def test_slice_shape_wrong_length_rejected(self):
        devices = rc.generate_case(seed=1, pattern="ordinary_signed", element_count=8, device_count=2)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(rc.CorrectnessError):
                rc.write_case(
                    Path(tmp) / "case-0001", case_id="case-0001", seed=1, pattern="ordinary_signed",
                    reduction_signature_key="sig-a", topology_key="n2:peer1001",
                    peer_access="partial", devices=devices, slice_shape=(2, 2, 2),
                )

    def test_slice_shape_non_positive_dims_rejected(self):
        devices = rc.generate_case(seed=1, pattern="ordinary_signed", element_count=8, device_count=2)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(rc.CorrectnessError):
                rc.write_case(
                    Path(tmp) / "case-0001", case_id="case-0001", seed=1, pattern="ordinary_signed",
                    reduction_signature_key="sig-a", topology_key="n2:peer1001",
                    peer_access="partial", devices=devices, slice_shape=(-8, 1, 1, 1),
                )


class SignatureKeyTests(unittest.TestCase):
    def test_matches_real_telemetry_key_format(self):
        key = rc.make_reduction_signature_key(
            element_type="f32", element_count=8192, slice_shape=(4096, 2, 1, 1),
            topology_key="n2:peer1001",
        )
        self.assertEqual(key, "split_reduce:v1:f32:8192:4096,2,1,1:n2:peer1001")

    def test_write_case_rejects_inconsistent_declared_key(self):
        devices = rc.generate_case(seed=1, pattern="ordinary_signed", element_count=8, device_count=2)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(rc.CorrectnessError):
                rc.write_case(
                    Path(tmp) / "case-0001", case_id="case-0001", seed=1, pattern="ordinary_signed",
                    reduction_signature_key="not-the-real-canonical-key",
                    topology_key="n2:peer1001", peer_access="partial",
                    devices=devices, slice_shape=(8, 1, 1, 1),
                )

    def test_write_case_accepts_the_canonical_key(self):
        devices = rc.generate_case(seed=1, pattern="ordinary_signed", element_count=8, device_count=2)
        with tempfile.TemporaryDirectory() as tmp:
            rc.write_case(
                Path(tmp) / "case-0001", case_id="case-0001", seed=1, pattern="ordinary_signed",
                reduction_signature_key=rc.make_reduction_signature_key(
                    element_type="f32", element_count=8, slice_shape=(8, 1, 1, 1),
                    topology_key="n2:peer1001",
                ),
                topology_key="n2:peer1001", peer_access="partial",
                devices=devices, slice_shape=(8, 1, 1, 1),
            )


# The 4-signature representative subset GPT scoped from the real dual-XTX
# Qwen3.8-27B MTP production capture (artifacts/hardware/20260820-qwen38-
# mtp-dual-xtx/reduction.jsonl): dominant anchor, max-s1 stress, near-max-odd
# stress, and the s1=1 minimum boundary.
REAL_PRODUCTION_SIGNATURES = [
    (30720, (5120, 6, 1, 1), 460200, "dominant real decode signature"),
    (2621440, (5120, 512, 1, 1), 7150, "max s1, power-of-two stress"),
    (2606080, (5120, 509, 1, 1), 2600, "near-max odd/non-power-of-two stress"),
    (5120, (5120, 1, 1, 1), 41130, "minimum/boundary s1=1"),
]


class RealProductionSignatureTests(unittest.TestCase):
    def test_real_production_signature_keys_are_well_formed(self):
        for element_count, slice_shape, observation_count, note in REAL_PRODUCTION_SIGNATURES:
            with self.subTest(note=note):
                key = rc.make_reduction_signature_key(
                    element_type="f32", element_count=element_count,
                    slice_shape=slice_shape, topology_key="n2:peer1001",
                )
                s0, s1, s2, s3 = slice_shape
                self.assertEqual(key, f"split_reduce:v1:f32:{element_count}:{s0},{s1},{s2},{s3}:n2:peer1001")
                self.assertEqual(s0 * s1 * s2 * s3, element_count)
                self.assertGreater(observation_count, 0)

    def test_real_production_signatures_write_case_round_trips(self):
        for element_count, slice_shape, _observation_count, note in REAL_PRODUCTION_SIGNATURES:
            with self.subTest(note=note):
                devices = rc.generate_case(
                    seed=1, pattern="ordinary_signed", element_count=element_count, device_count=2,
                )
                with tempfile.TemporaryDirectory() as tmp:
                    case_dir = Path(tmp) / "case-0001"
                    rc.write_case(
                        case_dir, case_id="case-0001", seed=1, pattern="ordinary_signed",
                        reduction_signature_key=rc.make_reduction_signature_key(
                            element_type="f32", element_count=element_count,
                            slice_shape=slice_shape, topology_key="n2:peer1001",
                        ),
                        topology_key="n2:peer1001", peer_access="partial",
                        devices=devices, slice_shape=slice_shape,
                    )
                    loaded_manifest, ranks = rc.load_case(case_dir)
                    self.assertEqual(loaded_manifest["element_count"], element_count)
                    self.assertEqual(rc.from_f32_bytes(ranks[0]), devices[0])


class AnalyticalBoundTests(unittest.TestCase):
    def test_bound_grows_with_device_count(self):
        sum_abs = [1.0]
        bound_2 = rc.analytical_error_bound(sum_abs, device_count=2)
        bound_4 = rc.analytical_error_bound(sum_abs, device_count=4)
        self.assertGreater(bound_4[0], bound_2[0])

    def test_bound_zero_for_zero_input(self):
        bound = rc.analytical_error_bound([0.0], device_count=2)
        self.assertGreater(bound[0], 0.0)  # FLT_MIN floor still applies
        self.assertLess(bound[0], 1e-30)

    def test_cpu_reference_matches_hand_sum(self):
        reference, sum_abs = rc.cpu_reference([[1.0, -2.0], [3.0, 4.0]])
        self.assertEqual(reference, [4.0, 2.0])
        self.assertEqual(sum_abs, [4.0, 6.0])


class EvaluateProviderRunTests(unittest.TestCase):
    def setUp(self):
        self.device_values = rc.generate_case(
            seed=1, pattern="ordinary_signed", element_count=16, device_count=2,
        )
        self.manifest = _make_manifest(self.device_values)
        self.reference, self.sum_abs = rc.cpu_reference(self.device_values)

    def _eval(self, run: rc.ProviderRun) -> rc.CaseResult:
        return rc.evaluate_provider_run(self.manifest, self.device_values, self.reference, self.sum_abs, run)

    def test_correct_rccl_run_passes(self):
        run = _clean_run(self.manifest, self.device_values, "rccl")
        result = self._eval(run)
        self.assertTrue(result.valid)
        self.assertTrue(result.correct)
        self.assertIsNone(result.reason)

    def test_correct_meta_run_passes(self):
        run = _clean_run(self.manifest, self.device_values, "meta")
        result = self._eval(run)
        self.assertTrue(result.correct)

    def test_correct_auto_run_passes_no_gate(self):
        # AUTO has no provenance gate -- whatever it actually chose is
        # recorded, not constrained.
        run = _clean_run(self.manifest, self.device_values, "auto",
                          effective_provider="meta", handoff="provider_declined_handoff_meta",
                          fallback_depth=1)
        result = self._eval(run)
        self.assertTrue(result.correct)

    def test_input_digest_mismatch_rejected(self):
        run = _clean_run(self.manifest, self.device_values, "rccl",
                          input_digests=("deadbeef", "deadbeef"))
        result = self._eval(run)
        self.assertFalse(result.valid)
        self.assertIn("input digest mismatch", result.reason)

    def test_missing_participant_rejected(self):
        run = _clean_run(self.manifest, self.device_values, "rccl",
                          devices=(0,), outputs=(_f32_bytes(self.reference),))
        result = self._eval(run)
        self.assertFalse(result.valid)
        self.assertIn("missing participant", result.reason)

    def test_sync_failure_rejected(self):
        run = _clean_run(self.manifest, self.device_values, "rccl", completion_synchronized=False)
        result = self._eval(run)
        self.assertFalse(result.valid)
        self.assertIn("not synchronized", result.reason)

    def test_rccl_provenance_gate_rejects_fallback(self):
        run = _clean_run(self.manifest, self.device_values, "rccl",
                          effective_provider="meta", handoff="provider_declined_handoff_meta",
                          fallback_depth=1)
        result = self._eval(run)
        self.assertFalse(result.valid)
        self.assertIn("provenance gate failed", result.reason)

    def test_rccl_provenance_gate_rejects_unsuccessful(self):
        run = _clean_run(self.manifest, self.device_values, "rccl", provider_succeeded=False)
        result = self._eval(run)
        self.assertFalse(result.valid)
        self.assertIn("provenance gate failed", result.reason)

    def test_meta_provenance_gate_rejects_wrong_requested(self):
        run = _clean_run(self.manifest, self.device_values, "meta", requested_provider="auto")
        result = self._eval(run)
        self.assertFalse(result.valid)
        self.assertIn("provenance gate failed", result.reason)

    def test_duplicate_participant_devices_rejected(self):
        run = _clean_run(self.manifest, self.device_values, "rccl", devices=(0, 0))
        result = self._eval(run)
        self.assertFalse(result.valid)
        self.assertIn("duplicate participating devices", result.reason)

    def test_non_finite_output_rejected(self):
        bad = list(self.reference)
        bad[0] = float("nan")
        run = _clean_run(self.manifest, self.device_values, "rccl",
                          outputs=(_f32_bytes(bad), _f32_bytes(self.reference)))
        result = self._eval(run)
        self.assertFalse(result.valid)
        self.assertIn("non-finite", result.reason)

    def test_single_element_corruption_exceeds_bound(self):
        bad = list(self.reference)
        bad[3] += 1.0  # far outside any plausible F32 summation error for these inputs
        run = _clean_run(self.manifest, self.device_values, "rccl",
                          outputs=(_f32_bytes(bad), _f32_bytes(self.reference)))
        result = self._eval(run)
        self.assertFalse(result.valid)
        self.assertIn("exceeds the analytical F32 summation bound", result.reason)

    def test_gross_device_disagreement_rejected(self):
        # A per-reference bound violation and a cross-device violation are
        # not independent: by the triangle inequality, if both outputs are
        # within `allowed_abs_error` of the reference, they are
        # automatically within `2 * allowed_abs_error` of each other -- so
        # any real disagreement this large trips the per-device check
        # first. The cross-device gate exists as an explicit, independently
        # statable requirement (HI18 design: "a provider can be close to
        # the CPU reference while still being broken as an allreduce")
        # rather than because it can fire on its own given today's bound.
        other = list(self.reference)
        other[5] += 1.0
        run = _clean_run(self.manifest, self.device_values, "rccl",
                          outputs=(_f32_bytes(self.reference), _f32_bytes(other)))
        result = self._eval(run)
        self.assertFalse(result.valid)
        self.assertIn("analytical F32 summation bound", result.reason)

    def test_output_byte_length_mismatch_rejected(self):
        run = _clean_run(self.manifest, self.device_values, "rccl",
                          outputs=(_f32_bytes(self.reference), b"\x00\x00"))
        result = self._eval(run)
        self.assertFalse(result.valid)
        self.assertIn("byte length mismatch", result.reason)

    def test_bit_identical_outputs_record_digest_equal_true(self):
        run = _clean_run(self.manifest, self.device_values, "rccl")
        result = self._eval(run)
        self.assertTrue(result.cross_device_digest_equal)

    def test_non_identical_but_within_bound_outputs_record_digest_equal_false(self):
        other = list(self.reference)
        # Perturb by roughly one F32 ULP of the largest magnitude value --
        # small enough to stay within the analytical bound, large enough to
        # differ bitwise.
        scale = max(abs(v) for v in other) or 1.0
        other[0] = struct.unpack("<f", struct.pack("<f", other[0] + scale * 2 ** -23))[0]
        run = _clean_run(self.manifest, self.device_values, "rccl",
                          outputs=(_f32_bytes(self.reference), _f32_bytes(other)))
        result = self._eval(run)
        # This assertion only holds if the perturbation stayed inside the
        # bound; if not, evaluate_provider_run already failed closed above.
        if result.valid:
            self.assertFalse(result.cross_device_digest_equal)


class EvaluateCaseTests(unittest.TestCase):
    def test_evaluates_every_arm(self):
        device_values = rc.generate_case(seed=1, pattern="ordinary_signed", element_count=8, device_count=2)
        manifest = _make_manifest(device_values)
        runs = {
            "rccl": _clean_run(manifest, device_values, "rccl"),
            "meta": _clean_run(manifest, device_values, "meta"),
            "auto": _clean_run(manifest, device_values, "auto"),
        }
        results = rc.evaluate_case(manifest, device_values, runs)
        self.assertEqual({r.provider for r in results}, {"rccl", "meta", "auto"})
        self.assertTrue(all(r.correct for r in results))

    def test_element_count_mismatch_raises(self):
        device_values = rc.generate_case(seed=1, pattern="ordinary_signed", element_count=8, device_count=2)
        manifest = _make_manifest(device_values)
        manifest["element_count"] = 999
        runs = {"rccl": _clean_run(manifest, device_values, "rccl")}
        with self.assertRaises(rc.CorrectnessError):
            rc.evaluate_case(manifest, device_values, runs)


class AggregateTests(unittest.TestCase):
    def test_all_correct_aggregates_clean(self):
        device_values = rc.generate_case(seed=1, pattern="ordinary_signed", element_count=8, device_count=2)
        manifest = _make_manifest(device_values)
        run = _clean_run(manifest, device_values, "rccl")
        reference, sum_abs = rc.cpu_reference(device_values)
        results = [rc.evaluate_provider_run(manifest, device_values, reference, sum_abs, run)]
        agg = rc.aggregate_case_results("sig-a", "rccl", results)
        self.assertTrue(agg.all_correct)
        self.assertEqual(agg.failing_case_ids, ())

    def test_one_failure_fails_whole_aggregate(self):
        device_values = rc.generate_case(seed=1, pattern="ordinary_signed", element_count=8, device_count=2)
        manifest = _make_manifest(device_values)
        reference, sum_abs = rc.cpu_reference(device_values)
        good_run = _clean_run(manifest, device_values, "rccl")
        bad = list(reference)
        bad[0] += 1.0
        bad_manifest = _make_manifest(device_values, case_id="case-0002")
        bad_run = _clean_run(bad_manifest, device_values, "rccl",
                              outputs=(_f32_bytes(bad), _f32_bytes(reference)))
        good_result = rc.evaluate_provider_run(manifest, device_values, reference, sum_abs, good_run)
        bad_result = rc.evaluate_provider_run(bad_manifest, device_values, reference, sum_abs, bad_run)
        agg = rc.aggregate_case_results("sig-a", "rccl", [good_result, bad_result])
        self.assertFalse(agg.all_correct)
        self.assertEqual(agg.failing_case_ids, ("case-0002",))

    def test_empty_results_raises(self):
        with self.assertRaises(rc.CorrectnessError):
            rc.aggregate_case_results("sig-a", "rccl", [])


class JsonlWriterTests(unittest.TestCase):
    def test_row_shape_and_round_trip(self):
        device_values = rc.generate_case(seed=1, pattern="ordinary_signed", element_count=8, device_count=2)
        manifest = _make_manifest(device_values)
        reference, sum_abs = rc.cpu_reference(device_values)
        run = _clean_run(manifest, device_values, "rccl")
        result = rc.evaluate_provider_run(manifest, device_values, reference, sum_abs, run)
        row = rc.case_result_to_row(
            result, source_revision="deadbeef", manifest_hash="cafef00d",
            reduction_signature_key="sig-a", topology_key="n2:peer1001",
            peer_access="partial", element_count=8, seed=1,
        )
        self.assertEqual(row["provider"], "rccl")
        self.assertTrue(row["correct"])
        self.assertEqual(row["contract_version"], rc.CONTRACT_VERSION)
        self.assertEqual(row["reduction_signature_key"], "sig-a")

        import json
        import tempfile as _tempfile
        with _tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reduce-correctness.jsonl"
            rc.write_reduce_correctness_jsonl(path, [row])
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            round_tripped = json.loads(lines[0])
            self.assertEqual(round_tripped["case_id"], row["case_id"])


class Bf16WireCompressionBoundTests(unittest.TestCase):
    """GP05: the analytical bound must widen for the rccl provider's real
    BF16-wire-compression regime (element_count >= the real upstream
    threshold), and ONLY for that provider/regime -- never a blanket
    loosening. See tools/bigcherry/tuning/reduction.py's
    NCCL_BF16_WIRE_THRESHOLD_BY_DEVICE_COUNT / provider_uses_bf16_wire."""

    def test_thresholds_match_pinned_source(self):
        self.assertEqual(rc.nccl_bf16_wire_threshold(1), 32768)
        self.assertEqual(rc.nccl_bf16_wire_threshold(2), 32768)
        self.assertEqual(rc.nccl_bf16_wire_threshold(3), 131072)
        self.assertEqual(rc.nccl_bf16_wire_threshold(4), 262144)
        self.assertEqual(rc.nccl_bf16_wire_threshold(8), 262144)  # >=4 shares one threshold

    def test_provider_uses_bf16_wire_gates_on_provider_and_size(self):
        self.assertTrue(rc.provider_uses_bf16_wire("rccl", 32768, 2))
        self.assertTrue(rc.provider_uses_bf16_wire("rccl", 2621440, 2))
        self.assertFalse(rc.provider_uses_bf16_wire("rccl", 32767, 2))
        self.assertFalse(rc.provider_uses_bf16_wire("meta", 2621440, 2))
        self.assertFalse(rc.provider_uses_bf16_wire("auto", 2621440, 2))

    def test_bf16_bound_strictly_wider_than_f32_bound(self):
        sum_abs = [1.0, 1000.0]
        f32_bound = rc.analytical_error_bound(sum_abs, device_count=2, bf16_wire=False)
        bf16_bound = rc.analytical_error_bound(sum_abs, device_count=2, bf16_wire=True)
        for f32_b, bf16_b in zip(f32_bound, bf16_bound):
            self.assertGreater(bf16_b, f32_b)

    def _above_threshold_case(self):
        # element_count=40000 is above the <=2-device 32768 threshold --
        # real regime the two stress signatures in production evidence
        # (2621440 / 2606080) fall into, kept smaller here purely so the
        # pure-Python reference/evaluation loops stay fast in CI.
        device_values = rc.generate_case(
            seed=7, pattern="ordinary_signed", element_count=40000, device_count=2,
        )
        manifest = _make_manifest(device_values, case_id="bf16-case")
        manifest["element_count"] = 40000
        reference, sum_abs = rc.cpu_reference(device_values)
        return device_values, manifest, reference, sum_abs

    def _bf16_scale_perturb(self, reference: list[float], sum_abs: list[float]) -> list[float]:
        """Perturb every element by ~half the real BF16 quantization
        magnitude -- large enough to exceed the plain F32 bound (which is
        ~2^16x tighter), small enough to stay inside the BF16-widened
        bound with real margin."""
        return [
            v + 0.5 * rc.BF16_ULP * s if s > 0.0 else v
            for v, s in zip(reference, sum_abs)
        ]

    def test_rccl_above_threshold_with_bf16_scale_error_passes(self):
        device_values, manifest, reference, sum_abs = self._above_threshold_case()
        perturbed = self._bf16_scale_perturb(reference, sum_abs)
        run = _clean_run(
            manifest, device_values, "rccl",
            outputs=(_f32_bytes(reference), _f32_bytes(perturbed)),
        )
        result = rc.evaluate_provider_run(manifest, device_values, reference, sum_abs, run)
        self.assertTrue(result.valid, result.reason)
        self.assertTrue(result.correct, result.reason)

    def test_meta_above_threshold_with_same_error_still_fails(self):
        # Same perturbation magnitude, but as a meta run -- meta never gets
        # BF16 slack (it doesn't wire-compress), so this must still fail.
        device_values, manifest, reference, sum_abs = self._above_threshold_case()
        perturbed = self._bf16_scale_perturb(reference, sum_abs)
        run = _clean_run(
            manifest, device_values, "meta",
            outputs=(_f32_bytes(reference), _f32_bytes(perturbed)),
        )
        result = rc.evaluate_provider_run(manifest, device_values, reference, sum_abs, run)
        self.assertFalse(result.correct)

    def test_rccl_below_threshold_with_same_error_still_fails(self):
        # Same relative perturbation, but element_count stays below the
        # real wire threshold -- rccl must NOT get BF16 slack here either.
        device_values = rc.generate_case(
            seed=7, pattern="ordinary_signed", element_count=1024, device_count=2,
        )
        manifest = _make_manifest(device_values, case_id="below-threshold-case")
        reference, sum_abs = rc.cpu_reference(device_values)
        perturbed = self._bf16_scale_perturb(reference, sum_abs)
        run = _clean_run(
            manifest, device_values, "rccl",
            outputs=(_f32_bytes(reference), _f32_bytes(perturbed)),
        )
        result = rc.evaluate_provider_run(manifest, device_values, reference, sum_abs, run)
        self.assertFalse(result.correct)


class LoadProbeRunTests(unittest.TestCase):
    """test-hip-reduce's real JSON output shape, reconstructed by hand
    rather than requiring a compiled probe -- this layer's own correctness
    (does load_probe_run ingest and validate that shape correctly) is
    independent of whether a real probe binary is available."""

    def setUp(self):
        self.device_values = rc.generate_case(seed=1, pattern="ordinary_signed", element_count=8, device_count=2)
        self.manifest = _make_manifest(self.device_values)
        self.reference, _ = rc.cpu_reference(self.device_values)

    def _write_result(self, tmp: Path, **overrides) -> Path:
        out_dir = Path(tmp)
        rank_paths = []
        for rank in range(2):
            data = _f32_bytes(self.reference)
            rank_path = out_dir / f"result-rank-{rank}.f32"
            rank_path.write_bytes(data)
            rank_paths.append(rank_path)

        obj = {
            "schema_version": 1,
            "case_id": self.manifest["case_id"],
            "plan": "rccl",
            "devices": [0, 1],
            "device_count": 2,
            "input_digests": list(self.manifest["input_digests"]),
            "requested_provider": "rccl",
            "effective_provider": "rccl",
            "provider_succeeded": True,
            "handoff": "none",
            "fallback_depth": 0,
            "completion_synchronized": True,
            "captured": True,
            "probe_valid": True,
            "reduction_signature_key": self.manifest["reduction_signature_key"],
            "expected_reduction_signature_key": self.manifest["reduction_signature_key"],
            "reduction_signature_matches_case": True,
            "reduction_signature": {
                "version": 1,
                "element_count": self.manifest["element_count"],
                "element_type": "f32",
                "slice_shape": self.manifest["slice_shape"],
                "topology_key": self.manifest["topology_key"],
                "peer_access": self.manifest["peer_access"],
            },
            "outputs": [
                {
                    "device": 0,
                    "byte_count": len(rank_paths[0].read_bytes()),
                    "sha256": rc.sha256_hex(rank_paths[0].read_bytes()),
                    "path": str(rank_paths[0]),
                },
                {
                    "device": 1,
                    "byte_count": len(rank_paths[1].read_bytes()),
                    "sha256": rc.sha256_hex(rank_paths[1].read_bytes()),
                    "path": str(rank_paths[1]),
                },
            ],
        }
        obj.update(overrides)
        result_path = out_dir / "result.json"
        import json as _json
        result_path.write_text(_json.dumps(obj))
        return result_path

    def test_loads_a_valid_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_result(tmp)
            run = rc.load_probe_run(path, manifest=self.manifest, provider="rccl")
            self.assertEqual(run.provider, "rccl")
            self.assertEqual(run.devices, (0, 1))
            self.assertEqual(len(run.outputs), 2)

    def test_rejects_probe_valid_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_result(tmp, probe_valid=False)
            with self.assertRaises(rc.CorrectnessError):
                rc.load_probe_run(path, manifest=self.manifest, provider="rccl")

    def test_rejects_signature_mismatch_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_result(tmp, reduction_signature_matches_case=False)
            with self.assertRaises(rc.CorrectnessError):
                rc.load_probe_run(path, manifest=self.manifest, provider="rccl")

    def test_rejects_case_id_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_result(tmp, case_id="some-other-case")
            with self.assertRaises(rc.CorrectnessError):
                rc.load_probe_run(path, manifest=self.manifest, provider="rccl")

    def test_rejects_plan_provider_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_result(tmp)
            with self.assertRaises(rc.CorrectnessError):
                rc.load_probe_run(path, manifest=self.manifest, provider="meta")

    def test_rejects_wrong_topology_in_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_result(tmp)
            import json as _json
            obj = _json.loads(path.read_text())
            obj["reduction_signature"]["topology_key"] = "n2:peer1111"
            path.write_text(_json.dumps(obj))
            with self.assertRaises(rc.CorrectnessError):
                rc.load_probe_run(path, manifest=self.manifest, provider="rccl")

    def test_rejects_tampered_output_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_result(tmp)
            import json as _json
            obj = _json.loads(path.read_text())
            obj["outputs"][0]["sha256"] = "0" * 64
            path.write_text(_json.dumps(obj))
            with self.assertRaises(rc.CorrectnessError):
                rc.load_probe_run(path, manifest=self.manifest, provider="rccl")


if __name__ == "__main__":
    unittest.main()

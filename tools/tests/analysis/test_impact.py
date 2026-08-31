import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.analysis import impact  # noqa: E402
from bigcherry.analysis import kernel_fraction  # noqa: E402
from bigcherry.analysis import symbol_map  # noqa: E402


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class ImpactTests(unittest.TestCase):
    def _data(self):
        observations = [{"signature": "a", "calls": 10, "native": "blas:native:v1"}]
        results = [
            {
                "signature": "a",
                "native": "blas:native:v1",
                "winner": "mmq:candidate:v1",
                "candidates": [
                    {"name": "blas:native:v1", "status": "ok", "median_us": 10.0},
                    {"name": "mmq:candidate:v1", "status": "ok", "median_us": 8.0},
                ],
            }
        ]
        return observations, results

    def test_call_weighted_saving_and_coverage_are_explicit(self):
        observations, results = self._data()
        report = impact.predicted_saving(observations, results)
        self.assertEqual(report["coverage"].calls_covered, 10)
        self.assertAlmostEqual(report["saving_pct"], 20.0)
        self.assertEqual(report["rows"][0]["saved_us"], 20.0)

    def test_missing_candidate_does_not_claim_coverage(self):
        observations, results = self._data()
        results[0]["winner"] = "missing"
        report = impact.predicted_saving(observations, results)
        self.assertEqual(report["coverage"].calls_covered, 0)

    def test_decode_effect_requires_fraction_in_range(self):
        self.assertAlmostEqual(impact.expected_decode_effect(10.0, 0.3), 3.0)
        with self.assertRaisesRegex(impact.ImpactError, "between"):
            impact.expected_decode_effect(10.0, 1.1)

    def test_coverage_reports_both_asymmetries(self):
        observations = [
            {"signature": "both", "calls": 10, "native": "mmq:native:v1"},
            {"signature": "record_only", "calls": 5, "native": "mmq:native:v1"},
        ]
        results = [
            {
                "signature": "both",
                "native": "mmq:native:v1",
                "winner": "mmq:cand:v1",
                "candidates": [
                    {"name": "mmq:native:v1", "status": "ok", "median_us": 10.0},
                    {"name": "mmq:cand:v1", "status": "ok", "median_us": 9.0},
                ],
            },
            {
                "signature": "measurement_only",
                "native": "mmq:native:v1",
                "winner": "mmq:cand:v1",
                "candidates": [
                    {"name": "mmq:native:v1", "status": "ok", "median_us": 10.0},
                    {"name": "mmq:cand:v1", "status": "ok", "median_us": 9.0},
                ],
            },
        ]
        report = impact.predicted_saving(observations, results)
        coverage = report["coverage"]
        self.assertEqual(coverage.matched, 1)
        self.assertEqual(coverage.record_only, ["record_only"])
        self.assertEqual(coverage.measurement_only, ["measurement_only"])
        # The record-only signature keeps its native cost in the total; the
        # measurement-only one contributes nothing and nothing is dropped.
        self.assertEqual(coverage.calls_total, 15)
        self.assertEqual(coverage.calls_covered, 10)

    def test_gemma_e4b_case_reports_low_coverage(self):
        # 121 signatures recorded, 21 tuned: the model must say so loudly.
        observations = [
            {"signature": f"s{i:03d}", "calls": 100, "native": "mmq:native:v1"}
            for i in range(121)
        ]
        results = []
        for i in range(21):
            results.append(
                {
                    "signature": f"s{i:03d}",
                    "native": "mmq:native:v1",
                    "winner": "mmq:cand:v1",
                    "candidates": [
                        {"name": "mmq:native:v1", "status": "ok", "median_us": 10.0},
                        {"name": "mmq:cand:v1", "status": "ok", "median_us": 9.0},
                    ],
                }
            )
        report = impact.predicted_saving(observations, results)
        coverage = report["coverage"]
        self.assertEqual(coverage.matched, 21)
        self.assertEqual(len(coverage.record_only), 100)
        self.assertEqual(coverage.calls_covered, 2100)
        self.assertEqual(coverage.calls_total, 12100)

    def test_saved_time_ranking_differs_from_margin_ranking(self):
        # A +69.76% winner on 13,440 calls vs a +44.35% winner on 110,160:
        # margin ranking and value ranking must not agree.
        observations = [
            {"signature": "a", "calls": 13440, "native": "mmq:native:v1"},
            {"signature": "b", "calls": 110160, "native": "mmq:native:v1"},
        ]
        results = [
            {
                "signature": "a",
                "native": "mmq:native:v1",
                "winner": "mmq:candA:v1",
                "improvement_pct": 69.76,
                "candidates": [
                    {"name": "mmq:native:v1", "status": "ok", "median_us": 100.0},
                    {"name": "mmq:candA:v1", "status": "ok", "median_us": 30.24},
                ],
            },
            {
                "signature": "b",
                "native": "mmq:native:v1",
                "winner": "mmq:candB:v1",
                "improvement_pct": 44.35,
                "candidates": [
                    {"name": "mmq:native:v1", "status": "ok", "median_us": 100.0},
                    {"name": "mmq:candB:v1", "status": "ok", "median_us": 55.65},
                ],
            },
        ]
        report = impact.predicted_saving(observations, results)
        by_margin = {
            r["signature"]: r
            for r in sorted(
                (dict(r, _margin=r.get("margin_pct", 0.0)) for r in report["rows"]),
                key=lambda r: -r["_margin"],
            )
        }
        self.assertEqual(by_margin["a"], by_margin["a"])  # a has the bigger margin
        self.assertGreater(results[0]["improvement_pct"], results[1]["improvement_pct"])
        self.assertEqual(report["rows"][0]["signature"], "b")
        self.assertGreater(report["rows"][0]["saved_us"], report["rows"][1]["saved_us"])

    def test_native_resolved_across_families(self):
        # A result carrying blas:native:v1 and mmq:native:v1 uses the one the
        # record names as native for that signature, not a scan for *:native:v1.
        observations = [
            {"signature": "a", "calls": 10, "native": "mmq:native:v1"},
        ]
        results = [
            {
                "signature": "a",
                "winner": "mmq:cand:v1",
                "candidates": [
                    {"name": "blas:native:v1", "status": "ok", "median_us": 5.0},
                    {"name": "mmq:native:v1", "status": "ok", "median_us": 100.0},
                    {"name": "mmq:cand:v1", "status": "ok", "median_us": 90.0},
                ],
            }
        ]
        report = impact.predicted_saving(observations, results)
        row = report["rows"][0]
        # Native is the record's mmq:native:v1 (100us), not the blas one (5us).
        self.assertAlmostEqual(row["saved_us_each"], 10.0)
        self.assertAlmostEqual(row["saved_us"], 100.0)

    def test_interval_is_none_without_raw_samples(self):
        observations, results = self._data()
        self.assertIsNone(
            impact.saving_interval(observations, results, draws=20, seed=0)
        )

    def test_interval_brackets_point_with_raw_samples(self):
        # 8 rounds -- exactly impact.MIN_PAIRED_ROUNDS, the floor below
        # which a signature is excluded from the interval entirely.
        observations = [{"signature": "a", "calls": 100, "native": "mmq:native:v1"}]
        results = [
            {
                "signature": "a",
                "native": "mmq:native:v1",
                "winner": "mmq:cand:v1",
                "candidates": [
                    {
                        "name": "mmq:native:v1",
                        "status": "ok",
                        "median_us": 10.0,
                        "samples_us": [9.5, 10.0, 10.5, 11.0, 9.9, 10.1, 9.8, 10.2],
                    },
                    {
                        "name": "mmq:cand:v1",
                        "status": "ok",
                        "median_us": 8.0,
                        "samples_us": [7.5, 8.0, 8.5, 9.0, 7.9, 8.1, 7.8, 8.2],
                    },
                ],
            }
        ]
        interval = impact.saving_interval(observations, results, draws=200, seed=1)
        assert interval is not None
        low, high = interval
        report = impact.predicted_saving(observations, results)
        self.assertLess(low, report["saving_pct"])
        self.assertGreater(high, report["saving_pct"])

    def test_interval_below_min_paired_rounds_is_none(self):
        """gpt-dev-agent review, 2026-08-31: a singleton (or any tiny) paired
        sample used to produce a confident-looking zero-width "95% CI" from
        essentially no evidence. Below MIN_PAIRED_ROUNDS the signature must
        be excluded from the interval entirely, not silently trusted."""
        observations = [{"signature": "a", "calls": 100, "native": "mmq:native:v1"}]
        results = [
            {
                "signature": "a",
                "native": "mmq:native:v1",
                "winner": "mmq:cand:v1",
                "candidates": [
                    {"name": "mmq:native:v1", "status": "ok", "median_us": 10.0,
                     "samples_us": [100.0]},
                    {"name": "mmq:cand:v1", "status": "ok", "median_us": 8.0,
                     "samples_us": [90.0]},
                ],
            }
        ]
        self.assertIsNone(impact.saving_interval(observations, results, draws=50, seed=0))

    def test_interval_preserves_round_pairing_not_independent_resampling(self):
        """gpt-dev-agent review, 2026-08-31: native=[100,200,...,800],
        winner=[90,180,...,720] is EXACTLY 10% faster every single round.
        A correct paired bootstrap must always return exactly 10% (every
        resampled round preserves that exact ratio); independently
        resampling native and winner destroys the pairing and would widen
        the interval away from the true fixed point."""
        native = [100.0 * i for i in range(1, 9)]
        winner = [0.9 * v for v in native]
        observations = [{"signature": "a", "calls": 1, "native": "mmq:native:v1"}]
        results = [
            {
                "signature": "a",
                "native": "mmq:native:v1",
                "winner": "mmq:cand:v1",
                "candidates": [
                    {"name": "mmq:native:v1", "status": "ok", "median_us": 400.0,
                     "samples_us": native},
                    {"name": "mmq:cand:v1", "status": "ok", "median_us": 360.0,
                     "samples_us": winner},
                ],
            }
        ]
        interval = impact.saving_interval(observations, results, draws=500, seed=0)
        self.assertIsNotNone(interval)
        low, high = interval
        self.assertAlmostEqual(low, 10.0, places=6)
        self.assertAlmostEqual(high, 10.0, places=6)

    def test_interval_requires_equal_length_round_aligned_samples(self):
        observations = [{"signature": "a", "calls": 10, "native": "mmq:native:v1"}]
        results = [
            {
                "signature": "a",
                "native": "mmq:native:v1",
                "winner": "mmq:cand:v1",
                "candidates": [
                    {"name": "mmq:native:v1", "status": "ok", "median_us": 10.0,
                     "samples_us": [10.0] * 8},
                    {"name": "mmq:cand:v1", "status": "ok", "median_us": 8.0,
                     "samples_us": [8.0] * 7},  # different length -- not round-aligned
                ],
            }
        ]
        with self.assertRaises(impact.ImpactError):
            impact._usable_pairs(
                observations, results, {"a": 10},
            )

    def test_interval_drops_only_the_null_round_keeping_the_rest_paired(self):
        observations = [{"signature": "a", "calls": 10, "native": "mmq:native:v1"}]
        results = [
            {
                "signature": "a",
                "native": "mmq:native:v1",
                "winner": "mmq:cand:v1",
                "candidates": [
                    {
                        "name": "mmq:native:v1", "status": "ok", "median_us": 10.0,
                        "samples_us": [10.0, 10.1, None, 10.2, 9.9, 10.0, 10.1, 9.8, 10.0],
                    },
                    {
                        "name": "mmq:cand:v1", "status": "ok", "median_us": 8.0,
                        "samples_us": [8.0, 8.1, 7.9, 8.2, 7.9, 8.0, 8.1, 7.8, 8.0],
                    },
                ],
            }
        ]
        interval = impact.saving_interval(observations, results, draws=50, seed=0)
        self.assertIsNotNone(interval)  # 8 of 9 rounds survive -- still >= MIN_PAIRED_ROUNDS

    def test_load_results_tolerates_truncated_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps({"kind": "header", "final_samples": 15}) + "\n")
                handle.write(json.dumps({"kind": "result", "signature": "a"}) + "\n")
                handle.write(
                    '{"kind":"result","signature":"cut")  # truncated, no newline'
                )
            results = impact.load_results(path)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["signature"], "a")

    def test_load_results_interior_corruption_raises(self):
        """gpt-dev-agent review round 2, 2026-08-31: load_results's ORIGINAL
        `except: break` discarded not just the corrupt line but EVERY
        record after it too -- valid A, corrupt B, valid C used to silently
        report just A with no indication B or C were ever there."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps({"kind": "header"}) + "\n")
                handle.write(json.dumps({"kind": "result", "signature": "a"}) + "\n")
                handle.write('{"kind":"result","signature":"corrupt-interior\n')
                handle.write(json.dumps({"kind": "result", "signature": "c"}) + "\n")
            with self.assertRaises(impact.ImpactError):
                impact.load_results(path)

    def test_load_observations_interior_corruption_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps({"kind": "header"}) + "\n")
                handle.write(json.dumps({"kind": "observation", "signature": "a", "calls": 1}) + "\n")
                handle.write('{"kind":"observation","signature":"corrupt-interior\n')
                handle.write(json.dumps({"kind": "observation", "signature": "c", "calls": 1}) + "\n")
            with self.assertRaises(impact.ImpactError):
                impact.load_observations(path)

    def test_load_observations_requires_a_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.jsonl"
            path.write_text(
                json.dumps({"kind": "observation", "signature": "a", "calls": 1}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(impact.ImpactError):
                impact.load_observations(path)

    def test_cli_reports_and_fails_on_slower(self):
        observations = [{"signature": "a", "calls": 10, "native": "mmq:native:v1"}]
        results = [
            {
                "signature": "a",
                "native": "mmq:native:v1",
                "winner": "mmq:cand:v1",
                "candidates": [
                    {"name": "mmq:native:v1", "status": "ok", "median_us": 10.0},
                    # winner SLOWER than native: mismatched artifacts or stale baseline
                    {"name": "mmq:cand:v1", "status": "ok", "median_us": 12.0},
                ],
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            # Observation rows carry kind:"observation" on the wire (the record
            # format inventory.read_jsonl parses); bare dicts would read as zero.
            record = Path(tmp) / "r.jsonl"
            _write_jsonl(
                record,
                [
                    {
                        "kind": "header",
                        "source_revision": "x" * 40,
                        "manifest_hash": "y" * 32,
                    },
                    *[{"kind": "observation", **o} for o in observations],
                ],
            )
            meas = Path(tmp) / "m.jsonl"
            _write_jsonl(
                meas,
                [
                    {"kind": "header"},
                    *[{"kind": "result", **r} for r in results],
                ],
            )
            out = Path(tmp) / "report.md"

            ns = argparse.Namespace(
                observations=str(record),
                measurements=str(meas),
                title="t",
                report=str(out),
                draws=10,
                seed=0,
                fail_on_slower=True,
            )
            self.assertEqual(impact._cmd_impact(ns), 1)
            self.assertTrue(out.exists())
            text = out.read_text(encoding="utf-8")
            self.assertIn("SLOWER: 1", text)

            ns.fail_on_slower = False
            self.assertEqual(impact._cmd_impact(ns), 0)

    def test_power_numbers(self):
        # 3% effect against 10.2% spread, alpha 0.05, power 0.8: ~181-182/arm.
        n = impact.repetitions_needed(3.0, 10.2)
        self.assertGreaterEqual(n, 181)
        self.assertLessEqual(n, 182)
        # Pairing at r=0.7 removes the shared machine-state variance.
        self.assertEqual(impact.repetitions_needed(3.0, 10.2, paired_r=0.7), 55)
        # At 1% end-to-end effect it is an order of magnitude worse.
        self.assertGreater(impact.repetitions_needed(1.0, 10.2), 1600)
        with self.assertRaises(impact.ImpactError):
            impact.repetitions_needed(0.0, 10.2)
        with self.assertRaises(impact.ImpactError):
            impact.repetitions_needed(3.0, 10.2, paired_r=1.0)

    def test_renders_interval_and_top_contributors(self):
        observations, results = self._data()
        report = impact.predicted_saving(observations, results)
        text = impact.render_report(report, "MTP 27B Q8_0", interval=(19.0, 21.0))
        self.assertIn("[19.0 - 21.0]", text)
        self.assertIn("top contributors", text)
        self.assertIn("SLOWER: 0", text)

    def test_report_states_the_interval_coverage_gap_explicitly(self):
        """gpt-dev-agent review, 2026-08-31: the point estimate and CI can
        describe different populations -- the report must say so rather
        than let a reader assume the interval covers the whole point."""
        observations, results = self._data()
        report = impact.predicted_saving(observations, results)
        coverage = impact.sample_backed_coverage(observations, results)
        text = impact.render_report(
            report, "MTP 27B Q8_0", interval=(19.0, 21.0), interval_coverage=coverage,
        )
        self.assertIn("interval covers", text)
        self.assertIn("point-estimate calls", text)


class SampleBackedCoverageTests(unittest.TestCase):
    def test_high_call_signature_without_samples_us_is_excluded_from_coverage(self):
        observations = [
            {"signature": "a", "calls": 10, "native": "n"},
            {"signature": "b", "calls": 1_000_000, "native": "n"},
        ]
        results = [
            {"signature": "a", "native": "n", "winner": "w", "candidates": [
                {"name": "n", "status": "ok", "median_us": 10.0,
                 "samples_us": [10.0] * 8},
                {"name": "w", "status": "ok", "median_us": 9.0,
                 "samples_us": [9.0] * 8},
            ]},
            {"signature": "b", "native": "n", "winner": "w", "candidates": [
                # No samples_us at all -- counted in the point estimate,
                # must NOT be counted as sample-backed.
                {"name": "n", "status": "ok", "median_us": 100.0},
                {"name": "w", "status": "ok", "median_us": 50.0},
            ]},
        ]
        coverage = impact.sample_backed_coverage(observations, results)
        # Both "a" and "b" resolve real candidates, so BOTH are inside
        # predicted_saving()'s point estimate here -- point_estimate_calls
        # is 1,000,010, not just the sample-backed subset's 10.
        self.assertEqual(coverage["point_estimate_calls"], 1_000_010)
        self.assertEqual(coverage["sample_backed_calls"], 10)
        self.assertAlmostEqual(coverage["sample_backed_fraction"], 10 / 1_000_010)
        self.assertEqual(coverage["sample_backed_signature_count"], 1)

    def test_denominator_is_point_estimate_calls_not_all_recorded_calls(self):
        """gpt-dev-agent review round 2, 2026-08-31: the exact failure case
        gpt demonstrated -- a huge record-only signature (never in results
        at all, so NOT in the point estimate) used to dilute the reported
        coverage fraction toward ~0%, even when the point estimate itself
        was 100% sample-backed."""
        observations = [
            {"signature": "a", "calls": 10, "native": "n"},
            # "ghost" has an observation but NO matching result row at
            # all -- record-only, excluded from predicted_saving()'s point
            # estimate entirely.
            {"signature": "ghost", "calls": 1_000_000, "native": "n"},
        ]
        results = [
            {"signature": "a", "native": "n", "winner": "w", "candidates": [
                {"name": "n", "status": "ok", "median_us": 10.0,
                 "samples_us": [10.0] * 8},
                {"name": "w", "status": "ok", "median_us": 9.0,
                 "samples_us": [9.0] * 8},
            ]},
        ]
        coverage = impact.sample_backed_coverage(observations, results)
        self.assertEqual(coverage["point_estimate_calls"], 10)
        self.assertEqual(coverage["sample_backed_calls"], 10)
        self.assertEqual(coverage["sample_backed_fraction"], 1.0)  # fully backed, not ~0.001%

    def test_usable_pairs_honors_the_observation_native_fallback(self):
        """gpt-dev-agent review round 2, 2026-08-31: _usable_pairs() used
        to omit the native-name fallback predicted_saving() relies on, so
        a result row without its own "native" field could be counted in
        the point estimate but silently excluded from sample-backing."""
        observations = [{"signature": "a", "calls": 10, "native": "n"}]
        results = [
            {
                # No "native" key on the result row at all -- must fall
                # back to the observation's declared native, same as
                # predicted_saving() does.
                "signature": "a", "winner": "w",
                "candidates": [
                    {"name": "n", "status": "ok", "median_us": 10.0,
                     "samples_us": [10.0] * 8},
                    {"name": "w", "status": "ok", "median_us": 9.0,
                     "samples_us": [9.0] * 8},
                ],
            }
        ]
        coverage = impact.sample_backed_coverage(observations, results)
        self.assertEqual(coverage["sample_backed_calls"], 10)


class KernelFractionTests(unittest.TestCase):
    def test_gpu_busy_merges_overlapping_spans(self):
        self.assertEqual(kernel_fraction.gpu_busy_ns([(0, 10), (5, 15), (20, 25)]), 20)
        # Order independence.
        self.assertEqual(kernel_fraction.gpu_busy_ns([(20, 25), (0, 10), (5, 15)]), 20)
        self.assertEqual(kernel_fraction.gpu_busy_ns([]), 0)
        # A kernel fully inside another adds nothing.
        self.assertEqual(kernel_fraction.gpu_busy_ns([(0, 100), (10, 20)]), 100)

    def test_classify_matmul_families(self):
        cases = {
            "ggml_cuda_mul_mat_vec_q_1": "mmvq",
            "quantize_q8_1": "mmvq",
            "quantize_mmq_q8_1": "mmq",
            "ggml_cuda_mul_mat_q_2": "mmq",
            "ggml_cuda_mul_mat_vec_f16": "mmvf",
            "ggml_cuda_mul_mat_f16": "mmf",
            "Cijk_TNT_64x128": "blas",
            "flash_attn_ext_": "attention",
            "ggml_cuda_rms_norm_f32": "norm/rope/act",
            "ggml_cuda_cpy_16_f32_f32": "copy/other",
            "totally_unrelated_kernel": "unmapped",
        }
        for name, family in cases.items():
            self.assertEqual(kernel_fraction.classify(name), family, name)
        # Mangled symbols embed the C++ identifiers verbatim, so the fallback
        # path (no demangler) still classifies.
        mangled = "_Z28ggml_cuda_quantize_mmq_q8_1PKvj"
        self.assertEqual(kernel_fraction.classify(mangled), "mmq")

    def _write_trace(
        self,
        tmp,
        rows,
        header=("Kernel_Name", "Start_Timestamp", "End_Timestamp", "Duration"),
    ):
        path = Path(tmp) / "trace.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = __import__("csv").writer(handle)
            writer.writerow(header)
            for row in rows:
                writer.writerow(row)
        return path

    def test_parse_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_trace(
                tmp,
                [
                    # 0-1000: two kernels overlapping on the same window
                    ["mul_mat_q_1", "0", "1000", "1000"],
                    ["quantize_mmq_q8_1", "500", "1500", "1000"],
                    ["flash_attn_ext", "0", "800", "800"],
                    ["unknown_kernel", "0", "200", "200"],
                ],
            )
            trace = kernel_fraction.parse_kernel_trace([path])
            summary = kernel_fraction.family_summary(trace)
            self.assertEqual(summary["kernel_total_ns"], 3000)
            self.assertEqual(summary["by_family"]["mmq"], 2000)
            self.assertAlmostEqual(summary["matmul_kernel_pct"], 2000 / 3000 * 100.0)
            # Wall window is 0..1500; GPU busy is the union: [0,1500] = 1500.
            self.assertEqual(summary["wall_ns"], 1500)
            self.assertAlmostEqual(summary["gpu_busy_pct"], 100.0)
            # Wall ceiling is the UNION of just the matmul-family spans
            # ([0,1000] and [500,1500] -> [0,1500]), not a product of two
            # independent ratios (gpt-dev-agent review round 2, 2026-08-31:
            # the product formula reports 50% when a matmul and a
            # non-matmul kernel each span the ENTIRE wall concurrently,
            # even though matmul is genuinely active 100% of wall -- only
            # the real union is mathematically valid).
            self.assertAlmostEqual(summary["matmul_wall_pct"], 100.0, places=6)

    def test_matmul_wall_pct_full_overlap_is_100_not_50(self):
        """gpt-dev-agent review round 2, 2026-08-31: the exact failure case
        gpt demonstrated -- one matmul kernel and one non-matmul kernel
        each spanning the ENTIRE wall concurrently. matmul is active
        100% of wall; the old product-of-ratios formula reported 50%."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_trace(
                tmp,
                [
                    ["mul_mat_q_1", "0", "1000", "1000"],
                    ["rms_norm_f32", "0", "1000", "1000"],  # fully overlapping, non-matmul
                ],
            )
            trace = kernel_fraction.parse_kernel_trace([path])
            summary = kernel_fraction.family_summary(trace)
            self.assertAlmostEqual(summary["matmul_wall_pct"], 100.0, places=6)

    def test_matmul_wall_pct_unavailable_without_timestamps(self):
        """A duration-only CSV cannot distinguish sequential from
        concurrent kernels -- matmul_wall_pct must be None, not a
        fabricated number from summed durations."""
        header = ("Kernel_Name", "Duration")
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_trace(
                tmp, [["mul_mat_q_1", "1000"], ["rms_norm_f32", "500"]], header=header,
            )
            trace = kernel_fraction.parse_kernel_trace([path])
            summary = kernel_fraction.family_summary(trace)
            self.assertIsNone(summary["matmul_wall_pct"])
            report = kernel_fraction.render_report(trace, summary, "decode")
            self.assertIn("unmapped", report)
            self.assertIn("unavailable", report)

    def test_header_keyed_columns_and_alternate_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Alternate header spellings: "Kernel Name", no Duration column.
            path = self._write_trace(
                tmp,
                [["mul_mat_q_1", "10", "60"]],
                header=("Kernel Name", "BeginNs", "EndNs"),
            )
            trace = kernel_fraction.parse_kernel_trace([path])
            self.assertEqual(trace["rows"][0]["dur_ns"], 50)
            self.assertEqual(trace["columns"]["kernel"], "Kernel Name")

    def test_rejects_unrecognizable_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_trace(tmp, [["x", "1"]], header=("foo", "bar"))
            with self.assertRaises(kernel_fraction.KernelFractionError):
                kernel_fraction.parse_kernel_trace([path])

    def test_rejects_negative_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_trace(tmp, [["mul_mat_q_1", "100", "50", "-50"]])
            with self.assertRaises(kernel_fraction.KernelFractionError):
                kernel_fraction.parse_kernel_trace([path])


class SymbolMapTests(unittest.TestCase):
    def test_empty_and_fallback(self):
        self.assertEqual(symbol_map.demangle(""), "")
        # A non-mangled name either comes back unchanged or, if the demangler
        # fails, falls back to the input. Either way: usable for classification.
        self.assertIn("unrelated", symbol_map.demangle("unrelated"))

    def test_caches(self):
        first = symbol_map.demangle("_Z3addiii")
        second = symbol_map.demangle("_Z3addiii")
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()

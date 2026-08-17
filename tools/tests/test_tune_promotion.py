"""Fresh-confirmation and experiment-wide promotion tests."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import tune_promotion  # noqa: E402


def result(dispatch: str, p: float, winner: float) -> dict:
    signature = dispatch
    return {
        "kind": "result",
        "dispatch": dispatch,
        "native": "native",
        "signature": signature,
        "winner": "candidate",
        "promotion_status": "pending_bh",
        "provisional_winner": "candidate",
        "schedule_seed": int.from_bytes(bytes.fromhex(signature)[:4], "little"),
        "schedule": {
            "schema_version": 1,
            "selection_algorithm": "seeded-rotation-v1",
            "confirmation_algorithm": "seeded-alternation-v1",
            "candidates": ["candidate", "native", "native#twin"],
        },
        "improvement_pct": 100.0 * (100.0 - winner) / 100.0,
        "confirmation": {
            "p_value": p,
            "effect_pct": 100.0 * (100.0 - winner) / 100.0,
            "wins": 12,
            "rounds": 12,
            "native_us": [100.0] * 12,
            "winner_us": [winner] * 12,
        },
    }


class PromotionTests(unittest.TestCase):
    HEADER = {
        "kind": "header",
        "artifact_version": 1,
        "source_revision": "a" * 40,
        "manifest_hash": "a" * 32,
    }

    def test_nonpositive_event_durations_are_not_bootstrap_evidence(self):
        low, high = tune_promotion.paired_bootstrap(
            [10.0, 10.0, 10.0],
            [9.0, -100.0, 9.0],
            seed=7,
            resamples=1000,
        )
        self.assertAlmostEqual(low, 10.0)
        self.assertAlmostEqual(high, 10.0)

    def test_bootstrap_resamples_ratio_of_medians(self):
        low, high = tune_promotion.paired_bootstrap(
            [10.0, 20.0, 30.0],
            [5.0, 10.0, 15.0],
            seed=7,
            resamples=1000,
        )
        self.assertAlmostEqual(low, 50.0)
        self.assertAlmostEqual(high, 50.0)

    def test_bootstrap_uses_even_sample_median_average(self):
        self.assertEqual(tune_promotion._median([10.0, 20.0, 30.0, 40.0]), 25.0)

    def test_confirmation_rejected_is_in_hypotheses_but_cannot_promote(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.jsonl", root / "promoted.jsonl"
            row = result("a" * 32, 0.001, 95.0)
            row["promotion_status"] = "confirmation_rejected"
            source.write_text(
                "".join(json.dumps(row_) + "\n" for row_ in [self.HEADER, row]),
                encoding="utf-8",
            )
            report = tune_promotion.promote(source, output, resamples=1000)
            self.assertEqual(report["hypotheses"], 1)
            promoted = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(promoted[1]["promotion_status"], "confirmation_rejected")

    def test_inconsistent_persisted_effect_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.jsonl", root / "promoted.jsonl"
            row = result("a" * 32, 0.001, 95.0)
            row["confirmation"]["effect_pct"] = 99.0
            source.write_text(
                "".join(json.dumps(row_) + "\n" for row_ in [self.HEADER, row]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                tune_promotion.PromotionError, "does not match"
            ):
                tune_promotion.promote(source, output, resamples=1000)

    def test_rejection_reason_is_classified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.jsonl", root / "promoted.jsonl"
            row = result("a" * 32, 0.001, 99.5)
            source.write_text(
                "".join(json.dumps(row_) + "\n" for row_ in [self.HEADER, row]),
                encoding="utf-8",
            )
            tune_promotion.promote(source, output, threshold_pct=1.0, resamples=1000)
            promoted = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(promoted[1]["promotion_status"], "rejected_effect")

    def test_bh_and_bootstrap_promote_only_supported_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.jsonl", root / "promoted.jsonl"
            rows = [
                self.HEADER,
                result("a" * 32, 0.001, 95.0),
                result("b" * 32, 0.9, 99.5),
            ]
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            report = tune_promotion.promote(source, output, resamples=1000)
            promoted = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(report["promoted"], 1)
            self.assertEqual(promoted[1]["promotion_status"], "promoted")
            self.assertEqual(promoted[2]["promotion_status"], "rejected_effect")
            self.assertEqual(promoted[2]["winner"], "native")
            self.assertAlmostEqual(promoted[1]["promotion"]["q_value"], 0.002)
            self.assertAlmostEqual(promoted[2]["promotion"]["q_value"], 0.9)

    def test_bh_adjusted_values_are_exact_and_monotone(self):
        adjusted = tune_promotion.benjamini_hochberg(
            [
                ("d", 0.04),
                ("a", 0.001),
                ("c", 0.03),
                ("b", 0.02),
            ]
        )
        self.assertEqual(
            adjusted,
            {
                "a": 0.004,
                "b": 0.04,
                "c": 0.04,
                "d": 0.04,
            },
        )

    def test_null_fdr_simulation_is_reproducible_and_controlled(self):
        first = tune_promotion.simulate_null_fdr(
            experiments=2000,
            hypotheses=41,
            q=0.05,
            seed=340024,
        )
        second = tune_promotion.simulate_null_fdr(
            experiments=2000,
            hypotheses=41,
            q=0.05,
            seed=340024,
        )
        self.assertEqual(first, second)
        self.assertLess(first["empirical_fdr"], 0.07)
        self.assertEqual(len(first["runs"]), 2000)

    def test_non_current_pending_state_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            row = result("a" * 32, 0.001, 95.0)
            row.pop("promotion_status")
            source.write_text(json.dumps(self.HEADER) + "\n" + json.dumps(row) + "\n")
            with self.assertRaisesRegex(tune_promotion.PromotionError, "pending_bh"):
                tune_promotion.promote(source, root / "out")

    def test_promotion_is_deterministic_for_same_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            rows = [
                self.HEADER,
                result("a" * 32, 0.001, 95.0),
                result("b" * 32, 0.9, 99.5),
            ]
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            first = tune_promotion.promote(source, root / "out1.jsonl", resamples=1000)
            second = tune_promotion.promote(source, root / "out2.jsonl", resamples=1000)
            self.assertEqual(first["content_hash"], second["content_hash"])

    def test_ranking_coverage_ignores_synthetic_twin(self):
        # The fixture schedule carries ["candidate", "native",
        # "native#twin"]: ranking coverage must be taken over the real
        # finalists only, so a decision naming exactly candidate+native is
        # complete.
        row = result("a" * 32, 0.001, 95.0)
        row["production_policy"] = {"name": "latency-v1", "version": 1}
        row["ranking_decisions"] = [
            {
                "policy_name": "latency-v1",
                "policy_version": 1,
                "is_production": True,
                "predicted_winner": "candidate",
                "candidates": [
                    {"name": "native", "verdict": "qualified"},
                    {"name": "candidate", "verdict": "winner"},
                ],
            }
        ]
        header = dict(self.HEADER, production_policy="latency-v1")
        tune_promotion._validate_policy_identity(row, header)

    def test_schedule_rejects_duplicate_unsuffixed_native(self):
        # The C++ bug this guards against: emitting raw stable names puts the
        # twin into the schedule as a second plain "native".
        row = result("a" * 32, 0.001, 95.0)
        row["schedule"]["candidates"] = ["candidate", "native", "native"]
        with self.assertRaisesRegex(tune_promotion.PromotionError, "drift"):
            tune_promotion.validate_schedule(row)

    def test_adaptive_evidence_counts_remain_registry_counts(self):
        # Candidate cardinality (three emitted rows incl. the synthetic twin)
        # and measurement-instance cardinality intentionally differ: the stage
        # funnel stays over the registry.
        row = result("a" * 32, 0.001, 95.0)
        row.update({"generated": 2, "applicable": 2, "eligible": 2, "measured": 2})
        row["candidates"] = [
            {"name": "native", "samples": 10},
            {"name": "native#twin", "samples": 10},
            {"name": "candidate", "samples": 10},
        ]
        tune_promotion.validate_adaptive_evidence(row, self.HEADER)

    def test_schedule_seed_and_position_drift_are_rejected(self):
        row = result("a" * 32, 0.001, 95.0)
        row["schedule_seed"] += 1
        with self.assertRaisesRegex(tune_promotion.PromotionError, "seed drift"):
            tune_promotion.validate_schedule(row)
        row = result("a" * 32, 0.001, 95.0)
        row["schedule"]["candidates"] = ["native", "candidate", "native#twin"]
        with self.assertRaisesRegex(tune_promotion.PromotionError, "position drift"):
            tune_promotion.validate_schedule(row)

    def test_adaptive_evidence_rejects_short_confirmation(self):
        row = result("a" * 32, 0.001, 95.0)
        row["confirmation"]["native_us"] = [100.0] * 7
        row["confirmation"]["winner_us"] = [95.0] * 7
        row["confirmation"]["rounds"] = 7
        row["confirmation"]["wins"] = 7
        with self.assertRaisesRegex(tune_promotion.PromotionError, "insufficient"):
            tune_promotion.validate_adaptive_evidence(row, self.HEADER)

    def test_adaptive_evidence_rejects_inconsistent_final_samples(self):
        row = result("a" * 32, 0.001, 95.0)
        row["candidates"] = [
            {"name": "candidate", "samples": 3, "samples_us": [95.0, 95.0]}
        ]
        header = dict(
            self.HEADER, final_samples=2, screen_samples=4, confirmation_samples=8
        )
        with self.assertRaisesRegex(tune_promotion.PromotionError, "samples_us"):
            tune_promotion.validate_adaptive_evidence(row, header)

    def test_adaptive_evidence_rejects_unresolved_canary_challenger(self):
        row = result("a" * 32, 0.001, 95.0)
        row.update(
            {
                "canary_state": "unresolved",
                "canary_retries": 1,
                "canary_pair": "native#twin",
                "canary_pct": 4.0,
            }
        )
        with self.assertRaisesRegex(tune_promotion.PromotionError, "unresolved"):
            tune_promotion.validate_adaptive_evidence(row, self.HEADER)

    def test_hi68_canary_terminal_state_matrix_is_accepted(self):
        # Producer/consumer contract (HI68): every terminal triple the C++
        # canary state machine can emit must validate here. The zero-probe
        # unresolved row is the P1 case: noise_canary_retries=0 makes it legal.
        legal = [
            {
                "canary_state": "not_available",
                "canary_retries": 0,
                "canary_fresh_block": False,
            },
            {
                "canary_state": "pass",
                "canary_retries": 0,
                "canary_fresh_block": False,
                "canary_pair": "native#twin",
                "canary_pct": 1.0,
            },
            {
                "canary_state": "unresolved",
                "canary_retries": 0,
                "canary_fresh_block": False,
                "provisional_winner": "native",
                "canary_pair": "native#twin",
                "canary_pct": 7.0,
            },
            {
                "canary_state": "unresolved",
                "canary_retries": 1,
                "canary_fresh_block": False,
                "provisional_winner": "native",
                "canary_pair": "native#twin",
                "canary_pct": 7.0,
            },
            {
                "canary_state": "retried_pass",
                "canary_retries": 1,
                "canary_fresh_block": True,
                "canary_pair": "native#twin",
                "canary_pct": 0.5,
            },
            {
                "canary_state": "unresolved",
                "canary_retries": 1,
                "canary_fresh_block": True,
                "provisional_winner": "native",
                "canary_pair": "native#twin",
                "canary_pct": 9.0,
            },
        ]
        for fields in legal:
            with self.subTest(fields=fields):
                row = result("a" * 32, 0.001, 95.0)
                row.update(fields)
                tune_promotion.validate_adaptive_evidence(row, self.HEADER)

    def test_hi68_illegal_canary_terminal_states_are_rejected(self):
        illegal = [
            {
                "canary_state": "pass",
                "canary_retries": 1,
                "canary_fresh_block": False,
                "canary_pair": "native#twin",
                "canary_pct": 1.0,
            },
            {
                "canary_state": "unresolved",
                "canary_retries": 0,
                "canary_fresh_block": True,
                "provisional_winner": "native",
                "canary_pair": "native#twin",
                "canary_pct": 7.0,
            },
            {
                "canary_state": "retried_pass",
                "canary_retries": 0,
                "canary_fresh_block": True,
                "canary_pair": "native#twin",
                "canary_pct": 0.5,
            },
            # Explicit fresh=false on retried_pass is the NEW schema saying the
            # producer contradicted itself: rejected (unlike the legacy row
            # below, where the field is simply absent).
            {
                "canary_state": "retried_pass",
                "canary_retries": 1,
                "canary_fresh_block": False,
                "canary_pair": "native#twin",
                "canary_pct": 0.5,
            },
        ]
        for fields in illegal:
            with self.subTest(fields=fields):
                row = result("a" * 32, 0.001, 95.0)
                row.update(fields)
                with self.assertRaises(tune_promotion.PromotionError):
                    tune_promotion.validate_adaptive_evidence(row, self.HEADER)

    def test_legacy_retried_pass_without_fresh_field_still_validates(self):
        # Pre-HI68 artifacts carry retried_pass with the pair-replacement
        # semantics and no canary_fresh_block key at all; the field's ABSENCE
        # (not a false value) is what marks them legacy.
        row = result("a" * 32, 0.001, 95.0)
        row.update(
            {
                "canary_state": "retried_pass",
                "canary_retries": 1,
                "canary_pair": "native#twin",
                "canary_pct": 0.5,
            }
        )
        tune_promotion.validate_adaptive_evidence(row, self.HEADER)

    def test_non_bool_canary_fresh_block_flag_is_rejected(self):
        row = result("a" * 32, 0.001, 95.0)
        row.update(
            {
                "canary_state": "retried_pass",
                "canary_retries": 1,
                "canary_fresh_block": 1,
                "canary_pair": "native#twin",
                "canary_pct": 0.5,
            }
        )
        with self.assertRaisesRegex(tune_promotion.PromotionError, "fresh-block flag"):
            tune_promotion.validate_adaptive_evidence(row, self.HEADER)

    def test_production_policy_hash_is_deterministic(self):
        first = tune_promotion.production_policy_hash("latency-v1", 1)
        second = tune_promotion.production_policy_hash("latency-v1", 1)
        self.assertEqual(first, second)
        self.assertNotEqual(
            first, tune_promotion.production_policy_hash("latency-v1", 2)
        )

    def test_ranking_coverage_requires_policy_identity_and_all_finalists(self):
        row = result("a" * 32, 0.001, 95.0)
        row["schedule"] = {"candidates": ["native", "candidate"]}
        row["production_policy"] = {"name": "latency-v1", "version": 1}
        row["ranking_decisions"] = [
            {
                "policy_name": "latency-v1",
                "policy_version": 1,
                "is_production": True,
                "predicted_winner": "candidate",
                "candidates": [
                    {"name": "native", "verdict": "qualified"},
                    {"name": "candidate", "verdict": "winner"},
                ],
            }
        ]
        header = dict(self.HEADER, production_policy="latency-v1")
        tune_promotion._validate_policy_identity(row, header)
        row["ranking_decisions"][0]["candidates"].pop()
        with self.assertRaisesRegex(tune_promotion.PromotionError, "coverage"):
            tune_promotion._validate_policy_identity(row, header)

    def test_native_only_ranking_decision_has_no_selection_schedule(self):
        row = result("a" * 32, 0.001, 95.0)
        row["native"] = "native"
        row["provisional_winner"] = "native"
        row.pop("schedule")
        row["production_policy"] = {"name": "latency-v1", "version": 1}
        row["ranking_decisions"] = [
            {
                "policy_name": "latency-v1",
                "policy_version": 1,
                "is_production": True,
                "predicted_winner": "native",
                "candidates": [{"name": "native", "verdict": "winner"}],
            }
        ]
        header = dict(self.HEADER, production_policy="latency-v1")
        tune_promotion._validate_policy_identity(row, header)

    def test_ranking_provisional_winner_and_status_are_consistent(self):
        row = result("a" * 32, 0.001, 95.0)
        row["provisional_winner"] = "native"
        with self.assertRaisesRegex(tune_promotion.PromotionError, "status"):
            tune_promotion._validate_provisional_status(row)

    def test_policy_hash_tampering_is_rejected(self):
        row = result("a" * 32, 0.001, 95.0)
        row["production_policy"] = {
            "name": "latency-v1",
            "version": 1,
            "policy_hash": "0" * 32,
        }
        header = dict(self.HEADER, production_policy="latency-v1")
        with self.assertRaisesRegex(tune_promotion.PromotionError, "hash"):
            tune_promotion._validate_policy_identity(row, header)

    def test_confirmation_reduction_excludes_aligned_ties(self):
        confirmation = {
            "rounds": 2,
            "wins": 2,
            "native_us": [100.0, 100.0, 100.0],
            "winner_us": [90.0, 100.0, 80.0],
        }
        native, winner = tune_promotion._paired_rounds(confirmation)
        self.assertEqual(native, [100.0, 100.0])
        self.assertEqual(winner, [90.0, 80.0])

    def test_effect_keeps_tied_samples_like_cpp_median(self):
        confirmation = {
            "rounds": 8,
            "wins": 8,
            "native_us": [100.0] * 9,
            "winner_us": [90.0] * 8 + [100.0],
            "effect_pct": 10.0,
        }
        self.assertAlmostEqual(tune_promotion._validated_effect(confirmation), 10.0)

    def test_native_retention_can_have_no_challenger_winner_verdict(self):
        row = result("b" * 32, 0.001, 100.0)
        row["provisional_winner"] = "native"
        row["winner"] = "native"
        row["promotion_status"] = "native"
        row["production_policy"] = {"name": "latency-v1", "version": 1}
        row["ranking_decisions"] = [
            {
                "policy_name": "latency-v1",
                "policy_version": 1,
                "is_production": True,
                "predicted_winner": "native",
                "candidates": [
                    {"name": "native", "verdict": "outside_tie_band"},
                    {"name": "candidate", "verdict": "near_tie_below_threshold"},
                ],
            }
        ]
        header = dict(self.HEADER, production_policy="latency-v1")
        tune_promotion._validate_policy_identity(row, header)


if __name__ == "__main__":
    unittest.main()

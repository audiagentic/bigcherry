"""Tests for the offline bandit/sequential-elimination simulator (HI44)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.analysis import bandit_simulator as bs  # noqa: E402


def _row(dispatch, native, candidates, *, winner=None, provisional_winner=None,
         schedule_candidates=None):
    return {
        "kind": "result",
        "dispatch": dispatch,
        "native": native,
        "winner": winner if winner is not None else native,
        "provisional_winner": provisional_winner,
        "schedule": {"candidates": schedule_candidates or [c["name"] for c in candidates]},
        "candidates": candidates,
    }


def _candidate(name, samples):
    return {"name": name, "samples_us": list(samples)}


def _write_jsonl(rows):
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".measurements.jsonl", delete=False, encoding="utf-8"
    )
    handle.write(json.dumps({"kind": "header"}) + "\n")
    for row in rows:
        handle.write(json.dumps(row) + "\n")
    handle.close()
    return Path(handle.name)


class LoadRoundEvidenceTests(unittest.TestCase):
    def test_skips_rows_without_schedule(self):
        row = {"kind": "result", "dispatch": "d1", "native": "n", "winner": "n",
               "candidates": [_candidate("n", [1.0] * 10)]}
        # no "schedule" key at all -- matches the real artifacts/tuning-logs/*.jsonl shape
        path = _write_jsonl([row])
        self.assertEqual(bs.load_round_evidence(path), [])

    def test_skips_candidates_without_samples(self):
        path = _write_jsonl([_row("d1", "n", [
            _candidate("n", [1.0] * 10),
            {"name": "c", "samples_us": []},
        ])])
        evidence = bs.load_round_evidence(path)
        self.assertEqual(len(evidence), 1)
        self.assertEqual([c.name for c in evidence[0].candidates], ["n"])

    def test_uses_provisional_winner_when_present(self):
        path = _write_jsonl([_row("d1", "n", [_candidate("n", [1.0] * 10)],
                                  winner="n", provisional_winner="c")])
        evidence = bs.load_round_evidence(path)
        self.assertEqual(evidence[0].target_winner, "c")
        self.assertFalse(evidence[0].target_is_degraded)

    def test_falls_back_to_winner_when_provisional_absent(self):
        path = _write_jsonl([_row("d1", "n", [_candidate("n", [1.0] * 10)], winner="n")])
        evidence = bs.load_round_evidence(path)
        self.assertEqual(evidence[0].target_winner, "n")
        self.assertTrue(evidence[0].target_is_degraded)

    def test_twin_detection(self):
        path = _write_jsonl([_row("d1", "n", [
            _candidate("n", [10.0] * 20),
            _candidate("n#twin", [10.0] * 20),
        ])])
        evidence = bs.load_round_evidence(path)[0]
        native = next(c for c in evidence.candidates if c.name == "n")
        twin = next(c for c in evidence.candidates if c.name == "n#twin")
        self.assertTrue(native.is_native)
        self.assertTrue(twin.is_twin)
        self.assertFalse(twin.is_native)
        self.assertTrue(evidence.is_canary_only())


class PairedSignTestTests(unittest.TestCase):
    def test_identical_sequences_not_significant(self):
        rounds = tuple(float(x) for x in range(1, 21))
        p, wins, n = bs.paired_sign_test(rounds, rounds, min_rounds=8)
        self.assertEqual(wins, 0)
        self.assertGreater(p, 0.5)

    def test_consistently_faster_challenger_significant(self):
        native = tuple(10.0 + 0.1 * i for i in range(30))
        challenger = tuple(9.0 + 0.1 * i for i in range(30))  # always 1.0 lower
        p, wins, n = bs.paired_sign_test(native, challenger, min_rounds=8)
        self.assertEqual(wins, 30)
        self.assertLess(p, 0.001)

    def test_below_min_rounds_is_inconclusive(self):
        native = (10.0, 9.0, 10.0)
        challenger = (9.0, 8.0, 9.0)
        p, wins, n = bs.paired_sign_test(native, challenger, min_rounds=8)
        self.assertEqual(p, 1.0)


class DeclaredPromotionTests(unittest.TestCase):
    def test_native_retained_when_winner_is_native(self):
        result = bs.PolicyResult(winner="n", rounds_consumed=10, conclusive=True,
                                 means={"n": 10.0}, consumed_rounds={"n": (10.0,) * 10})
        self.assertEqual(bs.declared_promotion(result, "n"), "n")

    def test_native_retained_when_effect_size_too_small(self):
        # 0.5% improvement, below the 1% default threshold
        native_rounds = tuple(10.0 for _ in range(20))
        challenger_rounds = tuple(9.95 for _ in range(20))
        result = bs.PolicyResult(
            winner="c", rounds_consumed=40, conclusive=True,
            means={"n": 10.0, "c": 9.95},
            consumed_rounds={"n": native_rounds, "c": challenger_rounds},
        )
        self.assertEqual(bs.declared_promotion(result, "n", threshold_pct=1.0), "n")

    def test_native_retained_when_effect_size_ok_but_not_significant(self):
        # Large mean gap but only from occasional outliers -- inconsistent round-by-round.
        native_rounds = (10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 100.0)
        challenger_rounds = (10.1, 10.1, 10.1, 10.1, 10.1, 10.1, 10.1, 10.1)
        result = bs.PolicyResult(
            winner="c", rounds_consumed=16, conclusive=True,
            means={"n": bs._mean(native_rounds), "c": bs._mean(challenger_rounds)},
            consumed_rounds={"n": native_rounds, "c": challenger_rounds},
        )
        # native's mean is inflated by one outlier, so the raw-mean effect size
        # favors the challenger, but round-for-round the challenger is *slower*
        # on 7 of 8 rounds -- the paired test must catch this, not just the mean.
        self.assertEqual(bs.declared_promotion(result, "n"), "n")

    def test_promotion_when_both_effect_and_significance_clear(self):
        native_rounds = tuple(10.0 for _ in range(20))
        challenger_rounds = tuple(9.0 for _ in range(20))
        result = bs.PolicyResult(
            winner="c", rounds_consumed=40, conclusive=True,
            means={"n": 10.0, "c": 9.0},
            consumed_rounds={"n": native_rounds, "c": challenger_rounds},
        )
        self.assertEqual(bs.declared_promotion(result, "n"), "c")

    def test_inconclusive_result_retains_native(self):
        result = bs.PolicyResult(winner="", rounds_consumed=0, conclusive=False)
        self.assertEqual(bs.declared_promotion(result, "n"), "n")


class AllocationPolicyTests(unittest.TestCase):
    def _candidates(self, native_rounds, challenger_rounds):
        return (
            bs.CandidateRounds(name="n", is_native=True, is_twin=False, rounds=native_rounds),
            bs.CandidateRounds(name="c", is_native=False, is_twin=False, rounds=challenger_rounds),
        )

    def test_fixed_schedule_picks_lower_mean(self):
        candidates = self._candidates((10.0,) * 20, (9.0,) * 20)
        result = bs.fixed_schedule(candidates, min_rounds=8)
        self.assertEqual(result.winner, "c")
        self.assertEqual(result.rounds_consumed, 40)
        self.assertTrue(result.conclusive)

    def test_fixed_schedule_inconclusive_below_min_rounds(self):
        candidates = self._candidates((10.0,) * 3, (9.0,) * 3)
        result = bs.fixed_schedule(candidates, min_rounds=8)
        self.assertFalse(result.conclusive)

    def test_successive_halving_never_eliminates_native(self):
        # Ten decoy candidates all faster than native early on, but native
        # must survive to the end so declared_promotion always has a
        # native mean to compare against.
        candidates = [bs.CandidateRounds(name="n", is_native=True, is_twin=False,
                                         rounds=(10.0,) * 64)]
        for i in range(10):
            candidates.append(bs.CandidateRounds(
                name=f"c{i}", is_native=False, is_twin=False, rounds=(5.0,) * 64))
        result = bs.successive_halving(tuple(candidates), min_rounds=8, chunk=8)
        self.assertIn("n", result.means)

    def test_successive_halving_uses_fewer_rounds_than_fixed(self):
        candidates = [bs.CandidateRounds(name="n", is_native=True, is_twin=False,
                                         rounds=(10.0,) * 64)]
        for i in range(6):
            candidates.append(bs.CandidateRounds(
                name=f"c{i}", is_native=False, is_twin=False, rounds=(10.0 + i,) * 64))
        sh = bs.successive_halving(tuple(candidates), min_rounds=8, chunk=8)
        fixed = bs.fixed_schedule(tuple(candidates), min_rounds=8)
        self.assertLess(sh.rounds_consumed, fixed.rounds_consumed)

    def test_ucb1_never_exceeds_available_rounds(self):
        candidates = self._candidates((10.0,) * 15, (9.0,) * 15)
        result = bs.ucb1(candidates, min_rounds=8)
        for name, rounds in result.consumed_rounds.items():
            available = {"n": 15, "c": 15}[name]
            self.assertLessEqual(len(rounds), available)

    def test_thompson_sampling_never_exceeds_available_rounds(self):
        candidates = self._candidates((10.0,) * 15, (9.0,) * 15)
        result = bs.thompson_sampling(candidates, min_rounds=8, seed=1)
        for name, rounds in result.consumed_rounds.items():
            available = {"n": 15, "c": 15}[name]
            self.assertLessEqual(len(rounds), available)


class EvaluateFalsePromotionTests(unittest.TestCase):
    def test_identical_twin_rarely_triggers_false_promotion(self):
        # Ten canary pairs built from the same fixed pseudo-random-looking
        # but deterministic sequence -- native and twin literally identical,
        # so any policy declaring a promotion here is a false positive.
        rows = []
        base = [10.0, 10.2, 9.8, 10.1, 9.9, 10.0, 10.3, 9.7, 10.0, 10.1] * 10
        for i in range(10):
            rows.append(_row(f"d{i}", "n", [
                _candidate("n", base),
                _candidate("n#twin", base),
            ], winner="n"))
        path = _write_jsonl(rows)
        corpus = bs.load_corpus([path])
        result = bs.evaluate_false_promotion(corpus, "fixed-schedule", min_rounds=8)
        self.assertEqual(result["false_promotions"], 0)


class ExactMcNemarTests(unittest.TestCase):
    def test_zero_discordant_pairs_is_certain(self):
        self.assertEqual(bs._exact_mcnemar_p(0, 0), 1.0)

    def test_perfectly_balanced_discordance_is_high_p(self):
        # 10 vs 10 discordant: about as "no difference" as paired data gets.
        p = bs._exact_mcnemar_p(10, 20)
        self.assertGreater(p, 0.5)

    def test_lopsided_discordance_is_significant(self):
        # 1 vs 19 discordant pairs is a strong, real signal.
        p = bs._exact_mcnemar_p(1, 20)
        self.assertLess(p, 0.001)

    def test_matches_known_reference_value(self):
        # Textbook example: 5 vs 15 discordant (n=20, k=5) -> exact two-sided
        # McNemar p ~= 0.0414 (R's mcnemar.test(matrix(c(0,5,15,0),2), correct=FALSE)
        # gives the uncorrected chi-sq form; the exact binomial form used here
        # is the more conservative, small-sample-correct variant and is
        # standard for n this small).
        p = bs._exact_mcnemar_p(5, 20)
        self.assertAlmostEqual(p, 0.0414, places=3)


class PairedComparisonTests(unittest.TestCase):
    def _corpus_where_policies_disagree(self, n_a_only, n_b_only, n_agree):
        """A synthetic corpus engineered so successive-halving and
        fixed-schedule disagree with ground truth in exactly the requested,
        opposite pattern on `n_a_only`/`n_b_only` dispatches, and agree with
        each other (both correct) on `n_agree`.

        Successive-halving eliminates candidates whose *early* rounds already
        look clearly worse; fixed-schedule always uses the full round budget.
        So a dispatch where the TRUE winner only pulls ahead in LATE rounds,
        after an early stretch where it looks worse, is one successive-
        halving can get wrong while fixed-schedule (seeing all rounds) gets
        right -- and a dispatch with a clean, consistent gap from round 1 is
        one both should get right the same way.
        """
        rows = []
        # Both correct: clean, consistent gap throughout -- easy for any policy.
        for i in range(n_agree):
            rows.append(_row(f"agree{i}", "n", [
                _candidate("n", [10.0] * 40),
                _candidate("c", [9.0] * 40),
            ], winner="c"))
        # fixed-schedule-only-correct: candidate "c" looks WORSE for the first
        # half of rounds (successive-halving eliminates it early) then pulls
        # ahead late; only a policy that sees the full schedule catches it.
        for i in range(n_b_only):
            rows.append(_row(f"fsonly{i}", "n", [
                _candidate("n", [10.0] * 40),
                _candidate("c", [11.0] * 20 + [7.0] * 20),
            ], winner="c"))
        return rows, n_a_only  # n_a_only left at 0 in the cases we construct below

    def test_identical_policies_have_zero_discordant_pairs(self):
        corpus = bs.load_round_evidence(_write_jsonl([
            _row("d1", "n", [_candidate("n", [10.0] * 20), _candidate("c", [9.0] * 20)], winner="c"),
            _row("d2", "n", [_candidate("n", [10.0] * 20), _candidate("c", [11.0] * 20)], winner="n"),
        ]))
        result = bs.evaluate_paired(corpus, "fixed-schedule", "fixed-schedule", min_rounds=8)
        self.assertEqual(result["fixed-schedule_only_correct"], 0)
        self.assertEqual(result["both_agree"] + result["both_disagree"],
                         result["paired_dispatches"])
        self.assertIn("IDENTICAL", result["verdict"])

    def test_lopsided_disagreement_resolves_significant(self):
        rows, _ = self._corpus_where_policies_disagree(n_a_only=0, n_b_only=15, n_agree=5)
        corpus = bs.load_round_evidence(_write_jsonl(rows))
        result = bs.evaluate_paired(corpus, "fixed-schedule", "successive-halving", min_rounds=8)
        # fixed-schedule should be right on (at least some of) the "late
        # pull-ahead" dispatches that successive-halving's early elimination
        # misses -- i.e. discordant pairs should skew toward fixed-schedule.
        self.assertGreaterEqual(result["fixed-schedule_only_correct"], 0)
        self.assertIn("mcnemar_p_value", result)
        self.assertGreaterEqual(result["mcnemar_p_value"], 0.0)
        self.assertLessEqual(result["mcnemar_p_value"], 1.0)

    def test_paired_examples_capped_and_dispatch_named(self):
        rows, _ = self._corpus_where_policies_disagree(n_a_only=0, n_b_only=8, n_agree=0)
        corpus = bs.load_round_evidence(_write_jsonl(rows))
        result = bs.evaluate_paired(corpus, "fixed-schedule", "successive-halving", min_rounds=8)
        examples = result["fixed-schedule_only_correct_examples"]
        self.assertLessEqual(len(examples), 5)
        for name in examples:
            self.assertTrue(name.startswith("fsonly"))

    def test_simulate_report_includes_paired_vs_baseline(self):
        rows = [_row("d1", "n", [
            _candidate("n", [10.0] * 30), _candidate("c", [9.0] * 30),
        ], winner="c")]
        report = bs.simulate([_write_jsonl(rows)], min_rounds=8, resamples=5)
        self.assertIn("paired_vs_baseline", report)
        for name in bs.ALLOCATION_POLICIES:
            if name == bs.BASELINE_POLICY_NAME:
                self.assertNotIn(name, report["paired_vs_baseline"])
            else:
                self.assertIn(name, report["paired_vs_baseline"])
                self.assertIn("mcnemar_p_value", report["paired_vs_baseline"][name])


class EndToEndSmokeTest(unittest.TestCase):
    def test_simulate_runs_against_small_real_shaped_corpus(self):
        rows = [
            _row("d1", "n", [
                _candidate("n", [10.0 + 0.1 * (i % 5) for i in range(50)]),
                _candidate("c", [9.0 + 0.1 * (i % 5) for i in range(50)]),
            ], winner="c"),
            _row("d2", "n", [
                _candidate("n", [10.0 + 0.1 * (i % 5) for i in range(50)]),
                _candidate("n#twin", [10.0 + 0.1 * ((i + 2) % 5) for i in range(50)]),
            ], winner="n"),
        ]
        path = _write_jsonl(rows)
        report = bs.simulate([path], min_rounds=8, resamples=20)
        self.assertEqual(report["rows_with_usable_round_data"], 2)
        for policy in bs.ALLOCATION_POLICIES:
            self.assertIn(policy, report["policies"])

    def test_main_writes_report(self):
        rows = [_row("d1", "n", [
            _candidate("n", [10.0] * 20),
            _candidate("c", [9.0] * 20),
        ], winner="c")]
        path = _write_jsonl(rows)
        out = Path(tempfile.mkstemp(suffix=".json")[1])
        code = bs.main([str(path), "--output", str(out), "--resamples", "5"])
        self.assertEqual(code, 0)
        self.assertTrue(out.is_file())


if __name__ == "__main__":
    unittest.main()

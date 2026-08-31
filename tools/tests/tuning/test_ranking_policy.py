"""Tests for the ranking-policy schema/validation module (HI50)."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.tuning import ranking as rp # noqa: E402


def _candidate_json(name, effective_us, status="ok", **extra):
    entry = {
        "name": name, "status": status, "effective_us": effective_us,
        "median_us": effective_us, "p95_us": effective_us,
    }
    entry.update(extra)
    return entry


def _result_json(*, native="native:v1", finalists=None, candidates=None,
                  decisions=None, provisional_winner=None):
    return {
        "kind": "result",
        "dispatch": "deadbeef",
        "native": native,
        "winner": native,
        "provisional_winner": provisional_winner,
        "schedule": {"candidates": finalists or [native]},
        "candidates": candidates or [_candidate_json(native, 10.0)],
        "ranking_decisions": decisions,
    }


class CandidateMetricsTests(unittest.TestCase):
    def test_from_json_marks_native(self):
        entry = _candidate_json("candidate:v2", 12.5)
        metrics = rp.CandidateMetrics.from_json(entry, native_name="native:v1")
        self.assertEqual(metrics.name, "candidate:v2")
        self.assertFalse(metrics.is_native)
        self.assertEqual(metrics.effective_us, 12.5)

        native_metrics = rp.CandidateMetrics.from_json(
            _candidate_json("native:v1", 10.0), native_name="native:v1"
        )
        self.assertTrue(native_metrics.is_native)


class FinalistMetricsTests(unittest.TestCase):
    def test_restricts_to_schedule_candidates(self):
        result = _result_json(
            finalists=["native:v1", "candidate:v2"],
            candidates=[
                _candidate_json("native:v1", 10.0),
                _candidate_json("candidate:v2", 9.0),
                _candidate_json("candidate:v3", 8.5),  # screened out, not in schedule
            ],
        )
        names = {c.name for c in rp.finalist_metrics(result)}
        self.assertEqual(names, {"native:v1", "candidate:v2"})

    def test_excludes_native_twin(self):
        result = _result_json(
            finalists=["native:v1", "native:v1#twin"],
            candidates=[
                _candidate_json("native:v1", 10.0),
                _candidate_json("native:v1#twin", 10.1),
            ],
        )
        names = {c.name for c in rp.finalist_metrics(result)}
        self.assertEqual(names, {"native:v1"})


class ParseRankingDecisionsTests(unittest.TestCase):
    def test_absent_field_returns_empty(self):
        result = _result_json(decisions=None)
        self.assertEqual(rp.parse_ranking_decisions(result), [])

    def test_parses_candidates_and_verdicts(self):
        result = _result_json(decisions=[
            {
                "policy_name": "latency-v1", "policy_version": 1,
                "is_production": True, "predicted_winner": "candidate:v2",
                "candidates": [
                    {"name": "native:v1", "effective_us": 10.0, "verdict": "qualified", "rejection_reason": ""},
                    {"name": "candidate:v2", "effective_us": 9.0, "verdict": "winner", "rejection_reason": ""},
                ],
            }
        ])
        decisions = rp.parse_ranking_decisions(result)
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].predicted_winner, "candidate:v2")
        self.assertTrue(decisions[0].is_production)
        self.assertEqual(len(decisions[0].candidates), 2)

    def test_rejects_duplicate_policy_names(self):
        result = _result_json(decisions=[
            {"policy_name": "latency-v1", "policy_version": 1, "is_production": True,
             "predicted_winner": "native:v1", "candidates": []},
            {"policy_name": "latency-v1", "policy_version": 1, "is_production": False,
             "predicted_winner": "native:v1", "candidates": []},
        ])
        with self.assertRaises(rp.RankingPolicyError):
            rp.parse_ranking_decisions(result)

    def test_rejects_multiple_production_policies(self):
        result = _result_json(decisions=[
            {"policy_name": "a", "policy_version": 1, "is_production": True,
             "predicted_winner": "native:v1", "candidates": []},
            {"policy_name": "b", "policy_version": 1, "is_production": True,
             "predicted_winner": "native:v1", "candidates": []},
        ])
        with self.assertRaises(rp.RankingPolicyError):
            rp.parse_ranking_decisions(result)


class PolicySpecValidationTests(unittest.TestCase):
    def _valid_spec(self):
        spec = {"schema_version": rp.POLICY_SCHEMA_VERSION, "name": "pareto-v1", "version": 1}
        spec["policy_hash"] = rp.policy_hash(spec)
        return spec

    def test_valid_spec_round_trips(self):
        spec = self._valid_spec()
        self.assertEqual(rp.validate_policy_spec(spec), spec)

    def test_tampered_spec_fails_hash_check(self):
        spec = self._valid_spec()
        tampered = copy.deepcopy(spec)
        tampered["name"] = "different"
        with self.assertRaises(rp.RankingPolicyError):
            rp.validate_policy_spec(tampered)

    def test_wrong_schema_version_rejected(self):
        spec = self._valid_spec()
        spec["schema_version"] = rp.POLICY_SCHEMA_VERSION + 1
        spec["policy_hash"] = rp.policy_hash(spec)
        with self.assertRaises(rp.RankingPolicyError):
            rp.validate_policy_spec(spec)

    def test_missing_name_rejected(self):
        spec = {"schema_version": rp.POLICY_SCHEMA_VERSION, "version": 1}
        spec["policy_hash"] = rp.policy_hash(spec)
        with self.assertRaises(rp.RankingPolicyError):
            rp.validate_policy_spec(spec)


def _decision_json(*, policy_name="latency-v1", policy_version=1, is_production=True,
                    predicted_winner="candidate_a", candidates=None):
    return {
        "policy_name": policy_name, "policy_version": policy_version,
        "is_production": is_production, "predicted_winner": predicted_winner,
        "candidates": candidates or [],
    }


class PolicyDecisionStrictTypingTests(unittest.TestCase):
    """gpt-dev-agent review, 2026-08-31: PolicyDecision.from_json used to
    coerce types instead of validating them, and this feeds
    tune_promotion.py's production-policy identity check directly -- a
    type-coerced field here is a real correctness-boundary bug."""

    def test_valid_decision_round_trips(self):
        decision = rp.PolicyDecision.from_json(_decision_json())
        self.assertEqual(decision.policy_name, "latency-v1")
        self.assertEqual(decision.policy_version, 1)
        self.assertIs(decision.is_production, True)

    def test_string_false_for_is_production_is_rejected_not_coerced_true(self):
        # bool("false") == True in Python -- this must not silently pass.
        with self.assertRaises(rp.RankingPolicyError):
            rp.PolicyDecision.from_json(_decision_json(is_production="false"))

    def test_boolean_policy_version_is_rejected_not_treated_as_int(self):
        # True == 1 in Python -- a boolean must not satisfy an int field.
        with self.assertRaises(rp.RankingPolicyError):
            rp.PolicyDecision.from_json(_decision_json(policy_version=True))

    def test_empty_policy_name_is_rejected(self):
        with self.assertRaises(rp.RankingPolicyError):
            rp.PolicyDecision.from_json(_decision_json(policy_name=""))

    def test_non_string_policy_name_is_rejected(self):
        with self.assertRaises(rp.RankingPolicyError):
            rp.PolicyDecision.from_json(_decision_json(policy_name=123))

    def test_non_list_candidates_is_rejected(self):
        with self.assertRaises(rp.RankingPolicyError):
            rp.PolicyDecision.from_json(_decision_json(candidates="not-a-list"))

    def test_malformed_decision_in_parse_ranking_decisions_fails_closed(self):
        result = {"ranking_decisions": [_decision_json(is_production="false")]}
        with self.assertRaises(rp.RankingPolicyError):
            rp.parse_ranking_decisions(result)


if __name__ == "__main__":
    unittest.main()

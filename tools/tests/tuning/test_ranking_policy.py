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


if __name__ == "__main__":
    unittest.main()

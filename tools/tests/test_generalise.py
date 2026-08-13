"""Policy-driven generalized key and proof tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import generalise  # noqa: E402


def policy() -> dict:
    excluded = ["ne1"]
    value = {
        "schema_version": 1, "key_version": 1, "family": "mmvq",
        "eligibility_predicate_version": 1,
        "included_fields": sorted(generalise.FIELDS - set(excluded)),
        "excluded_fields": excluded,
        "transforms": [{"field": "ne1", "indices": [1], "op": "zero"}],
        "source_experiment_ids": ["holdout"],
        "thresholds": dict(generalise.REQUIRED_THRESHOLDS),
        "status": "proposed",
    }
    value["policy_hash"] = generalise.policy_hash(value)
    return value


def signature(m: int) -> dict:
    value = {name: 0 for name in generalise.FIELDS}
    for name in ("nb0", "nb1", "nbd", "ne0", "ne1", "ned"):
        value[name] = [1, 1, 1, 1]
    value["ne1"][1] = m
    value["schema_version"] = 1
    return value


class GeneraliseTests(unittest.TestCase):
    def _winner(self):
        return generalise.build_generalised_winner(
            source_signature="a" * 32,
            coverage_delta={
                "baseline_coverage_pct": 20.0,
                "generalised_coverage_pct": 35.0,
                "added_coverage_pct": 15.0,
                "added_calls": 42,
            },
            source_revision="b" * 40, manifest_hash="c" * 32,
            build_descriptor_hash="d" * 32,
            evidence_references=[{"kind": "experiment", "ref": "bundle/experiment.json"}],
            winner="candidate:v1",
        )

    def test_generalised_winner_requires_complete_traceable_provenance(self):
        winner = self._winner()
        self.assertIs(generalise.validate_generalised_winner(winner), winner)

    def test_generalised_winner_rejects_missing_or_inconsistent_evidence(self):
        winner = self._winner()
        del winner["source_signature"]
        with self.assertRaisesRegex(generalise.GeneralisationError, "source_signature"):
            generalise.validate_generalised_winner(winner)
        winner = self._winner()
        winner["coverage_delta"]["added_coverage_pct"] = 14.0
        with self.assertRaisesRegex(generalise.GeneralisationError, "inconsistent"):
            generalise.validate_generalised_winner(winner)
        winner = self._winner()
        winner["evidence_references"] = []
        with self.assertRaisesRegex(generalise.GeneralisationError, "evidence"):
            generalise.validate_generalised_winner(winner)

    def test_masked_dimension_groups_but_hard_field_does_not(self):
        spec = policy()
        self.assertEqual(
            generalise.generalised_digest(signature(1), spec),
            generalise.generalised_digest(signature(5), spec),
        )
        changed = signature(5)
        changed["src0_type"] = 7
        self.assertNotEqual(
            generalise.generalised_digest(signature(1), spec),
            generalise.generalised_digest(changed, spec),
        )

    def test_hash_or_unknown_schema_tamper_is_rejected(self):
        spec = policy()
        spec["family"] = "mmq"
        with self.assertRaisesRegex(generalise.GeneralisationError, "hash"):
            generalise.validate_policy(spec)
        spec = policy()
        spec["schema_version"] = 2
        spec["policy_hash"] = generalise.policy_hash(spec)
        with self.assertRaisesRegex(generalise.GeneralisationError, "external_conversion"):
            generalise.validate_policy(spec)

    def test_holdout_proof_enforces_minimum_group_and_regret(self):
        spec = policy()
        rows = []
        calls = {}
        for index, m in enumerate((1, 2, 3)):
            digest = f"{index + 1:032x}"
            calls[digest] = 100
            rows.append({
                "kind": "result", "signature": digest, "canonical": signature(m),
                "winner": "candidate", "canary_state": "pass", "promotion_blocked": "",
                "candidates": [{"name": "candidate", "status": "ok", "effective_us": 99.8}],
            })
        evidence = generalise.prove(spec, rows, calls)
        self.assertTrue(evidence["passed"])
        self.assertEqual(evidence["proven_groups"], 1)


# ---------------------------------------------------------------------------
# HI36 steps 1/3: regret_table / group_representatives / --what-if
# ---------------------------------------------------------------------------

def _result_row(sig_id: str, family: str, k: int, m: int, ncols_dst: int,
                winner: str, candidates: dict[str, float], calls: int = 0) -> dict:
    canonical = signature(m)
    canonical["ne0"][0] = k
    canonical["ned"][1] = ncols_dst
    return {
        "kind": "result",
        "signature": sig_id,
        "native": f"{family}:native:v1",
        "winner": winner,
        "canonical": canonical,
        "candidates": [
            {"name": name, "status": "ok", "median_us": median}
            for name, median in candidates.items()
        ],
    }, calls


class RegretAnalysisTests(unittest.TestCase):
    GROUP_BY = ["family", "src0_type", "K", "ncols_dst"]

    def _rows_and_calls(self, specs):
        rows, calls = [], {}
        for sig_id, *rest in specs:
            row, call_count = _result_row(sig_id, *rest)
            rows.append(row)
            calls[sig_id] = call_count
        return rows, calls

    def test_representative_has_zero_regret(self):
        # Two signatures in the same group (K, ncols_dst identical -- M and
        # signature id differ); the higher-call one is the representative and
        # must show exactly zero regret against itself.
        rows, calls = self._rows_and_calls([
            ("sigA", "mmvq", 5120, 1, 5, "w5:winnerA", {"w5:winnerA": 100.0}, 1000),
            ("sigB", "mmvq", 5120, 2, 5, "w5:winnerB", {"w5:winnerB": 90.0, "w5:winnerA": 91.0}, 10),
        ])
        table = generalise.regret_table(rows, calls, self.GROUP_BY)
        by_sig = {row["signature"]: row for row in table}
        self.assertEqual(by_sig["sigA"]["status"], "ok")
        self.assertAlmostEqual(by_sig["sigA"]["regret_pct"], 0.0)

    def test_no_generalisation_yields_zero_regret(self):
        # Grouping by every field (down to the signature id itself) puts each
        # row alone in its own group, so the representative is always the
        # row's own winner and regret must be exactly zero everywhere.
        rows, calls = self._rows_and_calls([
            ("sigA", "mmq", 4096, 1, 1, "cand1", {"cand1": 50.0, "cand2": 55.0}, 5),
            ("sigB", "mmq", 8192, 1, 1, "cand2", {"cand2": 70.0, "cand1": 90.0}, 7),
        ])
        table = generalise.regret_table(rows, calls, self.GROUP_BY + ["M"])
        for row in table:
            self.assertEqual(row["status"], "ok")
            self.assertAlmostEqual(row["regret_pct"], 0.0)

    def test_ineligible_representative_is_reported_not_dropped(self):
        # The group's representative candidate does not appear in a member's
        # own candidate list -- reported as ineligible, not silently skipped.
        rows, calls = self._rows_and_calls([
            ("sigA", "mmvq", 5120, 1, 5, "onlyA", {"onlyA": 100.0}, 1000),
            ("sigB", "mmvq", 5120, 2, 5, "onlyB", {"onlyB": 90.0}, 10),
        ])
        table = generalise.regret_table(rows, calls, self.GROUP_BY)
        by_sig = {row["signature"]: row for row in table}
        self.assertEqual(by_sig["sigB"]["status"], "ineligible")
        self.assertIsNone(by_sig["sigB"]["regret_pct"])

    def test_group_representatives_picks_highest_call_weighted_member(self):
        rows, calls = self._rows_and_calls([
            ("sigA", "mmvq", 5120, 1, 5, "winnerA", {"winnerA": 10.0}, 1),
            ("sigB", "mmvq", 5120, 2, 5, "winnerB", {"winnerB": 10.0}, 1000),
        ])
        reps = generalise.group_representatives(rows, calls, self.GROUP_BY)
        [key] = reps.keys()
        self.assertEqual(reps[key], "winnerB")

    def test_what_if_converts_against_a_tuned_representative(self):
        rows, calls = self._rows_and_calls([
            ("sigA", "mmvq", 5120, 1, 5, "tunedWinner", {"tunedWinner": 10.0}, 1000),
        ])
        miss = {
            "signature": "missA", "native": "mmvq:native:v1",
            "fallback": "mmvq:native:v1", "canonical": signature(3),
            "candidates": [
                {"name": "mmvq:native:v1", "status": "ok", "median_us": 12.0},
                {"name": "tunedWinner", "status": "ok", "median_us": 11.0},
            ],
        }
        miss["canonical"]["ne0"][0] = 5120
        miss["canonical"]["ned"][1] = 5
        result = generalise.what_if([miss], rows, calls, self.GROUP_BY)
        self.assertEqual(result["converted"], 1)
        self.assertEqual(result["no_member_tuned"], 0)


if __name__ == "__main__":
    unittest.main()

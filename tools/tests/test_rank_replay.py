"""Tests for the ranking-policy offline report/replay harness (HI50)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import rank_replay  # noqa: E402

HEADER = {"kind": "header", "artifact_version": 1, "config": {"tie_pct": 0.5}}


def _candidate(name, effective_us, verdict="qualified", reason=""):
    return {"name": name, "effective_us": effective_us, "verdict": verdict, "rejection_reason": reason}


def _result(dispatch, *, native, winner, provisional_winner, decisions):
    return {
        "kind": "result",
        "dispatch": dispatch,
        "native": native,
        "winner": winner,
        "provisional_winner": provisional_winner,
        "promotion_status": "native" if winner == native else "pending_bh",
        "schedule": {"candidates": [native, winner] if winner != native else [native]},
        "candidates": [
            {"name": native, "effective_us": 10.0, "status": "ok"},
            {"name": winner, "effective_us": 9.0, "status": "ok"},
        ] if winner != native else [{"name": native, "effective_us": 10.0, "status": "ok"}],
        "ranking_decisions": decisions,
    }


def _write_jsonl(rows):
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".measurements.jsonl", delete=False, encoding="utf-8"
    )
    for row in rows:
        handle.write(json.dumps(row) + "\n")
    handle.close()
    return Path(handle.name)


AGREE_DECISION = [{
    "policy_name": "latency-v1", "policy_version": 1, "is_production": True,
    "predicted_winner": "candidate:v2",
    "candidates": [_candidate("native:v1", 10.0, "qualified"),
                   _candidate("candidate:v2", 9.0, "winner")],
}]
DISAGREE_DECISION = [{
    "policy_name": "latency-v1", "policy_version": 1, "is_production": True,
    "predicted_winner": "candidate:v2",
    "candidates": [_candidate("native:v1", 10.0, "near_tie_below_threshold"),
                   _candidate("candidate:v2", 9.5, "winner")],
}]


class ReadTests(unittest.TestCase):
    def test_reads_header_and_results(self):
        path = _write_jsonl([HEADER, _result(
            "d1", native="native:v1", winner="candidate:v2",
            provisional_winner="candidate:v2", decisions=AGREE_DECISION,
        )])
        header, results = rank_replay._read(path)
        self.assertEqual(header["kind"], "header")
        self.assertEqual(len(results), 1)

    def test_rejects_duplicate_header(self):
        path = _write_jsonl([HEADER, HEADER])
        with self.assertRaises(rank_replay.RankReplayError):
            rank_replay._read(path)

    def test_rejects_malformed_line(self):
        path = Path(tempfile.mkstemp(suffix=".jsonl")[1])
        path.write_text(json.dumps(HEADER) + "\nnot json\n", encoding="utf-8")
        with self.assertRaises(rank_replay.RankReplayError):
            rank_replay._read(path)


class BuildReportTests(unittest.TestCase):
    def test_agreement_and_disagreement_counted(self):
        results = [
            _result("d1", native="native:v1", winner="candidate:v2",
                    provisional_winner="candidate:v2", decisions=AGREE_DECISION),
            _result("d2", native="native:v1", winner="native:v1",
                    provisional_winner="native:v1", decisions=DISAGREE_DECISION),
        ]
        report = rank_replay.build_report(HEADER, results)
        bucket = report["policies"]["latency-v1"]
        self.assertEqual(bucket["total"], 2)
        self.assertEqual(bucket["agree"], 1)
        self.assertEqual(bucket["disagree"], 1)
        self.assertAlmostEqual(bucket["agreement_rate"], 0.5)
        self.assertEqual(len(report["disagreements"]), 1)
        self.assertEqual(report["disagreements"][0]["dispatch"], "d2")
        self.assertFalse(report["degraded_comparison"])

    def test_degraded_flag_when_provisional_winner_absent(self):
        result = _result("d1", native="native:v1", winner="candidate:v2",
                          provisional_winner="candidate:v2", decisions=AGREE_DECISION)
        del result["provisional_winner"]
        report = rank_replay.build_report(HEADER, [result])
        self.assertTrue(report["degraded_comparison"])


class VerifyParityTests(unittest.TestCase):
    def test_counts_matched_mismatched_excluded(self):
        matching = _result("d1", native="native:v1", winner="candidate:v2",
                            provisional_winner="candidate:v2", decisions=AGREE_DECISION)
        mismatching = _result("d2", native="native:v1", winner="native:v1",
                               provisional_winner="native:v1", decisions=DISAGREE_DECISION)
        no_provisional = _result("d3", native="native:v1", winner="native:v1",
                                  provisional_winner="native:v1", decisions=AGREE_DECISION)
        del no_provisional["provisional_winner"]

        report = rank_replay.verify_parity([matching, mismatching, no_provisional])
        self.assertEqual(report["matched"], 1)
        self.assertEqual(report["mismatched"], 1)
        self.assertEqual(report["excluded"], 1)
        self.assertEqual(report["mismatches"][0]["dispatch"], "d2")


class DispatchDetailTests(unittest.TestCase):
    def test_returns_full_candidate_list(self):
        result = _result("d1", native="native:v1", winner="candidate:v2",
                          provisional_winner="candidate:v2", decisions=AGREE_DECISION)
        detail = rank_replay.dispatch_detail([result], "d1")
        self.assertEqual(detail["dispatch"], "d1")
        self.assertEqual(len(detail["policies"]), 1)
        self.assertEqual(len(detail["policies"][0]["candidates"]), 2)

    def test_unknown_dispatch_raises(self):
        result = _result("d1", native="native:v1", winner="native:v1",
                          provisional_winner="native:v1", decisions=AGREE_DECISION)
        with self.assertRaises(rank_replay.RankReplayError):
            rank_replay.dispatch_detail([result], "missing")


PROTOTYPE_MODULE_SOURCE = '''
from bigcherry import ranking_policy as rp

SPEC = {"schema_version": rp.POLICY_SCHEMA_VERSION, "name": "always-native-v0", "version": 1}
SPEC["policy_hash"] = rp.policy_hash(SPEC)


def rank(native, candidates, config):
    ranked = [
        rp.RankedCandidate(name=c.name, effective_us=c.effective_us,
                            verdict="winner" if c.is_native else "rejected",
                            rejection_reason="" if c.is_native else "prototype always keeps native")
        for c in candidates
    ]
    return rp.PolicyDecision(policy_name=SPEC["name"], policy_version=SPEC["version"],
                             is_production=False, predicted_winner=native.name,
                             candidates=ranked)
'''


class MainSmokeTests(unittest.TestCase):
    def test_default_report_exit_zero(self):
        path = _write_jsonl([HEADER, _result(
            "d1", native="native:v1", winner="candidate:v2",
            provisional_winner="candidate:v2", decisions=AGREE_DECISION,
        )])
        self.assertEqual(rank_replay.main([str(path), "--json"]), 0)

    def test_verify_parity_exit_code_reflects_mismatches(self):
        matching = _write_jsonl([HEADER, _result(
            "d1", native="native:v1", winner="candidate:v2",
            provisional_winner="candidate:v2", decisions=AGREE_DECISION,
        )])
        self.assertEqual(rank_replay.main([str(matching), "--verify-parity"]), 0)

        mismatching = _write_jsonl([HEADER, _result(
            "d2", native="native:v1", winner="native:v1",
            provisional_winner="native:v1", decisions=DISAGREE_DECISION,
        )])
        self.assertEqual(rank_replay.main([str(mismatching), "--verify-parity"]), 1)

    def test_policy_module_prototype_runs_alongside_recorded(self):
        module_path = Path(tempfile.mkstemp(suffix=".py")[1])
        module_path.write_text(PROTOTYPE_MODULE_SOURCE, encoding="utf-8")
        data_path = _write_jsonl([HEADER, _result(
            "d1", native="native:v1", winner="candidate:v2",
            provisional_winner="candidate:v2", decisions=AGREE_DECISION,
        )])
        code = rank_replay.main([
            str(data_path), "--policy-module", str(module_path), "--json",
        ])
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()

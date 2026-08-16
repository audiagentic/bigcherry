"""Unit tests for tools/residency_gates.py (HI34 B1 gate evaluation).

Synthetic artifacts pin down the gate semantics that the real B1 evidence
rests on, in particular:
- sign conventions of hot_adv / cold_adv (a positive value means the named
  side is FASTER), and that a crossover requires BOTH sides material;
- sub-material margins, ties, and non-crossover flips do not fail gate 2;
- incomplete arms (poison/fatal/failed/run rejected/disabled) exclude only
  their own arm's observation from the pool;
- winner-missing candidates are reported, not treated as crossovers;
- artifact parsing is fail-closed on malformed / duplicate / unknown rows.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import residency_gates as rg  # noqa: E402


def header(flush_l2=0, flush_mb=256, pre_sample_mode=None):
    row = {
        "kind": "header",
        "artifact_version": 1,
        "source_revision": "rev",
        "manifest_hash": "man",
        "flush_l2": flush_l2,
        "flush_evict_mb": flush_mb,
    }
    if pre_sample_mode is not None:
        row["pre_sample_mode"] = pre_sample_mode
    return row


def result(signature, winner, candidates, reason="native retained"):
    return {
        "kind": "result",
        "signature": signature,
        "winner": winner,
        "reason": reason,
        "candidates": candidates,
    }


def cand(name, median_us, status="ok"):
    c = {"name": name, "status": status}
    if status == "ok":
        c["median_us"] = median_us
    return c


def write_artifact(tmp: Path, name: str, header_row, results):
    path = tmp / name
    lines = [json.dumps(header_row)] + [json.dumps(r) for r in results]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _one_result():
    return [result("s1", "w", [cand("w", 1.0)])]


class TestArmProvenance(unittest.TestCase):
    """HI65 pre-run requirement: declared arms must match headers fail-closed."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rg-arm-"))

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_arms(self, h1_mode=None, cold_mode=None, h2_mode=None, c_mb=256):
        p1 = write_artifact(
            self.tmp,
            "h1.jsonl",
            header(flush_l2=0 if h1_mode is None else 0, pre_sample_mode=h1_mode),
            _one_result(),
        )
        pc = write_artifact(
            self.tmp,
            "cold.jsonl",
            header(
                flush_l2=0 if cold_mode is None else 1,
                flush_mb=c_mb,
                pre_sample_mode=cold_mode,
            ),
            _one_result(),
        )
        p2 = write_artifact(
            self.tmp,
            "h2.jsonl",
            header(flush_l2=0, pre_sample_mode=h2_mode),
            _one_result(),
        )
        return p1, pc, p2

    def test_matching_arms_pass(self):
        p1, pc, p2 = self._write_arms("none", "evict", "none", c_mb=128)
        expected = {
            "h1": {"pre_sample_mode": "none", "flush_evict_mb": None},
            "cold": {"pre_sample_mode": "evict", "flush_evict_mb": 128},
            "h2": {"pre_sample_mode": "none", "flush_evict_mb": None},
        }
        report = rg.evaluate(p1, pc, p2, expected_arms=expected)
        self.assertEqual(report["header_check"]["status"], "ok")
        self.assertTrue(report["verdict"]["header_check_pass"])

    def test_missing_pre_sample_mode_fails(self):
        p1, pc, p2 = self._write_arms(None, None, None)  # legacy headers
        expected = {
            "cold": {"pre_sample_mode": "evict", "flush_evict_mb": 256}
        }
        report = rg.evaluate(p1, pc, p2, expected_arms=expected)
        self.assertEqual(report["header_check"]["status"], "fail")
        self.assertIn(
            "pre_sample_mode",
            report["header_check"]["mismatches"][0],
        )

    def test_evict_vs_evict_rewarm_detected_despite_identical_flush_l2(self):
        # Both arms carry the legacy flush_l2=1 mirror; only the resolved
        # mode string can tell them apart.
        p1, pc, p2 = self._write_arms("none", "evict", "none")
        expected = {
            "cold": {"pre_sample_mode": "evict_rewarm", "flush_evict_mb": 256}
        }
        report = rg.evaluate(p1, pc, p2, expected_arms=expected)
        self.assertEqual(report["header_check"]["status"], "fail")
        self.assertTrue(
            any("pre_sample_mode" in m for m in report["header_check"]["mismatches"])
        )

    def test_flush_mb_mismatch_fails(self):
        p1, pc, p2 = self._write_arms("none", "evict", "none", c_mb=128)
        expected = {
            "cold": {"pre_sample_mode": "evict", "flush_evict_mb": 256}
        }
        report = rg.evaluate(p1, pc, p2, expected_arms=expected)
        self.assertEqual(report["header_check"]["status"], "fail")
        self.assertTrue(
            any("flush_evict_mb" in m for m in report["header_check"]["mismatches"])
        )

    def test_source_revision_differs_across_arms_fails(self):
        p1, pc, p2 = self._write_arms("none", "evict", "none")
        bad = header(flush_l2=0, pre_sample_mode="none")
        bad["source_revision"] = "other-rev"
        p3 = write_artifact(self.tmp, "h2b.jsonl", bad, _one_result())
        expected = {
            "cold": {"pre_sample_mode": "evict", "flush_evict_mb": 256}
        }
        report = rg.evaluate(p1, pc, p3, expected_arms=expected)
        self.assertEqual(report["header_check"]["status"], "fail")
        self.assertTrue(
            any("source_revision" in m for m in report["header_check"]["mismatches"])
        )

    def test_manifest_hash_missing_fails(self):
        p1, pc, p2 = self._write_arms("none", "evict", "none")
        bad = header(flush_l2=0, pre_sample_mode="none")
        del bad["manifest_hash"]
        p3 = write_artifact(self.tmp, "h2b.jsonl", bad, _one_result())
        expected = {
            "cold": {"pre_sample_mode": "evict", "flush_evict_mb": 256}
        }
        report = rg.evaluate(p1, pc, p3, expected_arms=expected)
        self.assertEqual(report["header_check"]["status"], "fail")
        self.assertTrue(
            any("manifest_hash" in m for m in report["header_check"]["mismatches"])
        )

    def test_no_declared_arms_skips_check(self):
        p1, pc, p2 = self._write_arms(None, None, None)
        report = rg.evaluate(p1, pc, p2)
        self.assertEqual(report["header_check"]["status"], "skipped")
        self.assertTrue(report["verdict"]["header_check_pass"])

    def test_cli_exits_2_on_provenance_mismatch(self):
        p1, pc, p2 = self._write_arms("none", "evict", "none")
        rc = rg.main(
            [str(p1), str(pc), str(p2), "--arm", "cold=evict_rewarm@256"]
        )
        self.assertEqual(rc, 2)

    def test_cli_exits_0_on_matching_declaration(self):
        p1, pc, p2 = self._write_arms("none", "evict", "none")
        rc = rg.main([str(p1), str(pc), str(p2), "--arm", "cold=evict@256"])
        self.assertEqual(rc, 0)

    def test_parse_arm_spec_valid(self):
        name, spec = rg.parse_arm_spec("cold=evict@128")
        self.assertEqual(name, "cold")
        self.assertEqual(spec, {"pre_sample_mode": "evict", "flush_evict_mb": 128})
        name, spec = rg.parse_arm_spec("h1=none")
        self.assertEqual(spec, {"pre_sample_mode": "none", "flush_evict_mb": None})

    def test_parse_arm_spec_invalid(self):
        for bad in ("evict@256", "cold=", "cold=bogus", "cold=evict@x", "cold=evict@-4"):
            with self.assertRaises(ValueError, msg=bad):
                rg.parse_arm_spec(bad)


class TestArtifactLoading(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_loads_header_and_results(self):
        p = write_artifact(
            self.tmp, "a.jsonl", header(), [result("s1", "w", [cand("w", 1.0)])]
        )
        h, r = rg.load_artifact(p)
        self.assertEqual(h["flush_l2"], 0)
        self.assertIn("s1", r)

    def test_malformed_line_raises(self):
        p = self.tmp / "bad.jsonl"
        p.write_text(json.dumps(header()) + "\n{not json\n", encoding="utf-8")
        with self.assertRaises(rg.ResidencyGateError):
            rg.load_artifact(p)

    def test_duplicate_signature_raises(self):
        p = write_artifact(
            self.tmp,
            "dup.jsonl",
            header(),
            [result("s", "w", []), result("s", "w", [])],
        )
        with self.assertRaises(rg.ResidencyGateError):
            rg.load_artifact(p)

    def test_unknown_kind_raises(self):
        p = write_artifact(
            self.tmp,
            "u.jsonl",
            header(),
            [dict(result("s", "w", []), kind="observation")],
        )
        with self.assertRaises(rg.ResidencyGateError):
            rg.load_artifact(p)

    def test_missing_header_raises(self):
        p = write_artifact(self.tmp, "nh.jsonl", None, [result("s", "w", [])])
        # header None -> json.dumps(None) == 'null' -> unknown kind path; use a
        # file with only result rows instead:
        p.write_text(json.dumps(result("s", "w", [])) + "\n", encoding="utf-8")
        with self.assertRaises(rg.ResidencyGateError):
            rg.load_artifact(p)


class TestCompleteness(unittest.TestCase):
    def test_completed_reasons(self):
        self.assertTrue(rg.is_completed({"reason": "native retained"}))
        self.assertTrue(
            rg.is_completed(
                {
                    "reason": "native retained (fresh confirmation rejected provisional winner)"
                }
            )
        )

    def test_incomplete_reasons_excluded(self):
        for reason in (
            "poisoned by twin mismatch",
            "native timing unstable; run rejected",
            "candidate fatal error",
            "run failed mid-screen",
            "tuner disabled",
        ):
            self.assertFalse(rg.is_completed({"reason": reason}), reason)


class TestCandidateMedians(unittest.TestCase):
    def test_only_ok_positive_medians(self):
        r = result(
            "s",
            "w",
            [cand("a", 1.0), cand("b", 2.0, status="rejected"), cand("c", -5.0)],
        )
        self.assertEqual(rg.candidate_medians(r), {"a": 1.0})

    def test_ok_candidate_without_name_raises(self):
        r = result("s", "w", [{"status": "ok", "median_us": 1.0}])
        with self.assertRaises(rg.ResidencyGateError):
            rg.candidate_medians(r)


class TestGate1(unittest.TestCase):
    def test_flips_counted_over_completed_intersection_only(self):
        h1 = {
            "s1": result("s1", "a", []),
            "s2": result("s2", "b", []),
            "s3": result("s3", "c", [], reason="native timing unstable; run rejected"),
        }
        h2 = {
            "s1": result("s1", "a", []),
            "s2": result("s2", "d", []),
            "s3": result("s3", "x", []),
        }
        report = rg.gate1_hot_repeatability(h1, h2)
        self.assertEqual(report.pairs, 2)  # s3 excluded (incomplete in h1)
        self.assertEqual([f["signature"] for f in report.flips], ["s2"])
        self.assertAlmostEqual(report.flip_rate_pct, 50.0)


class TestGate2Crossover(unittest.TestCase):
    def _arms(
        self,
        hot_winner="blas:native:v1",
        cold_winner="mmvf:native:v1",
        hot_med=(100.0, 110.0),
        cold_med=(130.0, 105.0),
    ):
        """hot: hot_winner faster by 10% (hot_adv +10%);
        cold: cold_winner faster by ~24% (cold_adv +24%)."""

        def mk(winner, hw, cw):
            return result("s", winner, [cand(hw, hw), cand(cw, cw)])

        h1 = {"s": mk(hot_winner, hot_med[0], cold_med[0])}
        cold = {"s": mk(cold_winner, hot_med[0] * 1.0 + 30, cold_med[1])}
        # rebuild with explicit per-context medians for clarity:
        h1 = {
            "s": result(
                "s", hot_winner, [cand(hot_winner, 100.0), cand(cold_winner, 110.0)]
            )
        }
        h2 = {
            "s": result(
                "s", hot_winner, [cand(hot_winner, 101.0), cand(cold_winner, 112.0)]
            )
        }
        cold = {
            "s": result(
                "s", cold_winner, [cand(hot_winner, 130.0), cand(cold_winner, 105.0)]
            )
        }
        return h1, cold, h2

    def test_true_crossover_detected(self):
        h1, cold, h2 = self._arms()
        details = rg.gate2_material_reversals(h1, cold, h2)
        self.assertEqual(len(details), 1)
        d = details[0]
        self.assertTrue(d.replicated)
        self.assertTrue(d.crosses_any_hot_arm)
        for info in d.per_arm:
            # hot winner 100 vs 110 -> hot_adv +10%; cold 130 vs 105 -> cold_adv +23.8%
            self.assertGreater(info["hot_adv"], 0.05)
            self.assertGreater(info["cold_adv"], 0.05)
            self.assertTrue(info["crossover"])

    def test_no_crossover_when_hot_winner_stays_faster_on_cold(self):
        # cold context: hot winner still fastest -> selection flip without
        # median crossover (the four non-crossover B1 flips).
        h1 = {
            "s": result(
                "s",
                "blas:native:v1",
                [cand("blas:native:v1", 100.0), cand("mmvf:native:v1", 120.0)],
            )
        }
        h2 = {
            "s": result(
                "s",
                "blas:native:v1",
                [cand("blas:native:v1", 101.0), cand("mmvf:native:v1", 121.0)],
            )
        }
        cold = {
            "s": result(
                "s",
                "mmvf:native:v1",
                [cand("blas:native:v1", 98.0), cand("mmvf:native:v1", 104.0)],
            )
        }
        details = rg.gate2_material_reversals(h1, cold, h2)
        self.assertEqual(len(details), 1)
        d = details[0]
        self.assertTrue(d.replicated)
        self.assertFalse(d.crosses_any_hot_arm)
        # hot_adv positive (hot winner faster on hot) but cold_adv NEGATIVE
        for info in d.per_arm:
            self.assertGreater(info["hot_adv"], 0.05)
            self.assertLess(info["cold_adv"], -0.05)
            self.assertFalse(info["crossover"])

    def test_sub_material_margins_do_not_cross(self):
        h1 = {"s": result("s", "a:v1", [cand("a:v1", 100.0), cand("b:v1", 103.0)])}
        h2 = {"s": result("s", "a:v1", [cand("a:v1", 100.0), cand("b:v1", 103.0)])}
        cold = {"s": result("s", "b:v1", [cand("a:v1", 104.0), cand("b:v1", 102.5)])}
        details = rg.gate2_material_reversals(h1, cold, h2)
        self.assertFalse(details[0].crosses_any_hot_arm)

    def test_tie_is_not_a_crossover(self):
        h1 = {"s": result("s", "a:v1", [cand("a:v1", 100.0), cand("b:v1", 120.0)])}
        h2 = {"s": result("s", "a:v1", [cand("a:v1", 100.0), cand("b:v1", 120.0)])}
        cold = {"s": result("s", "b:v1", [cand("a:v1", 120.0), cand("b:v1", 120.0)])}
        details = rg.gate2_material_reversals(h1, cold, h2)
        self.assertFalse(details[0].crosses_any_hot_arm)

    def test_missing_median_reported_not_crossover(self):
        h1 = {"s": result("s", "a:v1", [cand("a:v1", 100.0), cand("b:v1", 120.0)])}
        h2 = {"s": result("s", "a:v1", [cand("a:v1", 100.0), cand("b:v1", 120.0)])}
        # b (cold winner) rejected in the cold arm: no median to cross with.
        cold = {
            "s": result(
                "s", "b:v1", [cand("a:v1", 95.0), cand("b:v1", 90.0, status="rejected")]
            )
        }
        details = rg.gate2_material_reversals(h1, cold, h2)
        d = details[0]
        self.assertFalse(d.crosses_any_hot_arm)
        self.assertTrue(
            all(info["reason"].startswith("missing-median") for info in d.per_arm)
        )

    def test_incomplete_arm_excludes_signature(self):
        h1 = {"s": result("s", "a:v1", [cand("a:v1", 100.0), cand("b:v1", 120.0)])}
        h2 = {"s": result("s", "a:v1", [cand("a:v1", 100.0), cand("b:v1", 120.0)])}
        cold = {
            "s": result(
                "s",
                "b:v1",
                [cand("a:v1", 130.0), cand("b:v1", 105.0)],
                reason="native timing unstable; run rejected",
            )
        }
        self.assertEqual(rg.gate2_material_reversals(h1, cold, h2), [])

    def test_non_replicated_flip_still_listed_but_not_replicated(self):
        # cold agrees with h2, differs only from h1.
        h1 = {"s": result("s", "a:v1", [cand("a:v1", 100.0), cand("b:v1", 120.0)])}
        h2 = {"s": result("s", "b:v1", [cand("a:v1", 100.0), cand("b:v1", 95.0)])}
        cold = {"s": result("s", "b:v1", [cand("a:v1", 130.0), cand("b:v1", 105.0)])}
        details = rg.gate2_material_reversals(h1, cold, h2)
        self.assertEqual(len(details), 1)
        self.assertFalse(details[0].replicated)


class TestGate2HardGateAggregation(unittest.TestCase):
    """The locked hard gate requires replication AND crossover vs BOTH arms.

    These tests pin the distinction that a one-hot-arm-only crossover or a
    non-replicated flip must stay in diagnostics and never fail Gate 2, even
    when a material median crossover exists on one side.
    """

    def test_replicated_crossover_single_arm_only_not_survivor(self):
        # Constructed so the crossover is material vs h1 but NOT vs h2: on h2
        # the hot winner's cold-context median stays below the cold winner's.
        h1 = {"s": result("s", "a:v1", [cand("a:v1", 100.0), cand("b:v1", 110.0)])}
        h2 = {"s": result("s", "a:v1", [cand("a:v1", 100.0), cand("b:v1", 103.0)])}
        # cold: a=106, b=100 -> cold winner b beats a by ~5.9% (material).
        # vs h1: hot_adv=(110-100)/100=+10%, cold_adv=(106-100)/100=+6% -> CROSS
        # vs h2: hot_adv=(103-100)/100=+3% <5% -> NO cross
        cold = {"s": result("s", "b:v1", [cand("a:v1", 106.0), cand("b:v1", 100.0)])}
        details = rg.gate2_material_reversals(h1, cold, h2)
        d = details[0]
        self.assertTrue(d.replicated)
        self.assertTrue(d.crosses_any_hot_arm)  # diagnostic: one arm crosses
        self.assertFalse(d.crosses_both_hot_arms)
        self.assertFalse(d.hard_gate_survivor)  # must NOT fail the hard gate

    def test_non_replicated_material_crossover_not_survivor(self):
        # cold agrees with h2 (b); differs only from h1 (a). The b-vs-a
        # comparison is a material crossover on h1's side, but without
        # replication this cannot be a hard-gate survivor.
        h1 = {"s": result("s", "a:v1", [cand("a:v1", 100.0), cand("b:v1", 115.0)])}
        h2 = {"s": result("s", "b:v1", [cand("a:v1", 100.0), cand("b:v1", 92.0)])}
        cold = {"s": result("s", "b:v1", [cand("a:v1", 135.0), cand("b:v1", 96.0)])}
        details = rg.gate2_material_reversals(h1, cold, h2)
        d = details[0]
        self.assertFalse(d.replicated)
        # vs h1: hot_adv=(115-100)/100=+15%, cold_adv=(135-96)/96=+40.6% -> cross
        self.assertTrue(d.crosses_any_hot_arm)
        self.assertFalse(d.hard_gate_survivor)  # not replicated -> no survivor

    def test_replicated_crossover_both_arms_is_survivor(self):
        h1 = {"s": result("s", "a:v1", [cand("a:v1", 100.0), cand("b:v1", 110.0)])}
        h2 = {"s": result("s", "a:v1", [cand("a:v1", 101.0), cand("b:v1", 112.0)])}
        cold = {"s": result("s", "b:v1", [cand("a:v1", 130.0), cand("b:v1", 105.0)])}
        details = rg.gate2_material_reversals(h1, cold, h2)
        d = details[0]
        self.assertTrue(d.replicated)
        self.assertTrue(d.crosses_both_hot_arms)
        self.assertTrue(d.hard_gate_survivor)  # Gate 2 FAIL on this row


class TestEvaluateEndToEnd(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_full_report_shape_and_verdicts(self):
        # one stable signature, one true crossover signature.
        def mk(sig, winner, hw, cw, hot=True):
            if hot:
                return result(sig, "a:v1", [cand("a:v1", hw), cand("b:v1", cw)])
            return result(sig, "b:v1", [cand("a:v1", hw), cand("b:v1", cw)])

        h1 = [mk("stable", "a:v1", 100.0, 90.0), mk("cross", "a:v1", 100.0, 110.0)]
        h2 = [mk("stable", "a:v1", 100.5, 90.4), mk("cross", "a:v1", 101.0, 112.0)]
        cold_rows = [
            result("stable", "a:v1", [cand("a:v1", 101.0), cand("b:v1", 91.0)])
        ]
        # cross: a hot winner 100/110 on hot; on cold a=130, b=105 -> crossover
        cold_rows.append(
            result("cross", "b:v1", [cand("a:v1", 130.0), cand("b:v1", 105.0)])
        )

        p1 = write_artifact(self.tmp, "h1.jsonl", header(flush_l2=0), h1)
        pc = write_artifact(self.tmp, "cold.jsonl", header(flush_l2=1), cold_rows)
        p2 = write_artifact(self.tmp, "h2.jsonl", header(flush_l2=0), h2)

        report = rg.evaluate(p1, pc, p2)
        self.assertEqual(report["headers"]["cold"]["flush_l2"], 1)
        self.assertEqual(report["gate1"]["pairs"], 2)
        self.assertEqual(report["gate1"]["flips"], [])
        self.assertFalse(report["verdict"]["gate2_hard_pass"])
        survivors = report["gate2"]["survivors"]
        self.assertEqual([s["signature"] for s in survivors], ["cross"])
        # 'stable' has no winner difference at all, so it never appears in
        # gate2 details (winner_differences covers only differing winners).
        self.assertNotIn("stable", report["gate2"]["replicated"])


if __name__ == "__main__":
    unittest.main()

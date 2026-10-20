"""RAD01 PR 7: radiance KL results and component activation markers become campaign evidence."""

import dataclasses
import re
import unittest

from bigcherry.core import components, engines, paths
from bigcherry.experiment import engine_quality as eq

# --kld-out documents ('all' section) shaped after radiance KL runs on the 27B, two XTX (kqa2 and kqg2, 2026-10-12).
_GOOD = {"all": {"positions": 42761,
                 "kld": {"mean": 0.003533, "median": 0.001621, "p90": 0.008, "p95": 0.012, "p99": 0.02516, "p99_9": 0.1265,
                         "p99_99": 0.7817, "max": 1.3892},
                 "top1_agreement": 0.96801, "ppl": {"candidate": 8.5583, "reference": 8.5610, "ratio": 0.99968}},
         "by_source": {"docs": {"kld": {"mean": 0.004}}}}
_BROKEN = {"all": {"kld": {"mean": 6.499024, "median": 6.366023, "p99": 14.04798, "p99_9": 16.4834, "p99_99": 18.3643,
                           "max": 20.6620},
                   "top1_agreement": 0.07322, "ppl": {"candidate": 4903.6748, "reference": 8.5610, "ratio": 572.8}}}
_LIMITS = eq.QualityThresholds(
    max_kld_mean_increase=0.002, max_kld_median_increase=0.001, max_kld_p99_increase=0.02, max_kld_p999_increase=0.1,
    max_top1_drop_points=0.5, max_ppl_ratio_increase=0.005)


class KldTests(unittest.TestCase):
    def test_parses_one_result(self):
        result = eq.parse_radiance_kld_json(_GOOD)
        self.assertEqual(result.kld_mean, 0.003533)
        self.assertEqual(result.kld_median, 0.001621)
        self.assertEqual(result.kld_p99, 0.02516)
        self.assertEqual(result.kld_p999, 0.1265)
        self.assertEqual(result.top1_percent, 96.801)
        self.assertAlmostEqual(result.ppl_ratio, 8.5583 / 8.5610)

    def test_refuses_incomplete_documents(self):
        for document in ({}, {"all": {}}, {"all": {**_GOOD["all"], "kld": {**_GOOD["all"]["kld"], "p99_9": None}}},
                         {"all": {**_GOOD["all"], "ppl": {"candidate": 8.5583}}},
                         {"all": {k: v for k, v in _GOOD["all"].items() if k != "top1_agreement"}}):
            with self.subTest(document=document), self.assertRaises(eq.QualityParseError):
                eq.parse_radiance_kld_json(document)

    def test_gate_passes_a_neutral_arm_and_fails_the_broken_delta_net(self):
        control = eq.parse_radiance_kld_json(_GOOD)
        self.assertEqual(eq.quality_gate(control, control, _LIMITS), (True, []))
        passed, problems = eq.quality_gate(control, eq.parse_radiance_kld_json(_BROKEN), _LIMITS)
        self.assertFalse(passed)
        self.assertEqual(len(problems), 6)

    def test_gate_names_a_tail_only_regression(self):
        control = eq.parse_radiance_kld_json(_GOOD)
        tail = dataclasses.replace(control, kld_p999=control.kld_p999 + 0.5)
        passed, problems = eq.quality_gate(control, tail, _LIMITS)
        self.assertFalse(passed)
        self.assertEqual(len(problems), 1)
        self.assertIn("P99.9", problems[0])


class TokenKldTests(unittest.TestCase):
    def test_known_distribution(self):
        kl = [float(i) for i in range(1001)]  # 0..1000: median 500, P99 990, P99.9 999
        agrees = [i % 4 != 0 for i in range(1001)]
        stats = eq.token_kld(kl, agrees)
        self.assertEqual(stats.count, 1001)
        self.assertAlmostEqual(stats.mean, 500.0)
        self.assertAlmostEqual(stats.median, 500.0)
        self.assertAlmostEqual(stats.p99, 990.0)
        self.assertAlmostEqual(stats.p999, 999.0)
        self.assertAlmostEqual(stats.top1_percent, 100.0 * 750 / 1001)

    def test_subset_excludes_tokens(self):
        kl = [0.0, 0.0, 0.0, 100.0]
        agrees = [True, True, True, False]
        text = eq.token_kld(kl, agrees, [True, True, True, False])
        self.assertEqual((text.count, text.mean, text.p99, text.top1_percent), (3, 0.0, 0.0, 100.0))
        self.assertEqual(eq.token_kld(kl, agrees).top1_percent, 75.0)

    def test_rejects_mismatched_or_empty_input(self):
        with self.assertRaises(ValueError):
            eq.token_kld([1.0], [True, False])
        with self.assertRaises(ValueError):
            eq.token_kld([1.0], [True], [False])


class ActivationTests(unittest.TestCase):
    def setUp(self):
        self.libr3 = components.load_all(paths.REPO_ROOT, engines.RADIANCE)["libr3"]

    def test_declared_markers_are_printed_by_the_source(self):
        """A marker is a line in the served process's logs, so it may be printed by any component's source (libr3 declares
        R3_MOE_FP16, which the moe-remote helper reports)."""
        declared = components.load_all(paths.REPO_ROOT, engines.RADIANCE)
        source = chr(10).join(
            p.read_text(encoding="utf-8", errors="replace")
            for component in declared.values()
            for p in component.root.rglob("*") if p.is_file() and p.suffix in {".cpp", ".hip", ".rw", ".h"}
        )
        for component in declared.values():
            for setting in component.settings.values():
                if setting.activation:
                    literal = re.sub(r"[\^$]", "", setting.activation).replace(chr(92), "")
                    self.assertIn(literal.split(" ")[0], source, setting.name)

    def test_executed_only_when_the_subject_logs_the_marker_and_the_control_does_not(self):
        log = "libr3: R3_AR_WIRE4 active for M>=512\n"
        self.assertEqual(eq.setting_activation(self.libr3, "R3_AR_WIRE4", control_log="", subject_log=log).status,
                         "executed")
        self.assertEqual(eq.setting_activation(self.libr3, "R3_AR_WIRE4", control_log=log, subject_log=log).status,
                         "not_executed")
        self.assertEqual(eq.setting_activation(self.libr3, "R3_AR_WIRE4", control_log="", subject_log="").status,
                         "not_executed")

    def test_a_setting_without_a_marker_is_unobservable(self):
        self.assertEqual(
            eq.setting_activation(self.libr3, "R3_ATTN_NATIVE", control_log="", subject_log="").status, "unobservable")


if __name__ == "__main__":
    unittest.main()

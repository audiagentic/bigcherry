"""RAD01 PR 7: radiance KL results and component activation markers become campaign evidence."""

import dataclasses
import re
import unittest

from bigcherry.core import components, engines, paths
from bigcherry.experiment import engine_quality as eq

# Verbatim lines from radiance KL runs on the 27B, two XTX (runs kqa2 and kqg2, 2026-10-12).
_GOOD = (
    "I kld: KLD mean 0.003533  median 0.001621  p99 0.02516  p99.9 0.1265  p99.99 0.7817  max 1.3892\n"
    "I kld: top-1 agreement 96.801%   PPL 8.5583 vs reference 8.5610 (x0.9997)\n"
)
_BROKEN = (
    "I kld: KLD mean 6.499024  median 6.366023  p99 14.04798  p99.9 16.4834  p99.99 18.3643  max 20.6620\n"
    "I kld: top-1 agreement 7.322%   PPL 4903.6748 vs reference 8.5610 (x572.7913)\n"
)
_LIMITS = eq.QualityThresholds(
    max_kld_mean_increase=0.002, max_kld_median_increase=0.001, max_kld_p99_increase=0.02, max_kld_p999_increase=0.1,
    max_top1_drop_points=0.5, max_ppl_ratio_increase=0.005)


class KldTests(unittest.TestCase):
    def test_parses_one_result(self):
        result = eq.parse_radiance_kld(_GOOD)
        self.assertEqual(result.kld_mean, 0.003533)
        self.assertEqual(result.kld_median, 0.001621)
        self.assertEqual(result.kld_p99, 0.02516)
        self.assertEqual(result.kld_p999, 0.1265)
        self.assertEqual(result.top1_percent, 96.801)
        self.assertAlmostEqual(result.ppl_ratio, 8.5583 / 8.5610)

    def test_refuses_zero_or_two_results(self):
        with self.assertRaises(eq.QualityParseError):
            eq.parse_radiance_kld("no kld here\n")
        with self.assertRaises(eq.QualityParseError):
            eq.parse_radiance_kld(_GOOD + _GOOD)

    def test_gate_passes_a_neutral_arm_and_fails_the_broken_delta_net(self):
        control = eq.parse_radiance_kld(_GOOD)
        self.assertEqual(eq.quality_gate(control, control, _LIMITS), (True, []))
        passed, problems = eq.quality_gate(control, eq.parse_radiance_kld(_BROKEN), _LIMITS)
        self.assertFalse(passed)
        self.assertEqual(len(problems), 6)

    def test_gate_names_a_tail_only_regression(self):
        control = eq.parse_radiance_kld(_GOOD)
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
        source = "\n".join(
            p.read_text(encoding="utf-8", errors="replace")
            for p in self.libr3.root.rglob("*") if p.is_file() and p.suffix in {".cpp", ".hip", ".rw", ".h"}
        )
        for setting in self.libr3.settings.values():
            if setting.activation:
                literal = re.sub(r"[\^$]", "", setting.activation).replace("\\", "")
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

"""tools/bigcherry/experiment/perplexity.py -- the shared real-perplexity
comparison primitive extracted from RD17's correctness producer so RD13
can reuse it without duplication. Hardware-free via a faked subprocess.run."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.experiment import perplexity as ppl  # noqa: E402


def _completed(returncode: int, stdout: str = "", stderr: str = ""):
    result = MagicMock()
    result.returncode = returncode
    result.stdout = stdout
    result.stderr = stderr
    return result


class RunPerplexityTests(unittest.TestCase):
    def test_parses_the_final_estimate_line(self):
        stdout = "warmup\nFinal estimate: PPL = 12.3456 +/- 0.05678\nmore\n"

        def runner(argv, **kwargs):
            return _completed(0, stdout=stdout)

        run = ppl.run_perplexity(
            Path("binary"), model=Path("m.gguf"), corpus=Path("corpus.txt"), runner=runner,
        )
        self.assertAlmostEqual(run.ppl, 12.3456)
        self.assertAlmostEqual(run.uncertainty, 0.05678)

    def test_nonzero_exit_fails_closed(self):
        def runner(argv, **kwargs):
            return _completed(1, stderr="boom")

        with self.assertRaises(ppl.PerplexityError):
            ppl.run_perplexity(
                Path("binary"), model=Path("m.gguf"), corpus=Path("corpus.txt"), runner=runner,
            )

    def test_missing_ppl_line_fails_closed(self):
        def runner(argv, **kwargs):
            return _completed(0, stdout="no ppl line\n")

        with self.assertRaises(ppl.PerplexityError):
            ppl.run_perplexity(
                Path("binary"), model=Path("m.gguf"), corpus=Path("corpus.txt"), runner=runner,
            )


class PerplexityComparisonTests(unittest.TestCase):
    def test_within_combined_uncertainty_is_ok(self):
        comparison = ppl.PerplexityComparison(
            subject=ppl.PerplexityRun(10.00, 0.05, "", ""),
            control=ppl.PerplexityRun(10.02, 0.05, "", ""),
        )
        self.assertTrue(comparison.ok)

    def test_far_outside_combined_uncertainty_fails(self):
        comparison = ppl.PerplexityComparison(
            subject=ppl.PerplexityRun(10.0, 0.01, "", ""),
            control=ppl.PerplexityRun(11.0, 0.01, "", ""),
        )
        self.assertFalse(comparison.ok)


class RequirePplEqualityTests(unittest.TestCase):
    def test_passes_when_within_tolerance(self):
        def runner(argv, **kwargs):
            return _completed(0, stdout="Final estimate: PPL = 8.0000 +/- 0.0100\n")

        comparison = ppl.require_ppl_equality(
            subject_binary=Path("subj"), control_binary=Path("ctrl"),
            model=Path("m.gguf"), corpus=Path("c.txt"), runner=runner,
        )
        self.assertTrue(comparison.ok)

    def test_raises_with_specific_reason_on_mismatch(self):
        calls = {"n": 0}

        def runner(argv, **kwargs):
            calls["n"] += 1
            value = "8.0000" if calls["n"] == 1 else "9.0000"
            return _completed(0, stdout=f"Final estimate: PPL = {value} +/- 0.0100\n")

        with self.assertRaises(ppl.PerplexityError) as ctx:
            ppl.require_ppl_equality(
                subject_binary=Path("subj"), control_binary=Path("ctrl"),
                model=Path("m.gguf"), corpus=Path("c.txt"), runner=runner,
            )
        self.assertIn("ppl_equality failed", str(ctx.exception))


class ComparisonToDictTests(unittest.TestCase):
    def test_full_shape(self):
        comparison = ppl.PerplexityComparison(
            subject=ppl.PerplexityRun(10.0, 0.05, "", ""),
            control=ppl.PerplexityRun(10.02, 0.05, "", ""),
        )
        doc = ppl.comparison_to_dict(comparison)
        self.assertTrue(doc["ok"])
        self.assertIn("sigma", doc)


if __name__ == "__main__":
    unittest.main()

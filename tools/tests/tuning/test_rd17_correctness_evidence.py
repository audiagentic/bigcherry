"""RD17 correctness producer's control-transform and PPL-comparison logic,
hardware-free via a faked subprocess.run and a real temp directory for the
checked-replace control edit -- no real HIP hardware or a compiled
llama-perplexity binary needed for this layer's own correctness
(materialize_rd17_variants' worktree mechanics are patch_source_isolation's
responsibility, same division of labor as RD08's own test file)."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

_tools_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_tools_root))
_repo_root = _tools_root.parent
_rd17_path = (
    _repo_root / "patches" / "1207_rd17_moe_topk_down_fold" / "validation" / "rd17_correctness.py"
)
_rd17_spec = importlib.util.spec_from_file_location("rd17_package_validation", _rd17_path)
if _rd17_spec is None or _rd17_spec.loader is None:
    raise ImportError(f"cannot load RD17 package validation: {_rd17_path}")
rd17 = importlib.util.module_from_spec(_rd17_spec)
sys.modules[_rd17_spec.name] = rd17
_rd17_spec.loader.exec_module(rd17)


def _completed(returncode: int, stdout: str = "", stderr: str = ""):
    result = MagicMock()
    result.returncode = returncode
    result.stdout = stdout
    result.stderr = stderr
    return result


class ApplyNoFusionControlTests(unittest.TestCase):
    def _write_source(self, tmp: Path, content: str) -> None:
        target = tmp / "ggml" / "src" / "ggml-cuda" / "ggml-cuda.cu"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def test_reverts_the_detection_block_to_the_bare_anchor(self):
        tmp = Path(tempfile.mkdtemp())
        self._write_source(tmp, "before\n" + rd17._DETECT_NEW + "\nafter\n")
        rd17.apply_no_fusion_control(tmp)
        text = (tmp / "ggml/src/ggml-cuda/ggml-cuda.cu").read_text(encoding="utf-8")
        self.assertNotIn("x_scale_channel_dst = true", text)
        self.assertIn(rd17._DETECT_ANCHOR, text)

    def test_missing_block_fails_closed(self):
        tmp = Path(tempfile.mkdtemp())
        self._write_source(tmp, "no detection block here at all\n")
        with self.assertRaises(rd17.Rd17CorrectnessError):
            rd17.apply_no_fusion_control(tmp)

    def test_duplicated_block_fails_closed(self):
        tmp = Path(tempfile.mkdtemp())
        self._write_source(tmp, rd17._DETECT_NEW + rd17._DETECT_NEW)
        with self.assertRaises(rd17.Rd17CorrectnessError):
            rd17.apply_no_fusion_control(tmp)

    def test_path_escape_rejected(self):
        tmp = Path(tempfile.mkdtemp())
        # No file written -- the escape check must fire before any read.
        evil_edits = (
            (Path("../../etc/passwd"), rd17._DETECT_NEW, rd17._DETECT_ANCHOR),
        )
        original = rd17._CONTROL_EDITS
        rd17._CONTROL_EDITS = evil_edits
        try:
            with self.assertRaises(rd17.Rd17CorrectnessError):
                rd17.apply_no_fusion_control(tmp)
        finally:
            rd17._CONTROL_EDITS = original


class RunPerplexityTests(unittest.TestCase):
    def test_parses_the_final_estimate_line(self):
        stdout = "some warmup output\nFinal estimate: PPL = 12.3456 +/- 0.05678\nmore output\n"

        def runner(argv, **kwargs):
            return _completed(0, stdout=stdout)

        run = rd17.run_perplexity(
            Path("binary"), model=Path("m.gguf"), corpus=Path("corpus.txt"), runner=runner,
        )
        self.assertAlmostEqual(run.ppl, 12.3456)
        self.assertAlmostEqual(run.uncertainty, 0.05678)

    def test_nonzero_exit_fails_closed(self):
        def runner(argv, **kwargs):
            return _completed(1, stderr="boom")

        with self.assertRaises(rd17.Rd17CorrectnessError):
            rd17.run_perplexity(
                Path("binary"), model=Path("m.gguf"), corpus=Path("corpus.txt"), runner=runner,
            )

    def test_missing_ppl_line_fails_closed(self):
        def runner(argv, **kwargs):
            return _completed(0, stdout="no ppl line in here\n")

        with self.assertRaises(rd17.Rd17CorrectnessError):
            rd17.run_perplexity(
                Path("binary"), model=Path("m.gguf"), corpus=Path("corpus.txt"), runner=runner,
            )

    def test_command_shape(self):
        seen = {}

        def runner(argv, **kwargs):
            seen["argv"] = argv
            return _completed(0, stdout="Final estimate: PPL = 1.0 +/- 0.1\n")

        rd17.run_perplexity(
            Path("bin"), model=Path("m.gguf"), corpus=Path("c.txt"), ctx_size=4096, runner=runner,
        )
        argv = seen["argv"]
        self.assertEqual(argv[0], "bin")
        self.assertIn("-m", argv)
        self.assertIn("m.gguf", argv)
        self.assertIn("-f", argv)
        self.assertIn("c.txt", argv)
        self.assertIn("-c", argv)
        self.assertIn("4096", argv)


class Rd17PplComparisonTests(unittest.TestCase):
    def test_identical_ppl_is_ok(self):
        comparison = rd17.Rd17PplComparison(
            subject=rd17.PerplexityRun(10.0, 0.01, "", ""),
            control=rd17.PerplexityRun(10.0, 0.01, "", ""),
        )
        self.assertTrue(comparison.ok)
        self.assertEqual(comparison.sigma, 0.0)

    def test_within_combined_uncertainty_is_ok(self):
        comparison = rd17.Rd17PplComparison(
            subject=rd17.PerplexityRun(10.00, 0.05, "", ""),
            control=rd17.PerplexityRun(10.02, 0.05, "", ""),
        )
        self.assertTrue(comparison.ok)

    def test_far_outside_combined_uncertainty_fails(self):
        comparison = rd17.Rd17PplComparison(
            subject=rd17.PerplexityRun(10.0, 0.01, "", ""),
            control=rd17.PerplexityRun(11.0, 0.01, "", ""),
        )
        self.assertFalse(comparison.ok)
        self.assertGreater(comparison.sigma, comparison.max_sigma)

    def test_zero_uncertainty_and_zero_delta_is_ok(self):
        comparison = rd17.Rd17PplComparison(
            subject=rd17.PerplexityRun(10.0, 0.0, "", ""),
            control=rd17.PerplexityRun(10.0, 0.0, "", ""),
        )
        self.assertTrue(comparison.ok)

    def test_zero_uncertainty_and_nonzero_delta_fails(self):
        comparison = rd17.Rd17PplComparison(
            subject=rd17.PerplexityRun(10.0, 0.0, "", ""),
            control=rd17.PerplexityRun(10.1, 0.0, "", ""),
        )
        self.assertFalse(comparison.ok)


class RequireRd17PplEqualityTests(unittest.TestCase):
    def test_passes_and_returns_comparison_when_within_tolerance(self):
        def runner(argv, **kwargs):
            return _completed(0, stdout="Final estimate: PPL = 8.0000 +/- 0.0100\n")

        comparison = rd17.require_rd17_ppl_equality(
            subject_binary=Path("subj"), control_binary=Path("ctrl"),
            model=Path("m.gguf"), corpus=Path("c.txt"), runner=runner,
        )
        self.assertTrue(comparison.ok)

    def test_raises_with_specific_reason_on_mismatch(self):
        calls = {"n": 0}

        def runner(argv, **kwargs):
            calls["n"] += 1
            ppl = "8.0000" if calls["n"] == 1 else "9.0000"
            return _completed(0, stdout=f"Final estimate: PPL = {ppl} +/- 0.0100\n")

        with self.assertRaises(rd17.Rd17CorrectnessError) as ctx:
            rd17.require_rd17_ppl_equality(
                subject_binary=Path("subj"), control_binary=Path("ctrl"),
                model=Path("m.gguf"), corpus=Path("c.txt"), runner=runner,
            )
        message = str(ctx.exception)
        self.assertIn("ppl_equality failed", message)
        self.assertIn("sigma=", message)


class ComparisonToDictTests(unittest.TestCase):
    def test_full_shape(self):
        comparison = rd17.Rd17PplComparison(
            subject=rd17.PerplexityRun(10.0, 0.05, "", ""),
            control=rd17.PerplexityRun(10.02, 0.05, "", ""),
        )
        doc = rd17.comparison_to_dict(comparison)
        self.assertTrue(doc["ok"])
        self.assertAlmostEqual(doc["subject"]["ppl"], 10.0)
        self.assertAlmostEqual(doc["control"]["ppl"], 10.02)
        self.assertIn("sigma", doc)


if __name__ == "__main__":
    unittest.main()

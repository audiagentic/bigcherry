"""Tests for the paired native-versus-replay end-to-end runner."""

from __future__ import annotations

import re
import json
import itertools
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.campaign import benchmark as ab_benchmark # noqa: E402


def _python_executable() -> str:
    """Return a launchable interpreter even when sys.executable is an alias."""
    if Path(sys.executable).is_file():
        return sys.executable
    return shutil.which("python") or sys.executable


class Pairing(unittest.TestCase):
    def test_order_alternates(self):
        self.assertEqual(ab_benchmark.pair_modes(0), ("native", "replay"))
        self.assertEqual(ab_benchmark.pair_modes(1), ("replay", "native"))
        orders = ab_benchmark.schedule(6, include_stock=True, seed=7)
        self.assertEqual(set(orders), set(itertools.permutations(("stock", "native", "replay"))))
        self.assertEqual(orders, ab_benchmark.schedule(6, include_stock=True, seed=7))

    def test_structured_metrics_preserve_repetitions(self):
        values = ab_benchmark.extract_structured_metrics(
            'noise\n{"schema_version":1,"kind":"benchmark_result",'
            '"metrics":{"pp_ts":[100,101,102,103,104]}}\n'
        )
        self.assertEqual(values["pp_ts"], [100.0, 101.0, 102.0, 103.0, 104.0])

    def test_environment_is_decontaminated(self):
        env = ab_benchmark.sanitize_environment({
            "PATH": "x", "GGML_HIP_FORCE_FAKE": "1", "GGML_HIP_TUNE_WARMUP": "99",
            "GGML_HIP_DISPATCH_CACHE": "old",
        }, "native")
        self.assertEqual(env, {"PATH": "x", "GGML_HIP_DISPATCH_MODE": "native"})

    def test_deterministic_bootstrap_detects_effect(self):
        runs = []
        for pair in range(1, 7):
            runs.extend([
                {"pair": pair, "mode": "stock", "repetitions": {"rate": [100] * 5}},
                {"pair": pair, "mode": "native", "repetitions": {"rate": [102] * 5}},
                {"pair": pair, "mode": "replay", "repetitions": {"rate": [105] * 5}},
            ])
        result = ab_benchmark.block_bootstrap_effect(runs, "replay", "stock", "rate", seed=9)
        self.assertAlmostEqual(result["geometric_effect_pct"], 5.0)
        self.assertEqual(result, ab_benchmark.block_bootstrap_effect(runs, "replay", "stock", "rate", seed=9))
        report = ab_benchmark.render_report({
            "command": ["bench"], "pairs": 6, "settle_seconds": 0,
            "cache": "cache", "comparisons": {"replay_vs_stock": {"rate": {
                "mean_pct": 5, "median_pct": 5, "pairs": 6,
            }}},
            "decisions": {"replay_vs_stock": {"rate": result}},
        })
        self.assertIn("Decision-grade intervals", report)
        self.assertIn("+5.00%", report)

    def test_metric_selector_requires_one_capture(self):
        with self.assertRaisesRegex(ValueError, "exactly one capture"):
            ab_benchmark.parse_metric_specs(["rate=tok/s"])
        with self.assertRaisesRegex(ValueError, "exactly one capture"):
            ab_benchmark.parse_metric_specs(["rate=(a)(b)"])

    def test_last_metric_value_is_used(self):
        value = ab_benchmark.extract_metrics(
            "warmup 12.5\nfinal 18.25", {"rate": re.compile(r"(?:warmup|final) ([0-9.]+)")}
        )
        self.assertEqual(value, {"rate": 18.25})

    def test_paired_delta_respects_lower_is_better(self):
        runs = [
            {"pair": 1, "mode": "native", "metrics": {"rate": 100.0, "latency": 10.0}},
            {"pair": 1, "mode": "replay", "metrics": {"rate": 105.0, "latency": 9.0}},
        ]
        summary = ab_benchmark.paired_summary(runs, {"latency": "lower"})
        self.assertAlmostEqual(float(summary["rate"]["mean_pct"]), 5.0)
        self.assertAlmostEqual(float(summary["latency"]["mean_pct"]), 10.0)

    def test_incomplete_pair_is_not_compared(self):
        summary = ab_benchmark.paired_summary(
            [{"pair": 1, "mode": "native", "metrics": {"rate": 100.0}}], {}
        )
        self.assertEqual(summary, {})

    def test_stock_comparisons_share_the_same_round(self):
        runs = [
            {"pair": 1, "mode": "stock", "metrics": {"rate": 100.0}},
            {"pair": 1, "mode": "native", "metrics": {"rate": 101.0}},
            {"pair": 1, "mode": "replay", "metrics": {"rate": 105.0}},
        ]
        comparisons = ab_benchmark.comparison_summaries(runs, {}, include_stock=True)
        self.assertAlmostEqual(float(comparisons["native_vs_stock"]["rate"]["mean_pct"]), 1.0)
        self.assertAlmostEqual(float(comparisons["replay_vs_stock"]["rate"]["mean_pct"]), 5.0)
        self.assertAlmostEqual(float(comparisons["replay_vs_native"]["rate"]["mean_pct"]), 100.0 * 4.0 / 101.0)

    def test_cmake_parity_rejects_mismatched_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def cache(path: Path, targets: str) -> None:
                path.write_text("\n".join([
                    "CMAKE_BUILD_TYPE:STRING=Release", "CMAKE_C_COMPILER:FILEPATH=clang",
                    "CMAKE_CXX_COMPILER:FILEPATH=clang++", "GGML_HIP:BOOL=ON",
                    "GGML_HIP_RCCL:BOOL=ON", f"AMDGPU_TARGETS:STRING={targets}",
                ]), encoding="utf-8")
            stock, patched = root / "stock.txt", root / "patched.txt"
            cache(stock, "gfx1100;gfx1201;gfx1030")
            cache(patched, "gfx1100")
            with self.assertRaisesRegex(ValueError, "AMDGPU_TARGETS"):
                ab_benchmark.validate_build_parity(stock, patched)

    def test_coverage_requires_only_exact_current_v2_resolutions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "coverage.json"
            path.write_text(
                '{"total_dispatched": 12, "total_executed": 12, '
                '"replay": {"schema_version": 2, "entries": 4, "exact": 12, '
                '"candidate_unavailable": 0, "rerun_required": 0, '
                '"incompatible": 0, "misses": 0}}',
                encoding="utf-8",
            )
            result = ab_benchmark.validate_replay_coverage(path)
            self.assertEqual(result["entries"], 4)
            path.write_text(
                '{"total_dispatched": 12, "total_executed": 12, '
                '"replay": {"schema_version": 2, "entries": 4, "exact": 11, '
                '"candidate_unavailable": 0, "rerun_required": 0, '
                '"incompatible": 0, "misses": 1}}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "misses=1"):
                ab_benchmark.validate_replay_coverage(path)

    def test_runner_preserves_pair_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "dispatch.cache"
            output = root / "result"
            cache.write_text("cache", encoding="utf-8")
            child = (
                "import json, os; "
                "p=os.environ.get('GGML_HIP_DISPATCH_COVERAGE'); "
                "p and open(p, 'w').write(json.dumps({'total_dispatched': 1, "
                "'total_executed': 1, 'replay': {'schema_version': 2, 'entries': 1, "
                "'exact': 1, 'candidate_unavailable': 0, 'rerun_required': 0, "
                "'incompatible': 0, 'misses': 0}})); "
                "print('rate=100.0')"
            )
            status = ab_benchmark.main([
                "--cache", str(cache), "--output", str(output), "--pairs", "1",
                "--settle-seconds", "0", "--metric", r"rate=rate=([0-9.]+)",
                "--", _python_executable(), "-c", child,
            ])
            self.assertEqual(status, 0)
            result = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertEqual([run["mode"] for run in result["runs"]], ["native", "replay"])
            self.assertTrue((output / "report.md").is_file())


if __name__ == "__main__":
    unittest.main()

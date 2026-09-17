"""VA04 hardware-free preflight slice (post-PA36 migration #2): the
RD04-scoped paired benchmark evidence producer (the deleted
run_rd04_benchmark_evidence()) and its --run-rd04-benchmark CLI wiring
moved to the patch-local producer at
patches/1202_rd04_bf16_flash_attn_tile/validation/producer.py -- its
measurement semantics (forced -fa/-ctk/-ctv flags, control/subject
alternation, real sha binding, fail-closed nonzero arms) are now
covered by test_patch_validation_campaign_rd04_correctness.py, and its
CLI surface by test_patch_validation_campaign_rd04_contract_cli.py.

What remains in THIS file is the generic, still-current VA04 invariant:
the fallback "benchmark" validator's contract -- a bound performance
artifact with non-empty metrics + passed=true makes the performance AND
controls checks PASS, and a missing artifact stays honestly blocked
(never fabricated).
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _rd04_performance_doc() -> dict[str, object]:
    """The exact shape the migrated producer persists in its
    rd04-performance-<arch>.json artifact (legacy document shape,
    producer-built PPL pair identities)."""
    return {
        "passed": True,
        "campaign_id": "campaign123",
        "model": "m.gguf",
        "architecture": "gfx1100",
        "validation_build_identities": {
            "control": {"build_id": "ppl-pair-control-build"},
            "subject": {"build_id": "ppl-pair-subject-build"},
        },
        "commands": {
            "decode": {
                "control": ["c-bench", "-m", "m.gguf"],
                "subject": ["s-bench", "-m", "m.gguf"],
            },
            "prefill": {
                "control": ["c-bench", "-m", "m.gguf"],
                "subject": ["s-bench", "-m", "m.gguf"],
            },
        },
        "raw_logs": [
            {"path": "artifacts/rd04-benchmark-decode.log", "sha256": "0" * 64},
            {"path": "artifacts/rd04-benchmark-prefill.log", "sha256": "0" * 64},
        ],
        "metrics": {
            "decode": {
                "metric": "tg128",
                "stats": {"geometric_effect_pct": 3.41, "p_value": 0.02},
                "runs": [
                    {"metric": "tg128", "role": "control", "value": 100.0},
                    {"metric": "tg128", "role": "subject", "value": 103.4},
                ],
            },
            "prefill": {
                "metric": "pp512",
                "stats": {"geometric_effect_pct": 2.97, "p_value": 0.04},
                "runs": [
                    {"metric": "pp512", "role": "control", "value": 500.0},
                    {"metric": "pp512", "role": "subject", "value": 514.8},
                ],
            },
        },
    }


class BenchmarkArtifactBindingTests(unittest.TestCase):
    """Proves binding a producer-shaped real artifact into
    ctx.performance_evidence makes both the real "performance" and
    "controls" checks (validator="benchmark") reach PASS."""

    def test_performance_and_controls_checks_both_pass_from_the_bound_artifact(
        self,
    ) -> None:
        from bigcherry.patch import validation as pv

        run_dir = Path(tempfile.mkdtemp())
        performance_path = run_dir / "performance.json"
        performance_path.write_text(
            json.dumps(_rd04_performance_doc(), indent=2),
            encoding="utf-8",
        )
        performance_evidence = {
            "artifact": {
                "path": "performance.json",
                "sha256": hashlib.sha256(performance_path.read_bytes()).hexdigest(),
            },
        }
        # The benchmark validator only reads performance_evidence +
        # run_dir -- the namespace cast is this test file's established
        # pattern for that seam (a full ValidationContext needs a real
        # PatchDescriptor, which is irrelevant here).
        ctx = cast(
            "pv.ValidationContext",
            SimpleNamespace(run_dir=run_dir, performance_evidence=performance_evidence),
        )
        performance_spec = pv.CheckSpec(
            "performance", "performance", "benchmark", True, {}
        )
        controls_spec = pv.CheckSpec("controls", "controls", "benchmark", True, {})
        performance_result = pv.evaluate_check(performance_spec, ctx)
        controls_result = pv.evaluate_check(controls_spec, ctx)
        self.assertEqual(performance_result.status, pv.PASS, performance_result.summary)
        self.assertEqual(controls_result.status, pv.PASS, controls_result.summary)

    def test_missing_benchmark_evidence_is_blocked_not_fabricated(self) -> None:
        from bigcherry.patch import validation as pv

        ctx = cast(
            "pv.ValidationContext",
            SimpleNamespace(run_dir=Path(tempfile.mkdtemp()), performance_evidence={}),
        )
        spec = pv.CheckSpec("performance", "performance", "benchmark", True, {})
        result = pv.evaluate_check(spec, ctx)
        self.assertNotEqual(result.status, pv.PASS)


if __name__ == "__main__":
    unittest.main()

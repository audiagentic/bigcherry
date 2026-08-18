"""RE12-min (RV50-locked scope): comparisons.py run_comparison() and
promotion.py's report-backed pointer_from_comparison_report().

Two arms represent two different tuned builds (real production-class
descriptor-backed runtime-bundle/binary artifacts, differing intentionally
in build/effective-build identity, declared via allowed_differences). Only
the runtime-binary subprocess launch is patched out (same rationale as
test_re10_lifecycle.py); everything downstream -- ArtifactStore
verification, provenance derivation, ab_benchmark's real statistics,
promotion's report cross-check -- runs unmodified.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import comparisons, provenance  # noqa: E402
from bigcherry.artifacts import ArtifactStore  # noqa: E402
from bigcherry.promotion import PromotionError, pointer_from_comparison_report  # noqa: E402

_METRIC_PATTERNS = {"pp256": re.compile(r"pp256\s+([0-9.]+)")}


def _production_doc(*, build_plan_id: str, effective_build_id: str, run_id: str,
                     source_slice_id: str = "s1") -> provenance.ProvenanceV2:
    return provenance.ProvenanceV2.from_document(provenance.make(
        project={"provenance_class": "production", "bigcherry_revision": "r1"},
        source={
            "source_plan_id": "sp1", "materialization_plan_id": "mp1",
            "source_slice_id": source_slice_id, "patch_set_id": "ps1",
        },
        build={
            "build_plan_id": build_plan_id, "effective_build_id": effective_build_id,
            "binary_hash": "b" * 32, "runtime_bundle_hash": "c" * 32,
        },
        workload={"workload_id": "w1"},
        campaign={"run_id": run_id},
    ))


class _Fixture:
    def __init__(self, directory: Path):
        self.directory = directory
        self.store = ArtifactStore(directory / "store")
        self.run_id = "run1"

        self.left = self._publish_arm("left", build_plan_id="bp-left", effective_build_id="eb-left")
        self.right = self._publish_arm(
            "right", build_plan_id="bp-right", effective_build_id="eb-right", with_replay_cache=True)

    def _publish_arm(self, side: str, *, build_plan_id: str, effective_build_id: str,
                      with_replay_cache: bool = False) -> comparisons.BenchmarkArm:
        doc = _production_doc(
            build_plan_id=build_plan_id, effective_build_id=effective_build_id, run_id=self.run_id)
        prefix = f"builds/s1/{build_plan_id}"

        script_path = self.directory / f"{side}-entrypoint.py"
        script_path.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        script_digest = self.store.publish_file(f"{prefix}/entrypoint.py", script_path)
        manifest = {"entrypoint": "entrypoint.py", "members": {"entrypoint.py": script_digest}}
        bundle_ref = self.store.publish_json_ref(
            f"{prefix}/runtime-bundle.json", manifest, kind="runtime-bundle", provenance=doc)
        binary_ref = self.store.publish_bytes_ref(
            f"{prefix}/llama-bench", f"{side}-binary-bytes".encode(),
            kind="binary", provenance=doc)

        replay_cache_id = None
        if with_replay_cache:
            cache_ref = self.store.publish_bytes_ref(
                f"{prefix}/replay.cache", b"fake-cache-bytes", kind="replay-cache", provenance=doc)
            replay_cache_id = cache_ref.artifact_id

        return comparisons.BenchmarkArm(
            name=side, runtime_bundle_artifact_id=bundle_ref.artifact_id,
            binary_artifact_id=binary_ref.artifact_id, replay_cache_artifact_id=replay_cache_id,
            source_slice_id="s1", build_plan_id=build_plan_id, effective_build_id=effective_build_id,
            workload_id="w1", environment=(), device="",
        )

    def publish_replay_coverage(self, *, matches_right: bool) -> str:
        build_plan_id = self.right.build_plan_id if matches_right else "bp-mismatched"
        effective_build_id = self.right.effective_build_id if matches_right else "eb-mismatched"
        doc = provenance.ProvenanceV2.from_document(provenance.make(
            project={"provenance_class": "development", "bigcherry_revision": "r1"},
            source={"source_slice_id": "s1"},
            build={"build_plan_id": build_plan_id, "effective_build_id": effective_build_id},
            workload={},
            campaign={"run_id": self.run_id, "producer_stage": "replay-validate"},
        ))
        ref = self.store.publish_bytes_ref(
            "runs/run1/replay-validate/coverage.json", b'{"total_dispatched": 1}',
            kind="replay-coverage", provenance=doc)
        return ref.artifact_id


def _fake_run(*, left_value: float = 100.0, right_value: float = 110.0,
              replay_exact: bool = True):
    def run(argv, *, cwd, text, capture_output, env):
        side = "left" if "bp-left" in argv[0] else "right"
        value = left_value if side == "left" else right_value
        if env.get("GGML_HIP_DISPATCH_MODE") == "replay":
            coverage_path = env["GGML_HIP_DISPATCH_COVERAGE"]
            if replay_exact:
                replay = {"schema_version": 2, "exact": 1, "candidate_unavailable": 0,
                          "rerun_required": 0, "incompatible": 0, "misses": 0}
            else:
                replay = {"schema_version": 2, "exact": 0, "candidate_unavailable": 0,
                          "rerun_required": 1, "incompatible": 0, "misses": 0}
            Path(coverage_path).write_text(json.dumps(
                {"total_dispatched": 1, "total_executed": 1, "replay": replay}), encoding="utf-8")
        result = Mock()
        result.returncode = 0
        result.stdout = f"pp256 {value}\n"
        result.stderr = ""
        return result
    return Mock(side_effect=run)


class RunComparisonTests(unittest.TestCase):
    def test_balanced_comparison_produces_a_valid_decision_grade_report(self):
        with tempfile.TemporaryDirectory() as directory:
            fx = _Fixture(Path(directory))
            with patch("bigcherry.ab_benchmark.subprocess.run", _fake_run()):
                report_ref = comparisons.run_comparison(
                    comparisons.plan_pair(
                        fx.left, fx.right, label="left-vs-right",
                        allowed_differences=frozenset({
                            "build_plan_id", "effective_build_id",
                            "runtime_bundle_artifact_id", "binary_artifact_id",
                            "replay_cache_artifact_id"}),
                    ),
                    store=fx.store, run_id=fx.run_id, model_args=["-m", "model.gguf"],
                    output=fx.directory / "compare-out", pairs=4,
                    metric_patterns=_METRIC_PATTERNS, practical_threshold_pct=1.0,
                    resamples=200, decision_grade=True,
                    campaign_plan_id="cp1", comparison_plan_id="left-vs-right",
                    local_provenance_class="production",
                )
            self.assertEqual(report_ref.kind, "comparison-report")
            report = json.loads(report_ref.path.read_text())
            self.assertTrue(report["valid"])
            self.assertTrue(report["decision_grade"])
            self.assertEqual(report["effects"]["pp256"]["decision"], "improved")
            self.assertEqual(report["replay_arm"], "right")

            # every raw evidence artifact ID rehydrates
            for side_evidence in report["raw_evidence_artifact_ids"].values():
                for pair_evidence in side_evidence.values():
                    for artifact_id in pair_evidence.values():
                        rehydrated = fx.store.rehydrate(artifact_id, expected_kind="comparison-raw-evidence")
                        self.assertTrue(rehydrated.content_hash)

    def test_source_build_mismatch_rejects_before_any_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            fx = _Fixture(Path(directory))
            with self.assertRaises(comparisons.ComparisonError):
                comparisons.plan_pair(fx.left, fx.right, label="undeclared")

    def test_environment_mismatch_rejects_before_executable_invocation(self):
        with tempfile.TemporaryDirectory() as directory:
            fx = _Fixture(Path(directory))
            import dataclasses
            mismatched = dataclasses.replace(fx.right, environment=(("X", "1"),),
                                              build_plan_id=fx.left.build_plan_id,
                                              effective_build_id=fx.left.effective_build_id,
                                              runtime_bundle_artifact_id=fx.left.runtime_bundle_artifact_id,
                                              binary_artifact_id=fx.left.binary_artifact_id,
                                              replay_cache_artifact_id=fx.left.replay_cache_artifact_id)
            with self.assertRaises(comparisons.ComparisonError):
                comparisons.plan_pair(fx.left, mismatched, label="env-mismatch")

    def test_device_mismatch_rejects_before_executable_invocation(self):
        with tempfile.TemporaryDirectory() as directory:
            fx = _Fixture(Path(directory))
            import dataclasses
            mismatched = dataclasses.replace(fx.right, device="1",
                                              build_plan_id=fx.left.build_plan_id,
                                              effective_build_id=fx.left.effective_build_id,
                                              runtime_bundle_artifact_id=fx.left.runtime_bundle_artifact_id,
                                              binary_artifact_id=fx.left.binary_artifact_id,
                                              replay_cache_artifact_id=fx.left.replay_cache_artifact_id)
            with self.assertRaises(comparisons.ComparisonError):
                comparisons.plan_pair(fx.left, mismatched, label="device-mismatch")

    def test_tampered_runtime_bundle_member_rejects_before_any_subprocess_call(self):
        with tempfile.TemporaryDirectory() as directory:
            fx = _Fixture(Path(directory))
            (fx.store.root / "builds/s1/bp-right/entrypoint.py").write_text("TAMPERED", encoding="utf-8")
            plan = comparisons.plan_pair(
                fx.left, fx.right, label="tampered",
                allowed_differences=frozenset({
                    "build_plan_id", "effective_build_id", "runtime_bundle_artifact_id",
                    "binary_artifact_id", "replay_cache_artifact_id"}))
            with patch("bigcherry.ab_benchmark.subprocess.run", _fake_run()) as fake_run:
                with self.assertRaises(comparisons.ComparisonError):
                    comparisons.run_comparison(
                        plan, store=fx.store, run_id=fx.run_id, model_args=["-m", "model.gguf"],
                        output=fx.directory / "compare-out", pairs=2,
                        metric_patterns=_METRIC_PATTERNS, resamples=200,
                        local_provenance_class="production",
                    )
                fake_run.assert_not_called()

    def test_non_exact_replay_coverage_makes_the_report_invalid_not_an_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            fx = _Fixture(Path(directory))
            plan = comparisons.plan_pair(
                fx.left, fx.right, label="left-vs-right",
                allowed_differences=frozenset({
                    "build_plan_id", "effective_build_id", "runtime_bundle_artifact_id",
                    "binary_artifact_id", "replay_cache_artifact_id"}))
            with patch("bigcherry.ab_benchmark.subprocess.run", _fake_run(replay_exact=False)):
                report_ref = comparisons.run_comparison(
                    plan, store=fx.store, run_id=fx.run_id, model_args=["-m", "model.gguf"],
                    output=fx.directory / "compare-out", pairs=2,
                    metric_patterns=_METRIC_PATTERNS, resamples=200,
                    campaign_plan_id="cp1", comparison_plan_id="left-vs-right",
                    local_provenance_class="production",
                )
            report = json.loads(report_ref.path.read_text())
            self.assertFalse(report["valid"])
            self.assertFalse(report["decision_grade"])
            self.assertTrue(report["issues"])


class PointerFromComparisonReportTests(unittest.TestCase):
    def _valid_report(self, fx: _Fixture) -> str:
        plan = comparisons.plan_pair(
            fx.left, fx.right, label="left-vs-right",
            allowed_differences=frozenset({
                "build_plan_id", "effective_build_id", "runtime_bundle_artifact_id",
                "binary_artifact_id", "replay_cache_artifact_id"}))
        with patch("bigcherry.ab_benchmark.subprocess.run", _fake_run()):
            report_ref = comparisons.run_comparison(
                plan, store=fx.store, run_id=fx.run_id, model_args=["-m", "model.gguf"],
                output=fx.directory / "compare-out", pairs=4,
                metric_patterns=_METRIC_PATTERNS, resamples=200, decision_grade=True,
                campaign_plan_id="cp1", comparison_plan_id="left-vs-right",
                local_provenance_class="production",
            )
        return report_ref.artifact_id

    def test_a_valid_report_produces_a_release_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            fx = _Fixture(Path(directory))
            report_artifact_id = self._valid_report(fx)
            coverage_id = fx.publish_replay_coverage(matches_right=True)
            pointer = pointer_from_comparison_report(
                store=fx.store, report_artifact_id=report_artifact_id, release_tag="b1",
                replay_coverage_artifact_id=coverage_id, required_architectures=("gfx1100",),
            )
            self.assertEqual(pointer.schema_version, 3)
            self.assertEqual(pointer.report_artifact_id, report_artifact_id)
            self.assertEqual(pointer.document()["report_artifact_id"], report_artifact_id)

    def test_a_caller_cannot_substitute_a_mismatched_replay_result(self):
        with tempfile.TemporaryDirectory() as directory:
            fx = _Fixture(Path(directory))
            report_artifact_id = self._valid_report(fx)
            mismatched_coverage_id = fx.publish_replay_coverage(matches_right=False)
            with self.assertRaises(PromotionError):
                pointer_from_comparison_report(
                    store=fx.store, report_artifact_id=report_artifact_id, release_tag="b1",
                    replay_coverage_artifact_id=mismatched_coverage_id,
                    required_architectures=("gfx1100",),
                )

    def test_an_invalid_report_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fx = _Fixture(Path(directory))
            plan = comparisons.plan_pair(
                fx.left, fx.right, label="left-vs-right",
                allowed_differences=frozenset({
                    "build_plan_id", "effective_build_id", "runtime_bundle_artifact_id",
                    "binary_artifact_id", "replay_cache_artifact_id"}))
            with patch("bigcherry.ab_benchmark.subprocess.run", _fake_run(replay_exact=False)):
                report_ref = comparisons.run_comparison(
                    plan, store=fx.store, run_id=fx.run_id, model_args=["-m", "model.gguf"],
                    output=fx.directory / "compare-out", pairs=2,
                    metric_patterns=_METRIC_PATTERNS, resamples=200,
                    campaign_plan_id="cp1", comparison_plan_id="left-vs-right",
                    local_provenance_class="production",
                )
            coverage_id = fx.publish_replay_coverage(matches_right=True)
            with self.assertRaises(PromotionError):
                pointer_from_comparison_report(
                    store=fx.store, report_artifact_id=report_ref.artifact_id, release_tag="b1",
                    replay_coverage_artifact_id=coverage_id, required_architectures=("gfx1100",),
                )


if __name__ == "__main__":
    unittest.main()

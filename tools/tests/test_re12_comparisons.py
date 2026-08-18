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


def _production_doc(
    *,
    build_plan_id: str,
    effective_build_id: str,
    run_id: str,
    source_slice_id: str = "s1",
    provenance_class: str = "production",
) -> provenance.ProvenanceV2:
    return provenance.ProvenanceV2.from_document(
        provenance.make(
            project={"provenance_class": provenance_class, "bigcherry_revision": "r1"},
            source={
                "source_plan_id": "sp1",
                "materialization_plan_id": "mp1",
                "source_slice_id": source_slice_id,
                "patch_set_id": "ps1",
                # GPT audit fix (item 6): the pointer's revision axis comes from
                # the replay arm's binary source provenance -- the fixture must
                # carry a real upstream revision for that to be checkable.
                "upstream_revision": "abcdef1234567890",
            },
            build={
                "build_plan_id": build_plan_id,
                "effective_build_id": effective_build_id,
                "binary_hash": "b" * 32,
                "runtime_bundle_hash": "c" * 32,
            },
            workload={"workload_id": "w1"},
            campaign={"run_id": run_id},
        )
    )


class _Fixture:
    def __init__(self, directory: Path):
        self.directory = directory
        self.store = ArtifactStore(directory / "store")
        self.run_id = "run1"
        self._arm_artifacts: dict[str, dict[str, str | None]] = {}

        self.left = self._publish_arm(
            "left", build_plan_id="bp-left", effective_build_id="eb-left"
        )
        self.right = self._publish_arm(
            "right",
            build_plan_id="bp-right",
            effective_build_id="eb-right",
            with_replay_cache=True,
        )

    def _publish_arm(
        self,
        side: str,
        *,
        build_plan_id: str,
        effective_build_id: str,
        with_replay_cache: bool = False,
    ) -> comparisons.BenchmarkArm:
        doc = _production_doc(
            build_plan_id=build_plan_id,
            effective_build_id=effective_build_id,
            run_id=self.run_id,
        )
        prefix = f"builds/s1/{build_plan_id}"

        script_path = self.directory / f"{side}-entrypoint.py"
        script_path.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        # GPT audit fix (item 1): the entrypoint the benchmark executes must
        # BE the binary the pointer records -- same bytes, same content
        # hash. The old fixture deliberately used different bytes, which was
        # exactly the substitution hole the audit named.
        binary_bytes = script_path.read_bytes()
        script_digest = self.store.publish_file(f"{prefix}/entrypoint.py", script_path)
        manifest = {
            "entrypoint": "entrypoint.py",
            "members": {"entrypoint.py": script_digest},
        }
        bundle_ref = self.store.publish_json_ref(
            f"{prefix}/runtime-bundle.json",
            manifest,
            kind="runtime-bundle",
            provenance=doc,
        )
        binary_ref = self.store.publish_bytes_ref(
            f"{prefix}/llama-bench", binary_bytes, kind="binary", provenance=doc
        )
        self._arm_artifacts[side] = {
            "bundle": bundle_ref.artifact_id,
            "binary": binary_ref.artifact_id,
            "cache": None,
        }

        replay_cache_id = None
        if with_replay_cache:
            cache_ref = self.store.publish_bytes_ref(
                f"{prefix}/replay.cache",
                b"fake-cache-bytes",
                kind="replay-cache",
                provenance=doc,
            )
            replay_cache_id = cache_ref.artifact_id
            self._arm_artifacts[side]["cache"] = cache_ref.artifact_id

        return comparisons.BenchmarkArm(
            name=side,
            runtime_bundle_artifact_id=bundle_ref.artifact_id,
            binary_artifact_id=binary_ref.artifact_id,
            replay_cache_artifact_id=replay_cache_id,
            source_slice_id="s1",
            build_plan_id=build_plan_id,
            effective_build_id=effective_build_id,
            workload_id="w1",
            environment=(),
            device="",
        )

    def publish_replay_coverage(
        self,
        *,
        matches_right: bool = True,
        provenance_class: str = "production",
        parent_ids: tuple[str, ...] | None = None,
        coverage_bytes: bytes | None = None,
    ) -> str:
        # GPT audit fix (items 3/4/5): the old helper published a
        # DEVELOPMENT-class coverage with no parent artifact IDs and
        # unvalidated bytes, and the old happy-path test accepted it -- the
        # audit's items 3 and 4 literally. The fixture now mirrors what
        # execute_replay_validation_stage actually publishes: production
        # class, exact bundle+cache parent IDs, validator-passing bytes.
        build_plan_id = self.right.build_plan_id if matches_right else "bp-mismatched"
        effective_build_id = (
            self.right.effective_build_id if matches_right else "eb-mismatched"
        )
        if parent_ids is None:
            right = self._arm_artifacts["right"]
            parent_ids = tuple(sorted([right["bundle"] or "", right["cache"] or ""]))
        if coverage_bytes is None:
            coverage_bytes = json.dumps(
                {
                    "total_dispatched": 1,
                    "total_executed": 1,
                    "replay": {
                        "schema_version": 2,
                        "exact": 1,
                        "candidate_unavailable": 0,
                        "rerun_required": 0,
                        "incompatible": 0,
                        "misses": 0,
                    },
                }
            ).encode("utf-8")
        doc = provenance.ProvenanceV2.from_document(
            provenance.make(
                project={
                    "provenance_class": provenance_class,
                    "bigcherry_revision": "r1",
                },
                source={"source_slice_id": "s1"},
                build={
                    "build_plan_id": build_plan_id,
                    "effective_build_id": effective_build_id,
                },
                workload={},
                campaign={
                    "run_id": self.run_id,
                    "producer_stage": "replay-validate",
                    "producer_artifact_ids": list(parent_ids),
                },
            )
        )
        ref = self.store.publish_bytes_ref(
            "runs/run1/replay-validate/coverage.json",
            coverage_bytes,
            kind="replay-coverage",
            provenance=doc,
        )
        return ref.artifact_id


def _fake_run(
    *, left_value: float = 100.0, right_value: float = 110.0, replay_exact: bool = True
):
    def run(argv, *, cwd, text, capture_output, env):
        side = "left" if "bp-left" in argv[0] else "right"
        value = left_value if side == "left" else right_value
        if env.get("GGML_HIP_DISPATCH_MODE") == "replay":
            coverage_path = env["GGML_HIP_DISPATCH_COVERAGE"]
            if replay_exact:
                replay = {
                    "schema_version": 2,
                    "exact": 1,
                    "candidate_unavailable": 0,
                    "rerun_required": 0,
                    "incompatible": 0,
                    "misses": 0,
                }
            else:
                replay = {
                    "schema_version": 2,
                    "exact": 0,
                    "candidate_unavailable": 0,
                    "rerun_required": 1,
                    "incompatible": 0,
                    "misses": 0,
                }
            Path(coverage_path).write_text(
                json.dumps(
                    {"total_dispatched": 1, "total_executed": 1, "replay": replay}
                ),
                encoding="utf-8",
            )
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
                        fx.left,
                        fx.right,
                        label="left-vs-right",
                        allowed_differences=frozenset(
                            {
                                "build_plan_id",
                                "effective_build_id",
                                "runtime_bundle_artifact_id",
                                "binary_artifact_id",
                                "replay_cache_artifact_id",
                            }
                        ),
                    ),
                    store=fx.store,
                    run_id=fx.run_id,
                    model_args=["-m", "model.gguf"],
                    output=fx.directory / "compare-out",
                    pairs=4,
                    metric_patterns=_METRIC_PATTERNS,
                    practical_threshold_pct=1.0,
                    resamples=200,
                    decision_grade=True,
                    campaign_plan_id="cp1",
                    comparison_plan_id="left-vs-right",
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
                        rehydrated = fx.store.rehydrate(
                            artifact_id, expected_kind="comparison-raw-evidence"
                        )
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

            mismatched = dataclasses.replace(
                fx.right,
                environment=(("X", "1"),),
                build_plan_id=fx.left.build_plan_id,
                effective_build_id=fx.left.effective_build_id,
                runtime_bundle_artifact_id=fx.left.runtime_bundle_artifact_id,
                binary_artifact_id=fx.left.binary_artifact_id,
                replay_cache_artifact_id=fx.left.replay_cache_artifact_id,
            )
            with self.assertRaises(comparisons.ComparisonError):
                comparisons.plan_pair(fx.left, mismatched, label="env-mismatch")

    def test_device_mismatch_rejects_before_executable_invocation(self):
        with tempfile.TemporaryDirectory() as directory:
            fx = _Fixture(Path(directory))
            import dataclasses

            mismatched = dataclasses.replace(
                fx.right,
                device="1",
                build_plan_id=fx.left.build_plan_id,
                effective_build_id=fx.left.effective_build_id,
                runtime_bundle_artifact_id=fx.left.runtime_bundle_artifact_id,
                binary_artifact_id=fx.left.binary_artifact_id,
                replay_cache_artifact_id=fx.left.replay_cache_artifact_id,
            )
            with self.assertRaises(comparisons.ComparisonError):
                comparisons.plan_pair(fx.left, mismatched, label="device-mismatch")

    def test_entrypoint_and_binary_hash_mismatch_rejects_before_execution(self):
        # GPT audit fix (item 1): the benchmarked entrypoint must be the
        # same bytes as the binary the pointer will record. The old fixture
        # shipped different bytes for each -- a valid pointer could have
        # claimed 'binary X validated' when the run executed binary Y.
        with tempfile.TemporaryDirectory() as directory:
            fx = _Fixture(Path(directory))
            # Republish the RIGHT bundle with a DIFFERENT entrypoint member
            # (the binary stays the arm's original bytes).
            other_path = fx.directory / "other-entrypoint.py"
            other_path.write_text(
                "#!/usr/bin/env python3\n# other bytes\n", encoding="utf-8"
            )
            other_digest = fx.store.publish_file(
                "builds/s1/bp-right/entrypoint-alt.py", other_path
            )
            original_manifest = json.loads(
                fx._arm_artifacts["right"]["bundle"]
                and (
                    fx.store.root / "builds/s1/bp-right" / "runtime-bundle.json"
                ).read_text()
            )
            manifest = {
                "entrypoint": "entrypoint-alt.py",
                "members": dict(
                    original_manifest["members"], **{"entrypoint-alt.py": other_digest}
                ),
            }
            bundle_ref = fx.store.publish_json_ref(
                "builds/s1/bp-right/runtime-bundle-alt.json",
                manifest,
                kind="runtime-bundle",
                provenance=fx.store.rehydrate(
                    fx.right.runtime_bundle_artifact_id, expected_kind="runtime-bundle"
                ).provenance,
            )
            import dataclasses

            swapped = dataclasses.replace(
                fx.right, runtime_bundle_artifact_id=bundle_ref.artifact_id
            )
            plan = comparisons.plan_pair(
                fx.left,
                swapped,
                label="mismatched-binary",
                allowed_differences=frozenset(
                    {
                        "build_plan_id",
                        "effective_build_id",
                        "runtime_bundle_artifact_id",
                        "binary_artifact_id",
                        "replay_cache_artifact_id",
                    }
                ),
            )
            with patch(
                "bigcherry.ab_benchmark.subprocess.run", _fake_run()
            ) as fake_run:
                with self.assertRaises(comparisons.ComparisonError):
                    comparisons.run_comparison(
                        plan,
                        store=fx.store,
                        run_id=fx.run_id,
                        model_args=["-m", "m.gguf"],
                        output=fx.directory / "compare-out",
                        pairs=1,
                        metric_patterns=_METRIC_PATTERNS,
                        resamples=200,
                        local_provenance_class="production",
                    )
                fake_run.assert_not_called()

    def test_declared_identity_mismatch_rejects_before_execution(self):
        # GPT audit fix (item 1): the BenchmarkArm dataclass is a caller
        # claim, not evidence -- it must match the stored bundle/binary
        # provenance identity.
        with tempfile.TemporaryDirectory() as directory:
            fx = _Fixture(Path(directory))
            import dataclasses
            # build_plan_id is allowed to differ between arms at the
            # plan level -- so plan_pair passes and the rejection must
            # come from the stored-provenance cross-check, not the plan.
            lied = dataclasses.replace(fx.right, build_plan_id="bp-lied")
            plan = comparisons.plan_pair(
                fx.left,
                lied,
                label="liar",
                allowed_differences=frozenset(
                    {
                        "build_plan_id",
                        "effective_build_id",
                        "runtime_bundle_artifact_id",
                        "binary_artifact_id",
                        "replay_cache_artifact_id",
                    }
                ),
            )
            with patch(
                "bigcherry.ab_benchmark.subprocess.run", _fake_run()
            ) as fake_run:
                with self.assertRaises(comparisons.ComparisonError):
                    comparisons.run_comparison(
                        plan,
                        store=fx.store,
                        run_id=fx.run_id,
                        model_args=["-m", "m.gguf"],
                        output=fx.directory / "compare-out",
                        pairs=1,
                        metric_patterns=_METRIC_PATTERNS,
                        resamples=200,
                        local_provenance_class="production",
                    )
                fake_run.assert_not_called()

    def test_imported_legacy_bundle_taints_the_report(self):
        # GPT audit fix (item 2): the runtime bundle and replay cache are
        # real execution authorities -- if either is imported-legacy, the
        # report it participates in must be tainted, even when every
        # binary is production. This is a new laundering route the old
        # report (parents=binary only) silently allowed.
        with tempfile.TemporaryDirectory() as directory:
            fx = _Fixture(Path(directory))
            # Republish the RIGHT arm's bundle and cache as imported-legacy
            # with the same bytes (verification still passes -- this is the
            # taint vector, not the tamper vector).
            import dataclasses

            imported_doc = provenance.ProvenanceV2.from_document(
                provenance.make(
                    project={
                        "provenance_class": "imported-legacy",
                        "bigcherry_revision": "r1",
                    },
                    source={
                        "source_plan_id": "sp1",
                        "materialization_plan_id": "mp1",
                        "source_slice_id": "s1",
                        "patch_set_id": "ps1",
                        "upstream_revision": "abcdef1234567890",
                    },
                    build={
                        "build_plan_id": fx.right.build_plan_id,
                        "effective_build_id": fx.right.effective_build_id,
                        "binary_hash": "b" * 32,
                        "runtime_bundle_hash": "c" * 32,
                    },
                    workload={"workload_id": "w1"},
                    campaign={"run_id": fx.run_id},
                )
            )
            bundle_bytes = (
                fx.store.root / "builds/s1/bp-right" / "runtime-bundle.json"
            ).read_bytes()
            cache_bytes = (
                fx.store.root / "builds/s1/bp-right" / "replay.cache"
            ).read_bytes()
            new_bundle = fx.store.publish_bytes_ref(
                "builds/s1/bp-right/runtime-bundle-imported.json",
                bundle_bytes,
                kind="runtime-bundle",
                provenance=imported_doc.document(),
            )
            new_cache = fx.store.publish_bytes_ref(
                "builds/s1/bp-right/replay-imported.cache",
                cache_bytes,
                kind="replay-cache",
                provenance=imported_doc.document(),
            )
            swapped = dataclasses.replace(
                fx.right,
                runtime_bundle_artifact_id=new_bundle.artifact_id,
                replay_cache_artifact_id=new_cache.artifact_id,
            )
            plan = comparisons.plan_pair(
                fx.left,
                swapped,
                label="tainted",
                allowed_differences=frozenset(
                    {
                        "build_plan_id",
                        "effective_build_id",
                        "runtime_bundle_artifact_id",
                        "binary_artifact_id",
                        "replay_cache_artifact_id",
                    }
                ),
            )
            with patch("bigcherry.ab_benchmark.subprocess.run", _fake_run()):
                report_ref = comparisons.run_comparison(
                    plan,
                    store=fx.store,
                    run_id=fx.run_id,
                    model_args=["-m", "m.gguf"],
                    output=fx.directory / "compare-out",
                    pairs=2,
                    metric_patterns=_METRIC_PATTERNS,
                    resamples=200,
                    decision_grade=True,
                    campaign_plan_id="cp1",
                    comparison_plan_id="left-vs-right",
                    local_provenance_class="production",
                )
            report_doc = provenance.ProvenanceV2.from_document(report_ref.provenance)
            self.assertEqual(report_doc.project.provenance_class, "imported-legacy")

    def test_tampered_runtime_bundle_member_rejects_before_any_subprocess_call(self):
        with tempfile.TemporaryDirectory() as directory:
            fx = _Fixture(Path(directory))
            (fx.store.root / "builds/s1/bp-right/entrypoint.py").write_text(
                "TAMPERED", encoding="utf-8"
            )
            plan = comparisons.plan_pair(
                fx.left,
                fx.right,
                label="tampered",
                allowed_differences=frozenset(
                    {
                        "build_plan_id",
                        "effective_build_id",
                        "runtime_bundle_artifact_id",
                        "binary_artifact_id",
                        "replay_cache_artifact_id",
                    }
                ),
            )
            with patch(
                "bigcherry.ab_benchmark.subprocess.run", _fake_run()
            ) as fake_run:
                with self.assertRaises(comparisons.ComparisonError):
                    comparisons.run_comparison(
                        plan,
                        store=fx.store,
                        run_id=fx.run_id,
                        model_args=["-m", "model.gguf"],
                        output=fx.directory / "compare-out",
                        pairs=2,
                        metric_patterns=_METRIC_PATTERNS,
                        resamples=200,
                        local_provenance_class="production",
                    )
                fake_run.assert_not_called()

    def test_non_exact_replay_coverage_makes_the_report_invalid_not_an_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            fx = _Fixture(Path(directory))
            plan = comparisons.plan_pair(
                fx.left,
                fx.right,
                label="left-vs-right",
                allowed_differences=frozenset(
                    {
                        "build_plan_id",
                        "effective_build_id",
                        "runtime_bundle_artifact_id",
                        "binary_artifact_id",
                        "replay_cache_artifact_id",
                    }
                ),
            )
            with patch(
                "bigcherry.ab_benchmark.subprocess.run", _fake_run(replay_exact=False)
            ):
                report_ref = comparisons.run_comparison(
                    plan,
                    store=fx.store,
                    run_id=fx.run_id,
                    model_args=["-m", "model.gguf"],
                    output=fx.directory / "compare-out",
                    pairs=2,
                    metric_patterns=_METRIC_PATTERNS,
                    resamples=200,
                    campaign_plan_id="cp1",
                    comparison_plan_id="left-vs-right",
                    local_provenance_class="production",
                )
            report = json.loads(report_ref.path.read_text())
            self.assertFalse(report["valid"])
            self.assertFalse(report["decision_grade"])
            self.assertTrue(report["issues"])


class PointerFromComparisonReportTests(unittest.TestCase):
    def _valid_report(self, fx: _Fixture) -> str:
        plan = comparisons.plan_pair(
            fx.left,
            fx.right,
            label="left-vs-right",
            allowed_differences=frozenset(
                {
                    "build_plan_id",
                    "effective_build_id",
                    "runtime_bundle_artifact_id",
                    "binary_artifact_id",
                    "replay_cache_artifact_id",
                }
            ),
        )
        with patch("bigcherry.ab_benchmark.subprocess.run", _fake_run()):
            report_ref = comparisons.run_comparison(
                plan,
                store=fx.store,
                run_id=fx.run_id,
                model_args=["-m", "model.gguf"],
                output=fx.directory / "compare-out",
                pairs=4,
                metric_patterns=_METRIC_PATTERNS,
                resamples=200,
                decision_grade=True,
                campaign_plan_id="cp1",
                comparison_plan_id="left-vs-right",
                local_provenance_class="production",
            )
        return report_ref.artifact_id

    def test_a_valid_report_produces_a_release_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            fx = _Fixture(Path(directory))
            report_artifact_id = self._valid_report(fx)
            coverage_id = fx.publish_replay_coverage(matches_right=True)
            pointer = pointer_from_comparison_report(
                store=fx.store,
                report_artifact_id=report_artifact_id,
                release_tag="b1",
                replay_coverage_artifact_id=coverage_id,
                required_architectures=("gfx1100",),
            )
            self.assertEqual(pointer.schema_version, 3)
            self.assertEqual(pointer.report_artifact_id, report_artifact_id)
            self.assertEqual(
                pointer.document()["report_artifact_id"], report_artifact_id
            )

    def test_development_class_replay_coverage_is_rejected(self):
        # GPT audit fix (item 3): the old happy path accepted a
        # development-class coverage artifact because the pointer boundary
        # used ordinary rehydrate(). A release pointer demands promotable
        # (production-class) replay evidence.
        with tempfile.TemporaryDirectory() as directory:
            fx = _Fixture(Path(directory))
            report_artifact_id = self._valid_report(fx)
            coverage_id = fx.publish_replay_coverage(provenance_class="development")
            with self.assertRaises(PromotionError):
                pointer_from_comparison_report(
                    store=fx.store,
                    report_artifact_id=report_artifact_id,
                    release_tag="b1",
                    replay_coverage_artifact_id=coverage_id,
                    required_architectures=("gfx1100",),
                )

    def test_a_caller_cannot_substitute_a_mismatched_replay_result(self):
        with tempfile.TemporaryDirectory() as directory:
            fx = _Fixture(Path(directory))
            report_artifact_id = self._valid_report(fx)
            mismatched_coverage_id = fx.publish_replay_coverage(matches_right=False)
            with self.assertRaises(PromotionError):
                pointer_from_comparison_report(
                    store=fx.store,
                    report_artifact_id=report_artifact_id,
                    release_tag="b1",
                    replay_coverage_artifact_id=mismatched_coverage_id,
                    required_architectures=("gfx1100",),
                )

    def test_an_invalid_report_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fx = _Fixture(Path(directory))
            plan = comparisons.plan_pair(
                fx.left,
                fx.right,
                label="left-vs-right",
                allowed_differences=frozenset(
                    {
                        "build_plan_id",
                        "effective_build_id",
                        "runtime_bundle_artifact_id",
                        "binary_artifact_id",
                        "replay_cache_artifact_id",
                    }
                ),
            )
            with patch(
                "bigcherry.ab_benchmark.subprocess.run", _fake_run(replay_exact=False)
            ):
                report_ref = comparisons.run_comparison(
                    plan,
                    store=fx.store,
                    run_id=fx.run_id,
                    model_args=["-m", "model.gguf"],
                    output=fx.directory / "compare-out",
                    pairs=2,
                    metric_patterns=_METRIC_PATTERNS,
                    resamples=200,
                    campaign_plan_id="cp1",
                    comparison_plan_id="left-vs-right",
                    local_provenance_class="production",
                )
            coverage_id = fx.publish_replay_coverage(matches_right=True)
            with self.assertRaises(PromotionError):
                pointer_from_comparison_report(
                    store=fx.store,
                    report_artifact_id=report_ref.artifact_id,
                    release_tag="b1",
                    replay_coverage_artifact_id=coverage_id,
                    required_architectures=("gfx1100",),
                )


if __name__ == "__main__":
    unittest.main()

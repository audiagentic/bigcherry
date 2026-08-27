"""HI130: pure-Python helpers in tuning/workflow.py -- the parts testable
without a real GPU/build (the full run_tune_campaign() orchestration needs
real hardware and is validated live on Brutus, not here).
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.tuning import workflow  # noqa: E402

_EVIDENCE_DIR = Path(__file__).resolve().parents[3] / "docs" / "evidence" / "2026-08-27-dual-xtx-tune-run"
_REAL_PROMOTED = _EVIDENCE_DIR / "e2e_dualxtx_tune.promoted2.jsonl"


class CountMissingCorrectnessEvidenceTests(unittest.TestCase):
    def test_real_evidence_file_has_exactly_the_one_known_unsupported_row_missing(self):
        # This file is the SECOND promote() pass, run after correctness
        # evidence generation for every SUPPORTED candidate. Exactly one of
        # the 39 provisional winners was honestly skipped as an unsupported
        # signature domain (a non-routed/dense GLU fusion -- HI119 step 16,
        # not yet written; see the real session log: "SKIPPED (unsupported
        # signature): ... only the MoE-routed (MUL_MAT_ID-based) fused GLU
        # case is supported this slice") -- that row legitimately still has
        # no evidence and must still count as missing, not be silently
        # treated as resolved.
        if not _REAL_PROMOTED.is_file():
            self.skipTest("real evidence file not present in this checkout")
        count = workflow._count_missing_correctness_evidence(_REAL_PROMOTED)
        self.assertEqual(count, 1)

    def test_counts_rows_missing_evidence(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "promoted.jsonl"
            rows = [
                {"kind": "header"},
                {"kind": "result", "promotion_status": "promoted"},
                {"kind": "result", "promotion_status": "rejected_no_correctness_evidence"},
                {"kind": "result", "promotion_status": "rejected_no_correctness_evidence"},
                {"kind": "result", "promotion_status": "native"},
            ]
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            self.assertEqual(workflow._count_missing_correctness_evidence(path), 2)

    def test_zero_when_nothing_missing(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "promoted.jsonl"
            rows = [
                {"kind": "header"},
                {"kind": "result", "promotion_status": "promoted"},
                {"kind": "result", "promotion_status": "native"},
            ]
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            self.assertEqual(workflow._count_missing_correctness_evidence(path), 0)


class StageReplayExportTests(unittest.TestCase):
    def test_exports_against_the_supplied_target_manifest_not_some_other_one(self):
        # HI130 regression (req_ec659ded425c4335): _stage_replay_export must
        # bind the cache to whatever manifest/source_root it is GIVEN -- the
        # bug was the caller handing it tune's manifest instead of replay's,
        # not anything inside this function, but a wrong rename here would
        # silently reintroduce the same class of defect.
        from unittest.mock import patch
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            workdir = Path(directory)
            target_manifest_path = workdir / "replay-manifest.json"
            target_manifest_path.write_text("{}", encoding="utf-8")
            target_source_root = workdir / "replay-source-root"
            (target_source_root / "ggml" / "include").mkdir(parents=True)
            (target_source_root / "ggml" / "include" / "ggml.h").write_text("", encoding="utf-8")
            promoted_path = workdir / "promoted.jsonl"
            promoted_path.write_text("", encoding="utf-8")
            dispatch_db = workdir / "tune.sqlite"

            with patch.object(workflow, "replay_mod") as fake_replay_mod:
                fake_replay_mod.build.return_value = b"cache-bytes"
                result = workflow._stage_replay_export(
                    promoted_path=promoted_path,
                    target_manifest_path=target_manifest_path,
                    target_source_root=target_source_root,
                    dispatch_db=dispatch_db,
                    workdir=workdir,
                )

            fake_replay_mod.build.assert_called_once_with(
                promoted_path, target_manifest_path,
                target_source_root / "ggml" / "include" / "ggml.h",
                dispatch_db=dispatch_db,
            )
            self.assertEqual(result.read_bytes(), b"cache-bytes")


class StageReplayVerifyTests(unittest.TestCase):
    def _run_with_fake_coverage(self, coverage: dict):
        from unittest.mock import MagicMock, patch
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            workdir = Path(directory)
            (workdir / "coverage.json").write_text(json.dumps(coverage), encoding="utf-8")

            fake_lane_result = MagicMock()
            fake_lane_result.binary_ref.path = "/fake/bin/llama-server"
            fake_profile = MagicMock()
            fake_profile.production_context = 4096
            fake_profile.server_args = ()

            with patch.object(workflow, "ServerRunner") as fake_runner_cls:
                fake_runner = MagicMock()
                fake_runner_cls.return_value.__enter__.return_value = fake_runner
                return workflow._stage_replay_verify(
                    lane_result=fake_lane_result, model_path=Path("/fake/model.gguf"),
                    devices="0,1", runtime_profile=fake_profile,
                    dispatch_cache=workdir / "dispatch.cache", workdir=workdir,
                )

    def test_raises_on_stale_coverage(self):
        # This is exactly the HI130 defect this fix targets: a stale cache
        # must never be reported as a quiet success.
        with self.assertRaises(workflow.TuneCampaignError):
            self._run_with_fake_coverage({"stale": True, "rerun_required": 0})

    def test_raises_on_rerun_required(self):
        with self.assertRaises(workflow.TuneCampaignError):
            self._run_with_fake_coverage({"stale": False, "rerun_required": 3})

    def test_returns_coverage_when_clean(self):
        coverage = {"stale": False, "rerun_required": 0, "exact": 64, "misses": 0}
        result = self._run_with_fake_coverage(coverage)
        self.assertEqual(result, coverage)


class RunTuneCampaignReplayOrderingTests(unittest.TestCase):
    def test_replay_is_built_before_export_and_export_targets_replays_own_manifest(self):
        # HI130's actual root-cause bug: the cache used to be exported
        # against tune's manifest_ref BEFORE the replay lane was even built,
        # so every entry was stamped with a hash the replay binary could
        # never match. This pins the corrected order and data flow so it
        # cannot silently regress: _stage_replay_build must run first, and
        # _stage_replay_export's target_manifest_path/target_source_root
        # must come from THAT result, never from the tune stage.
        from unittest.mock import MagicMock, patch
        import tempfile

        def fake_lane_result(run_id, manifest_path, source_root):
            result = MagicMock()
            result.run_id = run_id
            result.source_slice_id = "slice1"
            result.build_plan_id = "plan1"
            result.manifest_ref.path = manifest_path
            result.source_root = Path(source_root)
            return result

        calls = []

        def fake_stage_replay_build(**kwargs):
            calls.append(("build", kwargs))
            return fake_lane_result("replay-run", "/replay/own-manifest.json", "/replay/own-source-root")

        def fake_stage_replay_export(**kwargs):
            calls.append(("export", kwargs))
            return Path("/fake/dispatch.cache")

        def fake_stage_replay_verify(**kwargs):
            calls.append(("verify", kwargs))
            return {"exact": 64, "stale": False, "rerun_required": 0}

        with tempfile.TemporaryDirectory() as directory:
            workdir = Path(directory)
            (workdir / "promoted.jsonl").write_text("", encoding="utf-8")

            tune_result = fake_lane_result("tune-run", "/tune/manifest.json", "/tune/source-root")
            record_result = fake_lane_result("record-run", None, "/record/source-root")
            record_result.manifest_ref = None

            with (
                patch.object(workflow, "_stage_record",
                             return_value=(record_result, workdir / "record.jsonl")),
                patch.object(workflow, "_stage_inventory_record",
                             return_value=(workdir / "inventory.json", workdir / "inventory.sqlite")),
                patch.object(workflow, "_stage_tune",
                             return_value=(tune_result, workdir / "tune.measurements.jsonl")),
                patch.object(workflow, "_stage_signature_verifier",
                             return_value=(
                                 fake_lane_result("verifier-run", "/verifier/manifest.json", "/verifier/source-root"),
                                 lambda canonical: "0" * 32,
                             )),
                patch.object(workflow, "_stage_load_and_promote",
                             return_value=(workdir / "tune.sqlite", {}, 19, 0)),
                patch.object(workflow, "_stage_replay_build", side_effect=fake_stage_replay_build),
                patch.object(workflow, "_stage_replay_export", side_effect=fake_stage_replay_export),
                patch.object(workflow, "_stage_replay_verify", side_effect=fake_stage_replay_verify),
                patch.object(workflow.gpu_mod, "preflight_context"),
            ):
                fake_profile = MagicMock()
                fake_profile.tune_context = 4096
                fake_profile.production_context = 64000
                fake_profile.server_args = ()
                fake_cfg = MagicMock()
                fake_cfg.runtime_profiles = {"production-dual-xtx": fake_profile}
                fake_context = MagicMock()
                fake_context.work_root = workdir

                workflow.run_tune_campaign(
                    context=fake_context, cfg=fake_cfg, store=MagicMock(),
                    model_path=Path("/fake/model.gguf"), platform_name="linux-multi",
                    devices="0,1", runtime_profile_name="production-dual-xtx",
                    run_id="test-run",
                )

        stage_order = [name for name, _ in calls]
        self.assertEqual(stage_order, ["build", "export", "verify"])

        export_kwargs = dict(calls[1][1])
        self.assertEqual(export_kwargs["target_manifest_path"], Path("/replay/own-manifest.json"))
        self.assertEqual(export_kwargs["target_source_root"], Path("/replay/own-source-root"))
        # The tune manifest must NOT leak into the export call.
        self.assertNotEqual(export_kwargs["target_manifest_path"], Path("/tune/manifest.json"))


class StageIdentityTests(unittest.TestCase):
    def test_uses_the_real_lane_run_id_not_a_reconstructed_one(self):
        # gpt review (2026-08-27): _stage_identity used to take a
        # SEPARATE run_id string reconstructed by the caller
        # (f"{campaign_run_id}-record") rather than the real per-lane
        # run_id run_campaign() actually used -- the two are different
        # strings, and only result.run_id matches the ArtifactStore
        # paths/provenance that build actually produced.
        from unittest.mock import MagicMock
        fake_result = MagicMock()
        fake_result.run_id = "real-lane-run-id-from-run-campaign"
        fake_result.source_slice_id = "slice1"
        fake_result.build_plan_id = "plan1"
        fake_result.manifest_ref = None
        fake_result.source_root = Path("/some/source/root")

        identity = workflow._stage_identity(fake_result)
        self.assertEqual(identity.run_id, "real-lane-run-id-from-run-campaign")


class StageLoadAndPromoteVerifierWiringTests(unittest.TestCase):
    """HI125 close-out step 6: production ingest is mandatory-strengthened
    -- signature_digest_verifier has no default, and a RecordError from
    load_measurements() (a verification failure or unsupported-domain
    signature) must abort as TuneCampaignError, never silently retry
    unverified."""

    def test_record_error_from_load_measurements_becomes_tune_campaign_error(self):
        from unittest.mock import patch
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            workdir = Path(directory)
            measurements = workdir / "tune.measurements.jsonl"
            measurements.write_text("", encoding="utf-8")
            manifest = workdir / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")

            def raising_load_measurements(*_args, **_kwargs):
                raise workflow.inv_mod.RecordError("signature verifier rejected a canonical")

            with patch.object(workflow.inv_mod, "load_measurements", side_effect=raising_load_measurements):
                with self.assertRaisesRegex(workflow.TuneCampaignError, "strengthened ingest failed"):
                    workflow._stage_load_and_promote(
                        tune_measurements=measurements, tune_manifest_path=manifest,
                        workdir=workdir, q=0.05, threshold_pct=1.0, resamples=100,
                        signature_digest_verifier=lambda _c: "0" * 32,
                    )

    def test_signature_digest_verifier_is_forwarded_to_load_measurements(self):
        from unittest.mock import patch
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            workdir = Path(directory)
            measurements = workdir / "tune.measurements.jsonl"
            measurements.write_text("", encoding="utf-8")
            manifest = workdir / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            promoted_path = workdir / "promoted.jsonl"
            promoted_path.write_text("", encoding="utf-8")

            captured = {}

            def fake_load_measurements(*_args, **kwargs):
                captured["signature_digest_verifier"] = kwargs.get("signature_digest_verifier")
                return {"results": 0, "measurements": 0, "candidates": 0}

            verifier = lambda _c: "0" * 32  # noqa: E731

            with (
                patch.object(workflow.inv_mod, "load_measurements", side_effect=fake_load_measurements),
                patch.object(workflow.tune_promotion, "promote", return_value={"promoted": 0}),
                patch.object(workflow, "_count_missing_correctness_evidence", return_value=0),
            ):
                workflow._stage_load_and_promote(
                    tune_measurements=measurements, tune_manifest_path=manifest,
                    workdir=workdir, q=0.05, threshold_pct=1.0, resamples=100,
                    signature_digest_verifier=verifier,
                )

            self.assertIs(captured["signature_digest_verifier"], verifier)


class StageSignatureVerifierTests(unittest.TestCase):
    def test_uses_record_build_with_hi105_correctness_experiment(self):
        # Must build from the SAME lane the HI125 verifier needs to be
        # RECORD-capable AND cover MUL_MAT_ID/routed-GLU (the hi105-
        # correctness experiment's extra patches) -- not the plain tune
        # build, which lacks GGML_HIP_AUTOTUNE_RECORD entirely.
        from unittest.mock import MagicMock, patch

        fake_result = MagicMock()
        fake_result.binary_ref.path = Path("/fake/bin/test-backend-ops")
        fake_result.source_root = Path("/fake/source-root")

        captured = {}

        def fake_plan_and_run(**kwargs):
            captured.update(kwargs)
            return fake_result

        with (
            patch.object(workflow, "_plan_and_run_one_lane", side_effect=fake_plan_and_run),
            patch.object(workflow.sdv, "make_signature_digest_verifier", return_value=lambda c: "0" * 32),
        ):
            lane_result, verifier = workflow._stage_signature_verifier(
                context=None, cfg=None, store=None, run_id="run-1",
                platform_name="linux-multi", source_name="bigcherry", devices="0",
            )

        self.assertEqual(captured["build_name"], "record")
        self.assertEqual(captured["binary_relative_path"], "bin/test-backend-ops")
        self.assertEqual(captured["experiment"], "hi105-correctness")
        self.assertIs(lane_result, fake_result)

    def test_gpu_scoped_runner_injects_hip_visible_devices_without_dropping_caller_env(self):
        from unittest.mock import patch

        runner = workflow._gpu_scoped_test_backend_ops_runner("0,1")
        captured = {}

        def fake_subprocess_run(argv, **kwargs):
            captured["argv"] = argv
            captured["env"] = kwargs.get("env")
            return "fake-result"

        with patch.object(workflow.subprocess, "run", side_effect=fake_subprocess_run):
            result = runner(["test-backend-ops"], env={"GGML_HIP_DISPATCH_DB": "/x"})

        self.assertEqual(result, "fake-result")
        self.assertEqual(captured["env"]["HIP_VISIBLE_DEVICES"], "0,1")
        self.assertEqual(captured["env"]["GGML_HIP_DISPATCH_DB"], "/x")


if __name__ == "__main__":
    unittest.main()

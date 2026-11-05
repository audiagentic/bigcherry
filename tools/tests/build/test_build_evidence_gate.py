"""MEN28 offline evidence gate: per-engine modes, one G4 authority, no GPU."""
from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from bigcherry.core import config
from bigcherry.campaign import lane
from bigcherry.build import logs
from bigcherry.patch import build_evidence, gates
from bigcherry.release import notes, pin_bump
from bigcherry.source.workspace import SourcePlan


class EvidenceGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        path = self.root / "recipes.toml"
        path.write_text(
            """version = 2
engine = "radiance"
pinned = "v1.17.0"
[patch-set.valid]
patches = ["rad0002_moe_remote_handoff", "rad0008_hc_upmix_rt_sweep"]
[patch-set.experimental]
patches = ["rad0005_mp4_ffmpeg6_writer"]
required-state = "untested"
[source.radiance]
overlay = false
patch-sets = ["valid", "experimental"]
""", encoding="utf-8")
        self.cfg = config.load(path)
        self.descriptors = {x: SimpleNamespace(patch_id=x) for x in (
            "rad0002_moe_remote_handoff", "rad0008_hc_upmix_rt_sweep"
        )}

    def _evaluate(self, responses, mode="enforced"):
        def g4(ctx):
            return responses[ctx.descriptor.patch_id]
        with (
            patch.object(build_evidence.registry, "load_registry", return_value=SimpleNamespace(by_id=self.descriptors)),
            patch.object(build_evidence.patchset, "resolve_exact", return_value=object()),
            patch.object(gates, "evaluate_evidence_gate", side_effect=g4) as shared,
        ):
            outcome = build_evidence.evaluate_source(
                engine="radiance", cfg=self.cfg, source_name="radiance",
                patches_dir=self.root, pinned_ref="v1.17.0",
                resolved_base_revision="a" * 40, mode=mode,
            )
        self.assertEqual(shared.call_count, 2)
        self.assertEqual(tuple(c.args[0].intent for c in shared.call_args_list),
                         (gates.GateIntent.BUILD,) * 2)
        return outcome

    def test_stale_and_missing_records_are_named_and_untested_is_exempt(self):
        bad = gates.GateResult(gates.GateId.G4, gates.GateStatus.FAIL, "evidence",
                               "patch.catalog", ("stale record: wrong pin",))
        missing = gates.GateResult(gates.GateId.G4, gates.GateStatus.BLOCKED,
                                   "evidence", "patch.catalog", ("no record",))
        outcome = self._evaluate(dict.fromkeys(self.descriptors, bad) |
                                 {"rad0008_hc_upmix_rt_sweep": missing})
        self.assertEqual(len(outcome.failed), 2)
        self.assertEqual(outcome.skipped, (("experimental", "untested"),))
        error = build_evidence.failure_text(outcome)
        self.assertIn("rad0002_moe_remote_handoff", error)
        self.assertIn("stale record: wrong pin", error)
        self.assertIn("rad0008_hc_upmix_rt_sweep", error)
        self.assertIn("no record", error)
        self.assertIn("patch-gates --intent build", error)
        self.assertIn("bigcherry-patch-qualification", error)
        self.assertIn("evidence gate skipped for set experimental (required-state untested)",
                      build_evidence.report_lines(outcome))

    def test_all_qualify_carried_forward_is_logged_with_original_pin(self):
        passed = gates.GateResult(gates.GateId.G4, gates.GateStatus.PASS,
                                  "evidence", "patch.catalog")
        carried = gates.GateResult(gates.GateId.G4, gates.GateStatus.PASS,
                                   "evidence", "patch.catalog",
                                   ("qualified at v1.14.0; not revalidated at v1.17.0",))
        result = self._evaluate({"rad0002_moe_remote_handoff": carried,
                                 "rad0008_hc_upmix_rt_sweep": passed})
        self.assertFalse(result.failed)
        self.assertEqual(result.record()["carried_forward"],
                         [{"patch": "rad0002_moe_remote_handoff", "qualified_at": "v1.14.0"}])
        self.assertIn("qualified-at v1.14.0", " ".join(build_evidence.report_lines(result)))

    def test_pending_warning_is_also_saved_in_build_summary_and_enforced_fails_pre_materialize(self):
        (self.root / "config").mkdir()
        (self.root / "config" / "release.toml").write_text("release fixture\n")
        ctx = SimpleNamespace(engine="radiance", project_root=self.root,
                              patches_root=self.root, upstream_repo=self.root,
                              work_root=self.root)
        spec = lane.CampaignLaneExecutionSpec(
            source_name="radiance", build_name="stock",
            platform_name="test", architectures=())
        audit = build_evidence.EvidenceAudit(
            "radiance", "radiance", "v1.17.0",
            "pending-requalification", tuple(self.descriptors),
            (("rad0002_moe_remote_handoff", "missing evidence"),),
            (), (("experimental", "untested"),))
        fake_logs = SimpleNamespace(
            lane="some-lane", summary={"lanes": {"some-lane": {}}}, save=MagicMock())
        token = logs.current.set(fake_logs)
        self.addCleanup(logs.current.reset, token)
        def invoke(mode):
            with (
                patch.object(lane.campaign_source, "source_plan_for",
                             return_value=SourcePlan("v1.17.0", False, tuple(self.descriptors))),
                patch.object(lane.UpstreamRepository, "resolve_ref", return_value="a" * 40),
                patch.object(lane.campaign_source, "resolve_materialization_identity", return_value={}),
                patch.object(lane.campaign_source, "materialization_plan_id", return_value="test-source"),
                patch.object(lane.release_notes, "load_config",
                             return_value=SimpleNamespace(evidence_gate=mode)),
                patch.object(lane.build_evidence, "evaluate_source",
                             return_value=build_evidence.EvidenceAudit(
                                 audit.engine, audit.source, audit.pin, mode,
                                 audit.validated, audit.failed, audit.carried_forward, audit.skipped)),
                patch.object(lane.campaign_plan, "materialize_stage_graph",
                             side_effect=LookupError("beyond the gate")),
                patch.object(lane.campaign_build, "materialize_source") as materializer,
            ):
                buffer = io.StringIO()
                with redirect_stderr(buffer):
                    if mode == "pending-requalification":
                        with self.assertRaisesRegex(LookupError, "beyond the gate"):
                            lane._execute_materialize_phase(
                                spec, cfg=self.cfg, context=ctx, store=object(), run_id="test",
                                campaign_root=self.root, resource_root=self.root,
                                project_revision="", local_provenance_class="development")
                    else:
                        with self.assertRaisesRegex(lane.campaign_build.CampaignBuildError,
                                                    "missing evidence"):
                            lane._execute_materialize_phase(
                                spec, cfg=self.cfg, context=ctx, store=object(), run_id="test",
                                campaign_root=self.root, resource_root=self.root,
                                project_revision="", local_provenance_class="development")
                materializer.assert_not_called()
                return buffer.getvalue()
        printed = invoke("pending-requalification")
        self.assertIn("evidence gate pending for radiance: 1 of 2 validated patches "
                      "do not qualify at v1.17.0 (MEN28)", printed)
        self.assertIn("rad0002_moe_remote_handoff", printed)
        self.assertEqual(fake_logs.summary["lanes"]["some-lane"]["evidence_gate"]["pending_count"], 1)
        self.assertEqual(fake_logs.summary["lanes"]["some-lane"]["evidence_gate"]["mode"],
                         "pending-requalification")
        self.assertEqual(fake_logs.summary["lanes"]["some-lane"]["evidence_gate"]["message"],
                         audit.pending_line)
        self.assertEqual(fake_logs.summary["lanes"]["some-lane"]["evidence_gate"]["first_five_patch_ids"],
                         ["rad0002_moe_remote_handoff"])
        invoke("enforced")

    def test_missing_mode_in_either_engine_is_a_load_error(self):
        root = Path(__file__).resolve().parents[3]
        text = (root / "config" / "release.toml").read_text()
        (self.root / "config").mkdir(exist_ok=True)
        (self.root / "config" / "release.toml").write_text(
            text.replace('evidence-gate = "pending-requalification"', ""))
        with self.assertRaisesRegex(notes.ReleaseNotesError, "llamacpp.*evidence-gate"):
            notes.load_config(self.root, "radiance")


class PinBumpEvidencePhaseTests(unittest.TestCase):
    def test_enforced_stops_and_resumes_after_evidence_is_repaired(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = SimpleNamespace(evidence_gate={}, save=MagicMock())
            failures = (("rad0002_moe_remote_handoff", "stale record"),)
            base = build_evidence.EvidenceAudit(
                "radiance", "radiance", "v1.17.0", "enforced",
                ("rad0002_moe_remote_handoff",), failures, (), ())
            def evaluate(*, repaired=False, **kw):
                return build_evidence.EvidenceAudit(
                    base.engine, base.source, base.pin, base.mode, base.validated,
                    () if repaired else base.failed, (), ())
            with (
                patch.object(pin_bump.release_notes, "load_config",
                             return_value=SimpleNamespace(evidence_gate="enforced")),
                patch.object(pin_bump.engines, "load",
                             return_value=SimpleNamespace(recipes_path=lambda root: root / "recipes.toml",
                                                         patches_root=lambda root: root / "patches",
                                                         promotion_source="radiance")),
                patch("bigcherry.core.config.load", return_value=SimpleNamespace(sources={"radiance": object(), "bigcherry": object()})),
                patch.object(pin_bump.releases, "record_for_checkout", return_value=record),
                patch.object(pin_bump.build_evidence, "evaluate_source", side_effect=evaluate) as shared,
            ):
                kw = dict(engine="radiance", source_name="radiance",
                          target_ref="v1.17.0", to_sha="a" * 40,
                          repo_root=root, vendor_root=root, report_dir=root / "report")
                with self.assertRaises(pin_bump.PinBumpStop) as stop:
                    pin_bump.run_phase_evidence(**kw)
                self.assertEqual(stop.exception.phase, "evidence")
                self.assertEqual(stop.exception.code, "EVIDENCE_GATE_STALE")
                self.assertIn("--resume", " ".join(stop.exception.recommended_actions))
                self.assertEqual(record.evidence_gate["pending_count"], 1)
                self.assertTrue((root / "report" / "evidence-gate.json").exists())
                shared.side_effect = lambda **kwargs: evaluate(repaired=True)
                result = pin_bump.run_phase_evidence(**kw)
                self.assertEqual(result["result"], "passed")
                self.assertEqual(result["pending_count"], 0)
                self.assertEqual(record.evidence_gate["mode"], "enforced")
                self.assertEqual(shared.call_count, 2)

    def test_pending_mode_carries_the_failure_count_without_stopping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = SimpleNamespace(evidence_gate={}, save=MagicMock())
            audit = build_evidence.EvidenceAudit(
                "llamacpp", "bigcherry", "b11552", "pending-requalification",
                ("0800_example", "0810_example"),
                (("0800_example", "stale"),), (), ())
            with (
                patch.object(pin_bump.release_notes, "load_config",
                             return_value=SimpleNamespace(evidence_gate="pending-requalification")),
                patch.object(pin_bump.engines, "load",
                             return_value=SimpleNamespace(recipes_path=lambda root: root / "recipes.toml",
                                                         patches_root=lambda root: root / "patches",
                                                         promotion_source="bigcherry")),
                patch("bigcherry.core.config.load", return_value=SimpleNamespace(sources={"radiance": object(), "bigcherry": object()})),
                patch.object(pin_bump.releases, "record_for_checkout", return_value=record),
                patch.object(pin_bump.build_evidence, "evaluate_source", return_value=audit),
            ):
                capture = io.StringIO()
                with redirect_stdout(capture):
                    result = pin_bump.run_phase_evidence(
                        engine="llamacpp", source_name="bigcherry",
                        target_ref="b11552", to_sha="a" * 40,
                        repo_root=root, vendor_root=root, report_dir=root / "report")
                self.assertEqual(result["pending_count"], 1)
                self.assertEqual(result["mode"], "pending-requalification")
                self.assertIn("evidence gate pending for llamacpp: 1 of 2", capture.getvalue())


if __name__ == "__main__":
    unittest.main()

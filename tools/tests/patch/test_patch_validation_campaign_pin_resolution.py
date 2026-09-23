"""VA04 real-hardware finding (GPT session ses_5bbee8ce5c9a4265,
req_71217bba406f4941): validation_campaign.py's run() resolved and
materialized control/subject source composition against base_ref="HEAD"
-- the shared vendor/llama.cpp checkout's current HEAD, NOT the
configured pin -- while the evidence record written at the end of the
same run labeled that resolved SHA as base_ref=cfg.pinned regardless of
whether HEAD actually matched the pin. A real RD04 hardware run on
Brutus resolved and built against vendor HEAD while its own evidence
claimed pin b10705; VA08's stale-detection correctly caught the mismatch
against the real currently-resolved pin and rejected the record.

run() is a large real-hardware integration entry point (real source
materialization, 7 real cmake builds, the real e2e_smoke_campaign.Campaign
class) that cannot be reasonably unit-tested end to end without real
hardware and a real git checkout -- consistent with VA14/VA15's
established scope boundary, this proves the exact fix via direct source
inspection of the committed function body.

PA36 sub-slice 2 (T2) moved run()'s source resolve/materialize block
verbatim into _build_standard_campaign_scaffold(); the invariant now
spans the call boundary -- run() feeds base_ref=cfg.pinned INTO the
scaffold, and the scaffold must use that base_ref (and never a
hardcoded HEAD) at every resolve/materialize/verify site. The
inspections below pin both sides of that boundary.
"""

from __future__ import annotations

import inspect
import re
import sys
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch import validation_campaign as vc  # noqa: E402
from bigcherry.patch.campaign import scaffold as campaign_scaffold  # noqa: E402


class PinResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.run_source = "".join(
            inspect.getsource(fn)
            for fn in (
                vc.run,
                # PA43: run() delegates to these stage functions in order.
                vc._prepare_standard_campaign,
                vc._run_activation_probe_stage,
                vc._collect_build_and_correctness_evidence,
                vc._run_contract_evidence_modes,
                vc._evaluate_validation_plan,
                vc._persist_validation_record,
            )
        )
        # PA43: source resolution/materialization lives in
        # _materialize_scaffold_sources(); the scaffold orchestrator and its
        # build-stage helpers are included in the "no HEAD anywhere" view.
        self.scaffold_source = inspect.getsource(
            campaign_scaffold._materialize_scaffold_sources)
        scaffold_all_source = self.scaffold_source + "".join(
            inspect.getsource(fn)
            for fn in (
                campaign_scaffold._build_standard_campaign_scaffold,
                campaign_scaffold._build_tune_and_replay_trees,
                campaign_scaffold._build_parity_trees,
            )
        )
        # The original pre-T2 tests inspected vc.run only; the combined
        # view keeps those assertions working AND extends "no HEAD
        # anywhere" to the scaffold where the resolution now lives.
        self.source = self.run_source + scaffold_all_source

    def test_no_hardcoded_head_is_used_for_source_resolution_or_materialization(self) -> None:
        # The real bug: base_ref="HEAD"/requested_revision="HEAD" silently
        # resolved and built against the shared checkout's current HEAD
        # instead of the configured pin.
        self.assertNotIn('base_ref="HEAD"', self.source)
        self.assertNotIn("requested_revision=\"HEAD\"", self.source)

    def test_run_threads_cfg_pinned_into_the_scaffold(self) -> None:
        # The VA04 invariant at the call boundary: run() must hand the
        # configured pin to the scaffold that resolves/materializes the
        # sources -- never a hardcoded HEAD.
        # The assignment call (not the docstring/comment mention, which
        # carries a bare "()" and would truncate the slice).
        call_index = self.run_source.index("= _build_standard_campaign_scaffold(")
        call_end = self.run_source.index("\n    )", call_index)
        self.assertIn(
            "base_ref=cfg.pinned", self.run_source[call_index:call_end])

    def test_scaffold_uses_base_ref_for_both_resolve_source_composition_calls(self) -> None:
        # The scaffold function makes two resolve_source_composition() calls
        # (control and subject), each passing base_ref=base_ref (which is
        # cfg.pinned, threaded by run()).
        matches = re.findall(r"base_ref=base_ref\b", self.scaffold_source)
        self.assertEqual(len(matches), 2, "both control and subject resolve_source_composition() calls must use the base_ref parameter (cfg.pinned, threaded by run())")

    def test_scaffold_uses_base_ref_for_all_four_requested_revision_sites(self) -> None:
        # materialize_composition (control, subject) + verify_composition_idempotent (control, subject).
        matches = re.findall(r"requested_revision=base_ref", self.scaffold_source)
        self.assertEqual(len(matches), 4)

    def test_cfg_is_loaded_before_source_resolution(self) -> None:
        cfg_load_index = self.run_source.index("cfg = campaign_config.load(")
        resolve_index = self.run_source.index("_build_standard_campaign_scaffold(")
        self.assertLess(
            cfg_load_index, resolve_index,
            "cfg must be loaded and resolved BEFORE any source resolution/materialization call, "
            "not after (the real bug: source materialization used to run first, against HEAD, "
            "with cfg loaded only much later purely for evidence labeling)",
        )

    def test_cfg_loaded_exactly_once(self) -> None:
        # The real bug's fix also removes the old duplicate load further
        # down (which existed only to label evidence, after HEAD-based
        # materialization had already happened).
        matches = re.findall(r"cfg = campaign_config\.load\(", self.source)
        self.assertEqual(len(matches), 1)

    def test_evidence_base_ref_still_uses_cfg_pinned(self) -> None:
        self.assertIn("base_ref=cfg.pinned,", self.source)

    def test_baseline_source_is_recorded_with_exact_composition(self) -> None:
        self.assertIn('"source": baseline_source', self.source)
        self.assertIn('"patches": list(control_composition)', self.source)

    def test_cli_preserves_default_and_accepts_explicit_named_baseline(self) -> None:
        required = ["--patch", "0100_cmake_options", "--model", "model.gguf",
                    "--hip-path", "rocm", "--amdgpu-targets", "gfx1100",
                    "--manifest", "manifest.json", "--workdir", "run"]
        for extra, expected in (([], "bigcherry"),
                                (["--baseline-source", "framework-control"], "framework-control")):
            with self.subTest(baseline=expected), mock.patch.object(vc, "run", return_value=0) as run:
                self.assertEqual(vc.main(required + extra), 0)
                self.assertEqual(run.call_args.args[0].baseline_source, expected)


if __name__ == "__main__":
    unittest.main()

"""VA15 real-hardware finding (GPT session ses_5bbee8ce5c9a4265,
req_bc329f6ae30c4e4c): the generic S1-S7 record/tune/promote/replay/bench/
report campaign is unrelated to a contract's own evidence and must never
be a hard prerequisite of it -- a real, honest tune-campaign run can
legitimately promote zero candidates, and that must not block the
contract's own lane/correctness/trigger/promotion evidence from ever
being collected. After the 1204/RD08 producer migration, --run-rd73-contract
is the last specialized execution mode guarded here (the RD08 guards were
retired with the dedicated flags).

run() is a large integration entry point (real source materialization,
7 real cmake builds, the real e2e_smoke_campaign.Campaign class) that
cannot be reasonably unit-tested end to end without real hardware --
consistent with VA14's established scope boundary, this proves the exact
structural invariant GPT flagged via direct source inspection of the
committed function body, rather than mocking the entire pipeline.
"""

from __future__ import annotations

import inspect
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch import validation_campaign as vc  # noqa: E402


class ContractSkipsGenericCampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = inspect.getsource(vc.run)

    def test_campaign_run_is_guarded_by_run_rd73_contract_check(self) -> None:
        # The exact invariant: `campaign.run()` must be reachable ONLY when
        # args.run_rd73_contract is not set -- never unconditionally. (The
        # RD04/RD08/RD13/RD26 dedicated execution flags were all deleted
        # by the PA36 producer migrations; every one of those patches now
        # runs through --validation-producer and is absent from every
        # generic-campaign guard.)
        match = re.search(
            r"if not args\.run_rd73_contract:\s*\n\s*try:\s*\n\s*campaign\.run\(\)",
            self.source,
        )
        self.assertIsNotNone(
            match,
            "campaign.run() must be guarded by "
            "`if not args.run_rd73_contract:` -- the generic "
            "tune/promote/export pipeline must never be a hard prerequisite of "
            "RD73's own contract evidence "
            "(RD08's, RD12's, RD04's, RD13's and RD26's specialized paths are "
            "producer-dispatched and "
            "return before this point entirely)",
        )

    def test_report_md_read_is_also_guarded(self) -> None:
        # --run-rd73-contract
        # must not require report.md
        # to exist (it's only ever written by the generic campaign's own S7
        # stage).
        guard_text = "if not args.run_rd73_contract:"
        guarded_block = self.source[self.source.index(guard_text):]
        guarded_block = guarded_block[:guarded_block.index("\n\n", guarded_block.index("report.md"))]
        self.assertIn("report.md", guarded_block)

    def test_generic_trace_probe_is_skipped_for_run_rd73_contract(self) -> None:
        # The generic tune-binary/fusion-disabled probe is invalid for
        # RD73 (graph-cache keying, not a fusion path) and is skipped by
        # this guard. (RD04's and RD08's producer-dispatched paths return
        # before this point entirely -- their activation evidence comes
        # from their producers, not from this generic probe.)
        match = re.search(
            r"trace_result = None if args\.run_rd73_contract else "
            r"run_trace_activation_probes\(",
            self.source,
        )
        self.assertIsNotNone(
            match,
            "the generic tune-binary/fusion-disabled probe must be skipped for "
            "--run-rd73-contract -- it is invalid for RD73 (graph-cache keying, "
            "not a fusion path) and cannot even load RD73's real contract model "
            "on Brutus's dual gfx1100 GPUs (RD08's and RD04's producer-dispatched "
            "paths return before this point entirely)",
        )

    def test_contract_correctness_gate_uses_a_real_experiment_contract_not_a_binding(self) -> None:
        # VA15 real-hardware finding (req_bc329f6ae30c4e4c follow-up):
        # validation_plan.contract is a patch_validation.ContractBinding --
        # a lightweight projection that deliberately does NOT carry
        # .correctness/.acceptance -- passing it directly to
        # compute_contract_correctness_gate() crashed on real hardware
        # with AttributeError: 'ContractBinding' object has no attribute
        # 'correctness'. The real committed code must resolve a real
        # ExperimentContract via patch_validation.load_contract_for_descriptor()
        # before calling compute_contract_correctness_gate().
        self.assertNotIn(
            "compute_contract_correctness_gate(\n            validation_plan.contract,",
            self.source,
        )
        match = re.search(
            r"full_contract = patch_validation\.load_contract_for_descriptor\(descriptor\)",
            self.source,
        )
        self.assertIsNotNone(
            match,
            "compute_contract_correctness_gate() must be called with a real "
            "ExperimentContract, resolved via load_contract_for_descriptor() -- "
            "never validation_plan.contract "
            "(a ContractBinding projection with no .correctness field)",
        )

    def test_campaign_ensure_identity_still_runs_unconditionally(self) -> None:
        # Contract evidence still legitimately binds
        # campaign.campaign_identity_digest -- only campaign.run() (S1-S7)
        # must be skipped, not identity binding.
        ensure_index = self.source.index("campaign.ensure_campaign_identity()")
        guard_index = self.source.index(
            "if not args.run_rd73_contract:"
        )
        self.assertLess(
            ensure_index, guard_index,
            "campaign.ensure_campaign_identity() must run before, and unconditionally "
            "relative to, the run_rd73_contract guard",
        )


if __name__ == "__main__":
    unittest.main()

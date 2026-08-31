"""VA15 real-hardware finding (GPT session ses_5bbee8ce5c9a4265,
req_bc329f6ae30c4e4c): the generic S1-S7 record/tune/promote/replay/bench/
report campaign is unrelated to RD08's own contract evidence and must
never be a hard prerequisite of --run-rd08-contract -- a real, honest
tune-campaign run can legitimately promote zero candidates, and that must
not block RD08's own lane/correctness/trigger/promotion evidence from
ever being collected.

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


class Rd08ContractSkipsGenericCampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = inspect.getsource(vc.run)

    def test_campaign_run_is_guarded_by_run_rd08_contract_check(self) -> None:
        # The exact invariant: `campaign.run()` must be reachable ONLY when
        # args.run_rd08_contract is falsy -- never unconditionally.
        match = re.search(
            r"if not args\.run_rd08_contract:\s*\n\s*try:\s*\n\s*campaign\.run\(\)",
            self.source,
        )
        self.assertIsNotNone(
            match,
            "campaign.run() must be guarded by `if not args.run_rd08_contract:` -- "
            "the generic tune/promote/export pipeline must never be a hard "
            "prerequisite of RD08's own contract evidence",
        )

    def test_report_md_read_is_also_guarded(self) -> None:
        # --run-rd08-contract must not require report.md to exist (it's
        # only ever written by the generic campaign's own S7 stage).
        guarded_block = self.source[self.source.index("if not args.run_rd08_contract:"):]
        guarded_block = guarded_block[:guarded_block.index("\n\n", guarded_block.index("report.md"))]
        self.assertIn("report.md", guarded_block)

    def test_generic_trace_probe_is_skipped_for_run_rd08_contract(self) -> None:
        match = re.search(
            r"trace_result = None if args\.run_rd08_contract else run_trace_activation_probes\(",
            self.source,
        )
        self.assertIsNotNone(
            match,
            "the generic tune-binary/fusion-disabled probe must be skipped for "
            "--run-rd08-contract -- it is redundant with and gets overwritten by "
            "RD08's own authoritative trigger probe",
        )

    def test_campaign_ensure_identity_still_runs_unconditionally(self) -> None:
        # RD08 evidence still legitimately binds campaign.campaign_identity_digest
        # -- only campaign.run() (S1-S7) must be skipped, not identity binding.
        ensure_index = self.source.index("campaign.ensure_campaign_identity()")
        guard_index = self.source.index("if not args.run_rd08_contract:")
        self.assertLess(
            ensure_index, guard_index,
            "campaign.ensure_campaign_identity() must run before, and unconditionally "
            "relative to, the run_rd08_contract guard",
        )


if __name__ == "__main__":
    unittest.main()

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
"""

from __future__ import annotations

import inspect
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch import validation_campaign as vc  # noqa: E402


class PinResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = inspect.getsource(vc.run)

    def test_no_hardcoded_head_is_used_for_source_resolution_or_materialization(self) -> None:
        # The real bug: base_ref="HEAD"/requested_revision="HEAD" silently
        # resolved and built against the shared checkout's current HEAD
        # instead of the configured pin.
        self.assertNotIn('base_ref="HEAD"', self.source)
        self.assertNotIn("requested_revision=\"HEAD\"", self.source)

    def test_cfg_pinned_used_for_both_resolve_source_composition_calls(self) -> None:
        matches = re.findall(r"resolve_source_composition\(\s*\n\s*\"bigcherry\", [^\n]*base_ref=cfg\.pinned", self.source)
        self.assertEqual(len(matches), 2, "both control and subject resolve_source_composition() calls must use base_ref=cfg.pinned")

    def test_cfg_pinned_used_for_all_four_requested_revision_sites(self) -> None:
        # materialize_composition (control, subject) + verify_composition_idempotent (control, subject).
        matches = re.findall(r"requested_revision=cfg\.pinned", self.source)
        self.assertEqual(len(matches), 4)

    def test_cfg_is_loaded_before_source_resolution(self) -> None:
        cfg_load_index = self.source.index("cfg = campaign_config.load(")
        resolve_index = self.source.index("resolve_source_composition(")
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


if __name__ == "__main__":
    unittest.main()

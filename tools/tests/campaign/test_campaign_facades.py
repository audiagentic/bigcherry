"""RA38/TR13 guard for the retired campaign graph compatibility facade."""

from __future__ import annotations

import importlib
import unittest
from pathlib import Path

from bigcherry.campaign.graph import CampaignGraph, CampaignGraphError, StageNode


class CampaignFacadeRetirementTests(unittest.TestCase):
    def test_canonical_graph_identity_and_exports(self) -> None:
        canonical = importlib.import_module("bigcherry.campaign.graph")
        self.assertIs(CampaignGraph, canonical.CampaignGraph)
        self.assertIs(CampaignGraphError, canonical.CampaignGraphError)
        self.assertIs(StageNode, canonical.StageNode)
        self.assertEqual(canonical.__name__, "bigcherry.campaign.graph")

    def test_retired_root_facade_stays_removed(self) -> None:
        tools_root = Path(__file__).resolve().parents[2] / "bigcherry"
        self.assertFalse(
            (tools_root / "campaign_graph.py").exists(),
            "campaign_graph.py was retired by RA38 and must not return",
        )


if __name__ == "__main__":
    unittest.main()

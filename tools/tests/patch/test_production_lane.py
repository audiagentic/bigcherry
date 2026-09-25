"""PVPS05: production dual-GPU MTP no-regression lane (offline pieces)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bigcherry.patch.campaign import production_lane as pl


class ProductionLaneTests(unittest.TestCase):
    def test_verdict_uses_the_interval_not_the_point_estimate(self) -> None:
        self.assertTrue(pl.verdict({"ci95_low_pct": -0.9, "geometric_effect_pct": -0.2})["passed"])
        failed = pl.verdict({"ci95_low_pct": -1.4, "geometric_effect_pct": +0.1})
        self.assertFalse(failed["passed"])
        self.assertIn("-1.0%", failed["reason"])
        self.assertFalse(pl.verdict({"ci95_low_pct": None})["passed"])

    def test_acceptance_mean_ignores_missing_values(self) -> None:
        rows = [{"draft_acceptance": 0.6}, {"draft_acceptance": 0.8}, {}]
        self.assertAlmostEqual(pl.mean_acceptance(rows), 0.7)
        self.assertIsNone(pl.mean_acceptance([{}]))

    def test_target_resolves_from_host_env_and_needs_two_devices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "m.gguf"
            model.write_bytes(b"x")
            with mock.patch.dict(os.environ, {"BIGCHERRY_PRODUCTION_MODEL": str(model),
                                              "BIGCHERRY_PRODUCTION_DEVICES": "0,1"}):
                self.assertEqual(pl.production_target(), (Path(str(model).replace("\\", "/")), (0, 1)))
            with mock.patch.dict(os.environ, {"BIGCHERRY_PRODUCTION_MODEL": str(model),
                                              "BIGCHERRY_PRODUCTION_DEVICES": "0"}):
                with self.assertRaises(pl.ProductionLaneError):
                    pl.production_target()

    def test_corpus_is_tracked(self) -> None:
        self.assertTrue(pl.CORPUS.is_file(), pl.CORPUS)

    def test_server_args_are_the_production_shape(self) -> None:
        args = " ".join(pl.PRODUCTION_SERVER_ARGS)
        for flag in ("-sm tensor", "--spec-type draft-mtp", "--spec-draft-n-max 4", "--flash-attn on",
                     "-ctkd q8_0", "-ctvd q8_0", "-ub 512", "-b 2048"):
            self.assertIn(flag, args)


if __name__ == "__main__":
    unittest.main()

"""Fail-closed promotion gate for the replay-cache exporter (HI34/P0).

Restores the pre-reset invariant that a non-native winner cannot reach a
replay cache without passing tune_promotion.py's experiment-wide BH
correction first. Tests `_validate_promotion_gate` directly rather than the
full `build()` round trip -- that needs a real manifest/ggml.h/variant-field
fixture unrelated to this gate, and is exercised end to end on real hardware
separately.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import replay_cache  # noqa: E402


def entry(dispatch: str, winner: str, native: str = "native", **extra) -> dict:
    row = {"kind": "result", "dispatch": dispatch, "winner": winner, "native": native}
    row.update(extra)
    return row


class PromotionGateTests(unittest.TestCase):

    def test_native_winner_always_exports(self):
        entries = {"a" * 32: entry("a" * 32, "native", "native")}
        replay_cache._validate_promotion_gate(entries)  # must not raise

    def test_promoted_non_native_exports(self):
        entries = {"a" * 32: entry("a" * 32, "candidate", "native",
                                   promotion_status="promoted")}
        replay_cache._validate_promotion_gate(entries)  # must not raise

    def test_raw_pending_bh_cannot_export(self):
        entries = {"a" * 32: entry("a" * 32, "candidate", "native",
                                   promotion_status="pending_bh")}
        with self.assertRaisesRegex(SystemExit, "promotion_status"):
            replay_cache._validate_promotion_gate(entries)

    def test_rejected_bh_cannot_export(self):
        entries = {"a" * 32: entry("a" * 32, "candidate", "native",
                                   promotion_status="rejected_bh")}
        with self.assertRaises(SystemExit):
            replay_cache._validate_promotion_gate(entries)

    def test_missing_promotion_metadata_cannot_export(self):
        # A non-native winner with no promotion_status at all -- e.g. a
        # measurements file from before this gate existed. Fail closed, not
        # a silent pass-through.
        entries = {"a" * 32: entry("a" * 32, "candidate", "native")}
        with self.assertRaises(SystemExit):
            replay_cache._validate_promotion_gate(entries)

    def test_missing_native_field_treated_as_non_native(self):
        entries = {"a" * 32: {"kind": "result", "dispatch": "a" * 32,
                              "winner": "candidate"}}
        with self.assertRaises(SystemExit):
            replay_cache._validate_promotion_gate(entries)

    def test_mixed_batch_reports_every_violation(self):
        entries = {
            "a" * 32: entry("a" * 32, "native", "native"),
            "b" * 32: entry("b" * 32, "candidate", "native",
                            promotion_status="promoted"),
            "c" * 32: entry("c" * 32, "candidate", "native",
                            promotion_status="pending_bh"),
            "d" * 32: entry("d" * 32, "candidate", "native",
                            promotion_status="rejected_bh"),
        }
        with self.assertRaisesRegex(SystemExit, "2 non-native winner"):
            replay_cache._validate_promotion_gate(entries)


if __name__ == "__main__":
    unittest.main()

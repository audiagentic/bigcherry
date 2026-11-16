"""Offline safety guards for handoff to adaptive checkpointing."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[2] / "lab/radiance"
sys.path.insert(0, str(TOOLS))
PATH = TOOLS / "iq6r_v4_adaptive_handoff.py"
spec = importlib.util.spec_from_file_location("iq6r_v4_adaptive_handoff", PATH)
assert spec and spec.loader
handoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handoff)


class HandoffTests(unittest.TestCase):
    def test_missing_worker_prevents_any_signal(self):
        with (
            patch.object(handoff, "OLD", SimpleNamespace(read_text=lambda: '{"pid": 123, "next_stage": 3, "status": "queued_for_exclusive_lease"}')),
            patch.object(handoff, "parent_alive", return_value=True),
            patch.object(handoff, "worker_alive", return_value=False),
            patch.object(handoff.os, "kill") as kill,
            self.assertRaisesRegex(RuntimeError, "not actively owned"),
        ):
            handoff.main()
        kill.assert_not_called()

    def test_sealed_stage_requires_hash_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(handoff, "MICRO_ROOT", root):
                self.assertFalse(handoff.stage_sealed(3))

    def test_missing_or_inconsistent_stage4_refuses_handoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "micro-0004.recipe").write_text("next stage")
            with (
                patch.object(handoff, "MICRO_ROOT", Path(tmp)),
                patch.object(handoff, "OLD", SimpleNamespace(read_text=lambda: '{"pid": 123, "next_stage": 3, "status": "queued_for_exclusive_lease"}')),
                patch.object(handoff, "parent_alive", return_value=True),
                patch.object(handoff, "worker_alive", return_value=True),
                patch.object(handoff.os, "kill") as kill,
                self.assertRaisesRegex(RuntimeError, "already started"),
            ):
                handoff.main()
            kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()

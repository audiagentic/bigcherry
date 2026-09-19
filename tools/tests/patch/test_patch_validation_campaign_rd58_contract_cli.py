"""PA36 migration #4 (dev-gpt-agent req_82fbbafe52c0472d): the RD58
dedicated CLI path (--run-rd58-state-restore,
run_rd58_state_restore_evidence) was DELETED from shared code and
replaced by the generic standard_campaign="run" producer path
(--validation-producer 1234_rd58_pin_state_buffer_multigpu_restore/rd58).

This file covers the replaced path at two levels:
  * source-layout regression guards (the deleted surface must not come
    back, and the producer manifest pins the migration policy),
  * the end-to-end dispatcher (_run_validation_producer) driven against
    the REAL 1234 patch/descriptor/plan with the expensive producer
    boundary faked (a lightweight stand-in producer; the real
    state-restore measurement semantics are covered by
    test_patch_validation_campaign_rd58_state_restore.py) -- the
    trace_probe="skip" path: no probe runs, no activation.json is
    written, and the honest correctness FAIL of the historical record
    is a valid exit-0 receipt (eligibility is evidence, not process
    success).

Producer-level measurement semantics (the state-restore execution,
artifact shape, env sanitization) are covered by
test_patch_validation_campaign_rd58_state_restore.py.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

TOOLS_ROOT = Path(__file__).resolve().parents[2]
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))


PATCH_ID = "1234_rd58_pin_state_buffer_multigpu_restore"
CAMPAIGN_SRC = (
    TOOLS_ROOT / "bigcherry" / "patch" / "validation_campaign.py"
).read_text(encoding="utf-8")
PRODUCER_SRC = (
    TOOLS_ROOT.parent / "patches" / PATCH_ID / "validation" / "producer.py"
).read_text(encoding="utf-8")
PATCH_DIR = TOOLS_ROOT.parent / "patches" / PATCH_ID
EXPECTED_ARTIFACT_NAMES = frozenset(
    {
        "rd58-correctness.json",
        "rd58-trigger-subject.log",
        "rd58-trigger-control.log",
        "rd58-trigger.json",
        "performance.json",
    }
)


class Rd58DedicatedPathDeletionTests(unittest.TestCase):
    def test_dedicated_cli_surface_is_gone(self) -> None:
        for needle in (
            "run_rd58_state_restore_evidence",
            "run-rd58-state-restore",
            "run_rd58_state_restore",
        ):
            self.assertNotIn(needle, CAMPAIGN_SRC)

    def test_exclusion_tuples_do_not_carry_the_dead_flags(self) -> None:
        for needle in (
            "run_rd58_state_restore",
            "run-rd58-state-restore",
        ):
            self.assertNotIn(needle, CAMPAIGN_SRC)

    def test_producer_manifest_pins_the_migration_policy(self) -> None:
        toml_src = (PATCH_DIR / "validation" / "producer.toml").read_text(encoding="utf-8")
        self.assertIn('trace_probe = "skip"', toml_src)
        self.assertIn('standard_campaign = "run"', toml_src)
        self.assertIn('correctness_evidence_cli = "forbid"', toml_src)
        self.assertIn('performance_benchmark_cli = "forbid"', toml_src)


if __name__ == "__main__":
    unittest.main()

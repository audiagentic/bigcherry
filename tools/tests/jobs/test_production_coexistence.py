from __future__ import annotations

import unittest

from bigcherry.jobs.production import (
    CoexistenceMode,
    ProductionClaimError,
    build_snapshot,
    decide_coexistence,
    parse_gpu_claim,
    snapshot_contaminated,
)
from bigcherry.jobs.window import (
    WindowRecord,
    WindowRequest,
    WindowState,
    deadline_expired,
    transition,
)


ARCH = {"gfx1100": ("gpu-a", "gpu-b"), "gfx1201": ("gpu-c",)}


class ProductionClaimTests(unittest.TestCase):
    def test_uuid_claim_is_exact(self):
        claim = parse_gpu_claim("uuid:gpu-b,uuid:gpu-a", architecture_devices=ARCH)
        self.assertFalse(claim.all_devices)
        self.assertEqual(claim.stable_device_ids, ("gpu-a", "gpu-b"))

    def test_arch_count_is_conservative_over_all_candidates(self):
        claim = parse_gpu_claim("arch:gfx1100,count=1", architecture_devices=ARCH)
        self.assertEqual(claim.stable_device_ids, ("gpu-a", "gpu-b"))

    def test_bad_or_missing_claim_fails_closed_in_snapshot(self):
        with self.assertRaises(ProductionClaimError):
            parse_gpu_claim(None, architecture_devices=ARCH)
        snapshot = build_snapshot(
            config_hash="cfg", claims=(None,), architecture_devices=ARCH
        )
        self.assertTrue(snapshot.potential_all)
        self.assertTrue(snapshot.ambiguities)
        decision = decide_coexistence(("gpu-c",), snapshot)
        self.assertEqual(decision.mode, CoexistenceMode.EXCLUSIVE_WINDOW)

    def test_dynamic_conflict_not_gpu_class_policy(self):
        snapshot = build_snapshot(
            config_hash="cfg",
            claims=("uuid:gpu-c",),
            architecture_devices=ARCH,
        )
        self.assertEqual(
            decide_coexistence(("gpu-c",), snapshot).mode,
            CoexistenceMode.EXCLUSIVE_WINDOW,
        )
        self.assertEqual(
            decide_coexistence(("gpu-a",), snapshot).mode,
            CoexistenceMode.IDLE_ATTESTATION,
        )

    def test_watchdog_detects_config_or_target_use(self):
        baseline = build_snapshot(
            config_hash="cfg",
            claims=("uuid:gpu-a",),
            architecture_devices=ARCH,
        )
        changed = build_snapshot(
            config_hash="cfg2",
            claims=("uuid:gpu-a",),
            architecture_devices=ARCH,
        )
        self.assertTrue(
            snapshot_contaminated(baseline, changed, target_stable_ids=("gpu-c",))[0]
        )
        target_use = build_snapshot(
            config_hash="cfg",
            claims=("uuid:gpu-a",),
            architecture_devices=ARCH,
            observed_devices=("gpu-c",),
        )
        contaminated, reason = snapshot_contaminated(
            baseline, target_use, target_stable_ids=("gpu-c",)
        )
        self.assertTrue(contaminated)
        self.assertIn("target GPU", reason)


class WindowStateTests(unittest.TestCase):
    def request(self):
        return WindowRequest(
            window_id="window-1",
            execution_id="run:a1",
            target_device_ids=("gpu-a",),
            inventory_hash="inv",
            production_config_hash="cfg",
            deadline_unix_ns=100,
        )

    def test_happy_path_and_close_idempotency(self):
        record = WindowRecord(WindowState.CLOSED)
        record = transition(record, "request", request=self.request())
        self.assertEqual(record, transition(record, "request", request=self.request()))
        record = transition(record, "begin-drain")
        record = transition(record, "activate")
        record = transition(record, "close", reason="complete")
        self.assertEqual(record, transition(record, "close"))
        record = transition(record, "closed", reason="production healthy")
        self.assertEqual(record.state, WindowState.CLOSED)
        self.assertIsNone(record.request)

    def test_crash_recovery_never_skips_recovery_state(self):
        record = transition(
            WindowRecord(WindowState.CLOSED), "request", request=self.request()
        )
        record = transition(record, "begin-drain")
        record = transition(record, "recover", reason="service crash")
        self.assertEqual(record.state, WindowState.RECOVERY)
        record = transition(record, "closed", reason="forced partition down and production restored")
        self.assertEqual(record.state, WindowState.CLOSED)

    def test_deadline_expiry(self):
        record = transition(
            WindowRecord(WindowState.CLOSED), "request", request=self.request()
        )
        self.assertFalse(deadline_expired(record, now_unix_ns=99))
        self.assertTrue(deadline_expired(record, now_unix_ns=100))


if __name__ == "__main__":
    unittest.main()

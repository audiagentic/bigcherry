"""Offline HI54 pool-cache protocol validation tests."""

from __future__ import annotations

import unittest

from bigcherry.pool_protocol import PoolProtocolError, validate_pool_protocol


def _isolated(*events: str) -> list[dict[str, str]]:
    return [{"stage": "isolated_workspace", "event": event} for event in events]


class TestPoolProtocol(unittest.TestCase):
    def test_isolated_trace_requires_clear_warmup_sync_rebase_before_timing(self):
        validate_pool_protocol(_isolated(
            "clear_cache", "warmup_begin", "warmup_complete", "synchronize",
            "rebase_peak", "timed_sample_begin", "timed_sample_end",
        ))

    def test_isolated_trace_rejects_clear_after_warmup(self):
        with self.assertRaisesRegex(PoolProtocolError, "out of order"):
            validate_pool_protocol(_isolated(
                "warmup_begin", "warmup_complete", "clear_cache",
                "synchronize", "rebase_peak", "timed_sample_begin",
            ))

    def test_isolated_trace_rejects_rebase_before_synchronization(self):
        with self.assertRaisesRegex(PoolProtocolError, "out of order"):
            validate_pool_protocol(_isolated(
                "clear_cache", "warmup_begin", "warmup_complete",
                "rebase_peak", "synchronize", "timed_sample_begin",
            ))

    def test_isolated_trace_rejects_timing_before_rebase(self):
        with self.assertRaisesRegex(PoolProtocolError, "before pool rebase"):
            validate_pool_protocol(_isolated(
                "clear_cache", "warmup_begin", "warmup_complete",
                "synchronize", "timed_sample_begin", "rebase_peak",
            ))

    def test_isolated_trace_requires_a_complete_timed_sample(self):
        base = _isolated(
            "clear_cache", "warmup_begin", "warmup_complete", "synchronize",
            "rebase_peak",
        )
        with self.assertRaisesRegex(PoolProtocolError, "no timed samples"):
            validate_pool_protocol(base)
        with self.assertRaisesRegex(PoolProtocolError, "without an end"):
            validate_pool_protocol(base + [{"stage": "isolated_workspace",
                                            "event": "timed_sample_begin"}])

    def test_isolated_trace_rejects_duplicate_lifecycle_or_unpaired_timing(self):
        base = _isolated(
            "clear_cache", "warmup_begin", "warmup_complete", "synchronize",
            "rebase_peak", "timed_sample_begin", "timed_sample_end",
        )
        with self.assertRaisesRegex(PoolProtocolError, "exactly one clear_cache"):
            validate_pool_protocol(base[:1] + base)
        with self.assertRaisesRegex(PoolProtocolError, "without a begin"):
            validate_pool_protocol(base + [{"stage": "isolated_workspace",
                                            "event": "timed_sample_end"}])

    def test_isolated_trace_allows_multiple_non_overlapping_samples(self):
        validate_pool_protocol(_isolated(
            "clear_cache", "warmup_begin", "warmup_complete", "synchronize",
            "rebase_peak", "timed_sample_begin", "timed_sample_end",
            "timed_sample_begin", "timed_sample_end",
        ))

    def test_final_and_confirmation_cannot_interleave_pool_isolation(self):
        events = [
            {"stage": "final", "event": "timed_sample_begin"},
            {"stage": "confirmation", "event": "timed_sample_end"},
        ]
        validate_pool_protocol(events)
        for stage in ("final", "confirmation"):
            with self.assertRaisesRegex(PoolProtocolError, "may not"):
                validate_pool_protocol([
                    {"stage": stage, "event": "clear_cache"},
                ])

    def test_unknown_or_empty_traces_fail_closed(self):
        with self.assertRaisesRegex(PoolProtocolError, "empty"):
            validate_pool_protocol([])
        with self.assertRaisesRegex(PoolProtocolError, "unknown event"):
            validate_pool_protocol([{
                "stage": "isolated_workspace", "event": "allocator_reset",
            }])
        with self.assertRaisesRegex(PoolProtocolError, "unknown pool protocol stage"):
            validate_pool_protocol([{
                "stage": "bogus", "event": "timed_sample_begin",
            }])


if __name__ == "__main__":
    unittest.main()

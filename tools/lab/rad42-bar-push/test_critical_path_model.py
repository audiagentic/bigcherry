"""Card-free safety, overlap and break-even checks."""
from dataclasses import replace
import unittest

from critical_path_model import Inputs, bandwidth_us, simulate, break_even_signal_us


class CriticalPathTest(unittest.TestCase):
    def test_exact_chunk_bytes_and_last_chunk(self):
        x = Inputs(rows=513, hidden=16, chunk_rows=256)
        self.assertEqual(x.sizes(), [8192, 8192, 32])
        self.assertEqual(sum(x.sizes()), x.rows*x.hidden*x.element_bytes)

    def test_reject_invalid_inputs(self):
        for changes in ({"rows":0}, {"cpu_push_gbps":0},
                        {"combine_us_per_chunk":-1},
                        {"gpu_pull_gbps":float("nan")},
                        {"cpu_wake_us_per_chunk":float("inf")},
                        {"chunk_rows":0}, {"rows":4097}):
            with self.assertRaises(ValueError):
                simulate(replace(Inputs(), **changes), "baseline")
        with self.assertRaises(ValueError):
            simulate(Inputs(), "invalid")
        with self.assertRaises(ValueError):
            bandwidth_us(-1, 1)
        with self.assertRaises(ValueError):
            break_even_signal_us(Inputs(), "baseline")

    def test_one_chunk_exact(self):
        x = Inputs(rows=256, hidden=2560, chunk_rows=256,
                   gpu_pull_gbps=6.8, cpu_push_gbps=10.3,
                   producer_first_us=11, combine_us_per_chunk=20,
                   gpu_add_us_per_chunk=7, cpu_wake_us_per_chunk=3,
                   signal_us_per_chunk=4)
        b = simulate(x, "baseline")
        c = simulate(x, "copy")
        self.assertAlmostEqual(b.makespan_us, 11+3+20+bandwidth_us(1310720,6.8)+7)
        self.assertAlmostEqual(c.makespan_us, 11+3+20+
                               bandwidth_us(1310720,10.3)+4+7)

    def test_baseline_overlaps_cpu_combine_with_gpu_read(self):
        x = Inputs(rows=512, hidden=2560, chunk_rows=256,
                   producer_first_us=0, producer_interval_us=0,
                   combine_us_per_chunk=100, gpu_add_us_per_chunk=0,
                   gpu_pull_gbps=10)
        r = simulate(x, "baseline")
        one_transfer = bandwidth_us(x.sizes()[0], 10)
        # first GPU read overlaps CPU combining the second chunk
        self.assertAlmostEqual(r.makespan_us,
                               100+one_transfer+max(0,100-one_transfer)+one_transfer)

    def test_copy_after_combine_can_lose_despite_faster_link(self):
        x = Inputs(rows=2048, producer_interval_us=100,
                   combine_us_per_chunk=150, gpu_add_us_per_chunk=30,
                   gpu_pull_gbps=6.8, cpu_push_gbps=10.3)
        self.assertLess(simulate(x, "baseline").makespan_us,
                        simulate(x, "copy").makespan_us)

    def test_small_decode_wake_overhead_can_dominate(self):
        x = Inputs(rows=1, chunk_rows=1, combine_us_per_chunk=0,
                   gpu_add_us_per_chunk=0, signal_us_per_chunk=4,
                   cpu_wake_us_per_chunk=0)
        self.assertGreater(simulate(x, "copy").makespan_us,
                           simulate(x, "baseline").makespan_us)

    def test_more_signal_overhead_never_improves_result(self):
        x = Inputs(rows=2048)
        for mode in ("copy", "direct"):
            self.assertGreaterEqual(
                simulate(replace(x, signal_us_per_chunk=20), mode).makespan_us,
                simulate(x, mode).makespan_us
            )

    def test_direct_zero_cost_is_only_optimistic_lower_bound(self):
        x = Inputs(rows=2048)
        self.assertLessEqual(simulate(x, "direct").makespan_us,
                             simulate(replace(x, direct_output_extra_us_per_chunk=100),
                                      "direct").makespan_us)

    def test_break_even_threshold(self):
        x = Inputs(rows=256, producer_interval_us=0,
                   combine_us_per_chunk=0, gpu_add_us_per_chunk=0,
                   gpu_pull_gbps=6.8, cpu_push_gbps=10.3)
        cutoff = bandwidth_us(x.sizes()[0], 6.8) - bandwidth_us(x.sizes()[0], 10.3)
        self.assertAlmostEqual(break_even_signal_us(x, "copy"), cutoff, places=2)
        loser = replace(x, cpu_push_gbps=2)
        self.assertEqual(break_even_signal_us(loser, "copy"), -1)

    def test_uncontended_baseline_often_wins(self):
        x = Inputs(rows=2048, cpu_push_gbps=10.3, gpu_pull_gbps=13.3,
                   producer_interval_us=0, combine_us_per_chunk=0,
                   gpu_add_us_per_chunk=0)
        self.assertLess(simulate(x, "baseline").makespan_us,
                        simulate(x, "copy").makespan_us)


if __name__ == "__main__":
    unittest.main()

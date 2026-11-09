"""Card-free integrity checks for compact live GPU-ready benchmark records."""
import unittest
from slice_analyze import expected_offsets, parse, report, provisional


def fixture(mode="PARALLEL", device=0, rows=768, iterations=100, complete=True, blocks=256):
    control=("hip-register-mapped-direct-parallel" if mode=="PARALLEL"
             else "hip-host-coherent-relay")
    delivery=("child-after-dma-sync-end" if mode=="PARALLEL"
              else "parent-after-socket-ack")
    lines=[f"exitCode: 0", "timedOut: false", "truncated: false",
           f"RAD50_LIVE_CONTEXT device={device} pci=0000:03:00.0 rows={rows} "
           f"bytes={rows*5120} iterations={iterations} cpu_child=persistent "
           "consumer_armed_before_write=1 dma_buf_sync=1 ready_scope=system "
           f"reuse_after_done=1 control={control} gpu_parallel_blocks={blocks}"]
    offsets=expected_offsets(iterations)
    for offset in offsets:
        gen=100000+rows*1000+offset
        lines.append(f"RAD50_{mode}_SAMPLE device={device} rows={rows} "
            f"bytes={rows*5120} gen={gen} control=host "
            f"delivery={delivery} gpu_parallel_blocks={blocks} "
            "arm_us=1.00 cpu_combine_us=5.00 dma_sync_us=0.30 "
            "helper_ack_us=6.00 post_ack_complete_us=3.00 "
            f"serial_us=10.00 gpu_seen_ready={gen} wait_loops=3 "
            "mismatches=0 status=0")
    if complete:
        lines.append(f"RAD50_{mode}_SUMMARY device={device} rows={rows} "
            f"bytes={rows*5120} iterations={iterations} verified={iterations} "
            f"gpu_parallel_blocks={blocks} "
            "p50_us=10.00 p95_us=10.00 p99_us=10.00 max_us=10.00 "
            f"emitted={len(offsets)} combine_p50_us=5.00 arm_p50_us=1.00 "
            "ack_p50_us=6.00 post_ack_complete_p50_us=3.00 status=0")
        lines.append(f"RAD50_{mode}_COMPLETE device={device} rows={rows} "
                     f"iterations={iterations} verified={iterations} status=0")
        lines.append(f"RAD50 RESULT test=dmabuf-live-{mode.lower()} exit=0")
    return "\n".join(lines)


class SliceAnalysisTests(unittest.TestCase):
    def test_compact_generation_schedule(self):
        self.assertEqual(list(range(1,11)),expected_offsets(10))
        self.assertEqual([1,2,3,10,20,30],expected_offsets(30))
        self.assertEqual([1,2,3,10,20,30,40,50,60,70,80,90,100],
                         expected_offsets(100))
        self.assertEqual(23,len(expected_offsets(200)))  # 3 early + 20 every-tenth samples

    def test_compact_100_pass(self):
        self.assertEqual(1,len(parse(fixture())))
        self.assertIn("100 10.00 10.00",report(fixture()))

    def test_30_compact_and_relay(self):
        self.assertEqual(2,len(parse(fixture(iterations=30)+"\n"+
                                     fixture(mode="RELAY",device=1,iterations=30))))

    def test_full_10_pass(self):
        self.assertEqual(1,len(parse(fixture(iterations=10))))

    def test_grid_sweep_keeps_bitwise_proof(self):
        for blocks in (1,32,64,128,256):
            self.assertEqual(1,len(parse(fixture(iterations=10,blocks=blocks))))

    def test_mixed_grid_metadata_is_rejected(self):
        text=fixture(blocks=32).replace("gpu_parallel_blocks=32",
                                         "gpu_parallel_blocks=64",1)
        with self.assertRaisesRegex(ValueError,"wrong/stale"):
            parse(text)

    def test_unsupported_grid_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"invalid GPU verifier"):
            parse(fixture(blocks=7))

    def test_grid_summary_must_match_samples(self):
        # Corrupt only summary grid metadata instead of the context.
        lines=fixture(blocks=32).splitlines()
        lines=[line.replace("gpu_parallel_blocks=32","gpu_parallel_blocks=64")
               if line.startswith("RAD50_PARALLEL_SUMMARY ") else line for line in lines]
        with self.assertRaisesRegex(ValueError,"incomplete or mismatched"):
            parse("\n".join(lines))

    def test_truncated_rejected_but_provisional(self):
        text=fixture(complete=False)
        with self.assertRaisesRegex(ValueError,"incomplete"):
            parse(text)
        self.assertIn("NOT_VERIFIED=1",provisional(text))

    def test_missing_early_sample(self):
        text=fixture()
        start=text.index("RAD50_PARALLEL_SAMPLE ")
        end=text.index("\n",start)
        text=text[:start]+text[end+1:]
        with self.assertRaises(ValueError):
            parse(text)

    def test_stale_ticket(self):
        text=fixture().replace("gen=868010","gen=868009")
        with self.assertRaisesRegex(ValueError,"wrong/stale"):
            parse(text)

    def test_wrong_gpu_ticket(self):
        text=fixture().replace("gpu_seen_ready=868001","gpu_seen_ready=0")
        with self.assertRaisesRegex(ValueError,"wrong/stale"):
            parse(text)

    def test_invalid_count_and_emitted(self):
        with self.assertRaises(ValueError):
            parse(fixture().replace("emitted=13","emitted=100"))

    def test_failed_hardware(self):
        with self.assertRaisesRegex(ValueError,"hardware failure"):
            parse(fixture()+"\nRAD50_PARALLEL_FAIL reason=timeout")

    def test_wrong_status_or_nonzero_process_exit(self):
        with self.assertRaisesRegex(ValueError,"nonzero"):
            parse(fixture().replace("exitCode: 0","exitCode: 3"))

    def test_missing_completion(self):
        with self.assertRaisesRegex(ValueError,"incomplete"):
            parse(fixture().replace("RAD50_PARALLEL_COMPLETE device=0 rows=768 iterations=100 verified=100 status=0",""))

    def test_bad_quantile_order(self):
        with self.assertRaisesRegex(ValueError,"quantiles"):
            parse(fixture().replace("p95_us=10.00","p95_us=9.00"))

    def test_forged_10_sample_median(self):
        with self.assertRaisesRegex(ValueError,"disagrees"):
            parse(fixture(iterations=10).replace("p50_us=10.00","p50_us=9.00"))

    def test_reject_marked_truncation(self):
        with self.assertRaisesRegex(ValueError,"truncated"):
            parse(fixture().replace("truncated: false","truncated: true"))


if __name__=="__main__":
    unittest.main()

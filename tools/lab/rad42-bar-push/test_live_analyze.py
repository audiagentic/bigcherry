"""Card-free strict vs provisional behavior for RAD50 Gate 4."""
from pathlib import Path
import unittest
from live_analyze import parse, provisional, report

EVIDENCE=Path(__file__).with_name("evidence")/"2026-11-09"

def fixture(rows=16,dev=0,iterations=10,complete=True,fail=False):
    size=rows*5120
    lines=[f"RAD50_LIVE_CONTEXT device={dev} pci=0000:03:00.0 "
           f"rows={rows} bytes={size} iterations={iterations} "
           "cpu_child=persistent consumer_armed_before_write=1 "
           "dma_buf_sync=1 ready_scope=system reuse_after_done=1"]
    for i in range(iterations):
        gen=100000+rows*1000+i+1
        lines.append(f"RAD50_LIVE_SAMPLE device={dev} rows={rows} "
                     f"bytes={size} gen={gen} armed_before_helper=1 "
                     "stale_ready_retained=1 cpu_combine_us=1.0 "
                     "dma_sync_us=0.5 helper_ack_after_arm_us=5.0 "
                     f"serial_us=25.0 gpu_seen_ready={gen} "
                     "wait_loops=2 mismatches=0 status=0")
    if fail:
        lines.append(f"RAD50_LIVE_FAIL device={dev} rows={rows} "
                     "reason=wrong-fp16-or-ready-timeout errors=4294967295")
    if complete:
        lines.append(f"RAD50_LIVE_SUMMARY device={dev} rows={rows} "
                     f"bytes={size} iterations={iterations} verified={iterations} "
                     "total_p50_us=25.0 total_p95_us=25.0 "
                     "cpu_combine_p50_us=1.0 sync_p50_us=0.5 status=0")
        lines.append(f"RAD50_LIVE_COMPLETE device={dev} rows={rows} "
                     f"iterations={iterations} verified={iterations} status=0")
    return "\n".join(lines)

class LiveAnalysis(unittest.TestCase):
    def test_full_completed(self):
        self.assertIn("PASS device=0 rows=16 verified=10",report(fixture()))
    def test_two_devices(self):
        self.assertEqual(2,len(parse(fixture()+"\n"+fixture(dev=1))))
    def test_missing_completion_rejected_but_visible(self):
        data=fixture(complete=False)
        with self.assertRaisesRegex(ValueError,"incomplete"):
            parse(data)
        self.assertIn("incomplete=1",provisional(data))
        self.assertIn("OBSERVED_SAMPLE",provisional(data))
    def test_gpu_failure_rejected_but_reported(self):
        data=fixture(fail=True)
        with self.assertRaisesRegex(ValueError,"hardware live failure"):
            parse(data)
        self.assertIn("FAILURE device=0",provisional(data))
    def test_stale_generation_rejected(self):
        data=fixture().replace("gen=116002","gen=116001")
        with self.assertRaisesRegex(ValueError,"wrong/stale"):
            parse(data)
    def test_wrong_gpu_ready_rejected(self):
        data=fixture().replace("gpu_seen_ready=116001","gpu_seen_ready=1")
        with self.assertRaisesRegex(ValueError,"GPU did not observe"):
            parse(data)
    def test_no_early_ready_claim(self):
        data=fixture().replace("armed_before_helper=1","armed_before_helper=0")
        with self.assertRaisesRegex(ValueError,"wrong/stale"):
            parse(data)
    def test_wrong_word_rejected(self):
        data=fixture().replace("mismatches=0","mismatches=1")
        with self.assertRaisesRegex(ValueError,"wrong/stale"):
            parse(data)
    def test_duplicate_sample_rejected(self):
        data=fixture().replace("gen=116002","gen=116001")
        with self.assertRaises(ValueError):parse(data)
    def test_nonzero_exit_rejected(self):
        with self.assertRaisesRegex(ValueError,"nonzero"):
            parse("exitCode: 3\n"+fixture())
    def test_shell_timeout_rejected(self):
        with self.assertRaisesRegex(ValueError,"timeout"):
            parse("timedOut: true\n"+fixture())
    def test_real_completed_one_row_log(self):
        data=(EVIDENCE/"rad50-live-xtx0-1row-10.log").read_text()
        self.assertIn("PASS device=0 rows=1 verified=10",report(data))
    def test_single_poller_failure_cpu_gpu_disagree(self):
        data=(EVIDENCE/"rad50-live-xtx0-768row-10-singlepoll.log").read_text()
        with self.assertRaises(ValueError):parse(data)
        summary=provisional(data)
        self.assertIn("host_ready=868001",summary)
        self.assertIn("gpu_seen_ready=0",summary)
        self.assertIn("errors=4294967295",summary)

    def test_real_768_failure_kept(self):
        data=(EVIDENCE/"rad50-live-xtx0-768row-10.log").read_text()
        with self.assertRaises(ValueError):parse(data)
        report_text=provisional(data)
        self.assertIn("errors=4294967295",report_text)
        self.assertIn("incomplete=1",report_text)

if __name__=="__main__":
    unittest.main()

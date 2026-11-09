"""RAD50 S1 strict CPU-only benchmark log verification (no hardware)."""
import unittest

from cpu_analyze import parse,report,scheduled,provisional

def fixture(iterations=10,rows=768,helpers=2,layout="hot",complete=True):
    lines=["exitCode: 0","timedOut: false","truncated: false",
           f"RAD50_CPU_CONTEXT rows={rows} width=2560 helpers={helpers} "
           f"layout={layout} iterations={iterations} "
           "paths=host-direct,host-stage-copy source=deterministic_bf16 "
           "bitwise_oracle=1 gpu=0"]
    for n in scheduled(iterations):
        lines.append(f"RAD50_CPU_SAMPLE rows={rows} helpers={helpers} "
                     f"layout={layout} iteration={n} direct_us=10.000 "
                     "stage_combine_us=10.000 stage_copy_us=2.000 "
                     "stage_total_us=12.000 mismatches=0 status=0")
    if complete:
        lines.append(f"RAD50_CPU_SUMMARY rows={rows} helpers={helpers} "
                     f"layout={layout} iterations={iterations} verified={iterations} "
                     f"emitted={len(scheduled(iterations))} direct_p50_us=10.000 "
                     "direct_p95_us=10.000 direct_max_us=10.000 "
                     "stage_combine_p50_us=10.000 stage_copy_p50_us=2.000 "
                     "stage_total_p50_us=12.000 stage_total_p95_us=12.000 "
                     "stage_total_max_us=12.000 status=0")
        lines.append(f"RAD50_CPU_COMPLETE rows={rows} helpers={helpers} "
                     f"layout={layout} verified={iterations} status=0")
        lines.append("RAD50 RESULT test=combine-cpu exit=0")
    return "\n".join(lines)

class S1CPUAnalysis(unittest.TestCase):
    def test_valid_full_and_compact(self):
        for n in (10,15,30,100,200):
            r=parse(fixture(n))
            self.assertEqual(len(scheduled(n)),len(r["samples"]))

    def test_stale_or_missing_ticket(self):
        with self.assertRaisesRegex(ValueError,"missing, wrong"):
            parse(fixture().replace("iteration=1 ","iteration=2 ",1))

    def test_incomplete_rejected_and_provisional(self):
        data=fixture(complete=False)
        with self.assertRaisesRegex(ValueError,"incomplete"):
            parse(data)
        self.assertIn("NOT_VERIFIED=1",provisional(data))

    def test_wrong_fp16_or_nonzero_exit(self):
        for data in (fixture().replace("mismatches=0","mismatches=1",1),
                     fixture().replace("exitCode: 0","exitCode: 3"),
                     fixture().replace("test=combine-cpu exit=0",
                                       "test=combine-cpu exit=3")):
            with self.assertRaises(ValueError):
                parse(data)

    def test_truncated_or_timed_out(self):
        for old,new in (("truncated: false","truncated: true"),
                        ("timedOut: false","timedOut: true")):
            with self.assertRaises(ValueError):
                parse(fixture().replace(old,new))

    def test_forged_gpu_or_layout(self):
        for old,new in (("gpu=0","gpu=1"),("layout=hot","layout=bad")):
            with self.assertRaises(ValueError):
                parse(fixture().replace(old,new))

    def test_stage_timing_consistency(self):
        with self.assertRaisesRegex(ValueError,"not additive"):
            parse(fixture().replace("stage_copy_us=2.000 ",
                                    "stage_copy_us=99.000 ",1))

    def test_missing_sample_and_emitted_count(self):
        with self.assertRaises(ValueError):
            parse(fixture().replace("iteration=10 ","iteration=9 ",1))
        with self.assertRaisesRegex(ValueError,"incomplete summary"):
            parse(fixture().replace("emitted=10","emitted=9"))

    def test_short_run_quantile_validation(self):
        with self.assertRaisesRegex(ValueError,"sample quantile"):
            parse(fixture().replace("direct_p50_us=10.000",
                                    "direct_p50_us=9.000"))

    def test_nan_and_negative_time_rejected(self):
        for value in ("nan","-1.0","inf"):
            with self.assertRaises(ValueError):
                parse(fixture().replace("direct_us=10.000",
                                        "direct_us="+value,1))

    def test_missing_envelope_rejected(self):
        with self.assertRaisesRegex(ValueError,"incomplete"):
            parse(fixture().replace("exitCode: 0",""))

if __name__=="__main__":
    unittest.main()

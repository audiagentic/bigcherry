"""Card-free strict/partial parsing of real FP16 combine A/B evidence."""
import unittest
from ab_analyze import COMBINE_SIZES, format_report, parse
from ab_partial import provisional_report
from test_ab_analyze import benchmark

class CombineABAnalysisTests(unittest.TestCase):
    def test_full_combine_direct_sweep(self):
        samples=benchmark(style="combine-direct",size_filter=0,sizes=list(COMBINE_SIZES))
        self.assertEqual(5,len(parse(samples)))
        self.assertIn("combine-direct",format_report(samples))

    def test_full_combine_staged_sweep(self):
        samples=benchmark(style="combine-staged",size_filter=0,sizes=list(COMBINE_SIZES))
        self.assertEqual(5,len(parse(samples)))

    def test_one_row_fails_wrong_size_family(self):
        with self.assertRaisesRegex(ValueError,"invalid benchmark byte filter"):
            parse(benchmark(style="combine-direct",size_filter=4096))

    def test_wrong_row_count_rejected(self):
        with self.assertRaisesRegex(ValueError,"invalid benchmark byte filter"):
            parse(benchmark(style="combine-direct",size_filter=10240))

    def test_reject_incomplete_combine_sweep(self):
        with self.assertRaisesRegex(ValueError,"incomplete size"):
            parse(benchmark(style="combine-direct",size_filter=0,sizes=[5120]))

    def test_partial_combine_retained(self):
        samples=benchmark(style="combine-direct",size_filter=5120,completion=False)
        self.assertIn("PROVISIONAL_PAIR",provisional_report(samples))
        self.assertIn("missing_success_marker=1",provisional_report(samples))
        with self.assertRaises(ValueError):
            parse(samples)

    def test_completed_repetition_pair(self):
        samples=benchmark(style="combine-direct",size_filter=3932160,iterations=100)
        self.assertEqual(1,len(parse(samples)))

    def test_failed_helper_response_not_accepted(self):
        samples=benchmark(style="combine-direct",size_filter=5120)
        with self.assertRaisesRegex(ValueError,"nonzero"):
            parse(samples+"\nRAD50 RESULT test=combine-ab exit=3")
        self.assertIn("OBSERVED_DIAGNOSTIC",provisional_report(samples+
                       "\nRAD50 RESULT test=combine-ab exit=3"))

if __name__=="__main__":
    unittest.main()

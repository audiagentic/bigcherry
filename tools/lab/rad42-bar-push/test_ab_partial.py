"""Card-free tests: partial logs remain visible but never imply a final pass."""
import unittest
from ab_partial import provisional_report
from ab_analyze import parse
from test_ab_analyze import benchmark, row, PULL


class ProvisionalTests(unittest.TestCase):
    def test_missing_completion_still_shows_paired_samples(self):
        log=benchmark(completion=False)
        result=provisional_report(log)
        self.assertIn("PROVISIONAL_PAIR",result)
        self.assertIn("missing_success_marker=1",result)
        self.assertIn("NOT_VERIFIED=1",result)
        with self.assertRaises(ValueError):
            parse(log)

    def test_single_path_is_retained_as_unpaired(self):
        log="RAD50_AB_CONTEXT device=0 iterations=30 style=direct-scalar bytes_filter=4096\n"
        log+=row(PULL, size=4096)+"\n"
        result=provisional_report(log)
        self.assertIn("OBSERVED_SAMPLE",result)
        self.assertIn("UNPAIRED_SAMPLE",result)
        self.assertIn("dmabuf-cpu-push",result)

    def test_incomplete_truncated_record_survives(self):
        result=provisional_report("RAD50_AB_CONTEXT device=0\nRAD50_AB device=0 bytes=4096")
        self.assertIn("INCOMPLETE_SAMPLE",result)
        self.assertIn("no complete per-path sample rows yet",result)

    def test_nonzero_exit_reported_without_silencing_earlier_samples(self):
        log=benchmark(completion=False)+"\nexitCode: 3\nRAD50 RESULT test=dmabuf-ab exit=3"
        result=provisional_report(log)
        self.assertIn("OBSERVED_SAMPLE",result)
        self.assertIn("OBSERVED_DIAGNOSTIC exitCode: 3",result)
        self.assertIn("OBSERVED_DIAGNOSTIC RAD50 RESULT",result)
        self.assertIn("NOT_VERIFIED=1",result)

    def test_successful_run_still_labelled_provisional(self):
        result=provisional_report(benchmark())
        self.assertIn("OBSERVED_COMPLETION",result)
        self.assertIn("NOT_VERIFIED=1",result)
        self.assertNotIn("missing_success_marker",result)


if __name__=="__main__":
    unittest.main()

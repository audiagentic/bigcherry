"""Card-free integrity checks for RAD50 paired A/B evidence."""
import unittest
from ab_analyze import SIZES, format_report, parse


def row(path, style="direct-scalar", device=0, iterations=30, size=65536,
        seq=200.0, sync=0.0):
    return (f"RAD50_AB device={device} bytes={size} iterations={iterations} "
            f"style={style} path={path} prepare_p50_us=10.0 "
            f"cpu_write_p50_us=30.0 sync_start_p50_us={sync} "
            f"sync_end_p50_us=2.0 child_total_p50_us=35.0 "
            f"roundtrip_p50_us=80.0 gpu_kernel_p50_us=50.0 "
            f"gpu_host_p50_us=100.0 seq_p50_us={seq} "
            f"seq_p95_us=240.0 seq_p99_us=260.0")

PULL = "mapped-host-pull"
PUSH = "dmabuf-cpu-push"


def benchmark(style="direct-scalar", device=0, iterations=30, size_filter=65536,
              pull_seq=200, push_seq=150, completion=True, status=0,
              sizes=None, duplicate=False):
    sizes = ([size_filter] if size_filter else list(SIZES)) if sizes is None else sizes
    lines = [
        f"RAD50_AB_CONTEXT device={device} pci=0000:03:00.0 "
        f"bytes_filter={size_filter} iterations={iterations} style={style}",
    ]
    for size in sizes:
        lines += [
            row(PULL, style=style, device=device, size=size,
                iterations=iterations, seq=pull_seq),
            row(PUSH, style=style, device=device, size=size,
                iterations=iterations, seq=push_seq, sync=3),
        ]
        if duplicate:
            lines.append(row(PULL,style=style,device=device,size=size,
                             iterations=iterations,seq=pull_seq))
        lines.append(f"RAD50_AB_COMPARE device={device} bytes={size} method={style} "
                     f"baseline_seq_p50_us={pull_seq} push_seq_p50_us={push_seq}")
    if completion:
        lines.append(f"RAD50_AB_COMPLETE device={device} iterations={iterations} "
                     f"style={style} bytes_filter={size_filter} status={status}")
    return "\n".join(lines)


class AnalysisTests(unittest.TestCase):
    def test_pairing_and_speedup(self):
        output = format_report(benchmark())
        self.assertIn("1.333", output)
        self.assertIn("+50.00", output)
        self.assertIn("5.00", output)

    def test_multiple_sizes_and_methods(self):
        text = "\n".join((benchmark(style="direct-scalar", size_filter=0),
                          benchmark(style="staged-memcpy", size_filter=0)))
        self.assertEqual(10,len(parse(text)))

    def test_picks_fastest_submitted_method(self):
        text = "\n".join((benchmark(style="direct-scalar",push_seq=125),
                          benchmark(style="staged-memcpy",push_seq=145)))
        self.assertIn("style=direct-scalar seq_p50_us=125.00",format_report(text))

    def test_reject_missing_pair(self):
        text = benchmark().replace(row(PUSH,seq=150,sync=3),"")
        with self.assertRaises(ValueError):
            parse(text)

    def test_reject_duplicate_sample(self):
        with self.assertRaises(ValueError):
            parse(benchmark(duplicate=True))

    def test_reject_different_iterations(self):
        text = benchmark().replace("iterations=30 style=direct-scalar "
                                   "path=dmabuf-cpu-push",
                                   "iterations=10 style=direct-scalar "
                                   "path=dmabuf-cpu-push")
        with self.assertRaises(ValueError):
            parse(text)

    def test_reject_invalid_latency(self):
        with self.assertRaises(ValueError):
            parse(benchmark(pull_seq=-1))

    def test_reject_unrelated_output(self):
        with self.assertRaises(ValueError):
            parse("RAD50 RESULT test=dmabuf-ab exit=3")

    def test_reject_unknown_style(self):
        with self.assertRaises(ValueError):
            parse(benchmark(style="other"))

    def test_reject_missing_completion(self):
        with self.assertRaisesRegex(ValueError,"truncated"):
            parse(benchmark(completion=False))

    def test_reject_nonzero_completion(self):
        with self.assertRaisesRegex(ValueError,"status"):
            parse(benchmark(status=3))

    def test_reject_incomplete_size_sweep(self):
        with self.assertRaisesRegex(ValueError,"incomplete size"):
            parse(benchmark(size_filter=0,sizes=[4096,65536]))

    def test_reject_failed_shell_exit_header(self):
        with self.assertRaisesRegex(ValueError,"nonzero shell"):
            parse("exitCode: 3\n"+benchmark())

    def test_reject_timedout_shell_log(self):
        with self.assertRaisesRegex(ValueError,"timed-out"):
            parse("timedOut: true\n"+benchmark())

    def test_reject_truncated_shell_log(self):
        with self.assertRaisesRegex(ValueError,"truncated"):
            parse("truncated: true\n"+benchmark())

    def test_reject_sample_without_context(self):
        with self.assertRaises(ValueError):
            parse(row(PULL))

    def test_reject_failed_later_run(self):
        combined = "\n".join((benchmark(),benchmark(device=1,completion=False)))
        with self.assertRaisesRegex(ValueError,"truncated"):
            parse(combined)

    def test_reject_nonzero_process_exit_even_if_complete(self):
        with self.assertRaisesRegex(ValueError,"nonzero"):
            parse(benchmark()+"\nRAD50 RESULT test=dmabuf-ab exit=3")

    def test_reject_duplicate_benchmark(self):
        with self.assertRaises(ValueError):
            parse(benchmark()+"\n"+benchmark())

    def test_reject_unpaired_comparison(self):
        with self.assertRaises(ValueError):
            parse(benchmark().replace("method=direct-scalar", "method=other"))

    def test_reject_unexpected_payload(self):
        with self.assertRaises(ValueError):
            parse(benchmark(size_filter=12345))

    def test_accept_successful_result_record(self):
        self.assertEqual(1,len(parse(benchmark()+"\n"
                          "RAD50 RESULT test=dmabuf-ab exit=0")))


if __name__ == "__main__":
    unittest.main()

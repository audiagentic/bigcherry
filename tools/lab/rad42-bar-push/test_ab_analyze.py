"""Card-free validation of strict paired RAD50 A/B log analysis."""
import unittest
from ab_analyze import format_report, parse

def row(path, style="direct-scalar", device=0, iterations=30, size=65536,
        seq=200.0, sync=0.0):
    return (f"RAD50_AB device={device} bytes={size} iterations={iterations} "
            f"style={style} path={path} prepare_p50_us=10.0 "
            f"cpu_write_p50_us=30.0 sync_start_p50_us={sync} "
            f"sync_end_p50_us=2.0 child_total_p50_us=35.0 "
            f"roundtrip_p50_us=80.0 gpu_kernel_p50_us=50.0 "
            f"gpu_host_p50_us=100.0 seq_p50_us={seq} "
            f"seq_p95_us=240.0 seq_p99_us=260.0")

PULL="mapped-host-pull"
PUSH="dmabuf-cpu-push"

class AnalysisTests(unittest.TestCase):
    def test_pairing_and_speedup(self):
        text="\n".join([row(PULL,seq=200),row(PUSH,seq=150,sync=3)])
        output=format_report(text)
        self.assertIn("1.333",output)
        self.assertIn("+50.00",output)
        self.assertIn("5.00",output)

    def test_picks_fastest_synthetic_push_style(self):
        text="\n".join([
            row(PULL,style="direct-scalar",seq=200),
            row(PUSH,style="direct-scalar",seq=125),
            row(PULL,style="staged-memcpy",seq=200),
            row(PUSH,style="staged-memcpy",seq=145),
        ])
        self.assertIn("style=direct-scalar seq_p50_us=125.00",
                      format_report(text))

    def test_multiple_sizes_and_methods(self):
        text="\n".join(row(path,style=style,size=size)
                       for style in ("direct-scalar","staged-memcpy")
                       for size in (4096,65536)
                       for path in (PUSH,PULL))
        self.assertEqual(4,len(parse(text)))

    def test_reject_missing_pair(self):
        with self.assertRaises(ValueError):
            parse(row(PULL))

    def test_reject_duplicate(self):
        with self.assertRaises(ValueError):
            parse("\n".join([row(PULL),row(PULL),row(PUSH)]))

    def test_reject_different_iterations(self):
        with self.assertRaises(ValueError):
            parse("\n".join([row(PULL,iterations=10),row(PUSH,iterations=30)]))

    def test_reject_invalid_latency(self):
        with self.assertRaises(ValueError):
            parse("\n".join([row(PULL,seq=-1),row(PUSH)]))

    def test_reject_unrelated_output(self):
        with self.assertRaises(ValueError):
            parse("RAD50 RESULT test=dmabuf-ab exit=3")

    def test_reject_unknown_style(self):
        with self.assertRaises(ValueError):
            parse("\n".join([row(PULL,style="shell"),row(PUSH,style="shell")]))

if __name__ == "__main__":
    unittest.main()

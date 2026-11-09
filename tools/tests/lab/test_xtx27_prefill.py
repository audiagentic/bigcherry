"""Card-free safety and evidence tests for the dual-XTX Radiance runner."""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lab.radiance import xtx27_prefill as xp


def row(depth: int, rep: int) -> dict:
    return {
        "depth": depth, "rep": rep, "prompt_tokens": depth,
        "completion_tokens": 4, "ttft_s": 2.0,
        "prefill_tps": depth / 2, "decode_tps": 20.0,
        "error": None,
    }


def result(depths=(512, 2048), reps=2):
    return {
        "served_model": "local-test-model",
        "server_error": None, "shutdown": {"clean": True},
        "rows": [row(d, i) for d in depths for i in range(reps)],
    }


class Xtx27PlanTests(unittest.TestCase):
    def test_arms_and_devices_fail_closed(self):
        self.assertEqual(xp.parse_arm("wht6:2048").name, "wht6-b2048")
        self.assertEqual(xp.parse_devices("0,1"), (0, 1))
        for value in ("wht8:2048", "exact:0", "wht6:999", "exact:-1", "wht6:1024:0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                xp.parse_arm(value)
        for value in ("0", "0,0", "0,1,2", "0;1", "-1,2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                xp.parse_devices(value)

    def test_server_flags_explicitly_disable_p2p_and_drafting(self):
        args = xp.args_for(xp.Arm("exact", 2048), context=8192, decode=64, profile=False)
        self.assertEqual(args[:8], ["--embedding-placement", "ram", "--tp", "2",
                                   "--tp-wire", "exact", "--p2p", "off"])
        self.assertIn("0", args)
        self.assertNotIn("--profile-ops", args)
        self.assertIn("--profile-ops", xp.args_for(xp.Arm("wht6", 4096),
                                                   context=8192, decode=64, profile=True))
        with self.assertRaises(ValueError):
            xp.args_for(xp.Arm("exact", 2048), context=96, decode=64, profile=False)

    def test_metrics_are_median_of_paired_reps_not_total_response_latency(self):
        measured = xp.checked_rows(result(), (512, 2048), 2)
        self.assertEqual(measured[2048]["prefill_tps_median"], 1024.0)
        self.assertEqual(measured[512]["ttft_s_median"], 2.0)
        self.assertEqual(measured[2048]["decode_tps_median"], 20.0)

    def test_incomplete_failed_and_duplicate_runs_rejected(self):
        failures = []
        no_rep = result()
        no_rep["rows"].pop()
        failures.append(no_rep)
        duplicate = result()
        duplicate["rows"][-1] = duplicate["rows"][0]
        failures.append(duplicate)
        bad_tp = result()
        bad_tp["rows"][0]["prompt_tokens"] = 0
        failures.append(bad_tp)
        fail_shutdown = result()
        fail_shutdown["shutdown"]["clean"] = False
        failures.append(fail_shutdown)
        server_error = result()
        server_error["server_error"] = "no GPU"
        failures.append(server_error)
        for candidate in failures:
            with self.subTest(record=candidate), self.assertRaises(ValueError):
                xp.checked_rows(candidate, (512, 2048), 2)

    def test_refuse_checkout_outputs_and_invalid_depths_without_gpu(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            build = root / "build"
            (build / "bin").mkdir(parents=True)
            (build / "radiance_home").mkdir()
            (build / "bin" / "radiance").write_bytes(b"fake")
            model = root / "model.rad"
            model.write_bytes(b"model")
            corpus = root / "corpus"
            corpus.write_text("example")
            base = ["--build", str(build), "--model", str(model), "--corpus", str(corpus),
                    "--devices", "0,1", "--output", str(root / "evidence")]
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(xp.main(base), 0)
            self.assertFalse((root / "evidence").exists())
            self.assertIn("time-to-first-text-token", output.getvalue())
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                xp.main([*base, "--depths", "100000"])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                xp.main([*base, "--devices", "0,0"])


if __name__ == "__main__":
    unittest.main()

"""VTP04 experiments are plan-first; no GPU is touched by card-free tests."""
from __future__ import annotations

import importlib.util
import io
import json
import pathlib
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

SOURCE = pathlib.Path(__file__).resolve().parents[2] / "lab" / "radiance" / "xtx27_vtp_micro.py"
SPEC = importlib.util.spec_from_file_location("xtx27_vtp_micro", SOURCE)
assert SPEC and SPEC.loader
vtp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vtp)


class VtpMicroTests(unittest.TestCase):
    def test_default_plan_does_not_invoke_any_subprocess(self):
        buffer = io.StringIO()
        with (
            patch.object(vtp.subprocess, "run", side_effect=AssertionError("ran external command")),
            redirect_stdout(buffer),
        ):
            self.assertEqual(vtp.main([]), 0)
        data = json.loads(buffer.getvalue())
        self.assertEqual(data["type"], "VTP04_NO_MODEL_MICROBENCH")
        self.assertEqual(len(data["cases"]), 7)
        self.assertEqual([case["kind"] for case in data["cases"]].count("host_collective"), 4)
        self.assertTrue(any("not native Q6/Q8 GEMM" in s for s in data["limitations"]))

    def test_case_boundaries(self):
        cases = vtp.plan(mib=(4, 64), rows=(1, 8, 2048), lanes=32768, iterations=7)
        self.assertEqual(len(cases), 5)
        self.assertEqual(cases[0]["label"], "local-4mib-32768lanes")
        self.assertEqual(cases[-1]["label"], "host-f32-2048rows")
        self.assertIn("--iterations", cases[0]["args"])
        self.assertIn("--iterations", cases[-1]["args"])
        rotated = vtp.plan(mib=(4,), rows=(1,), lanes=32768, iterations=3, reverse=True)
        self.assertTrue(all("--reverse" in case["args"] for case in rotated))

    def test_opt_in_single_r9700_three_physical_partials(self):
        cases = vtp.plan(mib=(4,), rows=(1, 4, 64), lanes=32768,
                         iterations=9, physical_tp3=True)
        physical = [c for c in cases if c["label"].startswith("host-f32-physical-tp3-")]
        self.assertEqual(len(physical), 3)
        self.assertTrue(all("--single-r9700" in c["args"] for c in physical))
        self.assertTrue(all("three" not in c["label"] for c in physical))
        old = vtp.plan(mib=(4,), rows=(1, 4, 64), lanes=32768, iterations=9)
        self.assertTrue(all("--single-r9700" not in c["args"] for c in old))
        rotated = vtp.plan(mib=(4,), rows=(1,), lanes=32768,
                           iterations=3, physical_tp3=True, reverse=True)
        self.assertTrue(all("--reverse" in c["args"] for c in rotated))

    def test_reject_unsafe_test_inputs(self):
        for val in ("", "0", "1,", "1,1", "2,999", "nan", "2;3", "-1", "1 2"):
            with self.subTest(val=val), self.assertRaises(ValueError):
                vtp.csv_positive(val, upper=256, name="test")
        for lanes in (0, 255, 513, 262400):
            with self.subTest(lanes=lanes), self.assertRaises(ValueError):
                vtp.plan(mib=(4,), rows=(1,), lanes=lanes, iterations=9)
        for iters in (0, 101):
            with self.assertRaises(ValueError):
                vtp.plan(mib=(4,), rows=(1,), lanes=5120, iterations=iters)

    def test_no_compile_or_gpu_without_opt_in(self):
        with patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as c:
                vtp.main(["--phase", "compile"])
            self.assertEqual(c.exception.code, 2)
            with self.assertRaises(SystemExit) as c:
                vtp.main(["--phase", "gpu"])
            self.assertEqual(c.exception.code, 2)

    def test_scratch_cannot_be_checkout(self):
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit) as c:
            # This fails before trying to build a binary.
            vtp.main(["--phase", "compile", "--run", "--scratch", str(vtp.ROOT)])
        self.assertEqual(c.exception.code, 2)

    def test_leased_gpu_run_refuses_wrong_owner_before_scratch(self):
        with (patch.dict(vtp.os.environ, {"VTP04_LEASED": "1"}),
              patch.object(vtp.subprocess, "run", return_value=SimpleNamespace(
                  stdout="gpu: held by other for \"another owner\"")) as calls,
              tempfile.TemporaryDirectory() as tmp):
            with self.assertRaisesRegex(RuntimeError, "not owned"):
                vtp.run_leased(scratch=pathlib.Path(tmp), binaries={}, cases=[], digest="abcd")
            self.assertEqual(calls.call_count, 1)
            self.assertFalse((pathlib.Path(tmp) / "runs").exists())

    def test_leased_gpu_run_refuses_active_kfd_before_scratch(self):
        good = 'gpu: held by owner for "VTP04: isolated virtual TP4 local/host microbench"'
        with (patch.dict(vtp.os.environ, {"VTP04_LEASED": "1"}),
              patch.object(vtp.subprocess, "run", side_effect=[
                  SimpleNamespace(stdout=good), SimpleNamespace(stdout="PID 123 radiance")
              ]) as calls, tempfile.TemporaryDirectory() as tmp):
            with self.assertRaisesRegex(RuntimeError, "KFD processes"):
                vtp.run_leased(scratch=pathlib.Path(tmp), binaries={}, cases=[], digest="abcd")
            self.assertEqual(calls.call_count, 2)
            self.assertFalse((pathlib.Path(tmp) / "runs").exists())

    def test_digest_tracks_each_hip_source(self):
        source_before = vtp.source_digest()
        self.assertEqual(len(source_before), 64)
        with tempfile.TemporaryDirectory() as temporary:
            replacements = {}
            for key, old in vtp.SOURCES.items():
                p = pathlib.Path(temporary) / old.name
                p.write_bytes(old.read_bytes())
                replacements[key] = p
            with patch.object(vtp, "SOURCES", replacements):
                self.assertEqual(vtp.source_digest(), source_before)
                replacements["host_collective"].write_bytes(
                    replacements["host_collective"].read_bytes() + b"\n//changed\n"
                )
                self.assertNotEqual(vtp.source_digest(), source_before)

    def test_gpu_scopes_and_real_oracles_present(self):
        local = vtp.SOURCES["local"].read_text()
        collective = vtp.SOURCES["host_collective"].read_text()
        self.assertIn("hipStreamWaitEvent", local)
        self.assertIn("independent CPU sample oracle mismatch", local)
        self.assertIn("mode output differs from fused", local)
        self.assertIn("gfx1201", local)
        self.assertIn("hipMemcpyDeviceToHost", collective)
        self.assertIn("hipMemcpyHostToDevice", collective)
        self.assertIn("exact F32 host collective oracle mismatch", collective)
        self.assertIn("local_add", collective)
        self.assertIn("--single-r9700", collective)
        self.assertIn("physical_tp3_single_r9700", collective)
        self.assertIn("single_r9700 ? 6.0f", collective)
        self.assertIn("single_r9700 ? (host[0][k] + host[1][k] + host[2][k])", collective)
        self.assertIn("single_r9700 || local ? 3 : 4", collective)
        self.assertNotIn("hipDeviceEnablePeerAccess", local + collective)
        self.assertNotIn("hipDeviceGetP2PAttribute", local + collective)


if __name__ == "__main__":
    unittest.main()

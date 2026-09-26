"""PVPS10 profiler wiring (offline: builds and rocprofv3 mocked)."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bigcherry.patch.campaign import profile


class ProfileWiringTests(unittest.TestCase):
    def test_both_arms_traced_with_workload_and_env(self) -> None:
        calls = []

        def fake_run(command, stdout=None, stderr=None, env=None, check=False):
            calls.append((list(command), dict(env or {})))
            if command[0] == "rocprofv3":
                trace_dir = Path(command[command.index("-d") + 1])
                (trace_dir / "host").mkdir(parents=True, exist_ok=True)
                (trace_dir / "host" / "trace_kernel_trace.csv").write_text("x\n", encoding="utf-8")
            else:
                Path(command[command.index("--output") + 1]).write_text("{}", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            bins = {"control": Path(tmp) / "c" / "llama-bench", "subject": Path(tmp) / "s" / "llama-bench"}
            with mock.patch.object(profile, "build_pair", return_value=bins), \
                 mock.patch.object(profile.subprocess, "run", side_effect=fake_run):
                rc = profile.main([
                    "--patch", "p", "--arch", "gfx1100", "--device", "2", "--model", "m.gguf",
                    "--workload", "decode", "--hip-path", "/rocm", "--worktree-root", tmp,
                    "--build-root", tmp, "--out", str(out), "--env", "GGML_CUDA_DQ_MMV=1",
                ])
            self.assertEqual(rc, 0)
            traced = [c for c in calls if c[0][0] == "rocprofv3"]
            self.assertEqual([c[0][c[0].index("--") + 1] for c in traced], [str(bins["control"]), str(bins["subject"])])
            for command, env in traced:
                self.assertIn("-n", command)
                self.assertEqual(command[command.index("-n") + 1], "128")
                self.assertEqual(env["HIP_VISIBLE_DEVICES"], "2")
                self.assertEqual(env["GGML_CUDA_DQ_MMV"], "1")
                self.assertEqual(env["BIGCHERRY_PATCH_TRACE"], "1")
                self.assertNotIn("ROCR_VISIBLE_DEVICES", env)
            doc = json.loads((out / "profile.json").read_text(encoding="utf-8"))
            self.assertEqual(set(doc["arms"]), {"control", "subject"})


if __name__ == "__main__":
    unittest.main()

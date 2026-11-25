"""Card-free tests of fn-profile.sh: profiling goes through the native owner and never kills by process name.

A fake tree holds the script beside fake `bc-serve`, `rocprofv3`, `arm_yaml.py` and `bigcherry` stand-ins, so the real
control flow runs with no GPU: the owner is started with `bc-serve serve`, the engine and helper PIDs come from the
session record, rocprofv3 attaches to those PIDs, and the session ends through `bc-serve stop` only.
"""
from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LAB = ROOT / "tools" / "lab" / "radiance"
SCRIPT = LAB / "fn-profile.sh"

FAKE_BC_SERVE = r"""#!/bin/bash
echo "$*" >> "$FAKE_CALLS"
case "$1" in
  serve)
    run=; while [ $# -gt 0 ]; do [ "$1" = --run-dir ] && run=$2; shift; done
    mkdir -p "$run"
    printf '{"owner_pid": %s, "state": "running", "processes": [{"role": "engine", "pid": 4001}, {"role": "helper0", "pid": 4002}, {"role": "other", "pid": 4003}]}' "$$" > "$run/session.json"
    # bounded: a test path that never stops the fake owner must not leave it polling forever
    n=0; until [ -e "$run/stop" ] || [ $((n+=1)) -gt 1200 ]; do sleep 0.05; done
    exit "${FAKE_OWNER_RC:-0}" ;;
  status) cat "$3/session.json" ;;
  stop) touch "$3/stop"; exit "${FAKE_STOP_RC:-0}" ;;
esac
"""
FAKE_ROCPROF = r"""#!/bin/bash
echo "rocprofv3 $*" >> "$FAKE_CALLS"
read -r _
exit "${FAKE_ROCPROF_RC:-0}"
"""


class FnProfileNative(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="fnprof-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        repo = self.tmp / "repo"
        lab = repo / "tools" / "lab" / "radiance"
        lab.mkdir(parents=True)
        (repo / "tools" / "bigcherry").mkdir()
        shutil.copy(SCRIPT, lab / "fn-profile.sh")
        (lab / "arm_yaml.py").write_text(
            "import sys\na = sys.argv\nopen(a[a.index('--out') + 1], 'w').write(open(a[a.index('--base') + 1]).read())\n")
        (repo / "tools" / "bigcherry" / "__main__.py").write_text(
            "import os, sys\nsys.exit(int(os.environ.get('FAKE_REQUEST_RC', '0')))\n")
        self.script = lab / "fn-profile.sh"
        build = self.tmp / "build"
        (build / "bin").mkdir(parents=True)
        (build / "radiance_home" / "bin").mkdir(parents=True)
        (build / "bin" / "radiance").write_text("#!/bin/sh\n")
        os.chmod(build / "bin" / "radiance", 0o755)
        bc = build / "radiance_home" / "bin" / "bc-serve"
        bc.write_text(FAKE_BC_SERVE)
        os.chmod(bc, 0o755)
        fakebin = self.tmp / "fakebin"
        fakebin.mkdir()
        (fakebin / "rocprofv3").write_text(FAKE_ROCPROF)
        os.chmod(fakebin / "rocprofv3", 0o755)
        (self.tmp / "cfg.yaml").write_text("server: {port: 1}\n")
        self.build, self.fakebin = build, fakebin
        self.calls = self.tmp / "calls.log"
        self.calls.write_text("")
        self.out = self.tmp / "out"

    def run_script(self, *, extra_env=None, args=None, timeout=60):
        env = dict(os.environ, PATH=f"{self.fakebin}:{os.environ['PATH']}", FAKE_CALLS=str(self.calls),
                   BC_WORK_ROOT=str(self.tmp / "work"), BC_MODEL_ROOT=str(self.tmp / "models"))
        env.update(extra_env or {})
        argv = args if args is not None else [
            "--build", str(self.build), "--native-config", str(self.tmp / "cfg.yaml"), "--name", "flashnext",
            "--out", str(self.out), "--port", "18999", "--prompt-tokens", "100"]
        return subprocess.run(["bash", str(self.script), *argv], env=env, capture_output=True, text=True, timeout=timeout)

    def test_script_has_no_broad_kill_patterns(self):
        text = SCRIPT.read_text()
        code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
        for pattern in (r"\bpkill\b", r"\bkillall\b", r"\bpgrep\b", r"\bkill\b(?! -0\b)", r"/dev/shm", r"\|\|\s*true\b",
                        r"\|\|\s*:"):
            self.assertIsNone(re.search(pattern, code), pattern)
        self.assertFalse((LAB / "fn-helper.sh").exists(), "the manual helper launcher is retired")

    def test_unrelated_helper_process_is_never_touched(self):
        decoy = self.tmp / "r3_moe_worker"
        shutil.copy("/bin/sleep", decoy)
        proc = subprocess.Popen([str(decoy), "120"])
        self.addCleanup(lambda: (proc.kill(), proc.wait()))
        shm = self.tmp / "r3-moe-helper"
        shm.write_text("x")
        done = self.run_script()
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIsNone(proc.poll(), "an unrelated r3_moe_worker must survive")
        self.assertTrue(shm.exists())

    def test_attaches_to_session_pids_and_stops_through_bc_serve(self):
        done = self.run_script()
        self.assertEqual(done.returncode, 0, done.stderr)
        calls = self.calls.read_text().splitlines()
        attached = sorted(re.search(r"--attach (\d+)", c).group(1) for c in calls if c.startswith("rocprofv3"))
        self.assertEqual(attached, ["4001", "4002"])   # engine and helper from the session record, not the other role
        self.assertTrue(any(re.match(r"stop --run-dir .*/out/run$", c) for c in calls))
        self.assertTrue(any(c.startswith("serve flashnext ") and "--native-config" in c for c in calls))

    def test_wrong_arguments_are_refused(self):
        for args in ([], ["--bogus"], ["--build", str(self.build)],
                     ["--build", str(self.build), "--native-config", str(self.tmp / "cfg.yaml"), "--name", "n",
                      "--out", str(self.out), "--port", "x", "--prompt-tokens", "100"],
                     ["--build", str(self.build), "--native-config", str(self.tmp / "cfg.yaml"), "--name", "n",
                      "--out", str(self.out), "--port", "1", "--prompt-tokens", "0"]):
            done = self.run_script(args=args)
            self.assertEqual(done.returncode, 2, (args, done.stderr))
        self.assertEqual(self.calls.read_text(), "", "a refused invocation starts nothing")

    def test_profiler_failure_is_nonzero_and_session_still_stops(self):
        done = self.run_script(extra_env={"FAKE_ROCPROF_RC": "3"})
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("rocprofv3", done.stderr)
        self.assertTrue(any(c.startswith("stop ") for c in self.calls.read_text().splitlines()))

    def test_request_failure_is_nonzero_and_session_still_stops(self):
        done = self.run_script(extra_env={"FAKE_REQUEST_RC": "1"})
        self.assertNotEqual(done.returncode, 0)
        self.assertTrue(any(c.startswith("stop ") for c in self.calls.read_text().splitlines()))

    def test_stop_failure_is_nonzero(self):
        done = self.run_script(extra_env={"FAKE_STOP_RC": "1"})
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("stop failed", done.stderr)


if __name__ == "__main__":
    unittest.main()

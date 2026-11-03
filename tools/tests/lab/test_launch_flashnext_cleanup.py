"""launch-flashnext-best.sh end to end with fakes (no cards): what happens to the processes it starts and the power
settings it applied when the start fails, is interrupted, or succeeds (GPT review of #538: B3 and B4).

The fakes: python3 (bigcherry subcommands), bc-serve, the engine and the helpers. Everything else is the real launcher."""
from __future__ import annotations

import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LAUNCHER = REPO / "tools" / "lab" / "radiance" / "launch-flashnext-best.sh"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_plan_check_launcher import FAKE_SERVE  # noqa: E402

FAKE_PYTHON = r"""#!/bin/bash
case "$*" in
  *"-m bigcherry host-check"*) exit 0 ;;
  *"-m bigcherry gpu-power apply"*) echo apply >> "$FAKE_LOG"; [ -n "${FAKE_POWER_APPLY_FAIL:-}" ] && exit 1; exit 0 ;;
  *"-m bigcherry gpu-power restore"*) echo restore >> "$FAKE_LOG"; exit 0 ;;
  *"-m bigcherry layout env"*) cat "$FAKE_LAYOUT_ENV"; exit 0 ;;
esac
exec "$REAL_PYTHON" "$@"
"""

FAKE_HELPER = r"""#!/bin/bash
echo $$ >> "$FAKE_PIDS"
[ "${FAKE_FAIL_HELPER:-}" = "$FAKE_ID" ] && { echo "r3_moe_worker: cannot start" ; exit 1; }
sleep "${FAKE_READY_DELAY:-0}"
echo "r3_moe_worker: ready (helper $FAKE_ID)"
echo "helper-ready $FAKE_ID $(date +%s.%N)" >> "$FAKE_TIMES"
exec "$REAL_PYTHON" -c 'import signal, sys, time; signal.signal(signal.SIGINT, lambda *a: sys.exit(0)); signal.signal(signal.SIGTERM, lambda *a: sys.exit(0)); time.sleep(600)'
"""

FAKE_ENGINE = r"""#!/bin/bash
echo $$ >> "$FAKE_PIDS"
plan=""; port=""
while [ $# -gt 0 ]; do
  case "$1" in --plan-memory) plan=$2; shift 2 ;; --port) port=$2; shift 2 ;; *) shift ;; esac
done
echo "engine-start ${plan:+measure} $(date +%s.%N)" >> "$FAKE_TIMES"
if [ -n "$plan" ]; then
  touch "$FAKE_MEASURING"
  [ -n "${FAKE_MEASURE_HANG:-}" ] && exec "$REAL_PYTHON" -c 'import signal, sys, time; signal.signal(signal.SIGINT, lambda *a: sys.exit(0)); signal.signal(signal.SIGTERM, lambda *a: sys.exit(0)); time.sleep(600)'
  [ -n "${FAKE_MEASURE_FAIL:-}" ] && exit 1
  echo '{"plan_version": 1}' > "$plan"; exit 0
fi
exec "$REAL_PYTHON" -c '
import http.server, json, os, signal, sys
signal.signal(signal.SIGINT, lambda *a: os._exit(0))
signal.signal(signal.SIGTERM, lambda *a: os._exit(0))
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length", 0)))
        text = os.environ.get("FAKE_ANSWER", " Paris. The capital of Germany is Berlin.")
        self.send_response(int(os.environ.get("FAKE_STATUS", "200"))); self.send_header("content-type", "application/json"); self.end_headers()
        if os.environ.get("FAKE_MALFORMED"):
            self.wfile.write(b"Traceback: not json"); return
        self.wfile.write(json.dumps({"choices": [{"text": text}]}).encode())
    def log_message(self, *a): pass
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()' "$port"
"""

LAYOUT_ENV = """LAYOUT_ENGINE_DEVICE=0
LAYOUT_HELPER_COUNT=3
LAYOUT_HELPER_0=(R3_MOE_WORKER_PLUGIN_FILE=libp.so FAKE_ID=0)
LAYOUT_HELPER_1=(R3_MOE_WORKER_PLUGIN_FILE=libp.so FAKE_ID=1)
LAYOUT_HELPER_2=(R3_MOE_WORKER_PLUGIN_FILE=libp.so FAKE_ID=2)
LAYOUT_COMMON=(FAKE_COMMON=1)
LAYOUT_ENGINE_PLUGIN=libp
LAYOUT_SUMMARY=fake
"""


def _free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LaunchCleanupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.work = self.root / "work"
        build = self.root / "build"
        home = build / "radiance_home"
        for d in (self.work / "heat", build / "bin", home / "helpers", home / "kernels", home / "bin", self.root / "fakebin"):
            d.mkdir(parents=True)
        (self.work / "heat" / "flashnext-mixed.hot-first").write_bytes(b"\0")
        (self.root / "model.rad").write_bytes(b"\0")
        (home / "kernels" / "libp.so").write_bytes(b"\0")
        self._exe(build / "bin" / "radiance", FAKE_ENGINE)
        self._exe(home / "helpers" / "r3_moe_worker", FAKE_HELPER)
        self._exe(home / "bin" / "bc-serve", FAKE_SERVE)
        self._exe(self.root / "fakebin" / "python3", FAKE_PYTHON)
        (self.root / "layout.env").write_text(LAYOUT_ENV)
        self.build = build
        self.log = self.root / "events.log"
        self.pids = self.root / "pids.log"
        self.measuring = self.root / "measuring"
        self.port = _free_port()
        self.env = {**os.environ, "BC_WORK_ROOT": str(self.work), "ROCM_PATH": "/opt/rocm", "MODEL": str(self.root / "model.rad"),
                    "PATH": f"{self.root / 'fakebin'}:{os.environ['PATH']}", "REAL_PYTHON": sys.executable,
                    "FAKE_LOG": str(self.log), "FAKE_LAYOUT_ENV": str(self.root / "layout.env"), "FAKE_PIDS": str(self.pids),
                    "FAKE_MEASURING": str(self.measuring), "FAKE_TIMES": str(self.root / "times.log")}
        self.addCleanup(self._kill_leftovers)

    @staticmethod
    def _exe(path: Path, text: str) -> None:
        path.write_text(text)
        path.chmod(path.stat().st_mode | stat.S_IXUSR)

    def _args(self) -> list[str]:
        return ["bash", str(LAUNCHER), "--build", str(self.build), "--port", str(self.port), "--gpu-power", "--record-heat", "off"]

    def _alive_pids(self) -> list[int]:
        out = []
        for line in (self.pids.read_text().split() if self.pids.exists() else []):
            try:
                os.kill(int(line), 0)
                out.append(int(line))
            except OSError:
                pass
        return out

    def _kill_leftovers(self) -> None:
        for pid in self._alive_pids():
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass

    def _events(self) -> list[str]:
        return self.log.read_text().split() if self.log.exists() else []

    def test_a_failed_measurement_stops_everything_and_restores_the_power_settings(self) -> None:
        done = subprocess.run(self._args(), env={**self.env, "FAKE_MEASURE_FAIL": "1"}, cwd=REPO, capture_output=True, text=True, timeout=120)
        self.assertEqual(done.returncode, 97, done.stdout + done.stderr)
        self.assertEqual(self._events(), ["apply", "restore"])
        self.assertEqual(self._alive_pids(), [])

    def test_a_helper_that_fails_readiness_fails_the_start_whichever_helper_it_is(self) -> None:
        for failing in ("0", "1", "2"):
            with self.subTest(helper=failing):
                if self.log.exists():
                    self.log.unlink()
                done = subprocess.run(self._args(), env={**self.env, "FAKE_FAIL_HELPER": failing}, cwd=REPO, capture_output=True, text=True, timeout=120)
                self.assertNotEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn(f"helper{failing} not ready", done.stdout)
                self.assertNotIn("ARGS", done.stdout)
                self.assertFalse(self.measuring.exists(), "the engine must not start with an incomplete helper set")
                self.assertEqual(self._events(), ["apply", "restore"])
                self.assertEqual(self._alive_pids(), [])

    def test_an_interrupt_during_the_measuring_start_stops_the_processes_and_restores_the_power_settings(self) -> None:
        proc = subprocess.Popen(self._args(), env={**self.env, "FAKE_MEASURE_HANG": "1"}, cwd=REPO,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.time() + 90
            while not self.measuring.exists() and time.time() < deadline and proc.poll() is None:
                time.sleep(0.2)
            self.assertTrue(self.measuring.exists(), "the measuring engine never started")
            self.assertGreaterEqual(len(self._alive_pids()), 4)   # three helpers and the engine
            proc.send_signal(signal.SIGINT)
            out, err = proc.communicate(timeout=60)
        finally:
            if proc.poll() is None:
                proc.kill()
        self.assertEqual(proc.returncode, 130, out + err)
        self.assertEqual(self._events(), ["apply", "restore"])
        self.assertEqual(self._alive_pids(), [])

    def test_a_partly_applied_power_setting_is_put_back(self) -> None:
        # apply fails after changing a card: it is restored at once, and again by the cleanup if the start then fails
        done = subprocess.run(self._args(), env={**self.env, "FAKE_POWER_APPLY_FAIL": "1", "FAKE_MEASURE_FAIL": "1"}, cwd=REPO,
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(done.returncode, 97, done.stdout + done.stderr)
        self.assertEqual(self._events()[0], "apply")
        self.assertIn("restore", self._events()[1:])
        self.assertEqual(self._alive_pids(), [])

    def test_the_parent_truncates_each_helper_log_before_it_spawns_the_helper(self) -> None:
        # the race (a stale 'ready' line read before the new helper's own redirection truncates the log) cannot be forced
        # from outside, so the ordering is asserted on the launcher itself: the truncation precedes the spawn in start_helpers
        body = LAUNCHER.read_text().split("start_helpers() {")[1].split("\n}\n")[0]
        self.assertIn(': > "$log"', body)
        self.assertLess(body.index(': > "$log"'), body.index('setsid ${HELPER_WRAP:-} "$HELPER"'))

    def test_a_stop_that_cannot_stop_a_process_is_not_reported_as_stopped(self) -> None:
        # stop_all returns non-zero when stop_procs does; the stop command then says so and exits 1
        text = LAUNCHER.read_text()
        self.assertIn("stop_procs || rc=1", text)
        self.assertIn('if stop_all; then echo "stopped"; exit 0; fi', text)
        self.assertIn("stop failed:", text)

    def test_the_real_start_waits_for_its_own_helpers_not_the_measuring_starts_logs(self) -> None:
        # a cache miss reuses the helper log names of the measuring start moments earlier; the real start's engine must not
        # start until each of ITS helpers has said ready (they take 3 s here)
        env = {**self.env, "FAKE_READY_DELAY": "3"}
        done = subprocess.run(self._args(), env=env, cwd=REPO, capture_output=True, text=True, timeout=180)
        try:
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            rows = [l.split() for l in (self.root / "times.log").read_text().splitlines()]
            engines = [r for r in rows if r[0] == "engine-start"]
            real = [r for r in engines if r[1] != "measure"]
            self.assertEqual(len(engines), 2, rows)
            real_start = float(real[0][-1])
            phase2_ready = [float(r[-1]) for r in rows if r[0] == "helper-ready" and float(r[-1]) > float([e for e in engines if e[1] == "measure"][0][-1])]
            self.assertEqual(len(phase2_ready), 3, rows)
            self.assertGreaterEqual(real_start, max(phase2_ready), "the real engine started before its own helpers were ready")
        finally:
            subprocess.run(["bash", str(LAUNCHER), "stop", "--build", str(self.build), "--port", str(self.port), "--gpu-power"],
                           env=self.env, cwd=REPO, capture_output=True, text=True, timeout=120)

    def test_a_second_launch_while_one_is_measuring_is_refused_and_disturbs_nothing(self) -> None:
        first = subprocess.Popen(self._args(), env={**self.env, "FAKE_MEASURE_HANG": "1"}, cwd=REPO,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.time() + 90
            while not self.measuring.exists() and time.time() < deadline and first.poll() is None:
                time.sleep(0.2)
            self.assertTrue(self.measuring.exists(), "the first launch never reached its measuring start")
            alive = self._alive_pids()
            second = subprocess.run([*self._args()[:-3], "--chunk", "1024", "--record-heat", "off"], env=self.env, cwd=REPO,
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(second.returncode, 98, second.stdout + second.stderr)
            self.assertIn("is in progress", second.stdout)
            self.assertEqual(self._alive_pids(), alive, "the refused launch must not stop or start anything")
            first.send_signal(signal.SIGINT)
            first.communicate(timeout=60)
        finally:
            if first.poll() is None:
                first.kill()
        self.assertEqual(self._alive_pids(), [])

    def test_stop_during_a_launch_is_refused_and_does_not_disturb_it(self) -> None:
        first = subprocess.Popen(self._args(), env={**self.env, "FAKE_MEASURE_HANG": "1"}, cwd=REPO,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.time() + 90
            while not self.measuring.exists() and time.time() < deadline and first.poll() is None:
                time.sleep(0.2)
            self.assertTrue(self.measuring.exists(), "the first launch never reached its measuring start")
            alive = self._alive_pids()
            stop = subprocess.run(["bash", str(LAUNCHER), "stop", "--build", str(self.build), "--port", str(self.port), "--gpu-power"],
                                  env=self.env, cwd=REPO, capture_output=True, text=True, timeout=60)
            self.assertEqual(stop.returncode, 98, stop.stdout + stop.stderr)
            self.assertNotIn("stopped", stop.stdout)
            self.assertEqual(self._alive_pids(), alive)
            self.assertEqual(self._events(), ["apply"], "the refused stop must not restore the power settings under the launch")
            first.send_signal(signal.SIGINT)
            first.communicate(timeout=60)
        finally:
            if first.poll() is None:
                first.kill()
        self.assertEqual(self._events(), ["apply", "restore"])
        self.assertEqual(self._alive_pids(), [])

    def test_an_engine_that_is_up_but_answers_garbage_fails_the_start_and_leaves_nothing(self) -> None:
        # the '!!!' class: /health is fine, every token is '!'. The canary must stop the launch before the hand-off.
        done = subprocess.run(self._args(), env={**self.env, "FAKE_ANSWER": "!!!!!!!!!!!!!!!!"}, cwd=REPO,
                              capture_output=True, text=True, timeout=180)
        self.assertEqual(done.returncode, 96, done.stdout + done.stderr)
        self.assertIn("start canary failed", done.stdout + done.stderr)
        self.assertNotIn("FLASHNEXT_EXIT=0", done.stdout)
        self.assertEqual(self._events(), ["apply", "restore"])
        self.assertEqual(self._alive_pids(), [])

    def test_a_failed_request_or_a_reply_that_is_not_a_completion_never_passes_the_canary(self) -> None:
        # valid completion text over a non-2xx status, and a non-JSON body that merely contains the expected word
        for name, extra, args in (("http 500 with the right text", {"FAKE_STATUS": "500"}, []),
                                  ("malformed body containing the expected word", {"FAKE_MALFORMED": "1"}, ["--canary-expect", "Traceback"])):
            with self.subTest(case=name):
                if self.log.exists():
                    self.log.unlink()
                done = subprocess.run([*self._args(), *args], env={**self.env, **extra}, cwd=REPO, capture_output=True, text=True, timeout=180)
                self.assertEqual(done.returncode, 96, done.stdout + done.stderr)
                self.assertIn("start canary failed", done.stdout + done.stderr)
                self.assertEqual(self._events(), ["apply", "restore"])
                self.assertEqual(self._alive_pids(), [])

    def test_the_canary_is_semantic_and_overridable(self) -> None:
        # a model that answers differently gets its own prompt/expectation; the match is case-insensitive substring
        done = subprocess.run([*self._args(), "--canary-prompt", "2+2=", "--canary-expect", "FOUR"],
                              env={**self.env, "FAKE_ANSWER": " four."}, cwd=REPO, capture_output=True, text=True, timeout=180)
        try:
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            self.assertIn("start canary: ok", done.stderr)
        finally:
            subprocess.run(["bash", str(LAUNCHER), "stop", "--build", str(self.build), "--port", str(self.port), "--gpu-power"],
                           env=self.env, cwd=REPO, capture_output=True, text=True, timeout=120)

    def test_a_successful_start_keeps_the_processes_and_the_power_settings_until_stop(self) -> None:
        done = subprocess.run(self._args(), env=self.env, cwd=REPO, capture_output=True, text=True, timeout=180)
        try:
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            self.assertIn("FLASHNEXT_EXIT=0", done.stdout)
            self.assertEqual(self._events(), ["apply"])
            self.assertGreaterEqual(len(self._alive_pids()), 4)
        finally:
            subprocess.run(["bash", str(LAUNCHER), "stop", "--build", str(self.build), "--port", str(self.port), "--gpu-power"],
                           env=self.env, cwd=REPO, capture_output=True, text=True, timeout=120)
        self.assertEqual(self._events(), ["apply", "restore"])
        self.assertEqual(self._alive_pids(), [])


if __name__ == "__main__":
    unittest.main()

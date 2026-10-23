"""MEN18: remote run preflight, launch metadata, status transitions, stall detection and wait.

The host is a local fake: the same bash script the controller would send over ssh runs in a local ``bash -se``, against a
real git checkout (with a bare local ``origin``), a stub ``bigcherry`` package and a stub ``pgrep``.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from bigcherry.cli import remote_run
from bigcherry.core import environment, tree_activity

ROOT = Path(__file__).resolve().parents[3]
BASH = shutil.which("bash")

_STUB_MAIN = '''import os, sys, time
a = sys.argv[1:]
print("ARGS", a, flush=True)
print("ENV", os.environ.get("ROCM_PATH"), os.environ.get("BIGCHERRY_ENVIRONMENT"), os.environ.get("PYTHONPATH"), flush=True)
if "--sleep" in a:
    time.sleep(float(a[a.index("--sleep") + 1]))
print("RUN-BODY-DONE", flush=True)
sys.exit(int(a[a.index("--rc") + 1]) if "--rc" in a else 0)
'''

_PGREP = """#!/usr/bin/env bash
# fake pgrep -x -l NAME: reports NAME as running when listed in $BC_FAKE_BUSY
name="${@: -1}"
for n in ${BC_FAKE_BUSY:-}; do [ "$n" = "$name" ] && echo "4242 $n" && exit 0; done
exit 1
"""


def posix(path: Path) -> str:
    text = path.resolve().as_posix()
    match = re.match(r"^([A-Za-z]):/(.*)$", text)
    return f"/{match.group(1).lower()}/{match.group(2)}" if match else text


class LocalBashTransport:
    def __init__(self, shim: Path, busy: str = "") -> None:
        self.shim, self.busy = shim, busy
        self.scripts: list[str] = []

    def run(self, script: str) -> remote_run.Completed:
        self.scripts.append(script)
        env = {**os.environ, "BIGCHERRY_REMOTE_PYTHON": Path(sys.executable).as_posix(),
               "PATH": posix(self.shim) + os.pathsep + os.environ["PATH"], "BC_FAKE_BUSY": self.busy}
        done = subprocess.run([BASH, "-se"], input=script.encode(), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, env=env)
        return remote_run.Completed(done.returncode, done.stdout.decode("utf-8", "replace"))


def git(cwd: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True, env=env).stdout.strip()


@unittest.skipUnless(BASH and shutil.which("git"), "needs bash and git")
class RemoteRunTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="bc-remote-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        self.tmp = tmp
        origin, self.tree, self.work = tmp / "origin.git", tmp / "tree", tmp / "work"
        self.work.mkdir()
        git(tmp, "init", "-q", "--bare", "-b", "main", str(origin))
        git(tmp, "clone", "-q", str(origin), str(self.tree))
        pkg = self.tree / "tools" / "bigcherry"
        (pkg / "core").mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        (pkg / "core" / "__init__.py").write_text("")
        shutil.copy(ROOT / "tools" / "bigcherry" / "core" / "tree_activity.py", pkg / "core" / "tree_activity.py")
        (pkg / "__main__.py").write_text(_STUB_MAIN)
        git(self.tree, "add", "-A")
        git(self.tree, "commit", "-q", "-m", "stub")
        git(self.tree, "push", "-q", "origin", "HEAD:main")
        self.first = git(self.tree, "rev-parse", "HEAD")
        # a newer commit that exists only on origin: the ref must be fetched before it can be checked out
        other = tmp / "other"
        git(tmp, "clone", "-q", str(origin), str(other))
        (other / "tools" / "bigcherry" / "NEWER").write_text("x")
        git(other, "add", "-A")
        git(other, "commit", "-q", "-m", "newer")
        git(other, "push", "-q", "origin", "HEAD:refs/heads/feat/newer")
        self.newer = git(other, "rev-parse", "HEAD")
        shim = tmp / "shim"
        shim.mkdir()
        (shim / "pgrep").write_text(_PGREP, newline="\n")
        self.transport = LocalBashTransport(shim)
        self.host = replace(
            environment.load(ROOT / "config" / "environment.example.toml").host(),
            name="lab", hostname="lab", repo_root=posix(self.tree), work_root=posix(self.work), rocm_root="/opt/fakerocm",
            environment_file=posix(tmp / "env.toml"), busy_processes=("llama-server", "ninja"),
        )
        self.launched: list[str] = []
        self.addCleanup(self._kill)

    def _kill(self) -> None:
        for run_id in self.launched:
            pid = self.work / "remote-runs" / run_id / "pid"
            if pid.is_file():
                subprocess.run([BASH, "-c", f"kill {pid.read_text().strip()} 2>/dev/null; true"])

    def start(self, *argv: str, ref: str = "main", **kwargs) -> dict:
        meta = remote_run.launch(self.host, self.transport, ref=ref, argv=argv or ("selftest",), **kwargs)
        self.launched.append(meta["id"])
        return meta

    def run_dir(self, meta: dict) -> Path:
        return self.work / "remote-runs" / meta["id"]

    def wait_done(self, meta: dict, timeout: float = 30.0):
        return remote_run.wait(self.host, self.transport, meta["id"], timeout=timeout, poll=0.2)

    # -- preflight refusals

    def test_busy_gpu_process_is_refused_and_nothing_launches(self) -> None:
        self.transport.busy = "ninja"
        with self.assertRaisesRegex(remote_run.PreflightRefused, "busy.*4242,? ninja"):
            self.start()
        self.assertFalse((self.work / "remote-runs").exists())
        self.assertEqual(git(self.tree, "rev-parse", "HEAD"), self.first)

    def test_allow_busy_skips_only_the_process_check(self) -> None:
        self.transport.busy = "llama-server"
        meta = self.start("--rc", "0", allow_busy=True)
        self.assertEqual(self.wait_done(meta).state, "done")

    def test_dirty_tracked_file_is_refused(self) -> None:
        (self.tree / "tools" / "bigcherry" / "__main__.py").write_text(_STUB_MAIN + "# edit\n")
        with self.assertRaisesRegex(remote_run.PreflightRefused, "uncommitted tracked changes"):
            self.start()

    def test_held_lease_is_refused_even_with_allow_busy(self) -> None:
        with tree_activity.Lease(self.work, self.tree, command="build", run_id="r1"):
            with self.assertRaisesRegex(remote_run.PreflightRefused, r"live lease.*build\(r1\)"):
                self.start(allow_busy=True)
        self.start("--rc", "0")  # released -> admitted

    def test_maintenance_lock_is_refused(self) -> None:
        with tree_activity.MaintenanceLock(self.work, self.tree):
            with self.assertRaisesRegex(remote_run.PreflightRefused, "maintenance"):
                self.start()

    def test_unknown_ref_is_refused(self) -> None:
        with self.assertRaisesRegex(remote_run.PreflightRefused, "ref not found"):
            self.start(ref="feat/does-not-exist")

    def test_host_without_busy_process_list_is_refused_before_ssh(self) -> None:
        with self.assertRaisesRegex(remote_run.RemoteRunError, "busy-processes"):
            remote_run.launch(replace(self.host, busy_processes=()), self.transport, ref="main", argv=("x",))
        self.assertEqual(self.transport.scripts, [])

    def test_bad_ref_and_empty_argv_are_rejected(self) -> None:
        for ref in ("--upload-pack=x", "a b", ""):
            with self.subTest(ref=ref), self.assertRaises(ValueError):
                self.start(ref=ref)
        with self.assertRaises(ValueError):
            remote_run.launch(self.host, self.transport, ref="main", argv=())

    # -- launch metadata

    def test_launch_checks_out_fetched_ref_detached_and_records_metadata(self) -> None:
        meta = self.start("--engine", "demo", "thing", "with space", "--rc", "0", ref="feat/newer")
        self.assertEqual(meta["sha"], self.newer)
        self.assertEqual(git(self.tree, "rev-parse", "HEAD"), self.newer)
        self.assertEqual(git(self.tree, "branch", "--show-current"), "")
        on_host = json.loads((self.run_dir(meta) / "meta.json").read_text())
        self.assertEqual(on_host["id"], meta["id"])
        self.assertEqual(on_host["ref"], "feat/newer")
        self.assertEqual(on_host["sha"], self.newer)
        self.assertEqual(on_host["pid"], meta["pid"])
        self.assertEqual(on_host["argv"], ["python", "-m", "bigcherry", "--engine", "demo", "thing", "with space", "--rc", "0"])
        self.assertTrue(on_host["log"].endswith(f"/remote-runs/{meta['id']}/run.log"))
        self.assertGreater(on_host["started-at"], 0)
        self.assertTrue((self.run_dir(meta) / "run.sh").is_file())

    def test_run_gets_rocm_environment_and_tools_on_pythonpath(self) -> None:
        meta = self.start("--engine", "demo", "arg with space")
        self.assertEqual(self.wait_done(meta).state, "done")
        log = (self.run_dir(meta) / "run.log").read_text()
        self.assertIn("ARGS ['--engine', 'demo', 'arg with space']", log)
        # msys rewrites POSIX-looking paths for the native stub interpreter; the values are otherwise as configured
        self.assertRegex(log, r"ENV \S*/opt/fakerocm \S*/env\.toml tools\n")

    # -- status transitions, stall, wait

    def test_running_then_done(self) -> None:
        meta = self.start("--sleep", "4")
        mid = remote_run.status(self.host, self.transport, meta["id"])
        self.assertEqual((mid.state, mid.rc), ("running", None))
        final = self.wait_done(meta)
        self.assertEqual((final.state, final.rc), ("done", 0))
        self.assertIn("[remote-run] done rc=0", remote_run.tail_log(self.host, self.transport, meta["id"], 5))

    def test_failed_reports_the_exit_code(self) -> None:
        meta = self.start("--rc", "7")
        final = self.wait_done(meta)
        self.assertEqual((final.state, final.rc), ("failed", 7))
        self.assertEqual(remote_run._exit_code(final), 7)

    def test_process_that_died_without_an_exit_code_is_failed(self) -> None:
        meta = self.start("--sleep", "60")
        subprocess.run([BASH, "-c", f"kill -9 {meta['pid']}"], check=True)
        deadline = 50
        while remote_run.status(self.host, self.transport, meta["id"]).state == "running" and deadline:
            deadline -= 1
            subprocess.run([BASH, "-c", "sleep 0.2"])
        final = remote_run.status(self.host, self.transport, meta["id"])
        self.assertEqual((final.state, final.rc), ("failed", None))
        self.assertEqual(remote_run._exit_code(final), 1)

    def test_stall_is_no_log_growth_for_the_configured_minutes(self) -> None:
        meta = self.start("--sleep", "60")
        log = posix(self.run_dir(meta) / "run.log")
        subprocess.run([BASH, "-c", f"touch -d '2 hours ago' {log}"], check=True)
        stalled = remote_run.status(self.host, self.transport, meta["id"], stall_minutes=30)
        self.assertEqual(stalled.state, "stalled")
        self.assertGreater(stalled.idle_seconds, 3600)
        fresh = remote_run.status(self.host, self.transport, meta["id"], stall_minutes=240)
        self.assertEqual(fresh.state, "running")
        self.assertEqual(remote_run._exit_code(stalled), remote_run.EXIT_STALLED)
        returned = remote_run.wait(self.host, self.transport, meta["id"], poll=0.1, stall_minutes=30)
        self.assertEqual(returned.state, "stalled")

    def test_wait_timeout_returns_none_while_running(self) -> None:
        meta = self.start("--sleep", "60")
        self.assertIsNone(remote_run.wait(self.host, self.transport, meta["id"], timeout=0.5, poll=0.2))
        self.assertEqual(remote_run._exit_code(None), remote_run.EXIT_TIMEOUT)

    def test_wait_polls_through_a_fake_clock(self) -> None:
        meta = self.start("--sleep", "3")
        ticks = []
        remote_run.wait(self.host, self.transport, meta["id"], poll=0.5, sleep=lambda s: (ticks.append(s), subprocess.run([BASH, "-c", f"sleep {s}"])))
        self.assertTrue(ticks)

    def test_unknown_and_malformed_run_ids(self) -> None:
        with self.assertRaisesRegex(remote_run.RemoteRunError, "unknown run"):
            remote_run.status(self.host, self.transport, "20260101T000000Z-abcdef")
        with self.assertRaises(ValueError):
            remote_run.status(self.host, self.transport, "../etc")

    def test_list_shows_recent_runs_newest_first(self) -> None:
        first = self.start("--rc", "0")
        self.wait_done(first)
        subprocess.run([BASH, "-c", "sleep 1.1"])
        second = self.start("--rc", "3")
        self.wait_done(second)
        ids = remote_run.recent(self.host, self.transport, 10)
        self.assertEqual(ids[:2], [second["id"], first["id"]])
        self.assertEqual(remote_run.recent(self.host, self.transport, 1), [second["id"]])


class RemoteCliTests(unittest.TestCase):
    def test_parser_keeps_inner_engine_and_remote_is_platform_scope(self) -> None:
        from bigcherry.cli import engine_scope
        from bigcherry.cli.main import build_parser

        parser = build_parser()
        args = parser.parse_args(["remote", "run", "--host", "h", "--ref", "r", "--allow-busy", "--",
                                  "--engine", "radiance", "helper-ab", "--x"])
        engine_scope.resolve(args, parser)
        self.assertEqual(engine_scope.scope("remote"), engine_scope.PLATFORM)
        self.assertEqual([a for a in args.argv if a != "--"], ["--engine", "radiance", "helper-ab", "--x"])
        self.assertTrue(args.allow_busy)
        refused = parser.parse_args(["--engine", "radiance", "remote", "list"])
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            engine_scope.resolve(refused, parser)


if __name__ == "__main__":
    unittest.main()

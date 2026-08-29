"""HI151: tree-activity lease/maintenance-lock protocol."""

from __future__ import annotations

import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.core import tree_activity as ta  # noqa: E402


class LeaseLifecycleTests(unittest.TestCase):
    def test_lease_written_on_enter_and_removed_on_exit(self):
        with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as project:
            work_root, project_root = Path(work), Path(project)
            with ta.Lease(work_root, project_root, command="tune-campaign", run_id="run1"):
                leases = ta.list_active_leases(work_root, project_root)
                self.assertEqual(len(leases), 1)
                self.assertEqual(leases[0].command, "tune-campaign")
                self.assertEqual(leases[0].run_id, "run1")
                self.assertEqual(leases[0].pid, os.getpid())
            self.assertEqual(ta.list_active_leases(work_root, project_root), [])

    def test_lease_removed_even_on_exception(self):
        with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as project:
            work_root, project_root = Path(work), Path(project)
            with self.assertRaises(ValueError):
                with ta.Lease(work_root, project_root, command="build", run_id="run2"):
                    raise ValueError("boom")
            self.assertEqual(ta.list_active_leases(work_root, project_root), [])


class MaintenanceLockTests(unittest.TestCase):
    def test_refuses_to_acquire_with_a_live_lease_present(self):
        with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as project:
            work_root, project_root = Path(work), Path(project)
            with ta.Lease(work_root, project_root, command="tune-campaign", run_id="run1"):
                lock = ta.MaintenanceLock(work_root, project_root)
                with self.assertRaises(ta.TreeActivityError):
                    lock.acquire()
                self.assertFalse(lock.path.is_dir())

    def test_acquires_cleanly_once_the_lease_is_released(self):
        with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as project:
            work_root, project_root = Path(work), Path(project)
            with ta.Lease(work_root, project_root, command="tune-campaign", run_id="run1"):
                pass
            lock = ta.MaintenanceLock(work_root, project_root)
            lock.acquire()
            try:
                self.assertTrue(lock.path.is_dir())
            finally:
                lock.release()
            self.assertFalse(lock.path.is_dir())

    def test_second_acquire_refused_while_first_still_held(self):
        with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as project:
            work_root, project_root = Path(work), Path(project)
            first = ta.MaintenanceLock(work_root, project_root)
            first.acquire()
            try:
                second = ta.MaintenanceLock(work_root, project_root)
                with self.assertRaises(ta.TreeActivityError):
                    second.acquire()
            finally:
                first.release()

    def test_release_is_a_noop_if_never_acquired(self):
        with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as project:
            lock = ta.MaintenanceLock(Path(work), Path(project))
            lock.release()  # must not raise


class StaleLeaseTests(unittest.TestCase):
    def _write_lease(self, work_root: Path, project_root: Path, *, pid: int, hostname: str) -> None:
        import json
        import time

        root = ta._tree_activity_root(work_root, project_root) / "leases"
        root.mkdir(parents=True, exist_ok=True)
        (root / "fake.json").write_text(json.dumps({
            "pid": pid, "hostname": hostname, "command": "build", "run_id": "r",
            "project_root": str(project_root), "started_at": time.time(),
        }), encoding="utf-8")

    def test_dead_pid_on_this_host_is_not_live(self):
        with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as project:
            work_root, project_root = Path(work), Path(project)
            # A PID essentially guaranteed not to exist.
            self._write_lease(work_root, project_root, pid=2**30 - 1, hostname=socket.gethostname())
            self.assertEqual(ta.list_live_leases(work_root, project_root), [])

    def test_pruning_removes_only_dead_leases_on_this_host(self):
        with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as project:
            work_root, project_root = Path(work), Path(project)
            self._write_lease(work_root, project_root, pid=2**30 - 1, hostname=socket.gethostname())
            removed = ta.prune_stale_leases(work_root, project_root)
            self.assertEqual(removed, ["fake"])
            self.assertEqual(ta.list_active_leases(work_root, project_root), [])

    def test_a_lease_from_a_different_hostname_is_never_treated_as_stale(self):
        with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as project:
            work_root, project_root = Path(work), Path(project)
            self._write_lease(work_root, project_root, pid=2**30 - 1, hostname="some-other-host")
            # Cannot verify a remote PID's liveness -- must fail closed (live).
            self.assertEqual(len(ta.list_live_leases(work_root, project_root)), 1)
            self.assertEqual(ta.prune_stale_leases(work_root, project_root), [])
            lock = ta.MaintenanceLock(work_root, project_root)
            with self.assertRaises(ta.TreeActivityError):
                lock.acquire()

    def test_own_live_pid_is_never_treated_as_stale(self):
        with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as project:
            work_root, project_root = Path(work), Path(project)
            self._write_lease(work_root, project_root, pid=os.getpid(), hostname=socket.gethostname())
            self.assertEqual(len(ta.list_live_leases(work_root, project_root)), 1)
            self.assertEqual(ta.prune_stale_leases(work_root, project_root), [])


class ProcScanTests(unittest.TestCase):
    def test_returns_empty_list_without_raising_on_non_linux_or_no_hits(self):
        with tempfile.TemporaryDirectory() as project:
            # Diagnostic-only helper must never raise even if /proc is
            # absent (non-Linux) or nothing references this fresh empty dir.
            result = ta.scan_proc_for_tree_usage(Path(project))
            self.assertIsInstance(result, list)


if __name__ == "__main__":
    unittest.main()

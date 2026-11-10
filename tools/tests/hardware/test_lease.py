from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bigcherry.hardware import lease


class LeaseTest(unittest.TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.d = Path(t.name) / "leases"
        self.d.mkdir()
        self.proc = Path(t.name) / "proc"
        for pid, ppid, state in ((100, 1, "S"), (200, 100, "S"), (300, 200, "S"), (999, 1, "S"), (150, 1, "Z")):
            (self.proc / str(pid)).mkdir(parents=True)
            (self.proc / str(pid) / "stat").write_text(f"{pid} (x y) {state} {ppid} 0 0\n")
        self.now = datetime(2026, 11, 10, 12, tzinfo=timezone.utc)

    def write(self, **kw):
        doc = {"instance": "bigcherry-1", "pids": [100], "expires": (self.now + timedelta(minutes=5)).isoformat()}
        doc.update(kw)
        (self.d / "gpu.json").write_text(json.dumps(doc))

    def held(self, pid=300, instance="bigcherry-1"):
        return lease.held_by_this_process("gpu", lease_dir=self.d, proc=self.proc, pid=pid, instance=instance, now=self.now)

    def test_descendant_of_a_live_lease_pid_holds_it(self):
        self.write()
        self.assertTrue(self.held())

    def test_unrelated_process_does_not(self):
        self.write()
        self.assertFalse(self.held(pid=999))

    def test_expired_other_instance_zombie_and_missing_all_refuse(self):
        self.write(expires=(self.now - timedelta(seconds=1)).isoformat())
        self.assertFalse(self.held())
        self.write(instance="someone-else")
        self.assertFalse(self.held())
        self.write(pids=[150])
        self.assertFalse(self.held(pid=150))
        (self.d / "gpu.json").unlink()
        self.assertFalse(self.held())

    def test_garbage_lease_file_refuses(self):
        (self.d / "gpu.json").write_text("{not json")
        self.assertFalse(self.held())


if __name__ == "__main__":
    unittest.main()

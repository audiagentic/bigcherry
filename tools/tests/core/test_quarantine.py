"""WSL03: quarantine / restore / purge move and delete exactly the harvested bulk inventory, journalled, in temporary roots only."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.core import harvest, quarantine as q, retention, run_manifest as rm, workspace  # noqa: E402

DAY = 86400.0
POLICY = retention.Policy(min_age_days=7.0, newest_k=0)


def harvested_run(root, *, age_days=30.0, refs=None, kind="stream"):
    h = workspace.create_run(root, kind, "t", plan_item="RAD12", engine="radiance", argv=["x"], references=refs,
                             now=time.time() - age_days * DAY)
    m = rm.read_manifest(h.path)
    m["state"] = "completed"
    m["execution"]["exit_code"] = 0
    rm.write_manifest(h.path, m)
    cell = h.path / "pair-1"
    cell.mkdir()
    (cell / "stream-bench.json").write_text(json.dumps({"rounds": []}))
    (cell / "cell.json").write_text("{}")
    (cell / "server.log").write_text("s" * 100)
    (cell / "helper0.log").write_text("h" * 50)
    harvest.harvest(root, h.path)
    return h.path


class Quarantine(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.root = Path(self._t.name)

    def tearDown(self):
        self._t.cleanup()

    def quarantine(self, run, **kw):
        kw.setdefault("apply", True)
        return q.quarantine(self.root, run, policy=POLICY, blocked=None, **kw)

    def test_dry_run_does_not_mutate(self):
        run = harvested_run(self.root)
        out = self.quarantine(run, apply=False)
        self.assertEqual((out["entries"], out["bytes"]), (2, 150))
        self.assertTrue((run / "pair-1" / "server.log").exists())
        self.assertFalse((self.root / "trash" / run.name).exists())
        self.assertEqual(rm.read_manifest(run)["retention"]["state"], "active")

    def test_moves_only_the_recorded_bulk_and_keeps_relative_paths(self):
        run = harvested_run(self.root)
        (run / "pair-1" / "unlisted.txt").write_text("not in the inventory")
        self.quarantine(run)
        trash = self.root / "trash" / run.name / "files" / "pair-1"
        self.assertEqual(sorted(p.name for p in trash.iterdir()), ["helper0.log", "server.log"])
        self.assertFalse((run / "pair-1" / "server.log").exists())
        self.assertTrue((run / "pair-1" / "unlisted.txt").exists())
        self.assertTrue((run / "pair-1" / "stream-bench.json").exists())
        self.assertTrue((run / rm.MANIFEST_NAME).exists())
        self.assertTrue(harvest.bundle_dir(self.root, run.name).is_dir())
        self.assertEqual(rm.read_manifest(run)["retention"]["state"], "quarantined")

    def test_refuses_an_unharvested_run_a_young_run_and_a_blocked_workspace(self):
        h = workspace.create_run(self.root, "stream", "u", plan_item=None, engine="radiance", argv=["x"],
                                 now=time.time() - 30 * DAY)
        m = rm.read_manifest(h.path)
        m["state"] = "completed"
        rm.write_manifest(h.path, m)
        with self.assertRaises(q.QuarantineError):
            self.quarantine(h.path)
        young = harvested_run(self.root, age_days=1.0)
        with self.assertRaises(q.QuarantineError):
            self.quarantine(young)
        old = harvested_run(self.root)
        with self.assertRaises(q.QuarantineError):
            q.quarantine(self.root, old, policy=POLICY, blocked="a runner lease is active", apply=True)
        self.assertTrue((old / "pair-1" / "server.log").exists())

    def test_refuses_when_the_bundle_no_longer_verifies_or_a_file_changed(self):
        run = harvested_run(self.root)
        (harvest.bundle_dir(self.root, run.name) / "files" / "pair-1" / "cell.json").write_text("tampered")
        with self.assertRaises(q.QuarantineError):
            self.quarantine(run)
        run2 = harvested_run(self.root)
        (run2 / "pair-1" / "server.log").write_text("grew after harvest" * 10)
        with self.assertRaises(q.QuarantineError):
            self.quarantine(run2)
        self.assertTrue((run2 / "pair-1" / "server.log").exists())

    def test_a_referenced_run_is_not_a_candidate_even_from_a_terminal_run(self):
        base = harvested_run(self.root)
        user = harvested_run(self.root, refs=[{"type": "run", "id": base.name, "role": "baseline"}])
        with self.assertRaises(q.QuarantineError) as ctx:
            self.quarantine(base)
        self.assertIn("referenced", str(ctx.exception))
        self.quarantine(user)              # the referencing run itself is free to go

    @unittest.skipIf(sys.platform == "win32", "symlinks need privileges on Windows")
    def test_a_symlinked_ancestor_is_refused(self):
        run = harvested_run(self.root)
        outside = self.root / "outside"
        outside.mkdir()
        os.replace(run / "pair-1", outside / "pair-1")
        (run / "pair-1").symlink_to(outside / "pair-1", target_is_directory=True)
        with self.assertRaises(q.QuarantineError):
            self.quarantine(run)
        self.assertTrue((outside / "pair-1" / "server.log").exists())

    def test_an_interrupted_move_resumes_from_the_disk(self):
        run = harvested_run(self.root)
        real = os.rename
        calls = []

        def flaky(a, b):
            calls.append(a)
            if len(calls) == 2:
                raise OSError("simulated crash")
            return real(a, b)

        q.os.rename = flaky
        try:
            with self.assertRaises(OSError):
                self.quarantine(run)
        finally:
            q.os.rename = real
        self.assertEqual(rm.read_manifest(run)["retention"]["state"], "active")
        out = self.quarantine(run)             # resumes: the first file is already in trash, the second moves now
        self.assertEqual(out["state"], "quarantined")
        trash = self.root / "trash" / run.name / "files" / "pair-1"
        self.assertEqual(sorted(p.name for p in trash.iterdir()), ["helper0.log", "server.log"])

    def test_a_second_quarantine_is_idempotent(self):
        run = harvested_run(self.root)
        self.quarantine(run)
        self.assertEqual(self.quarantine(run)["state"], "quarantined")

    def test_restore_returns_exactly_the_files_and_reopens_the_run(self):
        run = harvested_run(self.root)
        self.quarantine(run)
        q.restore(self.root, run, blocked=None)
        self.assertEqual((run / "pair-1" / "server.log").read_text(), "s" * 100)
        self.assertEqual(rm.read_manifest(run)["retention"]["state"], "active")
        self.quarantine(run)                   # and it can be quarantined again from a fresh plan
        self.assertEqual(rm.read_manifest(run)["retention"]["state"], "quarantined")

    def test_restore_refuses_to_overwrite_a_new_file(self):
        run = harvested_run(self.root)
        self.quarantine(run)
        (run / "pair-1").mkdir(exist_ok=True)
        (run / "pair-1" / "server.log").write_text("new")
        with self.assertRaises(q.QuarantineError):
            q.restore(self.root, run, blocked=None)
        self.assertEqual((run / "pair-1" / "server.log").read_text(), "new")

    def test_purge_waits_for_the_grace_period_then_deletes_only_the_journalled_files(self):
        run = harvested_run(self.root)
        self.quarantine(run)
        with self.assertRaises(q.QuarantineError):
            q.purge(self.root, run, grace_days=3, blocked=None)
        stray = self.root / "trash" / run.name / "files" / "pair-1" / "stray.txt"
        stray.write_text("not journalled")
        out = q.purge(self.root, run, grace_days=3, blocked=None, now=time.time() + 4 * DAY)
        self.assertEqual((out["state"], out["bytes"]), ("purged", 150))
        self.assertTrue(stray.exists(), "only journalled files are deleted")
        self.assertFalse((self.root / "trash" / run.name / "files" / "pair-1" / "server.log").exists())
        self.assertEqual(rm.read_manifest(run)["retention"]["state"], "purged")
        self.assertTrue(harvest.bundle_dir(self.root, run.name).is_dir())
        self.assertTrue((self.root / "trash" / run.name / q.JOURNAL).exists())

    def test_purge_refuses_a_reference_made_after_quarantine_and_a_hold(self):
        run = harvested_run(self.root)
        self.quarantine(run)
        later = workspace.create_run(self.root, "stream", "later", plan_item=None, engine="radiance", argv=["x"],
                                     references=[{"type": "run", "id": run.name, "role": "baseline"}])
        with self.assertRaises(q.QuarantineError) as ctx:
            q.purge(self.root, run, grace_days=0, blocked=None, now=time.time() + DAY)
        self.assertIn("referenced again", str(ctx.exception))
        self.assertTrue((self.root / "trash" / run.name / "files" / "pair-1" / "server.log").exists())
        self.assertTrue(later.path.exists())

    def test_operations_are_serialised_by_the_gc_lock(self):
        run = harvested_run(self.root)
        with q.gc_lock(self.root):
            with self.assertRaises(q.QuarantineError):
                self.quarantine(run)
        self.quarantine(run)                   # lock released

    def test_unsafe_inventory_paths_are_rejected(self):
        for bad in ("../x", "/abs", ""):
            with self.assertRaises(q.QuarantineError):
                q._rel_ok(bad)


if __name__ == "__main__":
    unittest.main()

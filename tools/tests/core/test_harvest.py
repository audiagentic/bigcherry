"""WSL02: harvest writes a hash-verified evidence bundle from a finished run and never deletes anything."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.core import harvest, run_manifest as rm, workspace  # noqa: E402

TRACE = ("TRACE,lane,0,seq,1,layer,3,rows,{rows},routes,40,local_us,5,wait_us,{wait},helper_gpu_us,{gpu},"
         "enter,{enter},wall_ns,{wall}\n")


def finished(root, kind="stream", state="completed", exit_code=0, **patch):
    h = workspace.create_run(root, kind, "t", plan_item="RAD12", engine="radiance", argv=["x"])
    m = rm.read_manifest(h.path)
    m["state"] = state
    m["execution"]["exit_code"] = exit_code
    m.update(patch)
    rm.write_manifest(h.path, m)
    return h.path


def add_cell(run, name="pair-1"):
    cell = run / name
    cell.mkdir()
    rounds = [{"streams": 2, "wall_start_ns": 1000, "wall_end_ns": 2000}]
    (cell / "stream-bench.json").write_text(json.dumps({"rounds": rounds}))
    (cell / "cell.json").write_text("{}")
    lines = [TRACE.format(rows=8, wait=w, gpu=50, enter=e, wall=1500) for e, w in ((1, 10), (2, 30))]
    (cell / "helper0.log").write_text("".join(lines))
    (cell / "server.log").write_text("x" * 1000)
    return cell


class Harvest(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.root = Path(self._t.name)

    def tearDown(self):
        self._t.cleanup()

    def test_bundle_keeps_derives_and_lists_bulk_without_deleting(self):
        run = finished(self.root)
        cell = add_cell(run)
        doc = harvest.harvest(self.root, run)
        out = harvest.bundle_dir(self.root, doc["run_id"])
        paths = {f["path"] for f in doc["files"]}
        self.assertIn("files/pair-1/stream-bench.json", paths)
        self.assertIn("derived/trace-windows.pair-1.json", paths)
        self.assertEqual([b["path"] for b in doc["bulk"]], ["pair-1/helper0.log", "pair-1/server.log"])
        self.assertEqual(doc["bulk_bytes"], sum(b["bytes"] for b in doc["bulk"]))
        self.assertFalse(doc["diagnostic_only"])
        self.assertEqual(harvest.verify_bundle(out), [])
        self.assertTrue((cell / "server.log").exists(), "harvest must not delete anything")
        windows = json.loads((out / "derived" / "trace-windows.pair-1.json").read_text())
        self.assertEqual(windows["2"]["layer_steps"], 2)
        m = rm.read_manifest(run)
        self.assertEqual(m["harvest"]["state"], "verified")
        self.assertEqual(m["harvest"]["bundle_id"], doc["bundle_id"])

    def test_verification_catches_a_changed_or_missing_file(self):
        run = finished(self.root)
        add_cell(run)
        doc = harvest.harvest(self.root, run)
        out = harvest.bundle_dir(self.root, doc["run_id"])
        target = out / "files" / "pair-1" / "cell.json"
        target.write_text("tampered")
        self.assertTrue(any("does not match" in p for p in harvest.verify_bundle(out)))
        target.unlink()
        self.assertTrue(any("missing" in p for p in harvest.verify_bundle(out)))

    def test_missing_required_evidence_fails_and_is_recorded(self):
        run = finished(self.root)
        with self.assertRaises(harvest.HarvestError):
            harvest.harvest(self.root, run)
        m = rm.read_manifest(run)
        self.assertEqual(m["harvest"]["state"], "failed")
        self.assertIn("required evidence", m["harvest"]["error"])
        add_cell(run)                                   # a failed harvest can be retried once the evidence exists
        harvest.harvest(self.root, run)
        self.assertEqual(rm.read_manifest(run)["harvest"]["state"], "verified")

    def test_unfinished_or_already_harvested_runs_are_refused(self):
        running = finished(self.root, state="running")
        add_cell(running)
        with self.assertRaises(harvest.HarvestError):
            harvest.harvest(self.root, running)
        done = finished(self.root)
        add_cell(done)
        harvest.harvest(self.root, done)
        with self.assertRaises(harvest.HarvestError):
            harvest.harvest(self.root, done)

    def test_failed_or_suspect_runs_are_diagnostic_only(self):
        failed = finished(self.root, state="failed", exit_code=1)
        add_cell(failed)
        self.assertTrue(harvest.harvest(self.root, failed)["diagnostic_only"])
        suspect = finished(self.root, suspect="host under load")
        add_cell(suspect)
        doc = harvest.harvest(self.root, suspect)
        self.assertTrue(doc["diagnostic_only"])
        self.assertEqual(doc["suspect"], "host under load")

    def test_profile_trace_is_summarised_before_the_bulk_is_listed(self):
        run = finished(self.root, kind="profile")
        (run / "a_kernel_stats.csv").write_text("Name,Calls\nk,1\n")
        (run / "a_kernel_trace.csv").write_text(
            "Kernel_Name,Start_Timestamp,End_Timestamp,Agent_Id\n"
            "mul_mat_q_foo,100,200,1\nsoftmax,200,260,1\n")
        doc = harvest.harvest(self.root, run)
        self.assertIn("derived/kernel-fraction.a_kernel_trace.json", {f["path"] for f in doc["files"]})
        self.assertEqual([b["path"] for b in doc["bulk"]], ["a_kernel_trace.csv"])

    def test_the_bundle_is_deterministic_and_has_a_stable_id(self):
        run = finished(self.root)
        add_cell(run)
        first = harvest.harvest(self.root, run)
        m = rm.read_manifest(run)
        m["harvest"]["state"] = "failed"
        rm.write_manifest(run, m)
        self.assertEqual(harvest.harvest(self.root, run)["bundle_id"], first["bundle_id"])

    def test_retry_never_deletes_the_previous_bundle_and_publishes_atomically(self):
        run = finished(self.root)
        add_cell(run)
        doc = harvest.harvest(self.root, run)
        out = harvest.bundle_dir(self.root, doc["run_id"])
        (out / "marker").write_text("keep me")
        m = rm.read_manifest(run)
        m["harvest"]["state"] = "harvesting"            # a crash left the run mid-harvest
        rm.write_manifest(run, m)
        harvest.harvest(self.root, run)
        prev = [p for p in out.parent.iterdir() if ".prev-" in p.name]
        self.assertEqual(len(prev), 1)
        self.assertEqual((prev[0] / "marker").read_text(), "keep me")
        self.assertFalse([p for p in out.parent.iterdir() if p.name.startswith(".staging-")])
        self.assertEqual(rm.read_manifest(run)["harvest"]["state"], "verified")

    @unittest.skipIf(sys.platform == "win32", "symlinks need privileges on Windows")
    def test_symlinked_directories_are_not_followed(self):
        run = finished(self.root)
        add_cell(run)
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "stream-bench.json").write_text("{}")
        (run / "pair-2").symlink_to(outside, target_is_directory=True)
        doc = harvest.harvest(self.root, run)
        self.assertFalse([f for f in doc["files"] if "pair-2" in f["path"]])

    def test_bundle_id_covers_the_bulk_inventory_and_verification_rejects_bad_inventories(self):
        run = finished(self.root)
        add_cell(run)
        doc = harvest.harvest(self.root, run)
        out = harvest.bundle_dir(self.root, doc["run_id"])
        bundle = json.loads((out / "bundle.json").read_text())
        for bad in ("../escape.log", "/abs.log", "pair-1/server.log"):
            tampered = dict(bundle, bulk=bundle["bulk"] + [{"path": bad, "bytes": 1}])
            (out / "bundle.json").write_text(json.dumps(tampered))
            self.assertTrue(harvest.verify_bundle(out), bad)

    def test_every_run_kind_has_a_recipe(self):
        self.assertEqual(set(harvest.DEFAULT_RECIPE), set(rm.RUN_KINDS))
        for rid in harvest.DEFAULT_RECIPE.values():
            self.assertIn(rid, harvest.RECIPES)


if __name__ == "__main__":
    unittest.main()

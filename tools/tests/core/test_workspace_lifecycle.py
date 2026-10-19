"""WSL01 / WSL03 skeleton: the managed work root, the run manifest, and the dry-run cleanup plan (temporary roots only)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.core import retention, run_manifest as rm, workspace  # noqa: E402

DAY = 86400.0


def make(root, kind="ab", topic="t", *, age_days=30.0, state="completed", harvest="verified", refs=None, hold=False,
         retain_until=None, provenance="recorded"):
    now = time.time()
    h = workspace.create_run(root, kind, topic, plan_item="RAD12", engine="radiance", argv=["x"], references=refs,
                             now=now - age_days * DAY)
    m = rm.read_manifest(h.path)
    m["state"] = state
    m["harvest"]["state"] = harvest
    m["retention"]["hold"] = hold
    m["retention"]["retain_until"] = retain_until
    m["provenance"] = provenance
    rm.write_manifest(h.path, m)
    return h


class Ids(unittest.TestCase):
    def test_run_id_unique_and_path_safe(self):
        ids = {workspace.new_run_id("Rad12 / lanes: 4,8;16 ../../x") for _ in range(200)}
        self.assertEqual(len(ids), 200)
        for i in ids:
            self.assertRegex(i, r"^\d{8}T\d{6}Z-[a-z0-9-]+-[0-9a-f]{12}$")
            self.assertNotIn("..", i)
            self.assertNotIn("/", i)
        with self.assertRaises(workspace.WorkspaceError):
            workspace.slug("///")
        with self.assertRaises(workspace.WorkspaceError):
            workspace.run_dir(Path("/w"), "ab", "../escape")
        with self.assertRaises(workspace.WorkspaceError):
            workspace.run_dir(Path("/w"), "nonsense", "x")

    def test_create_exclusive_under_concurrency(self):
        with tempfile.TemporaryDirectory() as td:
            made, errs = [], []

            def worker():
                try:
                    made.append(workspace.create_run(Path(td), "stream", "same topic", plan_item=None, engine="radiance",
                                                     argv=[], now=1_700_000_000.0).run_id)
                except Exception as exc:        # noqa: BLE001
                    errs.append(exc)

            ts = [threading.Thread(target=worker) for _ in range(24)]
            [t.start() for t in ts]
            [t.join() for t in ts]
            self.assertEqual(errs, [])
            self.assertEqual(len(set(made)), 24)                         # the same second and topic never collide
            self.assertEqual(len(list((Path(td) / "runs" / "stream").iterdir())), 24)

    def test_layout_and_unknown_kind(self):
        with tempfile.TemporaryDirectory() as td:
            workspace.ensure_layout(Path(td))
            for d in workspace.LAYOUT_DIRS:
                self.assertTrue((Path(td) / d).is_dir(), d)
            with self.assertRaises(workspace.WorkspaceError):
                workspace.create_run(Path(td), "scratch", "x", plan_item=None, engine="e", argv=[])


class Manifest(unittest.TestCase):
    def test_round_trip_and_unknown_version_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            h = make(Path(td))
            m = h.manifest()
            self.assertEqual(m["schema_version"], 1)
            self.assertEqual(m["state"], "completed")
            m["schema_version"] = 2
            (h.path / rm.MANIFEST_NAME).write_text(json.dumps(m))
            with self.assertRaises(rm.ManifestError):
                rm.read_manifest(h.path)

    def test_torn_or_missing_manifest_is_an_error_not_a_crash(self):
        with tempfile.TemporaryDirectory() as td:
            h = make(Path(td))
            (h.path / rm.MANIFEST_NAME).write_text('{"schema_version": 1, "run_id"')
            with self.assertRaises(rm.ManifestError):
                rm.read_manifest(h.path)
            (h.path / rm.MANIFEST_NAME).unlink()
            with self.assertRaises(rm.ManifestError):
                rm.read_manifest(h.path)

    def test_atomic_write_leaves_no_temp_files_and_survives_a_failed_write(self):
        with tempfile.TemporaryDirectory() as td:
            h = make(Path(td))
            self.assertEqual([p.name for p in h.path.iterdir() if ".tmp" in p.name], [])
            before = (h.path / rm.MANIFEST_NAME).read_text()
            bad = h.manifest()
            bad["kind"] = "not-a-kind"
            with self.assertRaises(rm.ManifestError):
                rm.write_manifest(h.path, bad)                       # validated before anything is written
            self.assertEqual((h.path / rm.MANIFEST_NAME).read_text(), before)

    def test_generation_conflict_and_transitions(self):
        with tempfile.TemporaryDirectory() as td:
            h = workspace.create_run(Path(td), "ab", "g", plan_item=None, engine="radiance", argv=[])
            m = rm.update_manifest(h.path, expected_generation=1, state="running")
            self.assertEqual((m["generation"], m["state"]), (2, "running"))
            self.assertTrue(m["started_at"])
            with self.assertRaises(rm.ManifestError):
                rm.update_manifest(h.path, expected_generation=1, state="completed")      # stale
            with self.assertRaises(rm.ManifestError):
                rm.update_manifest(h.path, expected_generation=2, state="created")        # illegal
            m = rm.update_manifest(h.path, expected_generation=2, state="completed", patch={"size_bytes": 5})
            self.assertTrue(m["ended_at"])
            with self.assertRaises(rm.ManifestError):
                rm.update_manifest(h.path, expected_generation=3, state="running")        # terminal
            with self.assertRaises(rm.ManifestError):
                rm.update_manifest(h.path, expected_generation=3, patch={"run_id": "other"})
            self.assertEqual(rm.update_manifest(h.path, expected_generation=3, harvest_state="harvesting")["harvest"]["state"],
                             "harvesting")
            with self.assertRaises(rm.ManifestError):
                rm.update_manifest(h.path, expected_generation=4, harvest_state="pending")  # harvest cannot rewind

    def test_run_harvest_and_retention_are_separate_fields(self):
        with tempfile.TemporaryDirectory() as td:
            m = make(Path(td), state="completed", harvest="pending").manifest()
            self.assertEqual((m["state"], m["harvest"]["state"], m["retention"]["state"]), ("completed", "pending", "active"))


class Cleanup(unittest.TestCase):
    def plan(self, td, **kw):
        inv = retention.inspect_workspace(Path(td))
        return {d.run_id: d for d in retention.plan_cleanup(inv, policy=retention.Policy(newest_k=kw.pop("k", 0)), **kw).decisions}

    def test_only_a_harvested_expired_unreferenced_run_is_a_candidate(self):
        with tempfile.TemporaryDirectory() as td:
            ok = make(Path(td))
            plan = self.plan(td)
            self.assertEqual(plan[ok.run_id].action, "candidate")

    def test_running_unharvested_young_held_retained_and_nonrecorded_runs_are_protected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cases = {
                "running": make(root, state="running"),
                "unharvested": make(root, harvest="pending"),
                "young": make(root, age_days=1.0),
                "held": make(root, hold=True),
                "retained": make(root, retain_until=rm.utc_stamp(time.time() + 3 * DAY)),
                "inferred": make(root, provenance="legacy_inferred"),
            }
            plan = self.plan(td)
            for name, h in cases.items():
                self.assertEqual(plan[h.run_id].action, "protected", name)

    def test_unmanaged_unknown_owner_and_invalid_manifests_are_protected_and_reported(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            workspace.ensure_layout(root)
            bare = root / "runs" / "ab" / "handmade-dir"
            bare.mkdir(parents=True)
            (bare / "big.bin").write_bytes(b"x" * 1000)
            bad = make(root)
            (bad.path / rm.MANIFEST_NAME).write_text("{not json")
            odd = root / "runs" / "scratch"
            odd.mkdir()
            (root / "stray").mkdir()
            inv = retention.inspect_workspace(root)
            plan = retention.plan_cleanup(inv)
            by = {d.run_id: d for d in plan.decisions}
            self.assertEqual(by["handmade-dir"].action, "protected")
            self.assertEqual(by["handmade-dir"].size_bytes, 1000)
            self.assertEqual(by[bad.run_id].action, "protected")
            unknown = {p.name for p in plan.unknown}
            self.assertEqual(unknown, {"scratch", "stray"})
            self.assertEqual(plan.totals()["candidate"]["runs"], 0)

    def test_a_manifest_that_names_another_run_is_not_trusted(self):
        with tempfile.TemporaryDirectory() as td:
            a, b = make(Path(td)), make(Path(td), topic="other")
            (b.path / rm.MANIFEST_NAME).write_text((a.path / rm.MANIFEST_NAME).read_text())
            plan = self.plan(td)
            self.assertEqual(plan[b.run_id].action, "protected")

    def test_referenced_runs_are_protected_transitively_and_plan_membership_pins_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            leaf = make(root, topic="leaf")
            mid = make(root, topic="mid", refs=[{"type": "run", "id": leaf.run_id, "role": "input"}])
            top = make(root, topic="top", state="running", refs=[{"type": "run", "id": mid.run_id, "role": "input"}])
            same_plan = make(root, topic="same-plan")                    # RAD12 like all the others: no pin from that alone
            plan = self.plan(td)
            for h in (leaf, mid, top):
                self.assertEqual(plan[h.run_id].action, "protected", h.run_id)
            self.assertEqual(plan[same_plan.run_id].action, "candidate")

    def test_newest_k_per_kind_is_kept(self):
        with tempfile.TemporaryDirectory() as td:
            hs = [make(Path(td), topic=f"r{i}", age_days=30 + i) for i in range(5)]
            inv = retention.inspect_workspace(Path(td))
            plan = {d.run_id: d for d in retention.plan_cleanup(inv, policy=retention.Policy(newest_k=2)).decisions}
            self.assertEqual([plan[h.run_id].action for h in hs], ["protected", "protected", "candidate", "candidate", "candidate"])

    def test_a_blocked_workspace_has_no_candidates(self):
        with tempfile.TemporaryDirectory() as td:
            make(Path(td))
            inv = retention.inspect_workspace(Path(td))
            plan = retention.plan_cleanup(inv, policy=retention.Policy(newest_k=0), blocked="a measurement is running")
            self.assertEqual(plan.totals()["candidate"]["runs"], 0)
            self.assertIn("a measurement is running", retention.render(plan, Path(td)))

    @unittest.skipUnless(hasattr(os, "symlink"), "needs symlinks")
    def test_symlinks_are_not_followed_and_not_counted(self):
        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as outside:
            root = Path(td)
            h = make(root)
            (Path(outside) / "huge.bin").write_bytes(b"x" * 5000)
            try:
                os.symlink(outside, h.path / "link")
                os.symlink(outside, root / "runs" / "ab" / "linked-run")
            except (OSError, NotImplementedError):
                self.skipTest("symlinks not permitted here")
            inv = retention.inspect_workspace(root)
            sizes = {e.run_id: e.size_bytes for e in inv.runs}
            self.assertLess(sizes[h.run_id], 5000)                       # the link target is not counted
            self.assertNotIn("linked-run", sizes)                        # a symlinked run directory is outside the layout
            self.assertTrue(any(p.name == "linked-run" for p in inv.unknown))

    def test_dry_run_never_mutates(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            make(root)
            (root / "runs" / "ab" / "handmade").mkdir(parents=True)

            def snapshot():
                return sorted((str(p.relative_to(root)), p.stat().st_size if p.is_file() else -1) for p in root.rglob("*"))

            before = snapshot()
            retention.render(retention.plan_cleanup(retention.inspect_workspace(root)), root, verbose=True)
            self.assertEqual(snapshot(), before)


class Cli(unittest.TestCase):
    def test_clean_refuses_apply_and_reports(self):
        from argparse import Namespace

        from bigcherry.cli import workspace_cmds

        with tempfile.TemporaryDirectory() as td:
            make(Path(td))
            base = dict(work_root=td, apply=False, min_age_days=7.0, keep_newest=0, no_size=True, verbose=False, json=True)
            self.assertEqual(workspace_cmds.cmd_clean(Namespace(**{**base, "apply": True})), 2)
            self.assertEqual(workspace_cmds.cmd_clean(Namespace(**base)), 0)
            self.assertEqual(workspace_cmds.cmd_clean(Namespace(**{**base, "work_root": td + "-missing"})), 2)


if __name__ == "__main__":
    unittest.main()

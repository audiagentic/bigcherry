"""WSL04/WSL09: safe build candidate inspection and locked build GC."""
from __future__ import annotations

import errno
import io
import json
import math
import os
import tempfile
import time
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from bigcherry.cli import workspace_cmds
from bigcherry.core import build_gc, retention, run_manifest, workspace
from bigcherry.core.resources import ResourceLock

DAY = 86400
REAL_PROC_SCAN = build_gc._proc_references


class CacheInventory(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "runs").mkdir()
        self.proc = patch.object(build_gc, "_proc_references", return_value=([], []))
        self.proc.start()
        self.addCleanup(self.proc.stop)
        # Historical race fixtures use symbolic names (recipe/old); test
        # content-addressed ID validation separately with actual 32-hex IDs.
        self.ids = patch.object(build_gc, "_cache_id", return_value=True)
        self.ids.start()
        self.addCleanup(self.ids.stop)

    def build(self, name, *, recipe="recipe", age=30, metadata=True):
        b = self.root / "builds" / recipe / name
        b.mkdir(parents=True)
        (b / "binary").write_bytes(b"binary")
        if metadata:
            sidecar = b / "bigcherry-build-metadata-radiance.json"
            sidecar.write_text('{"build": "known"}')
            os.utime(sidecar, (time.time() - age * DAY,) * 2)
        os.utime(b, (time.time() - age * DAY,) * 2)
        return b

    def decisions(self, newest=0):
        return {Path(e["path"]).name: e for e in retention.inspect_cache_roots(
            self.root, policy=retention.Policy(newest_k=newest))}

    def args(self, **kw):
        values = {"work_root": str(self.root), "apply": False, "builds": True, "json": True,
                  "no_size": False, "min_age_days": 7, "keep_newest": 0,
                  "legacy": False, "verbose": False}
        values.update(kw)
        return Namespace(**values)

    def run_cli(self, **kw):
        output = io.StringIO()
        with redirect_stdout(output):
            status = workspace_cmds.cmd_clean(self.args(**kw))
        return status, json.loads(output.getvalue()) if output.getvalue().lstrip().startswith("{") else output.getvalue()

    def test_aged_sidecar_and_aged_partial_are_candidates(self):
        old = self.build("old", age=30)
        partial = self.build("partial", age=20, metadata=False)
        fresh = self.build("young", age=2)
        entries = self.decisions()
        self.assertEqual(entries["old"]["action"], "candidate")
        self.assertEqual(entries["partial"]["action"], "candidate")
        self.assertEqual(entries["partial"]["metadata"], "missing")
        self.assertEqual(entries["young"]["action"], "protected")
        self.assertIn("younger than", entries["young"]["reasons"][0])
        self.assertEqual(entries["old"]["size_bytes"], sum(x.stat().st_size for x in old.iterdir()))
        self.assertTrue(partial.exists() and fresh.exists())

    def test_invalid_build_metadata_identity_is_protected(self):
        build = self.build("old", age=30)
        sidecar = build / "bigcherry-build-metadata-radiance.json"
        sidecar.write_text(json.dumps({"source_slice_id": "wrong",
                                       "build_plan_id": build.name}))
        os.utime(sidecar, (time.time() - 30 * DAY,) * 2)
        self.assertEqual(self.decisions()["old"]["action"], "protected")
        self.assertIn("identity disagrees", " ".join(self.decisions()["old"]["reasons"]))

    def test_many_unknown_runs_have_one_bounded_diagnostic(self):
        self.build("old")
        for index in range(12):
            (self.root / "runs" / f"unclassified-{index}").mkdir(parents=True)
        reasons = self.decisions()["old"]["reasons"]
        self.assertEqual(sum("unresolved run/release references" in item for item in reasons), 1)
        self.assertIn("12 observations", " ".join(reasons))

    def test_unknown_run_path_fails_closed(self):
        self.build("old")
        unknown = self.root / "runs" / "unsupported-kind"
        unknown.mkdir(parents=True)
        entry = self.decisions()["old"]
        self.assertEqual(entry["action"], "protected")
        self.assertIn("unreadable run manifest", " ".join(entry["reasons"]))

    def test_newest_k_per_recipe_even_if_old(self):
        self.build("older", age=40)
        self.build("newer", age=30)
        self.build("different", recipe="another", age=35)
        entries = self.decisions(newest=1)
        self.assertEqual(entries["older"]["action"], "candidate")
        self.assertIn("newest 1", " ".join(entries["newer"]["reasons"]))
        self.assertIn("newest 1", " ".join(entries["different"]["reasons"]))

    def test_metadata_newer_than_build_and_corrupt_metadata_are_protected(self):
        b = self.build("old", age=25)
        sidecar = b / "bigcherry-build-metadata-radiance.json"
        os.utime(sidecar, None)
        self.assertEqual(self.decisions()["old"]["action"], "protected")
        sidecar.write_text("{broken")
        self.assertEqual(self.decisions()["old"]["metadata"], "invalid")
        self.assertEqual(self.decisions()["old"]["action"], "protected")

    def test_run_reference_and_administrative_hold(self):
        b = self.build("pinned")
        handle = workspace.create_run(self.root, "ab", "held", plan_item="WSL09",
                                      engine="radiance", argv=[], references=[
                                          {"type": "build", "id": b.name, "role": "build"}])
        m = run_manifest.read_manifest(handle.path)
        m["retention"]["hold"] = True
        run_manifest.write_manifest(handle.path, m)
        decision = self.decisions()["pinned"]
        self.assertEqual(decision["action"], "protected")
        self.assertIn("held run", " ".join(decision["reasons"]))
        m["retention"]["hold"] = False
        run_manifest.write_manifest(handle.path, m)
        self.assertIn("run manifest", " ".join(self.decisions()["pinned"]["reasons"]))

    def test_release_record_build_plan_is_pinned(self):
        build_id = "a" * 32
        build = self.build(build_id, age=40)
        releases = self.root / "releases"
        releases.mkdir()
        (releases / "version.json").write_text(json.dumps(
            {"notes": f"pin-bump lane build: build_plan_id={build_id} binary_sha256=none"}))
        policy = retention.Policy(newest_k=0)
        entry = next(e for e in build_gc.inspect(self.root, policy=policy, release_root=releases)
                     if e["path"] == str(build))
        self.assertEqual(entry["action"], "protected")
        self.assertIn("release record", " ".join(entry["reasons"]))
        (releases / "version.json").unlink()
        entry = next(e for e in build_gc.inspect(self.root, policy=policy, release_root=releases)
                     if e["path"] == str(build))
        self.assertEqual(entry["action"], "candidate")

    def test_unreadable_manifest_and_lease_fail_closed(self):
        b = self.build("old")
        orphan = self.root / "runs" / "ab" / "unknown"
        orphan.mkdir(parents=True)
        self.assertIn("unreadable run manifest", " ".join(self.decisions()["old"]["reasons"]))
        orphan.rmdir()
        activity = self.root / "tree-activity" / "namespace" / "leases"
        activity.mkdir(parents=True)
        (activity / "other.json").write_text("unclassified")
        self.assertIn("cleanup blocked", " ".join(self.decisions()["old"]["reasons"]))
        self.assertTrue(b.exists())

    def test_live_open_fd_is_protected(self):
        b = self.build("old")
        fake_proc = self.root / "fake-proc"
        fake_proc.mkdir()
        (fake_proc / str(os.getpid())).symlink_to(f"/proc/{os.getpid()}", target_is_directory=True)
        with (b / "binary").open("rb"):
            with patch.object(build_gc, "_proc_references", REAL_PROC_SCAN):
                result = build_gc.inspect(self.root, policy=retention.Policy(newest_k=0),
                                          proc_root=fake_proc)
            old = next(e for e in result if e["path"] == str(b))
            self.assertEqual(old["action"], "protected")
            self.assertIn("live process", " ".join(old["reasons"]))
        self.assertEqual(self.decisions()["old"]["action"], "candidate")

    def test_scan_covers_only_the_work_root_owners_processes(self):
        fake_proc = self.root / "fake-proc"
        process = fake_proc / "4242"
        process.mkdir(parents=True)
        (process / "fd").write_text("not a directory")  # an unreadable surviving process
        owner = fake_proc.stat().st_uid
        _, errors = REAL_PROC_SCAN(fake_proc, owner)
        self.assertTrue(errors, "the owner's unreadable process must fail the scan closed")
        refs, errors = REAL_PROC_SCAN(fake_proc, owner + 1)
        self.assertEqual((refs, errors), ([], []), "another user's process is out of scope")

    def test_scan_skips_processes_outside_this_cgroup_namespace(self):
        fake_proc = self.root / "fake-proc"
        owner = self.root.stat().st_uid
        for pid, cgroup in (("4301", "0::/../../user.slice/user-1000.slice/init.scope\n"),
                            ("4302", "0::/\n"), ("4303", "")):
            process = fake_proc / pid
            process.mkdir(parents=True)
            (process / "fd").write_text("not a directory")  # unreadable survivor
            if cgroup:
                (process / "cgroup").write_text(cgroup)
        _, errors = REAL_PROC_SCAN(fake_proc, owner)
        seen = " ".join(errors)
        self.assertNotIn("4301", seen, "a process outside the cgroup namespace is out of scope")
        self.assertIn("4302", seen, "a process in this cgroup namespace fails the scan closed")
        self.assertIn("4303", seen, "an unreadable cgroup means unknown, so in scope")

    def test_uninspectable_proc_fails_closed(self):
        self.build("old")
        with patch.object(build_gc, "_proc_references", return_value=([], ["access denied to process maps"])):
            decision = self.decisions()["old"]
        self.assertEqual(decision["action"], "protected")
        self.assertIn("access denied", " ".join(decision["reasons"]))

    def test_existing_build_resource_lock_protects(self):
        b = self.build("old")
        lock = ResourceLock(self.root / "resource-locks", f"build-{b.name}")
        lock.acquire(timeout_seconds=0)
        try:
            self.assertIn("resource lock", " ".join(self.decisions()["old"]["reasons"]))
        finally:
            lock.release()

    def test_sources_and_orphans_remain_read_only(self):
        source = self.root / "sources" / "source"
        source.mkdir(parents=True)
        (source.parent / "source.metadata.json").write_text('{"source": "known"}')
        (source.parent / "orphan.metadata.json").write_text("{}")
        self.build("old")
        entries = retention.inspect_cache_roots(self.root, policy=retention.Policy(newest_k=0))
        src = [e for e in entries if e["kind"] == "source"]
        self.assertEqual(len(src), 2)
        self.assertTrue(all(e["action"] == "protected" for e in src))
        self.assertTrue(any(e["metadata"] == "present-unverified" for e in src))

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks required")
    def test_build_symlink_is_reported_without_traversal(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "important").write_text("do not delete")
        recipe = self.root / "builds" / "recipe"
        recipe.mkdir(parents=True)
        (recipe / "borrowed").symlink_to(outside, target_is_directory=True)
        self.assertEqual(self.decisions()["borrowed"]["action"], "protected")
        result = build_gc.apply(self.root, policy=retention.Policy(newest_k=0),
                                blocked_check=lambda: None)
        self.assertEqual(result["removed"], [])
        self.assertTrue((outside / "important").exists())

    def test_dry_run_has_no_effect_and_reports_age_size_reasons(self):
        old = self.build("old")
        young = self.build("young", age=1)
        before = [(str(p), p.stat().st_mtime) for p in self.root.rglob("*")]
        status, data = self.run_cli()
        self.assertEqual(status, 0)
        self.assertTrue(data["read_only"])
        self.assertEqual({d["action"] for d in data["entries"]}, {"candidate", "protected"})
        self.assertGreater(next(d for d in data["entries"] if d["path"] == str(old))["age_days"], 7)
        self.assertEqual(before, [(str(p), p.stat().st_mtime) for p in self.root.rglob("*")])
        self.assertTrue(old.exists() and young.exists())
        self.assertFalse((self.root / "cleanup-records").exists())

    def test_apply_only_candidates_with_audited_removal(self):
        old = self.build("old", recipe="partial", age=40)
        partial = self.build("partial", recipe="partial", age=45, metadata=False)
        kept = self.build("newest", recipe="partial", age=30)
        kept_old = self.build("older", recipe="partial", age=50)
        young = self.build("young", recipe="young", age=2)
        status, data = self.run_cli(apply=True, keep_newest=1)
        self.assertEqual(status, 0, data)
        self.assertFalse(old.exists())
        self.assertFalse(partial.exists())
        self.assertFalse(kept_old.exists())
        self.assertTrue(kept.exists() and young.exists())
        self.assertTrue(old.parent.exists())
        result = data["apply"]
        self.assertEqual(len(result["removed"]), 3)
        self.assertEqual(len(result["skipped"]), 0)
        journal = Path(result["record"])
        self.assertTrue(journal.is_file())
        lines = [json.loads(line) for line in journal.read_text().splitlines()]
        self.assertEqual(len(lines), 6)
        self.assertEqual({line["state"] for line in lines}, {"pending", "removed"})
        self.assertTrue(all(line["size_bytes"] > 0 and line["age_days"] >= 7
                            and line["reason"] for line in lines))

    def test_apply_rechecks_if_candidate_becomes_protected(self):
        b = self.build("old")
        original_inspect = build_gc.inspect
        checks = []

        def inspect(*args, **kwargs):
            result = original_inspect(*args, **kwargs)
            checks.append(1)
            if len(checks) == 2:
                for item in result:
                    if item["path"] == str(b):
                        item["action"] = "protected"
            return result

        with patch.object(build_gc, "inspect", side_effect=inspect):
            outcome = build_gc.apply(self.root, policy=retention.Policy(newest_k=0),
                                     blocked_check=lambda: None)
        self.assertTrue(b.exists())
        self.assertEqual(len(outcome["skipped"]), 1)
        self.assertFalse((self.root / "cleanup-records").exists())

    def test_apply_prunes_empty_recipe_directory(self):
        old = self.build("alone", recipe="unused")
        status, result = self.run_cli(apply=True)
        self.assertEqual(status, 0, result)
        self.assertFalse(old.parent.exists())

    def test_missing_symlinked_and_unreadable_runs_fail_closed_on_apply(self):
        old = self.build("old")
        runs = self.root / "runs"
        runs.rmdir()
        for label in ("missing", "symlink"):
            if label == "symlink":
                actual = self.root / "run-records"
                actual.mkdir()
                runs.symlink_to(actual, target_is_directory=True)
            self.assertEqual(self.decisions()["old"]["action"], "protected", label)
            outcome = build_gc.apply(self.root, policy=retention.Policy(newest_k=0),
                                     blocked_check=lambda: None)
            self.assertEqual(outcome["removed"], [], label)
            self.assertTrue(old.exists(), label)
        runs.unlink()
        runs.mkdir()
        with patch.object(retention, "inspect_workspace", side_effect=PermissionError("runs unreadable")):
            self.assertEqual(self.decisions()["old"]["action"], "protected")
            outcome = build_gc.apply(self.root, policy=retention.Policy(newest_k=0),
                                     blocked_check=lambda: None)
        self.assertEqual(outcome["removed"], [])
        self.assertTrue(old.exists())

    def test_installed_release_receipt_pins_exact_build_on_apply(self):
        pinned = self.build("a" * 32, age=60)
        other = self.build("b" * 32, age=60)
        release = self.root / "releases" / "bc-radiance-1.2.3"
        release.mkdir(parents=True)
        (release / "install.json").write_text(json.dumps({
            "tag": release.name, "source": "ci", "source_slice_id": pinned.parent.name,
            "plan_id": pinned.name, "build_dir": str(pinned),
        }))
        self.assertEqual(self.decisions()[pinned.name]["action"], "protected")
        self.assertEqual(self.decisions()[other.name]["action"], "candidate")
        outcome = build_gc.apply(self.root, policy=retention.Policy(newest_k=0),
                                 blocked_check=lambda: None)
        self.assertTrue(pinned.exists())
        self.assertFalse(other.exists())
        self.assertEqual([p["path"] for p in outcome["removed"]], [str(other)])

    def test_invalid_or_missing_installed_release_receipt_blocks_all_gc(self):
        old = self.build("old")
        release = self.root / "releases" / "bc-radiance-1.2.3"
        release.mkdir(parents=True)
        for invalid in (None, "{corrupt", json.dumps({"build_dir": "../escape"})):
            receipt = release / "install.json"
            if invalid is None:
                receipt.unlink(missing_ok=True)
            else:
                receipt.write_text(invalid)
            self.assertEqual(self.decisions()["old"]["action"], "protected")
            result = build_gc.apply(self.root, policy=retention.Policy(newest_k=0),
                                    blocked_check=lambda: None)
            self.assertEqual(result["removed"], [])
            self.assertTrue(old.exists())
        receipt.unlink()
        outside = self.root / "outside.json"
        outside.write_text("{}")
        receipt.symlink_to(outside)
        self.assertEqual(self.decisions()["old"]["action"], "protected")

    def test_nan_infinity_and_negative_retention_policy_rejected_before_gc(self):
        young = self.build("young", age=0)
        for number in (math.nan, math.inf, -math.inf, -1.0):
            with self.subTest(number=number):
                self.assertEqual(self.run_cli(min_age_days=number)[0], 2)
                policy = retention.Policy(min_age_days=number, newest_k=0)
                with self.assertRaises(ValueError):
                    build_gc.inspect(self.root, policy=policy)
                with self.assertRaises(ValueError):
                    build_gc.apply(self.root, policy=policy, blocked_check=lambda: None)
                self.assertTrue(young.exists())
        self.assertEqual(self.run_cli(keep_newest=-1)[0], 2)

    def test_nested_symlink_and_simulated_same_device_mount_refuse_candidate(self):
        old = self.build("old")
        external = self.root / "external"
        external.mkdir()
        important = external / "important"
        important.write_text("untouched")
        link = old / "inner" / "link"
        link.parent.mkdir()
        link.symlink_to(external, target_is_directory=True)
        os.utime(old, (time.time() - 30 * DAY,) * 2)
        self.assertEqual(self.decisions()["old"]["action"], "protected")
        self.assertEqual(build_gc.apply(
            self.root, policy=retention.Policy(newest_k=0),
            blocked_check=lambda: None)["removed"], [])
        link.unlink()
        fake_mount = old / "inner"
        # A same-device bind mount evades st_dev; inject a mount point read from mountinfo.
        info = self.root / "simulated-mountinfo"
        info.write_text(f"19 1 8:1 / {fake_mount} rw - ext4 /dev/x rw\n")
        real_mount_reader = build_gc._mount_points
        with patch.object(build_gc, "_mount_points", side_effect=lambda: real_mount_reader(info)):
            self.assertEqual(self.decisions()["old"]["action"], "protected")
            result = build_gc.apply(self.root, policy=retention.Policy(newest_k=0),
                                    blocked_check=lambda: None)
        self.assertEqual(result["removed"], [])
        self.assertTrue(old.exists())
        self.assertEqual(important.read_text(), "untouched")


    def _assert_synthetic_mount_boundary_protected(self, location):
        build = self.build("old")
        nested = build / "inner" / "target"
        nested.mkdir(parents=True)
        os.utime(build, (time.time() - 30 * DAY,) * 2)
        targets = {
            "builds": self.root / "builds",
            "recipe": build.parent,
            "build": build,
            "nested": nested,
        }
        mounted = targets[location]
        info = self.root / "synthetic-mountinfo"
        info.write_text(f"19 1 8:1 / {mounted} rw - ext4 /dev/x rw\n")
        reader = build_gc._mount_points
        with patch.object(build_gc, "_mount_points", side_effect=lambda: reader(info)):
            decision = self.decisions()["old"]
            outcome = build_gc.apply(self.root, policy=retention.Policy(newest_k=0),
                                     blocked_check=lambda: None)
        self.assertEqual(decision["action"], "protected", location)
        self.assertIn("mount point on build GC path", " ".join(decision["reasons"]), location)
        self.assertEqual(outcome["removed"], [], location)
        self.assertTrue(build.is_dir(), location)
        self.assertTrue((build / "binary").is_file(), location)

    def test_mount_at_builds_boundary_protects_build_and_apply(self):
        self._assert_synthetic_mount_boundary_protected("builds")

    def test_mount_at_recipe_protects_build_and_apply(self):
        self._assert_synthetic_mount_boundary_protected("recipe")

    def test_mount_at_build_protects_build_and_apply(self):
        self._assert_synthetic_mount_boundary_protected("build")

    def test_mount_at_nested_target_protects_build_and_apply(self):
        self._assert_synthetic_mount_boundary_protected("nested")



    def _assert_component_swap_refused(self, component):
        """Swap one build-cache path component after the fresh inspection."""
        old = self.build("old")
        victim_root = self.root / "external"
        victim = victim_root / ("recipe/old" if component == "builds" else "old")
        victim.mkdir(parents=True)
        important = victim / "important"
        important.write_text("do not delete")
        saved = {
            "builds": self.root / "builds_saved",
            "recipe": old.parent.parent / "recipe_saved",
            "build": old.parent / "old_saved",
        }[component]
        original = build_gc._remove_build_tree
        swaps = []

        def swap_then_remove(*args, **kwargs):
            swaps.append(component)
            current = {
                "builds": self.root / "builds",
                "recipe": old.parent,
                "build": old,
            }[component]
            current.rename(saved)
            destination = victim_root if component != "build" else victim
            current.symlink_to(destination, target_is_directory=True)
            return original(*args, **kwargs)

        with patch.object(build_gc, "_remove_build_tree", side_effect=swap_then_remove):
            outcome = build_gc.apply(self.root, policy=retention.Policy(newest_k=0),
                                     blocked_check=lambda: None)
        self.assertEqual(swaps, [component])
        self.assertEqual(outcome["removed"], [], component)
        self.assertTrue(important.is_file(), component)
        self.assertEqual(important.read_text(), "do not delete", component)
        self.assertTrue((saved / ("recipe/old/binary" if component == "builds"
                                   else "old/binary" if component == "recipe"
                                   else "binary")).is_file(), component)
        self.assertEqual(len(outcome["skipped"]), 1, component)
        self.assertIn("component", outcome["skipped"][0]["reason"], component)

    def test_swapped_builds_symlink_after_reinspection_cannot_redirect_gc(self):
        self._assert_component_swap_refused("builds")

    def test_swapped_recipe_symlink_after_reinspection_cannot_redirect_gc(self):
        self._assert_component_swap_refused("recipe")

    def test_swapped_build_symlink_after_reinspection_is_explicitly_refused(self):
        self._assert_component_swap_refused("build")

    def test_mount_appearing_at_recipe_before_removal_fails_closed(self):
        old = self.build("old")
        scans = []

        def appearing_mount():
            scans.append(1)
            return {old.parent} if len(scans) >= 3 else set()

        with patch.object(build_gc, "_mount_points", side_effect=appearing_mount):
            outcome = build_gc.apply(self.root, policy=retention.Policy(newest_k=0),
                                     blocked_check=lambda: None)
        self.assertGreaterEqual(len(scans), 3)
        self.assertEqual(outcome["removed"], [])
        self.assertTrue((old / "binary").exists())

    def test_mount_appearing_during_fd_anchored_removal_fails_closed(self):
        old = self.build("old")
        scans = []

        def appearing_mount():
            scans.append(1)
            # Two inventory scans and a removal preflight see no mount.
            # Mount the recipe only when the fd-backed traversal begins.
            return {old.parent} if len(scans) >= 4 else set()

        with patch.object(build_gc, "_mount_points", side_effect=appearing_mount):
            outcome = build_gc.apply(self.root, policy=retention.Policy(newest_k=0),
                                     blocked_check=lambda: None)
        self.assertGreaterEqual(len(scans), 4)
        self.assertEqual(outcome["removed"], [])
        self.assertTrue((old / "binary").exists())

    def test_mount_table_unreadable_during_fd_removal_fails_closed(self):
        old = self.build("old")
        scans = []

        def unreadable_mountinfo():
            scans.append(1)
            if len(scans) >= 4:
                raise build_gc.BuildGCError("mountinfo became unreadable")
            return set()

        with patch.object(build_gc, "_mount_points", side_effect=unreadable_mountinfo):
            outcome = build_gc.apply(self.root, policy=retention.Policy(newest_k=0),
                                     blocked_check=lambda: None)
        self.assertGreaterEqual(len(scans), 4)
        self.assertEqual(outcome["removed"], [])
        self.assertTrue((old / "binary").exists())

    def test_mount_at_or_above_work_root_is_not_an_implicit_blocker(self):
        old = self.build("old")
        policy = retention.Policy(newest_k=0)
        for mount in (self.root, self.root.parent):
            with self.subTest(mount=mount), patch.object(
                build_gc, "_mount_points", return_value={mount}
            ):
                entry = next(item for item in build_gc.inspect(
                    self.root, policy=policy, measure=False)
                    if item["path"] == str(old))
                self.assertEqual(entry["action"], "candidate")
        self.assertTrue(old.exists())

    def test_mountinfo_decodes_escaped_paths_and_traversal_protected(self):
        old = self.build("old")
        mounted = old / "space dir"
        mounted.mkdir()
        info = self.root / "fake-mountinfo"
        encoded = str(mounted).replace(" ", chr(92) + "040")
        info.write_text(f"19 1 8:1 / {encoded} rw - ext4 /dev/x rw\n")
        self.assertIn(mounted, build_gc._mount_points(info))
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "keep").write_text("ok")
        (old / "escape").symlink_to(outside / ".." / "outside", target_is_directory=True)
        os.utime(old, (time.time() - 30 * DAY,) * 2)
        self.assertEqual(self.decisions()["old"]["action"], "protected")
        self.assertEqual((outside / "keep").read_text(), "ok")

    def test_process_scan_error_refuses_apply(self):
        old = self.build("old")
        with patch.object(build_gc, "_proc_references",
                          return_value=([], ["cannot inspect surviving process"])):
            result = build_gc.apply(self.root, policy=retention.Policy(newest_k=0),
                                    blocked_check=lambda: None)
        self.assertEqual(result["removed"], [])
        self.assertTrue(old.exists())
        self.assertFalse((self.root / "cleanup-records").exists())

    def test_uninspectable_mount_table_fails_closed(self):
        old = self.build("old")
        with patch.object(build_gc, "_mount_points", side_effect=build_gc.BuildGCError("mountinfo denied")):
            self.assertEqual(self.decisions()["old"]["action"], "protected")
            self.assertEqual(build_gc.apply(
                self.root, policy=retention.Policy(newest_k=0),
                blocked_check=lambda: None)["removed"], [])
        self.assertTrue(old.exists())

    def test_non_build_apply_and_no_size_apply_refused(self):
        self.build("old")
        self.assertEqual(self.run_cli(apply=True, no_size=True)[0], 2)
        self.assertEqual(self.run_cli(apply=True, builds=False)[0], 2)
        self.assertTrue((self.root / "builds" / "recipe" / "old").is_dir())


    def test_nested_external_victim_reparent_after_open_never_deleted(self):
        """Old fd walk can delete a pre-existing external victim after it is moved back out."""
        old = self.build("old")
        (old / "inner").mkdir()
        os.utime(old, (time.time() - 30 * DAY,) * 2)
        outside = self.root / "external"
        victim = outside / "victim"
        victim.mkdir(parents=True)
        important = victim / "important"
        important.write_bytes(b"EXTERNAL VICTIM")
        inode = victim.stat().st_ino
        original_stat, original_scandir = os.stat, os.scandir
        seen, swapped = [], []

        def stat_hook(name, *args, **kwargs):
            if name == "inner" and kwargs.get("dir_fd") is not None and old.exists():
                seen.append("inner")
                if len(seen) == 2:
                    (old / "inner").rename(old / "inner-saved")
                    victim.rename(old / "inner")
                    swapped.append("into build")
            return original_stat(name, *args, **kwargs)

        def scandir_hook(target):
            if (isinstance(target, int) and len(swapped) == 1
                    and os.fstat(target).st_ino == inode):
                (old / "inner").rename(victim)
                (old / "inner").symlink_to(old / "inner-saved", target_is_directory=True)
                swapped.append("back outside after fd open")
            return original_scandir(target)

        with patch.object(build_gc.os, "stat", side_effect=stat_hook), patch.object(
                build_gc.os, "scandir", side_effect=scandir_hook):
            outcome = build_gc.apply(self.root, policy=retention.Policy(newest_k=0),
                                     blocked_check=lambda: None)
        self.assertEqual(seen, ["inner"])
        self.assertEqual(swapped, [], "the public old/inner path must be gone before descent")
        # With a quarantine, the shared original location is inaccessible at descent.
        self.assertEqual(important.read_bytes(), b"EXTERNAL VICTIM")
        self.assertEqual([entry["path"] for entry in outcome["removed"]], [str(old)])

    def _make_interrupted_quarantine(self):
        old = self.build("old")
        original_unlink = os.unlink
        interrupted = []

        def interrupt(name, *args, **kwargs):
            if name == "binary" and kwargs.get("dir_fd") is not None and not interrupted:
                interrupted.append(1)
                raise OSError("simulated interrupted quarantine removal")
            return original_unlink(name, *args, **kwargs)

        with patch.object(build_gc.os, "unlink", side_effect=interrupt):
            first = build_gc.apply(self.root, policy=retention.Policy(newest_k=0),
                                   blocked_check=lambda: None)
        self.assertEqual(interrupted, [1])
        self.assertEqual(first["removed"], [])
        self.assertFalse(old.exists())
        quarantine = self.root / "builds" / ".gc-quarantine"
        leftovers = list(quarantine.iterdir())
        self.assertEqual(len(leftovers), 1)
        self.assertEqual((leftovers[0] / "binary").read_bytes(), b"binary")
        return old, leftovers[0], first

    def test_interrupted_quarantine_is_protected_without_pins_or_locks(self):
        old, leftover, first = self._make_interrupted_quarantine()
        self.assertEqual(first["skipped"][0]["quarantine_path"], str(leftover))
        inventory = build_gc.inspect(self.root, policy=retention.Policy(newest_k=0))
        entry = next(e for e in inventory if e["path"] == str(leftover))
        self.assertEqual(entry["kind"], "quarantine")
        self.assertEqual(entry["action"], "protected")
        self.assertIn("verify and remove by hand", " ".join(entry["reasons"]))
        self.assertGreaterEqual(entry["size_bytes"], len(b"binary"))
        self.assertIsNotNone(entry["age_days"])
        resumed = build_gc.apply(self.root, policy=retention.Policy(newest_k=0),
                                 blocked_check=lambda: None)
        self.assertEqual(resumed["removed"], [])
        self.assertEqual((leftover / "binary").read_bytes(), b"binary")
        self.assertFalse(old.exists())

    def test_interrupted_quarantine_stays_protected_with_new_run_pin(self):
        old, leftover, _ = self._make_interrupted_quarantine()
        workspace.create_run(self.root, "ab", "pinned", plan_item="WSL09",
                             engine="radiance", argv=[], references=[
                                 {"type": "build", "id": old.name, "role": "build"}])
        entry = next(e for e in build_gc.inspect(
            self.root, policy=retention.Policy(newest_k=0))
                     if e["path"] == str(leftover))
        self.assertEqual(entry["action"], "protected")
        resumed = build_gc.apply(self.root, policy=retention.Policy(newest_k=0),
                                 blocked_check=lambda: None)
        self.assertEqual(resumed["removed"], [])
        self.assertEqual((leftover / "binary").read_bytes(), b"binary")

    def test_interrupted_quarantine_stays_protected_with_original_build_lock(self):
        old, leftover, _ = self._make_interrupted_quarantine()
        lock = ResourceLock(self.root / "resource-locks", f"build-{old.name}")
        lock.acquire(timeout_seconds=0)
        try:
            entry = next(e for e in build_gc.inspect(
                self.root, policy=retention.Policy(newest_k=0))
                         if e["path"] == str(leftover))
            self.assertEqual(entry["action"], "protected")
            resumed = build_gc.apply(self.root, policy=retention.Policy(newest_k=0),
                                     blocked_check=lambda: None)
            self.assertEqual(resumed["removed"], [])
            self.assertEqual((leftover / "binary").read_bytes(), b"binary")
        finally:
            lock.release()

    def test_recipe_prune_eacces_after_build_deletion_is_only_warning(self):
        old = self.build("old")
        original_rmdir = os.rmdir
        attempted = []

        def refuse_prune(name, *args, **kwargs):
            if name == "recipe" and kwargs.get("dir_fd") is not None:
                attempted.append(name)
                raise OSError(errno.EACCES, "injected prune permission denied")
            return original_rmdir(name, *args, **kwargs)

        with patch.object(build_gc.os, "rmdir", side_effect=refuse_prune):
            outcome = build_gc.apply(self.root, policy=retention.Policy(newest_k=0),
                                     blocked_check=lambda: None)
        self.assertEqual(attempted, ["recipe"])
        self.assertFalse(old.exists())
        self.assertEqual(outcome["skipped"], [])
        self.assertEqual([item["path"] for item in outcome["removed"]], [str(old)])
        self.assertIn("prune", outcome["removed"][0]["warning"])
        journal = [json.loads(line) for line in
                   Path(outcome["record"]).read_text(encoding="utf-8").splitlines()]
        self.assertEqual([item["state"] for item in journal], ["pending", "removed"])
        self.assertIn("warning", journal[-1])

    def test_completion_journal_failure_after_deletion_still_reports_removed(self):
        old = self.build("old")
        original_fsync = os.fsync
        calls = []

        def fail_second(fd):
            calls.append(fd)
            if len(calls) == 2:
                raise OSError(errno.EIO, "injected completion audit fsync failure")
            return original_fsync(fd)

        with patch.object(build_gc.os, "fsync", side_effect=fail_second):
            outcome = build_gc.apply(self.root, policy=retention.Policy(newest_k=0),
                                     blocked_check=lambda: None)
        self.assertFalse(old.exists())
        self.assertEqual(outcome["skipped"], [])
        self.assertEqual([item["path"] for item in outcome["removed"]], [str(old)])
        self.assertIn("completion journal", outcome["removed"][0]["warning"])

    def test_quarantine_symlink_and_mount_refuse_build_gc(self):
        for hazard in ("symlink", "mount"):
            with self.subTest(hazard=hazard):
                old = self.build("old", recipe=hazard)
                quarantine = self.root / "builds" / ".gc-quarantine"
                outside = self.root / "external-quarantine"
                if hazard == "symlink":
                    outside.mkdir()
                    important = outside / "important"
                    important.write_bytes(b"IMPORTANT")
                    quarantine.symlink_to(outside, target_is_directory=True)
                    reader = lambda: set()
                else:
                    quarantine.mkdir(mode=0o700)
                    reader = lambda quarantine=quarantine: {quarantine}
                with patch.object(build_gc, "_mount_points", side_effect=reader):
                    decision = next(e for e in build_gc.inspect(
                        self.root, policy=retention.Policy(newest_k=0)) if e["path"] == str(old))
                    outcome = build_gc.apply(self.root, policy=retention.Policy(newest_k=0),
                                             blocked_check=lambda: None)
                self.assertEqual(decision["action"], "protected")
                self.assertEqual(outcome["removed"], [])
                self.assertTrue((old / "binary").is_file())
                if hazard == "symlink":
                    self.assertEqual(important.read_bytes(), b"IMPORTANT")
                    quarantine.unlink()
                    important.unlink()
                    outside.rmdir()
                else:
                    quarantine.rmdir()

    def test_mount_introduced_after_quarantine_rename_stops_deletion(self):
        old = self.build("old")
        quarantine = self.root / "builds" / ".gc-quarantine"
        seen = []

        def appearing_mount():
            if quarantine.is_dir() and not old.exists():
                entries = list(quarantine.iterdir())
                if entries:
                    seen.append(entries[0])
                    return {entries[0]}
            return set()

        with patch.object(build_gc, "_mount_points", side_effect=appearing_mount):
            outcome = build_gc.apply(self.root, policy=retention.Policy(newest_k=0),
                                     blocked_check=lambda: None)
            self.assertEqual(outcome["removed"], [])
            self.assertTrue(seen)
            self.assertTrue((seen[0] / "binary").is_file())
            decision = next(e for e in build_gc.inspect(
                self.root, policy=retention.Policy(newest_k=0))
                if e["path"] == str(seen[0]))
            self.assertEqual(decision["action"], "protected")
        # The mount disappearing does not authorize recovery from quarantine.
        resumed = build_gc.apply(self.root, policy=retention.Policy(newest_k=0),
                                 blocked_check=lambda: None)
        self.assertEqual(resumed["removed"], [])
        self.assertTrue((seen[0] / "binary").is_file())

    def test_symlinked_leftover_quarantine_entry_is_never_followed(self):
        self.build("young", age=1)
        quarantine = self.root / "builds" / ".gc-quarantine"
        quarantine.mkdir(mode=0o700)
        outside = self.root / "external"
        outside.mkdir()
        important = outside / "important"
        important.write_bytes(b"DO NOT DELETE")
        entry = quarantine / ("gc-" + "a" * 32)
        entry.symlink_to(outside, target_is_directory=True)
        inventory = build_gc.inspect(self.root, policy=retention.Policy(newest_k=0))
        leftover = next(r for r in inventory if r["path"] == str(entry))
        self.assertEqual(leftover["action"], "protected")
        result = build_gc.apply(self.root, policy=retention.Policy(newest_k=0),
                                blocked_check=lambda: None)
        self.assertEqual(result["removed"], [])
        self.assertEqual(important.read_bytes(), b"DO NOT DELETE")

    def test_rename_replacement_inode_cannot_be_recovered_as_leftover(self):
        old = self.build("old")
        victim = self.root / "outside" / "victim"
        victim.mkdir(parents=True)
        important = victim / "important"
        important.write_bytes(b"EXTERNAL REPLACEMENT")
        saved = old.parent / "old-saved"
        original_rename = os.rename
        replaced = []

        def replace_at_rename(src, dst, *args, **kwargs):
            if kwargs.get("src_dir_fd") is not None and not replaced:
                old.rename(saved)
                os.utime(saved, None)  # original kept young on subsequent apply
                victim.rename(old)
                replaced.append(1)
            return original_rename(src, dst, *args, **kwargs)

        with patch.object(build_gc.os, "rename", side_effect=replace_at_rename):
            outcome = build_gc.apply(self.root, policy=retention.Policy(newest_k=0),
                                     blocked_check=lambda: None)
        self.assertEqual(replaced, [1])
        self.assertEqual(outcome["removed"], [])
        self.assertTrue((saved / "binary").is_file())
        quarantine = self.root / "builds" / ".gc-quarantine"
        residual = list(quarantine.iterdir())
        self.assertEqual(len(residual), 1)
        self.assertEqual((residual[0] / "important").read_bytes(), b"EXTERNAL REPLACEMENT")
        inventory = build_gc.inspect(self.root, policy=retention.Policy(newest_k=0))
        entry = next(r for r in inventory if r["path"] == str(residual[0]))
        self.assertEqual(entry["action"], "protected")
        self.assertIn("verify and remove by hand", " ".join(entry["reasons"]))
        retry = build_gc.apply(self.root, policy=retention.Policy(newest_k=0),
                               blocked_check=lambda: None)
        self.assertEqual(retry["removed"], [])
        self.assertEqual((residual[0] / "important").read_bytes(), b"EXTERNAL REPLACEMENT")

    def test_failed_quarantine_rename_removes_nothing(self):
        old = self.build("old")
        original_rename = os.rename
        attempted = []

        def rename_hook(src, dst, *args, **kwargs):
            if kwargs.get("src_dir_fd") is not None:
                attempted.append((src, dst))
                raise OSError("injected rename failure")
            return original_rename(src, dst, *args, **kwargs)

        with patch.object(build_gc.os, "rename", side_effect=rename_hook):
            outcome = build_gc.apply(self.root, policy=retention.Policy(newest_k=0),
                                     blocked_check=lambda: None)
        self.assertTrue(attempted)
        self.assertEqual(outcome["removed"], [])
        self.assertTrue((old / "binary").is_file())

    def test_native_install_release_pins_exact_build(self):
        pinned = self.build("a" * 32, recipe="b" * 32, age=40)
        other = self.build("c" * 32, recipe="b" * 32, age=40)
        release = self.root / "installed-radiance" / "releases" / "bc-radiance-1170.33.0"
        release.mkdir(parents=True)
        (release / "install.json").write_text(json.dumps({
            "schema": "bc-radiance-installed-v1", "tag": release.name,
            "record": {"source_slice_id": pinned.parent.name, "plan_id": pinned.name},
        }))
        self.assertIn("installed release", " ".join(self.decisions()[pinned.name]["reasons"]))
        self.assertEqual(self.decisions()[other.name]["action"], "candidate")

    def test_native_install_bad_receipts_block_all_gc(self):
        old = self.build("a" * 32, age=30)
        release = self.root / "installed-radiance" / "releases" / "bc-radiance-1170.33.0"
        release.mkdir(parents=True)
        receipt = release / "install.json"
        for invalid in (None, "{broken", json.dumps({"tag": release.name, "record": {}})):
            receipt.unlink(missing_ok=True)
            if invalid is not None:
                receipt.write_text(invalid)
            self.assertEqual(self.decisions()[old.name]["action"], "protected")
        receipt.unlink(missing_ok=True)
        (self.root / "outside.json").write_text("{}")
        receipt.symlink_to(self.root / "outside.json")
        self.assertEqual(self.decisions()[old.name]["action"], "protected")

    def test_immutable_run_record_build_identity_pins_build(self):
        from bigcherry.core.run_records import RunRecordStore
        pinned = self.build("a" * 32, age=30)
        other = self.build("b" * 32, age=30)
        store = RunRecordStore(self.root)
        sidecar = pinned / "bigcherry-build-metadata-radiance.json"
        sidecar.write_text(json.dumps({"build_id": "d" * 32,
                                       "source_slice_id": pinned.parent.name,
                                       "build_plan_id": pinned.name}))
        os.utime(sidecar, (time.time() - 30 * DAY,) * 2)
        blob = store.put_blob(b"sealed")
        record = {
            "schema_version": 1, "run_id": "measure-1", "recorded_at": "2026-11-18T12:00:00Z",
            "identity": {"engine": "radiance",  "build_id": "d" * 32,
                         "binary_sha256": "f" * 64, "source_revision": "rev",
                         "model": "model", "layout": "layout"},
            "workload": {"name": "decode", "definition": {}},
            "metrics": {"decode": {}, "prefill": {}, "ttft": {}},
            "metric_specs": {"decode": {}, "prefill": {}, "ttft": {}},
            "provenance": {"effective_config": blob, "argv": blob,
                           "environment": blob, "power_settings": blob, "inputs": []},
            "host_facts": __import__("bigcherry.core.host_facts", fromlist=["snapshot_at_start"])
                              .snapshot_at_start(
                                  {"kernel_release": {"value": "fixture", "readable": True}},
                                  captured_at="2026-11-18T11:59:00Z"),
            "feature_states": [],
        }
        store.put_record(record)
        self.assertIn("run record", " ".join(self.decisions()[pinned.name]["reasons"]))
        self.assertEqual(self.decisions()[other.name]["action"], "candidate")
        next(store.root.glob("records/*/*.json")).write_text("{broken")
        self.assertIn("run-record", " ".join(self.decisions()[other.name]["reasons"]))

    def test_stored_serving_manifest_runtime_build_is_pinned(self):
        pinned = self.build("a" * 32, age=40)
        other = self.build("b" * 32, age=40)
        site = self.root / "flashnext"
        site.mkdir()
        saved = site / "manifest-123.json"
        saved.write_text(json.dumps({
            "apiVersion": "bigcherry/v2", "kind": "ServingManifest",
            "spec": {"processes": {"engine": {"runtime": {
                "radiance_home": str(pinned / "radiance_home")}}}},
        }))
        self.assertIn("stored serving manifest", " ".join(self.decisions()[pinned.name]["reasons"]))
        self.assertEqual(self.decisions()[other.name]["action"], "candidate")
        saved.write_text("{invalid")
        self.assertIn("stored manifest", " ".join(self.decisions()[other.name]["reasons"]))

    def test_stored_v3_serving_manifest_runtime_build_is_pinned(self):
        pinned = self.build("a" * 32, age=40)
        other = self.build("b" * 32, age=40)
        site = self.root / "flashnext"
        site.mkdir()
        (site / "manifest-native-123.json").write_text(json.dumps({
            "apiVersion": "bigcherry/v3", "kind": "ServingManifest",
            "spec": {"processes": {"engine": {"runtime": {
                "radiance_home": str(pinned / "radiance_home")}}}},
        }))
        self.assertIn("stored serving manifest", " ".join(self.decisions()[pinned.name]["reasons"]))
        self.assertEqual(self.decisions()[other.name]["action"], "candidate")

    def test_symlinked_record_store_blocks_gc(self):
        old = self.build("a" * 32, age=35)
        elsewhere = self.root / "external"
        elsewhere.mkdir()
        (self.root / "run-records").symlink_to(elsewhere, target_is_directory=True)
        self.assertEqual(self.decisions()[old.name]["action"], "protected")
        self.assertIn("run-record", " ".join(self.decisions()[old.name]["reasons"]))

    def test_hash_only_plan_cache_does_not_pin_build_tree(self):
        old = self.build("a" * 32, age=35)
        cache = self.root / "plan-cache"
        cache.mkdir()
        (cache / ("f" * 64 + ".json")).write_text(json.dumps({
            "cache_version": 2, "key_inputs": {"engine_build": "e" * 64},
        }))
        self.assertEqual(self.decisions()[old.name]["action"], "candidate")


if __name__ == "__main__":
    unittest.main()

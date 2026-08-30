"""Git-tree source identity tests."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.source.identity import (  # noqa: E402
    SourceIdentityError, atomic_write_json, git_tree_oid, source_slice_id,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


class SourceIdentityTests(unittest.TestCase):
    def test_clean_tree_matches_commit_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _git(root.parent, "init", str(root))
            _git(root, "config", "user.email", "test@example.invalid")
            _git(root, "config", "user.name", "Test")
            (root / "source.txt").write_text("one\n", encoding="utf-8")
            _git(root, "add", "source.txt")
            _git(root, "commit", "-m", "initial")
            self.assertEqual(git_tree_oid(root), _git(root, "rev-parse", "HEAD^{tree}"))

    def test_unexpected_untracked_files_fail_and_allowed_files_are_included(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _git(root.parent, "init", str(root))
            _git(root, "config", "user.email", "test@example.invalid")
            _git(root, "config", "user.name", "Test")
            (root / "source.txt").write_text("one\n", encoding="utf-8")
            _git(root, "add", "source.txt")
            _git(root, "commit", "-m", "initial")
            (root / "overlay.txt").write_text("overlay\n", encoding="utf-8")
            with self.assertRaises(SourceIdentityError):
                git_tree_oid(root)
            with_allowed = git_tree_oid(root, allowed_untracked={"overlay.txt"})
            self.assertNotEqual(with_allowed, _git(root, "rev-parse", "HEAD^{tree}"))

    def test_modified_tracked_file_is_not_unexpected(self):
        # workspace.materialize() overlays files and applies patches AFTER
        # add_detached_worktree() checks out a pristine upstream revision --
        # both legitimately modify already-tracked upstream files (that is
        # the entire mechanism of a patch). A materialize() against any
        # source with real patches must not be rejected as "unexpected"
        # for exactly the modifications it is supposed to produce.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _git(root.parent, "init", str(root))
            _git(root, "config", "user.email", "test@example.invalid")
            _git(root, "config", "user.name", "Test")
            (root / "source.txt").write_text("one\n", encoding="utf-8")
            _git(root, "add", "source.txt")
            _git(root, "commit", "-m", "initial")
            # Simulate a patch: modify an already-tracked file in place.
            (root / "source.txt").write_text("one\npatched\n", encoding="utf-8")
            tree_oid = git_tree_oid(root)
            self.assertNotEqual(tree_oid, _git(root, "rev-parse", "HEAD^{tree}"))

    def test_deleted_tracked_file_is_not_unexpected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _git(root.parent, "init", str(root))
            _git(root, "config", "user.email", "test@example.invalid")
            _git(root, "config", "user.name", "Test")
            (root / "a.txt").write_text("a\n", encoding="utf-8")
            (root / "b.txt").write_text("b\n", encoding="utf-8")
            _git(root, "add", "a.txt", "b.txt")
            _git(root, "commit", "-m", "initial")
            (root / "b.txt").unlink()
            tree_oid = git_tree_oid(root)
            self.assertNotEqual(tree_oid, _git(root, "rev-parse", "HEAD^{tree}"))

    def test_modification_plus_genuinely_unexpected_untracked_file_still_rejected(self):
        # The fix must not become blanket-permissive: a real stray
        # untracked file alongside a legitimate patch modification is
        # still exactly what this check exists to catch.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _git(root.parent, "init", str(root))
            _git(root, "config", "user.email", "test@example.invalid")
            _git(root, "config", "user.name", "Test")
            (root / "source.txt").write_text("one\n", encoding="utf-8")
            _git(root, "add", "source.txt")
            _git(root, "commit", "-m", "initial")
            (root / "source.txt").write_text("one\npatched\n", encoding="utf-8")
            (root / "stray.txt").write_text("litter\n", encoding="utf-8")
            with self.assertRaises(SourceIdentityError):
                git_tree_oid(root)

    def test_slice_identity_is_content_based(self):
        first = source_slice_id(upstream_revision="a" * 40, tree_oid="b" * 40)
        same = source_slice_id(upstream_revision="a" * 40, tree_oid="b" * 40)
        changed = source_slice_id(upstream_revision="a" * 40, tree_oid="c" * 40)
        self.assertEqual(first, same)
        self.assertNotEqual(first, changed)


class AtomicWriteJsonTests(unittest.TestCase):
    """PA12 (source/patch identity hardening L6.1): shared atomic JSON
    writer -- temp file + fsync + os.replace(), so a crash mid-write cannot
    leave truncated metadata beside a valid source worktree."""

    def test_write_then_read_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.json"
            atomic_write_json(path, {"a": 1, "b": [1, 2, 3]})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"a": 1, "b": [1, 2, 3]})

    def test_no_temp_file_left_behind_on_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.json"
            atomic_write_json(path, {"a": 1})
            leftovers = [p for p in Path(directory).iterdir() if p != path]
            self.assertEqual(leftovers, [])

    def test_previous_valid_content_survives_a_write_interruption(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.json"
            atomic_write_json(path, {"version": 1})
            with mock.patch("os.replace", side_effect=OSError("simulated crash")):
                with self.assertRaises(OSError):
                    atomic_write_json(path, {"version": 2})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"version": 1})
            leftovers = [p for p in Path(directory).iterdir() if p != path]
            self.assertEqual(leftovers, [], "the failed temp file must be cleaned up, not left behind")

    def test_fsync_parent_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.json"
            atomic_write_json(path, {"a": 1}, fsync_parent=False)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"a": 1})


class PlanLockTests(unittest.TestCase):
    """PA12 (source/patch identity hardening L6.2): two concurrent
    materializations of the same plan must serialize, not race."""

    def test_two_threads_serialize_on_the_same_plan_id(self):
        import threading
        import time
        from bigcherry.source.identity import plan_lock

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events: list[tuple[str, str]] = []
            lock_guard = threading.Lock()

            def worker(name: str) -> None:
                with plan_lock(root, "shared-plan", timeout_seconds=10):
                    with lock_guard:
                        events.append((name, "enter"))
                    time.sleep(0.15)
                    with lock_guard:
                        events.append((name, "exit"))

            t1 = threading.Thread(target=worker, args=("a",))
            t2 = threading.Thread(target=worker, args=("b",))
            t1.start()
            time.sleep(0.03)
            t2.start()
            t1.join(timeout=10)
            t2.join(timeout=10)

            # Whichever thread entered first must also exit before the
            # other enters -- no interleaving of the two critical sections.
            first = events[0][0]
            self.assertEqual(events[1], (first, "exit"))

    def test_different_plan_ids_do_not_contend(self):
        from bigcherry.source.identity import plan_lock

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with plan_lock(root, "plan-a", timeout_seconds=5):
                with plan_lock(root, "plan-b", timeout_seconds=5):
                    pass  # must not deadlock or time out

    def test_timeout_raises_when_lock_is_held(self):
        import threading
        from bigcherry.source.identity import PlanLockTimeout, plan_lock

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            holder_ready = threading.Event()
            release = threading.Event()

            def holder() -> None:
                with plan_lock(root, "contended", timeout_seconds=5):
                    holder_ready.set()
                    release.wait(timeout=5)

            t = threading.Thread(target=holder)
            t.start()
            self.assertTrue(holder_ready.wait(timeout=5))
            with self.assertRaises(PlanLockTimeout):
                with plan_lock(root, "contended", timeout_seconds=0.3, poll_interval=0.05):
                    pass
            release.set()
            t.join(timeout=5)


if __name__ == "__main__":
    unittest.main()

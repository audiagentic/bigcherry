from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from bigcherry.core import build_gc, retention


def test_scratch_build_layout_is_not_managed():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "runs").mkdir()
        valid = root / "builds" / ("a" * 32) / ("b" * 32)
        valid.mkdir(parents=True)
        (valid / "bin").write_text("bin")
        scratch = root / "builds" / "rad42-bar-push"
        (scratch / "obj-rad50").mkdir(parents=True)
        (scratch / "obj-rad50" / "bin").write_text("keep")
        (scratch / "build.log").write_text("keep")
        odd = root / "builds" / ("a" * 32) / "obj-rad50"
        odd.mkdir()
        (odd / "bin").write_text("keep")
        policy = retention.Policy(min_age_days=0, newest_k=0)
        with patch.object(build_gc, "_proc_references", return_value=([], [])):
            entries = {
                Path(x["path"]): x for x in build_gc.inspect(root, policy=policy, measure=False)
            }
            assert entries[valid]["action"] == "candidate"
            for path in (scratch / "obj-rad50", scratch / "build.log", odd):
                assert entries[path]["action"] == "protected"
                assert entries[path]["kind"] == "unclassified"
            build_gc.apply(root, policy=policy, blocked_check=lambda: None)
        assert not valid.exists()
        assert (scratch / "obj-rad50" / "bin").exists()
        assert (scratch / "build.log").exists()
        assert (odd / "bin").exists()


def test_dead_runner_lease_requires_operator_action():
    import json
    from bigcherry.core import tree_activity

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "runs").mkdir()
        build = root / "builds" / ("a" * 32) / ("b" * 32)
        build.mkdir(parents=True)
        leases = root / "tree-activity" / "namespace" / "leases"
        leases.mkdir(parents=True)
        marker = leases / "dead.json"
        marker.write_text(
            json.dumps(
                {
                    "pid": 2147483647,
                    "hostname": "local",
                    "project_root": "/gone",
                    "command": "build",
                    "run_id": "unspecified",
                    "started_at": 0,
                }
            )
        )
        assert not tree_activity._pid_alive(2147483647)
        policy = retention.Policy(min_age_days=0, newest_k=0)
        with patch.object(build_gc, "_proc_references", return_value=([], [])):
            inspect = lambda: next(
                x
                for x in build_gc.inspect(root, policy=policy, measure=False)
                if x["path"] == str(build)
            )
            assert inspect()["action"] == "protected"
            result = build_gc.apply(
                root,
                policy=policy,
                blocked_check=lambda: tree_activity.cleanup_activity_blocker(root),
            )
            assert not result["removed"]
            assert build.exists()
            marker.unlink()  # Mutation of disposable fixture, never the real root.
            assert inspect()["action"] == "candidate"


def test_unreadable_foreign_process_does_not_block_cleanup():
    """A process the cleaner cannot inspect (another user's, or outside the container) cannot hold a build here."""
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "runs").mkdir()
        build = root / "builds" / ("a" * 32) / ("b" * 32)
        build.mkdir(parents=True)
        (build / "bin").write_text("keep")
        policy = retention.Policy(min_age_days=0, newest_k=0)
        error = "cannot inspect process 1 exe: [Errno 13] Permission denied"
        with patch.object(build_gc, "_proc_references", return_value=([], [error])):
            entry = next(x for x in build_gc.inspect(root, policy=policy, measure=False) if x["path"] == str(build))
            assert entry["action"] == "candidate"
            result = build_gc.apply(root, policy=policy, blocked_check=lambda: None)
            assert [Path(r["path"]) for r in result["removed"]] == [build]
        assert not build.exists()

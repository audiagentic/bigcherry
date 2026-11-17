"""Legacy flat runs are adopted without modifying their contents or weakening GC."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bigcherry.core import build_gc, legacy_runs, retention
from bigcherry.core import run_manifest as rm

A = "a" * 32
B = "b" * 32
C = "c" * 32
D = "d" * 32


def _root(tmp: str) -> Path:
    root = Path(tmp)
    (root / "runs").mkdir()
    return root


def _run(root: Path, name="old-run") -> Path:
    run = root / "runs" / name
    run.mkdir()
    return run


def _build(root: Path, recipe=A, build=B) -> Path:
    p = root / "builds" / recipe / build
    p.mkdir(parents=True)
    (p / "binary").write_bytes(b"x")
    return p


def _dry(root, *, apply=False):
    return legacy_runs.migrate(root, apply=apply)


def _entry(root, build):
    with patch.object(build_gc, "_proc_references", return_value=([], [])):
        records = build_gc.inspect(root, policy=retention.Policy(min_age_days=0, newest_k=0), measure=False)
    return next(x for x in records if x["path"] == str(build))


def test_manifest_v1_accepts_legacy_kind_without_adding_nested_run_producer():
    m = rm.new_manifest(run_id="old-run", kind="legacy", purpose="legacy scan",
                        plan_item=None, engine="unknown", argv=[], owner={})
    assert rm.validate_manifest(m)["schema_version"] == 1


def test_chunk_spanning_reference_and_relative_form_scanned_entirely():
    with TemporaryDirectory() as tmp:
        root = _root(tmp)
        run = _run(root)
        prefix = b"p" * (legacy_runs.CHUNK_SIZE - 2) + b" "
        (run / "data.bin").write_bytes(prefix + f"builds/{A}/{B}".encode() + b"\x00"
                                        + f"/other/work/builds/{C}/{D}/output".encode())
        result = _dry(root)
        assert result["totals"]["would_write"] == 1
        assert not (run / rm.MANIFEST_NAME).exists()
        assert {r["id"] for r in result["runs"][0]["references"]} == {f"{A}/{B}", f"{C}/{D}"}
        assert _dry(root, apply=True)["totals"]["written"] == 1
        m = rm.read_manifest(run)
        assert m["kind"] == "legacy"
        assert m["provenance"] == "legacy_inferred"
        assert m["legacy_scan"]["files_scanned"] == 1
        assert m["legacy_scan"]["bytes_scanned"] == len((run / "data.bin").read_bytes())
        assert {r["id"] for r in m["references"]} == {f"{A}/{B}", f"{C}/{D}"}
        assert (run / rm.MANIFEST_NAME).stat().st_mode & 0o777 == 0o644
        assert not (run / (rm.MANIFEST_NAME + ".lock")).exists()
        entry = next(e for e in retention.inspect_workspace(root, measure=False).runs if e.run_id == "old-run")
        assert entry.kind == "legacy" and entry.manifest


def test_unreadable_file_and_symlink_refuse_manifest():
    with TemporaryDirectory() as tmp:
        root = _root(tmp)
        bad = _run(root)
        f = bad / "secret.bin"
        f.write_text(f"builds/{A}/{B}")
        f.chmod(0)
        try:
            result = _dry(root, apply=True)
            assert result["totals"]["errors"] == 1
            assert not (bad / rm.MANIFEST_NAME).exists()
        finally:
            f.chmod(0o600)
        other = _run(root, "link-run")
        (other / "link").symlink_to(f)
        result = _dry(root, apply=True)
        assert any(x["status"] == "error" and x["run_id"] == "link-run" for x in result["runs"])
        assert not (other / rm.MANIFEST_NAME).exists()


def test_existing_manifest_is_never_overwritten():
    with TemporaryDirectory() as tmp:
        root = _root(tmp)
        run = _run(root)
        manifest = run / rm.MANIFEST_NAME
        manifest.write_bytes(b"existing-not-valid-json\n")
        for apply in (False, True):
            result = _dry(root, apply=apply)
            assert result["totals"]["existing"] == 1
            assert manifest.read_bytes() == b"existing-not-valid-json\n"


def test_legacy_build_pin_is_exact_and_unmanifested_run_still_blocks():
    with TemporaryDirectory() as tmp:
        root = _root(tmp)
        pinned = _build(root, A, B)
        other = _build(root, C, D)
        run = _run(root)
        (run / "log.txt").write_text(f"used builds/{A}/{B}/bin\n")
        assert _entry(root, other)["action"] == "protected"  # old flat run blocks all
        assert _dry(root, apply=True)["totals"]["written"] == 1
        assert _entry(root, pinned)["action"] == "protected"
        assert _entry(root, other)["action"] == "candidate"
        unmanaged = _run(root, "unmanaged")
        assert _entry(root, other)["action"] == "protected"
        (unmanaged / "log.txt").write_text("none")
        assert _dry(root, apply=True)["totals"]["written"] == 1
        assert _entry(root, other)["action"] == "candidate"


def test_legacy_no_references_does_not_block_other_build():
    with TemporaryDirectory() as tmp:
        root = _root(tmp)
        build = _build(root)
        run = _run(root)
        (run / "log").write_bytes(b"no build here")
        assert _dry(root, apply=True)["totals"]["written"] == 1
        assert rm.read_manifest(run)["references"] == []
        assert _entry(root, build)["action"] == "candidate"


def test_loose_regular_file_refs_are_preserved_and_unreadable_fails_closed():
    with TemporaryDirectory() as tmp:
        root = _root(tmp)
        pinned = _build(root)
        other = _build(root, C, D)
        loose = root / "runs" / "findings.md"
        loose.write_text(f"observed builds/{A}/{B} in benchmark")
        assert _entry(root, pinned)["action"] == "protected"
        assert _entry(root, other)["action"] == "candidate"
        loose.chmod(0)
        try:
            assert _entry(root, other)["action"] == "protected"
        finally:
            loose.chmod(0o600)


def test_parent_and_run_symlinks_rejected_before_mutation():
    with TemporaryDirectory() as tmp:
        base = Path(tmp)
        outside = base / "outside"
        outside.mkdir()
        (outside / "data").write_text("keep")
        root = base / "work"
        root.mkdir()
        (root / "runs").symlink_to(outside, target_is_directory=True)
        result = _dry(root, apply=True)
        assert result["totals"]["errors"] > 0
        assert not (outside / rm.MANIFEST_NAME).exists()
        (root / "runs").unlink()
        (root / "runs").mkdir()
        (root / "runs" / "external").symlink_to(outside, target_is_directory=True)
        result = _dry(root, apply=True)
        assert result["totals"]["errors"] > 0
        assert not (outside / rm.MANIFEST_NAME).exists()


def test_fatal_read_error_after_partial_scan_never_publishes():
    with TemporaryDirectory() as tmp:
        root = _root(tmp)
        run = _run(root)
        (run / "data").write_bytes(b"partially readable")
        original_read = legacy_runs.os.read
        calls = [0]

        def interrupted_read(fd, length):
            calls[0] += 1
            if calls[0] > 1:
                raise OSError(5, "simulated I/O error")
            return original_read(fd, 4)

        with patch.object(legacy_runs.os, "read", side_effect=interrupted_read):
            result = _dry(root, apply=True)
        assert calls[0] > 1
        assert result["totals"]["errors"] == 1
        assert not (run / rm.MANIFEST_NAME).exists()


def test_swap_flat_run_to_symlink_before_publication_leaves_external_victim_intact():
    with TemporaryDirectory() as tmp:
        base = Path(tmp)
        root = base / "work"
        root.mkdir()
        (root / "runs").mkdir()
        run = _run(root)
        (run / "data").write_text(f"builds/{A}/{B}")
        victim = base / "external"
        victim.mkdir()
        sentinel = victim / (rm.MANIFEST_NAME + ".lock")
        sentinel.write_text("external victim lock")
        original_scan = legacy_runs._scan_dir

        def move_after_scan(*args, **kwargs):
            original_scan(*args, **kwargs)
            run.rename(base / "moved-run")
            run.symlink_to(victim, target_is_directory=True)

        with patch.object(legacy_runs, "_scan_dir", side_effect=move_after_scan):
            result = _dry(root, apply=True)
        assert result["totals"]["errors"] == 1
        assert sentinel.read_text() == "external victim lock"
        assert not (victim / rm.MANIFEST_NAME).exists()
        assert not (base / "moved-run" / rm.MANIFEST_NAME).exists()


def test_racing_manifest_create_does_not_overwrite_existing_bytes():
    with TemporaryDirectory() as tmp:
        root = _root(tmp)
        run = _run(root)
        (run / "data").write_text("log")
        original_publish = legacy_runs._publish

        def race(fd, manifest):
            (run / rm.MANIFEST_NAME).write_bytes(b"existing bytes")
            return original_publish(fd, manifest)

        with patch.object(legacy_runs, "_publish", side_effect=race):
            result = _dry(root, apply=True)
        assert result["totals"]["errors"] == 1
        assert (run / rm.MANIFEST_NAME).read_bytes() == b"existing bytes"


def test_cli_dry_run_requires_explicit_apply_on_synthetic_root(capsys):
    from bigcherry.cli.main import build_parser

    with TemporaryDirectory() as tmp:
        root = _root(tmp)
        run = _run(root)
        (run / "evidence.txt").write_text(f"builds/{A}/{B}")
        parser = build_parser()
        args = parser.parse_args(["legacy-runs", "migrate", "--work-root", str(root), "--json"])
        assert args.func(args) == 0
        result = json.loads(capsys.readouterr().out)
        assert result["read_only"]
        assert result["totals"]["would_write"] == 1
        assert not (run / rm.MANIFEST_NAME).exists()
        args = parser.parse_args(["legacy-runs", "migrate", "--work-root", str(root), "--apply", "--json"])
        assert args.func(args) == 0
        assert json.loads(capsys.readouterr().out)["totals"]["written"] == 1
        assert rm.read_manifest(run)["references"] == [{"type": "build", "id": f"{A}/{B}",
                                                       "role": "legacy-scanned"}]


def test_chunk_edge_not_falsely_matched_before_suffix_is_read():
    with TemporaryDirectory() as tmp:
        root = _root(tmp)
        run = _run(root)
        ref = f"builds/{A}/{B}".encode()
        prefix = b"x" * (legacy_runs.CHUNK_SIZE - len(ref) - 1) + b" "
        (run / "binary").write_bytes(prefix + ref + b"g" + b"\x00" * 100)
        assert _dry(root)["runs"][0]["references"] == []


def test_dry_run_scans_preexisting_lock_file_but_writes_nothing():
    with TemporaryDirectory() as tmp:
        root = _root(tmp)
        run = _run(root)
        (run / (rm.MANIFEST_NAME + ".lock")).write_text(f"builds/{A}/{B}")
        result = _dry(root)
        assert result["totals"]["would_write"] == 1
        assert result["runs"][0]["references"][0]["id"] == f"{A}/{B}"
        assert not (run / rm.MANIFEST_NAME).exists()


def test_mutation_proof_chunk_overlap_and_gc_typed_guard_are_necessary():
    with TemporaryDirectory() as tmp:
        root = _root(tmp)
        run = _run(root)
        build = _build(root)
        data = b" " * (legacy_runs.CHUNK_SIZE - 3) + f"builds/{A}/{B}".encode()
        (run / "data").write_bytes(data)
        assert _dry(root)["runs"][0]["references"][0]["id"] == f"{A}/{B}"
        # Removing the chunk-overlap guarantee loses a valid reference.
        with patch.object(legacy_runs, "_OVERLAP", 1):
            assert not _dry(root)["runs"][0]["references"]
        assert _dry(root, apply=True)["totals"]["written"] == 1
        assert _entry(root, build)["action"] == "protected"
        # Removing the manifest reference guard makes this very build eligible.
        with patch.object(build_gc, "_manifest_refs", return_value=({}, [])):
            assert _entry(root, build)["action"] == "candidate"


def test_bind_mount_and_unreadable_mountinfo_fail_closed():
    from bigcherry.core import build_gc

    with TemporaryDirectory() as tmp:
        root = _root(tmp)
        run = _run(root)
        (run / "data").write_text("no references")
        for mount in (root / "runs", run, run / "nested"):
            with patch.object(build_gc, "_mount_points", return_value={mount}):
                result = _dry(root, apply=True)
            assert result["totals"]["errors"] == 1
            assert not (run / rm.MANIFEST_NAME).exists()
        with patch.object(build_gc, "_mount_points", side_effect=build_gc.BuildGCError("unreadable mountinfo")):
            result = _dry(root, apply=True)
        assert result["totals"]["errors"] == 1
        assert not (run / rm.MANIFEST_NAME).exists()


# WSL09 independent review regressions: deliberately fail on the pre-fix reader/writer.
def test_gc_rejects_unproven_or_symlinked_flat_manifest():
    with TemporaryDirectory() as tmp:
        root = _root(tmp)
        run = _run(root)
        (run / "data").write_text(f"builds/{A}/{B}")
        candidate = _build(root, C, D)
        m = rm.new_manifest(run_id=run.name, kind="legacy", purpose="not scanned",
                            plan_item=None, engine="unknown", argv=[], owner={},
                            references=[])
        path = run / rm.MANIFEST_NAME
        path.write_text(json.dumps(m))
        assert _entry(root, candidate)["action"] == "protected"  # missing provenance
        m["provenance"] = "legacy_inferred"
        m["legacy_scan"] = {"scanner_version": "not-our-scanner"}
        path.write_text(json.dumps(m))
        assert _entry(root, candidate)["action"] == "protected"  # wrong scanner
        path.unlink()
        assert _dry(root, apply=True)["totals"]["written"] == 1
        outside = root / "external-manifest"
        path.rename(outside)
        path.symlink_to(outside)
        assert _entry(root, candidate)["action"] == "protected"  # symlink to valid record


def test_scanned_file_rewritten_after_scan_prevents_publication():
    with TemporaryDirectory() as tmp:
        root = _root(tmp)
        run = _run(root)
        file = run / "data"
        file.write_text(f"builds/{A}/{B}")
        original = legacy_runs._scan_file

        def rewrite_after_scan(*args, **kwargs):
            original(*args, **kwargs)
            file.write_text(f"builds/{C}/{D}")

        with patch.object(legacy_runs, "_scan_file", side_effect=rewrite_after_scan):
            result = _dry(root, apply=True)
        assert result["totals"]["errors"] == 1
        assert not (run / rm.MANIFEST_NAME).exists()


def test_directory_membership_changed_after_scan_prevents_publication():
    with TemporaryDirectory() as tmp:
        root = _root(tmp)
        run = _run(root)
        (run / "data").write_text("hello")
        original = legacy_runs._scan_dir

        def add_after_scan(*args, **kwargs):
            original(*args, **kwargs)
            (run / "new-evidence").write_text(f"builds/{C}/{D}")

        with patch.object(legacy_runs, "_scan_dir", side_effect=add_after_scan):
            result = _dry(root, apply=True)
        assert result["totals"]["errors"] == 1
        assert not (run / rm.MANIFEST_NAME).exists()


def test_later_edit_invalidates_adopted_manifest_and_blocks_all_build_gc():
    with TemporaryDirectory() as tmp:
        root = _root(tmp)
        run = _run(root)
        old = _build(root, A, B)
        new = _build(root, C, D)
        file = run / "data"
        file.write_text(f"builds/{A}/{B}")
        assert _dry(root, apply=True)["totals"]["written"] == 1
        stamp = rm.read_manifest(run)["legacy_scan"]["tree_stamp"]
        assert stamp["file_count"] == 1 and stamp["total_bytes"] == file.stat().st_size
        assert stamp["max_mtime_ns"] == file.stat().st_mtime_ns
        assert _entry(root, new)["action"] == "candidate"
        file.write_text(f"builds/{C}/{D}")
        assert _entry(root, new)["action"] == "protected"
        assert _entry(root, old)["action"] == "protected"


def test_rename_after_publish_rolls_back_manifest_outside_runs():
    with TemporaryDirectory() as tmp:
        base = Path(tmp)
        root = base / "work"
        root.mkdir()
        (root / "runs").mkdir()
        run = _run(root)
        (run / "data").write_text(f"builds/{A}/{B}")
        moved = base / "outside-runs"
        original = legacy_runs._publish

        def move_before_publish(fd, manifest):
            run.rename(moved)
            return original(fd, manifest)

        with patch.object(legacy_runs, "_publish", side_effect=move_before_publish):
            result = _dry(root, apply=True)
        assert result["totals"]["errors"] == 1
        assert not (moved / rm.MANIFEST_NAME).exists()


def test_chunk_boundary_does_not_fabricate_reference_from_midword():
    with TemporaryDirectory() as tmp:
        root = _root(tmp)
        run = _run(root)
        ref = f"builds/{A}/{B}".encode()
        # Test either side of the retained boundary: one processed byte must
        # not be reinterpreted as a new match on the following chunk.
        for back in (1, 2):
            data = b"z" * (legacy_runs.CHUNK_SIZE - legacy_runs._OVERLAP - back) + b"X" + ref + b"\x00"
            (run / "binary").write_bytes(data)
            assert _dry(root)["runs"][0]["references"] == []


def test_targeted_status_resolves_corrupt_nested_run(capsys):
    from argparse import Namespace
    from bigcherry.cli import workspace_cmds

    with TemporaryDirectory() as tmp:
        root = _root(tmp)
        nested = root / "runs" / "accept" / "broken-123"
        nested.mkdir(parents=True)
        (nested / rm.MANIFEST_NAME).write_text("{corrupt json")
        assert workspace_cmds._find(root, "broken-123") == [nested]
        assert workspace_cmds.cmd_run_status(Namespace(work_root=str(root), run_id="broken-123")) == 1
        assert "manifest" in capsys.readouterr().err.lower()
        adopted = _run(root, "adopted-123")
        (adopted / "data").write_text("old evidence")
        assert _dry(root, apply=True)["totals"]["written"] == 1
        assert workspace_cmds._find(root, "adopted-123") == [adopted]

def test_nonmax_file_edit_with_same_total_bytes_still_blocks_gc():
    import os

    with TemporaryDirectory() as tmp:
        root = _root(tmp)
        run = _run(root)
        older = run / "older"
        later = run / "later"
        older.write_text(f"builds/{A}/{B}")
        later.write_text("future timestamp")
        # An edit to a non-max-mtime file need not affect the requested
        # count/bytes/max-mtime summary. The metadata digest must catch it.
        future = later.stat().st_mtime_ns + 10**12
        os.utime(later, ns=(future, future))
        candidate = _build(root, C, D)
        assert _dry(root, apply=True)["totals"]["written"] == 1
        before = rm.read_manifest(run)["legacy_scan"]["tree_stamp"]
        older.write_text(f"builds/{C}/{D}")
        assert _entry(root, candidate)["action"] == "protected"
        assert before["file_count"] == 2

"""HI77: overlay (src/) vs compiled tree (vendor/llama.cpp) drift detection.

src/ is this repo's tracked canonical source; vendor/llama.cpp (gitignored)
is what cmake actually compiles. `apply` mirrors the former onto the latter,
but nothing previously verified they stayed in sync -- an overlay edit could
be committed and pass every test while the compiled tree kept running the
old content. These tests pin the drift-detection check's behavior directly
against a synthetic overlay/checkout pair (monkeypatched paths.SRC_OVERLAY),
independent of this repo's real src/ and vendor/ contents.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bigcherry import paths, source_audit  # noqa: E402


def _make_overlay_and_checkout(tmp_path: Path):
    overlay = tmp_path / "overlay-src"
    checkout = tmp_path / "checkout"
    (overlay / "ggml" / "src" / "ggml-cuda").mkdir(parents=True)
    (checkout / "ggml" / "src" / "ggml-cuda").mkdir(parents=True)
    cuda = checkout / "ggml" / "src" / "ggml-cuda"
    return overlay, checkout, cuda


def _audit_overlay_sync(monkeypatch, overlay: Path, checkout: Path, cuda: Path):
    monkeypatch.setattr(paths, "SRC_OVERLAY", overlay)
    ctx = source_audit.AuditContext(
        root=checkout, cuda=cuda, instances=cuda / "template-instances")
    source_audit.check_overlay_sync(ctx)
    return ctx.results


def test_matching_files_pass(monkeypatch, tmp_path: Path):
    overlay, checkout, cuda = _make_overlay_and_checkout(tmp_path)
    relative = Path("ggml") / "src" / "ggml-cuda" / "hip-autotune-tuner.cu"
    (overlay / relative).write_text("int x = 1;\n", encoding="utf-8")
    (checkout / relative).write_text("int x = 1;\n", encoding="utf-8")

    results = _audit_overlay_sync(monkeypatch, overlay, checkout, cuda)

    assert len(results) == 1
    assert results[0].id == "overlay.vendor_sync"
    assert results[0].ok


def test_drifted_file_fails_and_is_named(monkeypatch, tmp_path: Path):
    overlay, checkout, cuda = _make_overlay_and_checkout(tmp_path)
    relative = Path("ggml") / "src" / "ggml-cuda" / "hip-autotune-tuner.cu"
    (overlay / relative).write_text("int x = 2;  // fixed\n", encoding="utf-8")
    (checkout / relative).write_text("int x = 1;  // stale\n", encoding="utf-8")

    results = _audit_overlay_sync(monkeypatch, overlay, checkout, cuda)

    assert len(results) == 1
    assert not results[0].ok
    assert "hip-autotune-tuner.cu" in results[0].detail
    assert "bigcherry apply" in results[0].detail
    assert results[0].actual == ["ggml/src/ggml-cuda/hip-autotune-tuner.cu"]


def test_overlay_only_file_is_not_drift(monkeypatch, tmp_path: Path):
    # A new or just-edited overlay file that has not been mirrored onto the
    # checkout yet (normal pre-`apply` state) must not be flagged -- only a
    # same-relative-path file present on BOTH sides with different content
    # is a real sync failure.
    overlay, checkout, cuda = _make_overlay_and_checkout(tmp_path)
    relative = Path("ggml") / "src" / "ggml-cuda" / "hip-autotune-tuner.cu"
    (overlay / relative).write_text("int x = 1;\n", encoding="utf-8")
    # checkout side deliberately left without this file.

    results = _audit_overlay_sync(monkeypatch, overlay, checkout, cuda)

    assert len(results) == 1
    assert results[0].ok
    assert results[0].detail.startswith("0 overlay file(s)")


def test_multiple_drifted_files_are_all_named(monkeypatch, tmp_path: Path):
    overlay, checkout, cuda = _make_overlay_and_checkout(tmp_path)
    for name in ("a.cu", "b.cu"):
        relative = Path("ggml") / "src" / "ggml-cuda" / name
        (overlay / relative).write_text("new\n", encoding="utf-8")
        (checkout / relative).write_text("old\n", encoding="utf-8")

    results = _audit_overlay_sync(monkeypatch, overlay, checkout, cuda)

    assert not results[0].ok
    assert "a.cu" in results[0].detail
    assert "b.cu" in results[0].detail
    assert len(results[0].actual) == 2


def test_overlay_sync_check_is_registered_in_full_audit():
    assert source_audit.check_overlay_sync in source_audit.ALL_CHECKS

"""upstream.py -- the remote-facing half of repinning.

Covers the incident from 2026-08-11: a hanging fetch, a fallback that never
wrote a checkout-able ref, and ~2000 stale locks from interrupted attempts.
These tests assert the specific things that made those bugs possible, so a
regression here is the bug coming back, not a coincidence.
"""

from __future__ import annotations

from pathlib import Path


from bigcherry import upstream


class _Recorder:
    """Fake ``_git`` that records calls and returns scripted output."""

    def __init__(self, outputs=None):
        self.calls: list[tuple] = []
        self._outputs = outputs or {}

    def __call__(self, root, *args, timeout=300):
        self.calls.append(args)
        return self._outputs.get(args, "")


# --------------------------------------------------------------- ensure_ref

def test_release_tag_uses_one_scoped_refspec_with_no_tags(monkeypatch, tmp_path):
    """The incident's actual fix. `--no-tags` and an explicit
    refspec::refspec are what keep this to one ref instead of ~2000."""
    recorder = _Recorder()
    monkeypatch.setattr(upstream, "_git", recorder)
    monkeypatch.setattr(upstream, "_has_ref", lambda root, ref: False)

    target = upstream.ensure_ref(tmp_path, "b10362")

    assert target == "b10362"
    assert len(recorder.calls) == 1
    call = recorder.calls[0]
    assert "--no-tags" in call
    assert "refs/tags/b10362:refs/tags/b10362" in call
    # Must never be the shorthand that hung in the incident.
    assert "tag" not in call


def test_non_tag_ref_fetches_plain_and_targets_fetch_head(monkeypatch, tmp_path):
    """A branch name or raw sha cannot get a named local ref from a refspec.
    The incident's other half: falling back to a plain fetch but then
    checking out `ref` (which plain fetch never names) failed with
    'pathspec did not match'. The caller must check out FETCH_HEAD instead."""
    recorder = _Recorder()
    monkeypatch.setattr(upstream, "_git", recorder)
    monkeypatch.setattr(upstream, "_has_ref", lambda root, ref: False)

    target = upstream.ensure_ref(tmp_path, "some-branch")

    assert target == "FETCH_HEAD"
    call = recorder.calls[0]
    assert "--no-tags" in call
    assert "some-branch" in call
    assert not any("refs/tags" in a for a in call), \
        "a non-tag ref must not be fetched via a tags refspec"


def test_already_present_ref_fetches_nothing(monkeypatch, tmp_path):
    recorder = _Recorder()
    monkeypatch.setattr(upstream, "_git", recorder)
    monkeypatch.setattr(upstream, "_has_ref", lambda root, ref: True)

    assert upstream.ensure_ref(tmp_path, "b10257") == "b10257"
    assert recorder.calls == []


def test_deepen_false_omits_depth_flag(monkeypatch, tmp_path):
    recorder = _Recorder()
    monkeypatch.setattr(upstream, "_git", recorder)
    monkeypatch.setattr(upstream, "_has_ref", lambda root, ref: False)

    upstream.ensure_ref(tmp_path, "b10362", deepen=False)

    assert "--depth" not in recorder.calls[0]


# ---------------------------------------------------------- stale locks

def test_clear_stale_locks_removes_only_lock_files(tmp_path):
    git = tmp_path / ".git"
    (git / "refs" / "tags").mkdir(parents=True)
    (git / "refs" / "tags" / "b10362.lock").write_text("", encoding="utf-8")
    (git / "refs" / "tags" / "b10000.lock").write_text("", encoding="utf-8")
    (git / "refs" / "tags" / "b10257").write_text("deadbeef\n", encoding="utf-8")
    (git / "index.lock").write_text("", encoding="utf-8")

    removed = upstream.clear_stale_locks(tmp_path)

    assert len(removed) == 3
    assert not (git / "refs" / "tags" / "b10362.lock").exists()
    assert not (git / "index.lock").exists()
    # A real ref -- untouched. This is the file that would corrupt the
    # checkout if a lock-clearing pass were ever careless about scope.
    assert (git / "refs" / "tags" / "b10257").read_text(encoding="utf-8") == "deadbeef\n"


def test_clear_stale_locks_on_a_clean_tree_removes_nothing(tmp_path):
    (tmp_path / ".git").mkdir()
    assert upstream.clear_stale_locks(tmp_path) == []


def test_clear_stale_locks_reports_paths_relative_to_git_dir(tmp_path):
    git = tmp_path / ".git" / "refs" / "tags"
    git.mkdir(parents=True)
    (git / "b1.lock").write_text("", encoding="utf-8")

    removed = upstream.clear_stale_locks(tmp_path)

    assert removed == [str(Path("refs") / "tags" / "b1.lock")]

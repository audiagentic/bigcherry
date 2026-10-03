"""Talking to the llama.cpp remote. Fetching only what a checkout-able ref
needs, and self-healing the stale-lock damage a killed fetch leaves behind."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

UPSTREAM_URL = "https://github.com/ggml-org/llama.cpp"
_RELEASE_TAG = re.compile(r"^b(\d+)$")
LATEST = "latest"


class UpstreamError(RuntimeError):
    pass


def _git(root: Path | None, *args: str, timeout: int = 300) -> str:
    cmd = ["git"] + (["-C", str(root)] if root else []) + list(args)
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        raise UpstreamError(f"{' '.join(cmd)}: {exc}") from exc
    if done.returncode != 0:
        raise UpstreamError(f"{' '.join(cmd)} failed: {done.stderr.strip() or done.stdout.strip()}")
    return done.stdout


def release_tags(root: Path | None = None, url: str = UPSTREAM_URL) -> list[str]:
    """Every b<n> release tag on the remote, oldest first. Asks the remote,
    not the local clone -- a shallow checkout only knows the handful of tags
    it happened to fetch, and answering locally gives a confidently stale
    number."""
    out = _git(root, "ls-remote", "--tags", url, "refs/tags/b*")
    numbered = []
    for line in out.splitlines():
        _, _, ref = line.partition("refs/tags/")
        match = _RELEASE_TAG.match(ref.strip())
        if match:
            numbered.append((int(match.group(1)), ref.strip()))
    if not numbered:
        raise UpstreamError(f"no b<n> release tags found at {url}")
    return [tag for _, tag in sorted(numbered)]  # sort numerically, NOT lexically -- b9999 < b10362


def latest_release(root: Path | None = None, url: str = UPSTREAM_URL) -> str:
    return release_tags(root, url)[-1]


def resolve_ref(ref: str, root: Path | None = None, url: str = UPSTREAM_URL) -> str:
    return latest_release(root, url) if ref == LATEST else ref


def ensure_ref(root: Path, ref: str, *, deepen: bool = True) -> str:
    """Make `ref` fetchable. Returns what to check out (usually `ref` itself,
    or "FETCH_HEAD" when a local ref cannot be named for it).

    Incident 2026-08-11: `fetch origin tag <ref>` hangs on this remote --
    reproduced directly, 2+ minutes, regardless of settings. Its old fallback
    wrote only FETCH_HEAD, never a local ref, so the checkout that followed
    always failed. Neither passed --no-tags, so whichever ran, git tried to
    update every tag reachable -- ~2000 ref writes for one request -- which is
    what turned interrupted attempts into ~2000 stale .lock files. Fixed: one
    scoped refspec + --no-tags always."""
    if _has_ref(root, ref):
        return ref
    depth = ["--depth", "1"] if deepen else []
    if _RELEASE_TAG.match(ref):
        _git(root, "fetch", "--no-tags", *depth, "origin", f"refs/tags/{ref}:refs/tags/{ref}")
        return ref
    _git(root, "fetch", "--no-tags", *depth, "origin", ref)
    return "FETCH_HEAD"


def _has_ref(root: Path, ref: str) -> bool:
    try:
        _git(root, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", timeout=30)
        return True
    except UpstreamError:
        return False


def clear_stale_locks(root: Path) -> list[str]:
    """Remove .lock files left by a killed git process. Only call when no
    git process for this checkout is believed running (caller's judgement --
    inherently racy to check here)."""
    removed = []
    for lock in sorted((root / ".git").rglob("*.lock")):
        removed.append(str(lock.relative_to(root / ".git")))
        lock.unlink()
    return removed


def is_shallow(root: Path) -> bool:
    try:
        return _git(root, "rev-parse", "--is-shallow-repository", timeout=30).strip() == "true"
    except UpstreamError:
        return False


def sync_mirror_ref(mirror: Path, ref: str, revision: str | None = None) -> str | None:
    """Make ``ref`` resolvable in the campaign build mirror.

    Builds resolve the pin only against refs already present (RE13), so the
    deliberate pin-update step (``pull``/``pin-bump``) must bring the new tag
    into the mirror as well as the vendor checkout. Returns ``None`` when the
    ref resolves (or there is no mirror yet), otherwise a warning message --
    never raises, so a network problem cannot fail an already-verified pull.
    """
    if not (mirror / "HEAD").is_file() and not (mirror / ".git").exists():
        return None
    if _has_ref(mirror, ref):
        return None
    errors: list[str] = []
    try:
        ensure_ref(mirror, ref)
    except UpstreamError as exc:
        errors.append(str(exc))
    if not _has_ref(mirror, ref) and revision and _RELEASE_TAG.match(ref):
        try:
            _git(mirror, "fetch", "--no-tags", "--depth", "1", "origin", revision)
            _git(mirror, "tag", ref, "FETCH_HEAD")
        except UpstreamError as exc:
            errors.append(str(exc))
    if _has_ref(mirror, ref):
        return None
    detail = "; ".join(errors) or "ref still unresolvable after fetch"
    return f"campaign mirror {mirror} still lacks {ref!r}: {detail}"

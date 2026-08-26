"""Anchored patching of upstream-owned files.

A unified diff encodes *where* a change goes as "line 1853, and here are three
lines of context". Upstream adds a function above it and the diff is rubbish.
Anchored edits encode *what* they attach to instead — a regex naming the
construct — so an edit survives everything except a change to the thing it
actually depends on, and when it does break it can say which construct went
missing.

Three properties matter more than cleverness here:

* **Idempotence.** Every edit carries a ``guard`` regex matching its own
  output. Re-running ``apply`` on a patched tree is a no-op, not a double
  insertion.
* **No silent ambiguity.** An anchor that matches twice is a bug in the anchor,
  not a licence to pick the first hit. Edits declare how many matches they
  expect and fail otherwise.
* **No anchoring into comments.** Anchors are matched against a noise-stripped
  copy of the file with comments and string literals blanked out, then applied
  at the same offsets in the real text.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ..core import csource

Mode = Literal["insert_after", "insert_before", "replace", "replace_all"]


class PatchError(RuntimeError):
    """An edit could not be applied. The message names the failing anchor."""


#: PA07 (source/patch identity hardening L1.2): the SHARED semantics every
#: patch's output depends on beyond its own implementation bytes -- anchor
#: matching, noise stripping (core/csource.py), guard handling, replace_all,
#: applies_if, span limits, newline/encoding behavior. A patch's own digest
#: (patch_implementation_digest()) cannot see a change here: apply.py or
#: core/csource.py changing behavior can change output bytes while every
#: patch.py's digest stays identical. Bump ONLY when a change here can
#: change materialised source output -- never for docstring/comment-only
#: edits, never for this constant's own reassignment.
PATCH_APPLICATION_SEMANTICS_VERSION = 1


def resolve_contained_target(root: Path, relative: str) -> Path:
    """Resolve ``relative`` against ``root``, refusing any path or symlink
    that would let the write land outside ``root``.

    Registry-level path validation (patch/registry.py) protects where a
    patch.py *definition* lives; it says nothing about the paths a patch's
    own edits write to. ``root`` here is typically a checkout of untrusted
    upstream source (llama.cpp), which could in principle carry a symlink
    that redirects an innocuous-looking relative path outside the isolated
    worktree -- git_tree_oid()/git_worktree_tree() only ever walk *inside*
    the tree, so such a write would be invisible to source-identity hashing
    while still mutating real filesystem state outside it.
    """
    if not relative or relative.strip() == "":
        raise PatchError("patch target path is empty")
    candidate = Path(relative)
    if candidate.is_absolute():
        raise PatchError(f"patch target path must be relative, got absolute path: {relative!r}")
    if ".." in candidate.parts:
        raise PatchError(f"patch target path must not contain '..' components: {relative!r}")

    resolved_root = root.resolve(strict=True)

    # Walk the parent chain from the root down, rejecting any component that
    # is itself a symlink -- a symlinked *directory* partway down the path
    # is exactly as dangerous as a symlinked leaf file, since everything
    # written "through" it lands wherever the link points.
    current = resolved_root
    for part in candidate.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise PatchError(
                f"patch target path {relative!r} passes through a symlink at "
                f"{current} -- refusing to write through a path component "
                f"that could redirect outside the source root"
            )

    target = resolved_root / candidate
    if target.is_symlink():
        raise PatchError(
            f"patch target {relative!r} is itself a symlink -- refusing to "
            f"write through it since its destination is not guaranteed to "
            f"be contained by the source root"
        )

    resolved_parent = target.parent.resolve() if target.parent.exists() else target.parent
    try:
        resolved_parent.relative_to(resolved_root)
    except ValueError as exc:
        raise PatchError(
            f"patch target {relative!r} resolves to {target}, which is not "
            f"contained by source root {resolved_root}"
        ) from exc

    return target


@dataclass(frozen=True)
class Edit:
    """One anchored modification to a file.

    ``anchor`` locates the attachment point; ``text`` is inserted after it,
    before it, or in place of it depending on ``mode``. ``guard`` recognises
    already-applied output and makes the edit idempotent.
    """

    id: str
    anchor: str
    text: str
    mode: Mode = "insert_after"
    guard: str | None = None
    expect_matches: int = 1
    occurrence: int = 0
    #: Why this anchor was chosen, for the human reading a failure report.
    rationale: str = ""
    #: Regex that must be present in the target for this edit to be relevant.
    #: This is the escape hatch for genuine upstream restructuring: when an
    #: edit cannot be written to span old and new trees, ship both, each
    #: probing for the shape it handles. Absent means "always relevant" --
    #: an edit that simply fails to find its anchor is still an error, and
    #: `applies_if` must never be used to paper over that.
    applies_if: str | None = None
    #: Reject a match spanning more than this many lines. An anchor is meant to
    #: name one construct; one that swallows half the file has almost certainly
    #: gone greedy, and the resulting edit lands nowhere near its intended home.
    #: Raise it deliberately for an edit that really does span a large block.
    max_span_lines: int = 40

    def guard_pattern(self) -> str:
        if self.guard is not None:
            return self.guard
        # Default guard: the first non-blank line of the inserted text. Good
        # enough when that line is a distinctive declaration, which is why
        # every edit below leads with one.
        for line in self.text.splitlines():
            if line.strip():
                return re.escape(line.strip())
        raise PatchError(f"edit {self.id!r} has neither guard nor text")


@dataclass(frozen=True)
class FilePatch:
    """An ordered set of edits against one file, relative to the checkout."""

    path: str
    edits: tuple[Edit, ...]
    #: Human-readable purpose, echoed in reports.
    description: str = ""
    #: Noise-stripping dialect: "c", "cmake" or "none". Inferred from the file
    #: name when left empty. It matters because the dialects disagree about
    #: whether string literals are noise -- in CMake they carry the content an
    #: anchor needs to see.
    language: str = ""

    def dialect(self) -> str:
        return self.language or csource.language_for(self.path)


@dataclass
class EditResult:
    edit_id: str
    status: Literal["applied", "already-applied", "not-applicable", "failed"]
    detail: str = ""


@dataclass
class PatchResult:
    path: str
    results: list[EditResult] = field(default_factory=list)
    changed: bool = False

    @property
    def failed(self) -> list[EditResult]:
        return [r for r in self.results if r.status == "failed"]

    @property
    def ok(self) -> bool:
        return not self.failed


def _find_all_anchors(stripped: str, edit: Edit) -> list[re.Match[str]]:
    """Every match, for `replace_all`.

    Forwarding one new argument through a dispatcher's sixteen cases is a
    genuinely repetitive edit, and writing it as sixteen near-identical Edit
    objects would be worse in every way -- longer, harder to read, and each one
    a separate thing to keep in step. `expect_matches` still has to be stated,
    so upstream adding or removing a case is caught rather than silently
    half-patched.
    """
    pattern = re.compile(edit.anchor, re.MULTILINE)
    matches = list(pattern.finditer(stripped))
    if len(matches) != edit.expect_matches:
        raise PatchError(
            f"anchor for edit {edit.id!r} matched {len(matches)} time(s), "
            f"expected exactly {edit.expect_matches}. "
            f"Anchor: {edit.anchor!r}."
            + (f" It attaches to: {edit.rationale}" if edit.rationale else ""))
    return matches


def _find_anchor(stripped: str, edit: Edit) -> re.Match[str]:
    # MULTILINE but deliberately *not* DOTALL. Under DOTALL a natural-looking
    # anchor such as `^option\(FOO.*$` matches from the first FOO to the end of
    # the file, and the edit silently lands hundreds of lines from where it was
    # meant to. An anchor that genuinely spans lines says so with an explicit
    # \n, which is legible in the anchor itself.
    pattern = re.compile(edit.anchor, re.MULTILINE)
    matches = list(pattern.finditer(stripped))
    if len(matches) != edit.expect_matches:
        raise PatchError(
            f"anchor for edit {edit.id!r} matched {len(matches)} time(s), "
            f"expected {edit.expect_matches}. "
            f"Anchor: {edit.anchor!r}."
            + (f" It attaches to: {edit.rationale}" if edit.rationale else ""))

    match = matches[edit.occurrence]
    span_lines = match.group(0).count("\n") + 1
    if span_lines > edit.max_span_lines:
        raise PatchError(
            f"anchor for edit {edit.id!r} matched {span_lines} lines, more "
            f"than the {edit.max_span_lines}-line limit. It has almost "
            f"certainly gone greedy and would place the edit far from where "
            f"it belongs. Anchor: {edit.anchor!r}.")
    return match


def apply_patch(patch: FilePatch, root: Path, *, dry_run: bool = False,
                texts: dict[str, str] | None = None) -> PatchResult:
    """Apply every edit in ``patch`` to ``root / patch.path``.

    ``texts`` is an in-memory view of files already modified by earlier patches
    in the same run. Patches legitimately depend on each other -- the coverage
    hook attaches to a parameter a forced-variant patch added -- and validating
    each one against the *on-disk* file makes such a patch impossible to place
    during a dry run, however correct it is. Threading the accumulated text
    through means the trial pass sees the tree as it will actually be.
    """
    result = PatchResult(path=patch.path)
    target = resolve_contained_target(root, patch.path)

    if texts is not None and patch.path in texts:
        text = texts[patch.path]
    else:
        if not target.is_file():
            result.results.append(EditResult(
                "<file>", "failed", f"target file does not exist: {patch.path}"))
            return result
        text = target.read_text(encoding="utf-8")
    original = text

    for edit in patch.edits:
        # Guard first: an already-applied edit must not be re-anchored, because
        # its own output may well have displaced or duplicated the anchor.
        if re.search(edit.guard_pattern(), text, re.MULTILINE):
            result.results.append(EditResult(
                edit.id, "already-applied", "guard matched; left unchanged"))
            continue

        stripped = csource.strip_noise(text, patch.dialect())

        if edit.applies_if is not None and not re.search(
                edit.applies_if, stripped, re.MULTILINE):
            result.results.append(EditResult(
                edit.id, "not-applicable",
                f"this release does not have the shape this edit handles "
                f"({edit.applies_if!r})"))
            continue
        if edit.mode == "replace_all":
            try:
                matches = _find_all_anchors(stripped, edit)
            except PatchError as exc:
                result.results.append(EditResult(edit.id, "failed", str(exc)))
                continue
            # Right to left, so each splice leaves earlier offsets valid.
            for match in reversed(matches):
                text = text[:match.start()] + match.expand(edit.text) \
                     + text[match.end():]
            result.results.append(EditResult(
                edit.id, "applied", f"replace_all ({len(matches)} sites)"))
            continue

        try:
            match = _find_anchor(stripped, edit)
        except PatchError as exc:
            result.results.append(EditResult(edit.id, "failed", str(exc)))
            continue

        if edit.mode == "insert_after":
            text = text[:match.end()] + edit.text + text[match.end():]
        elif edit.mode == "insert_before":
            text = text[:match.start()] + edit.text + text[match.start():]
        elif edit.mode == "replace":
            text = text[:match.start()] + edit.text + text[match.end():]
        else:  # pragma: no cover - Mode is a closed Literal
            raise PatchError(f"unknown mode {edit.mode!r} on edit {edit.id!r}")

        result.results.append(EditResult(edit.id, "applied", edit.mode))

    result.changed = text != original
    if texts is not None:
        texts[patch.path] = text
    if result.changed and result.ok and not dry_run:
        # newline="" keeps the LF endings upstream uses, on Windows too.
        target.write_text(text, encoding="utf-8", newline="")
    return result


def apply_all(patches: list[FilePatch], root: Path, *,
              dry_run: bool = False) -> list[PatchResult]:
    """Apply patches, writing nothing unless every one of them can be placed.

    Two passes. The first applies every patch in order to an in-memory copy of
    the tree, proving each edit can be placed *given the ones before it*. Only
    if that succeeds does the second pass touch disk.

    The in-memory copy is what makes patch order meaningful. An earlier design
    validated each patch against the on-disk file, which quietly forbade any
    patch from depending on another's output -- so the coverage hook, which
    attaches to a parameter the forced-variant patch adds, could never pass the
    trial no matter how correct it was.

    A tree left half-patched because edit 7 of 9 lost its anchor is far worse to
    diagnose than one that was never touched, hence writing only at the end.
    """
    simulated: dict[str, str] = {}
    trial = [apply_patch(p, root, dry_run=True, texts=simulated)
             for p in patches]
    if any(not r.ok for r in trial) or dry_run:
        return trial

    # Replay against disk. The results are recomputed rather than reused so a
    # tree changed between the two passes is caught rather than assumed.
    #
    # PA07 (source/patch identity hardening L1.1): the real-write pass below
    # can still fail partway -- a later patch's anchor no longer matching
    # once an earlier patch's write actually landed, or a genuine I/O error.
    # Without a rollback, that leaves the tree (and the content-addressed
    # cache directory workspace.materialize() writes it into) half-patched,
    # which is worse than untouched: it looks like a completed worktree with
    # no valid metadata. Snapshot every file this run *could* touch before
    # writing anything for real, and restore all of them the instant one
    # patch's real pass fails.
    # Preserve patch order (not set iteration order, which CPython does not
    # guarantee to match it) and resolve+snapshot each target ONCE here,
    # keeping the already-validated Path rather than the bare relative
    # string. Adversarial-review follow-up: _restore() previously called
    # resolve_contained_target() again for every file, including ones
    # already restored -- if an EARLIER patch's real write left something
    # that makes a LATER file's path newly unsafe (e.g. it created a
    # symlink another touched path now resolves through), that second
    # resolution can raise and abort the loop with earlier files still
    # unrestored. Resolving once, before any real write happens, removes
    # that window entirely: restoration never re-derives a path that could
    # have been invalidated by the very writes it is undoing.
    touched_paths: list[str] = []
    seen: set[str] = set()
    for p in patches:
        if p.path not in seen:
            seen.add(p.path)
            touched_paths.append(p.path)
    backup: list[tuple[Path, str | None]] = []
    for relative in touched_paths:
        target = resolve_contained_target(root, relative)
        original = target.read_text(encoding="utf-8") if target.is_file() else None
        backup.append((target, original))

    def _restore() -> None:
        # Undo in reverse application order, and never let one file's
        # restore failure stop the others -- a partial rollback that stops
        # at the first error is exactly the partial-tree state this whole
        # mechanism exists to prevent. Collect failures and raise once all
        # restorable files have been attempted.
        errors: list[BaseException] = []
        for target, original in reversed(backup):
            try:
                if original is None:
                    if target.is_file():
                        target.unlink()
                else:
                    # Adversarial-review follow-up: resolving `target` to a
                    # Path once at snapshot time does NOT pin it against a
                    # filesystem race -- Path.write_text() re-resolves the
                    # path (and follows any symlink present) at the moment
                    # it actually opens the file, which happens here, well
                    # after snapshotting. If something (an earlier patch's
                    # own write, or an unrelated concurrent mutation)
                    # replaced this exact path with a symlink in between,
                    # a plain write_text() would follow it and write the
                    # restored contents somewhere outside root. Open with
                    # O_NOFOLLOW so the OS itself refuses to open through a
                    # symlink (ELOOP) instead of silently following one --
                    # this collapses the check-then-write gap into a single
                    # atomic syscall rather than trusting a is_symlink()
                    # check a moment before the write. O_NOFOLLOW does not
                    # exist on Windows; the remaining risk there is
                    # unchanged from before this fix (Windows symlink
                    # creation itself requires elevated privilege, unlike
                    # POSIX, so this is a materially smaller residual gap).
                    nofollow = getattr(os, "O_NOFOLLOW", 0)
                    fd = os.open(
                        str(target),
                        os.O_WRONLY | os.O_CREAT | os.O_TRUNC | nofollow,
                        0o644,
                    )
                    with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                        handle.write(original)
            except OSError as exc:
                errors.append(exc)
        if errors:
            raise PatchError(
                f"rollback failed to restore {len(errors)} file(s) after a "
                f"partial apply_all() failure: {errors!r}"
            )

    try:
        results = [apply_patch(p, root) for p in patches]
    except Exception:
        _restore()
        raise
    if any(not r.ok for r in results):
        _restore()
    return results


def format_results(results: list[PatchResult]) -> str:
    lines: list[str] = []
    for result in results:
        counts = {status: sum(1 for r in result.results if r.status == status)
                  for status in ("applied", "already-applied", "not-applicable")}
        state = "FAIL" if result.failed else "ok  "
        summary = f"{counts['applied']} applied, {counts['already-applied']} already applied"
        if counts["not-applicable"]:
            summary += f", {counts['not-applicable']} n/a"
        lines.append(f"  [{state}] {result.path} ({summary})")
        for edit in result.failed:
            lines.append(f"           {edit.edit_id}: {edit.detail}")
    return "\n".join(lines)

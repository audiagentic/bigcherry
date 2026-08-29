"""Revision-specific patch compatibility probing and known-good quarantine
(PA16).

The pin-consistency guard (``release/pin_status.py``, RE48) answers "is the
vendor checkout at the revision the pin declares?" -- a pure revision-identity
question. It says nothing about whether the anchored patches in ``patches/``
still find their anchors in that revision's source. This module answers that
second, orthogonal question: given the CURRENT vendor revision, which patches
still apply cleanly, which are legitimately not-applicable-by-design, and
which need a human to reconcile a moved/renamed/restructured anchor.

The probe runs in an isolated detached worktree (``source/workspace.py``'s
existing machinery) and never mutates the real vendor checkout. It is
observational: running it does not advance any release stage and does not
rewrite a patch's catalog STATE -- a patch failing against ONE upstream
revision is revision-specific evidence, not a global lifecycle judgment
(that stays in each package's ``patch.toml`` lifecycle metadata, set by a human).

A report this module writes may later be handed back to ``apply
--rebase-report PATH --known-good`` to apply exactly the dependency-closed
subset that reproved clean -- but only while every bound identity in the
report (upstream revision, BigCherry revision, patch implementation digests,
overlay digest, the patch-application semantics version, and the report
schema itself) still matches the live tree exactly. Any drift is a stale
report and apply fails closed, same as it always has for a plain ``apply``.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import apply as patcher
from . import patchset
from . import registry as patch_registry
from .. import pin_transition
from ..core import paths
from .apply import PATCH_APPLICATION_SEMANTICS_VERSION
from .apply import PatchError
from ..release import records as releases
from ..source.workspace import UpstreamRepository, WorkspaceError

REPORT_SCHEMA_VERSION = 1

# --- patch-level status -------------------------------------------------
STATUS_CLEAN = "CLEAN"
STATUS_CLEAN_NOOP = "CLEAN_NOOP"
STATUS_NOT_APPLICABLE = "NOT_APPLICABLE_BY_DESIGN"
STATUS_FAILED = "FAILED_NEEDS_RECONCILIATION"
STATUS_BLOCKED = "BLOCKED_BY_DEPENDENCY"
STATUS_QUARANTINED = "QUARANTINED"

_FAILURE_STATUSES = (STATUS_FAILED, STATUS_BLOCKED, STATUS_QUARANTINED)

# --- edit-level status ----------------------------------------------------
EDIT_APPLIED = "applied-clean"
EDIT_ALREADY_APPLIED = "already-applied"
EDIT_NOT_APPLICABLE = "not-applicable-by-design"
EDIT_FAILED = "failed"

# --- structured reason codes ----------------------------------------------
REASON_TARGET_MISSING = "target-file-missing"
REASON_ANCHOR_NO_MATCH = "anchor-no-match"
REASON_ANCHOR_MATCH_COUNT = "anchor-match-count"
REASON_ANCHOR_SPAN_TOO_WIDE = "anchor-span-too-wide"
REASON_INVALID_ANCHOR_REGEX = "invalid-anchor-regex"
REASON_UNSAFE_TARGET = "unsafe-target"
REASON_OTHER = "other-patch-error"

_MATCH_COUNT_RE = re.compile(r"matched (\d+) time\(s\), expected(?: exactly)? (\d+)")
_SPAN_RE = re.compile(r"matched (\d+) lines, more than the (\d+)-line limit")


class RebaseCheckError(RuntimeError):
    """The probe itself could not run (bad selection, worktree failure, ...).

    Distinct from a patch failing its own anchor -- that is reported data,
    not a raised error.
    """


class StaleRebaseReportError(RebaseCheckError):
    """A ``--rebase-report`` no longer matches the live tree it would be
    applied against. Every bound identity below must match exactly; this is
    intentionally strict since a report can silently outlive the state it
    describes (a new commit landed, a patch was edited, the overlay
    changed) and applying a stale known-good set would apply the WRONG
    subset with no anchor re-validation at all."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RebaseCheckError(
            f"git {' '.join(args)} in {root} failed: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _git_ok(root: Path, *args: str) -> str | None:
    try:
        return _git(root, *args)
    except RebaseCheckError:
        return None


def overlay_digest(overlay_root: Path | None = None) -> str:
    """Content digest over every file under ``src/``, independent of
    filesystem iteration order. Part of a rebase report's bound identity:
    an overlay edit changes what an anchored patch sees at apply time
    exactly as much as a patch edit does.

    Hashes ``read_text()`` (universal-newline-translated) content, NOT raw
    bytes -- ``_copy_overlay()``'s real write path is
    ``source.read_text(encoding="utf-8")`` then
    ``target.write_text(text, newline="")``, which normalizes CRLF to LF
    before it ever reaches a probed or patched file. Hashing raw bytes here
    would make this function (and a stale-report check built on it) see a
    difference between two overlay files that materialize identically, or
    miss one that doesn't -- the digest must describe what actually gets
    applied, not what happens to be on disk."""
    overlay_root = overlay_root or paths.SRC_OVERLAY
    texts: dict[str, str] = {}
    if overlay_root.is_dir():
        for source in sorted(overlay_root.rglob("*")):
            if not source.is_file():
                continue
            relative = str(source.relative_to(overlay_root)).replace("\\", "/")
            texts[relative] = source.read_text(encoding="utf-8")
    return _digest_from_texts(texts)


def _overlay_texts(overlay_root: Path | None = None) -> dict[str, str]:
    """The overlay's post-install content, keyed by repo-relative path --
    the same seed ``_apply_selection``'s dry-run path feeds ``apply_all``
    (PA16 follow-up to the overlay/dry-run parity fix).

    Deliberately ``read_text()`` (universal-newline translation), matching
    exactly what ``_copy_overlay()`` actually writes -- a probe reading raw
    bytes here would see CRLF an anchor could fail on, while the real write
    normalizes it to LF and the anchor would have matched. See
    :func:`overlay_digest`'s docstring for the same reasoning applied to
    the staleness digest."""
    overlay_root = overlay_root or paths.SRC_OVERLAY
    texts: dict[str, str] = {}
    if overlay_root.is_dir():
        for source in sorted(overlay_root.rglob("*")):
            if not source.is_file():
                continue
            relative = str(source.relative_to(overlay_root)).replace("\\", "/")
            texts[relative] = source.read_text(encoding="utf-8")
    return texts


def _digest_from_texts(texts: dict[str, str]) -> str:
    """Same algorithm as :func:`overlay_digest`, but over an in-memory
    snapshot already taken -- so a report's ``overlay_digest`` describes
    EXACTLY the bytes every round of the probe actually saw, not a second,
    separately-timed disk read that a concurrent edit on this shared,
    multi-agent working tree could see differently (adversarial-review
    follow-up: report-generation snapshot stability)."""
    hasher = hashlib.sha256()
    for relative in sorted(texts):
        hasher.update(relative.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(texts[relative].encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def _load_module_patches(module: "patchset.PatchModule") -> list["patcher.FilePatch"]:
    """Load one patch's implementation bound EXACTLY to the digest already
    recorded for it (``module.content_hash``) via
    ``registry.load_implementation``'s own ``expected_digest`` check.

    ``patchset.load_resolved`` re-reads the registry and loads by ID with no
    digest binding at all -- fine for a normal apply where the caller just
    resolved the selection a moment ago, but this module runs across a
    probe that can take real time on a shared, multi-agent working tree,
    and the known-good apply path binds to a REPORT's digest, possibly
    written much earlier. Going through the digest-checked loader directly
    closes that TOCTOU window instead of silently executing whatever bytes
    happen to be on disk at load time."""
    root = module.catalog_root or paths.PATCHES
    registry = patch_registry.load_registry(root)
    descriptor = registry.get(module.patch_id)
    return patch_registry.load_implementation(
        descriptor, root=root, expected_digest=module.content_hash,
    )


def _selection_patch_ids(
    *, recipe_name: str | None, all_patches: bool,
) -> tuple[str, ...]:
    if bool(recipe_name) == bool(all_patches):
        raise RebaseCheckError(
            "patch-rebase-check: pass exactly one of --recipe NAME or --all"
        )
    modules = patchset.catalog()
    if all_patches:
        return tuple(m.patch_id for m in modules if m.state != "rejected")
    from .. import recipes as recipes_module  # noqa: PLC0415 (leaf import, avoids a cycle)

    try:
        recipe = recipes_module.get(recipe_name)
    except recipes_module.RecipeError as exc:
        raise RebaseCheckError(f"patch-rebase-check: {exc}") from exc
    ids = tuple(
        m.patch_id for m in modules
        if (recipe.groups is None or m.group in recipe.groups)
        and (recipe.states is None or m.state in recipe.states)
        and m.state != "rejected"
    )
    return ids


def _partition_conflict_free(
    ids: tuple[str, ...], modules: dict[str, "patchset.PatchModule"],
) -> list[tuple[str, ...]]:
    """Split ``ids`` into groups with no `conflicts` relationship inside any
    one group. `--all` probing must never hand two mutually-exclusive
    alternatives (e.g. two candidate patches for the same problem) to
    ``resolve_exact``/``quarantine_fixed_point`` together -- that treats the
    input as ONE proposed build composition (correct for a real recipe) and
    ``quarantine_fixed_point`` applies every member to the SAME worktree, so
    genuinely conflicting patches would clash on real anchors, not just fail
    a policy check. Found live on pin-bump's first real bump: `--all`
    crashed uncaught the moment the registry contained two intentional
    alternative candidates (1205/1207) at once, which it always will."""
    groups: list[list[str]] = []
    for pid in ids:
        conflicts = set(modules[pid].conflicts)
        for group in groups:
            if conflicts.isdisjoint(group) and not any(
                pid in modules[member].conflicts for member in group
            ):
                group.append(pid)
                break
        else:
            groups.append([pid])
    return [tuple(group) for group in groups]


def resolve_selection(
    *, recipe_name: str | None, all_patches: bool,
) -> patchset.ResolvedPatchSet:
    """The exact, dependency-complete, topologically-ordered selection to
    probe. Deliberately goes through ``resolve_exact`` (not the flattening
    ``load_patches``) so logical patch identity survives into the report and
    a dependency-incomplete selection fails closed here, before probing,
    rather than silently probing an unsound subset."""
    ids = _selection_patch_ids(recipe_name=recipe_name, all_patches=all_patches)
    try:
        return patchset.resolve_exact(ids, allow_rejected=False)
    except ValueError as exc:
        raise RebaseCheckError(f"patch-rebase-check: {exc}") from exc


# --------------------------------------------------------------- probing


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _context_snippet(text: str, offset: int, *, context_lines: int) -> str:
    lines = text.splitlines()
    line_no = _line_number(text, offset)
    start = max(0, line_no - 1 - context_lines)
    end = min(len(lines), line_no + context_lines)
    numbered = [f"{i + 1:>6}: {lines[i]}" for i in range(start, end)]
    return "\n".join(numbered)


def _diff_context(previous_revision: str | None, revision: str, root: Path, relative: str) -> str | None:
    """Bounded ``git diff`` of one file across the pin bump, when the
    previous revision is known. Purely informational -- never authoritative,
    just what a human reconciling a moved anchor would run by hand anyway."""
    if not previous_revision:
        return None
    diff = _git_ok(
        root, "diff", f"{previous_revision}..{revision}", "--", relative,
    )
    if not diff:
        return None
    lines = diff.splitlines()
    if len(lines) > 200:
        lines = lines[:200] + ["... (truncated)"]
    return "\n".join(lines)


def _reason_for(detail: str) -> tuple[str, int | None]:
    """Map ``apply.py``'s prose failure detail to a structured reason code
    plus the actual match count when the message names one. The prose is a
    small, stable set of templates owned by this same codebase (apply.py),
    not third-party text, so pattern-matching it here is safe -- and far
    less risky than re-deriving anchor-matching logic a second time."""
    if detail.startswith("target file does not exist"):
        return REASON_TARGET_MISSING, None
    span = _SPAN_RE.search(detail)
    if span:
        return REASON_ANCHOR_SPAN_TOO_WIDE, int(span.group(1))
    count = _MATCH_COUNT_RE.search(detail)
    if count:
        actual = int(count.group(1))
        if actual == 0:
            return REASON_ANCHOR_NO_MATCH, 0
        return REASON_ANCHOR_MATCH_COUNT, actual
    return REASON_OTHER, None


@dataclass
class EditProbe:
    edit_id: str
    status: str
    reason_code: str | None
    anchor: str
    expect_matches: int
    actual_matches: int | None
    applies_if: str | None
    rationale: str
    context: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "edit_id": self.edit_id,
            "status": self.status,
            "reason_code": self.reason_code,
            "anchor": self.anchor,
            "expect_matches": self.expect_matches,
            "actual_matches": self.actual_matches,
            "applies_if": self.applies_if,
            "rationale": self.rationale,
            "context": self.context,
        }


@dataclass
class FileProbe:
    path: str
    edits: list[EditProbe] = field(default_factory=list)

    def status(self) -> str:
        statuses = {e.status for e in self.edits}
        if EDIT_FAILED in statuses:
            return STATUS_FAILED
        if statuses <= {EDIT_NOT_APPLICABLE}:
            return STATUS_NOT_APPLICABLE
        if statuses <= {EDIT_ALREADY_APPLIED}:
            return STATUS_CLEAN_NOOP
        return STATUS_CLEAN

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "edits": [e.to_dict() for e in self.edits]}


def _probe_file_patch(
    patch: "patcher.FilePatch",
    root: Path,
    texts: dict[str, str],
    *,
    context_lines: int,
    previous_revision: str | None,
    revision: str,
) -> FileProbe:
    probe = FileProbe(path=patch.path)
    try:
        target = patcher.resolve_contained_target(root, patch.path)
    except PatchError as exc:
        probe.edits.append(EditProbe(
            edit_id="<file>", status=EDIT_FAILED, reason_code=REASON_UNSAFE_TARGET,
            anchor="", expect_matches=0, actual_matches=None, applies_if=None,
            rationale=str(exc),
        ))
        return probe

    if patch.path in texts:
        text = texts[patch.path]
    elif target.is_file():
        text = target.read_text(encoding="utf-8")
        texts[patch.path] = text
    else:
        for edit in patch.edits:
            probe.edits.append(EditProbe(
                edit_id=edit.id, status=EDIT_FAILED, reason_code=REASON_TARGET_MISSING,
                anchor=edit.anchor, expect_matches=edit.expect_matches,
                actual_matches=None, applies_if=edit.applies_if,
                rationale=edit.rationale,
            ))
        return probe

    for edit in patch.edits:
        try:
            guard_hit = bool(re.search(edit.guard_pattern(), text, re.MULTILINE))
        except re.error as exc:
            probe.edits.append(EditProbe(
                edit_id=edit.id, status=EDIT_FAILED, reason_code=REASON_INVALID_ANCHOR_REGEX,
                anchor=edit.anchor, expect_matches=edit.expect_matches,
                actual_matches=None, applies_if=edit.applies_if,
                rationale=f"invalid guard regex: {exc}",
            ))
            continue
        if guard_hit:
            probe.edits.append(EditProbe(
                edit_id=edit.id, status=EDIT_ALREADY_APPLIED, reason_code=None,
                anchor=edit.anchor, expect_matches=edit.expect_matches,
                actual_matches=None, applies_if=edit.applies_if,
                rationale=edit.rationale,
            ))
            continue

        try:
            single = patcher.FilePatch(path=patch.path, edits=(edit,), language=patch.language)
            result = patcher.apply_patch(single, root, dry_run=True, texts=dict(texts))
        except re.error as exc:
            probe.edits.append(EditProbe(
                edit_id=edit.id, status=EDIT_FAILED, reason_code=REASON_INVALID_ANCHOR_REGEX,
                anchor=edit.anchor, expect_matches=edit.expect_matches,
                actual_matches=None, applies_if=edit.applies_if,
                rationale=f"invalid anchor regex: {exc}",
            ))
            continue

        edit_result = result.results[0]
        if edit_result.status == "not-applicable":
            probe.edits.append(EditProbe(
                edit_id=edit.id, status=EDIT_NOT_APPLICABLE, reason_code=None,
                anchor=edit.anchor, expect_matches=edit.expect_matches,
                actual_matches=None, applies_if=edit.applies_if,
                rationale=edit.rationale,
            ))
            continue
        if edit_result.status == "already-applied":
            probe.edits.append(EditProbe(
                edit_id=edit.id, status=EDIT_ALREADY_APPLIED, reason_code=None,
                anchor=edit.anchor, expect_matches=edit.expect_matches,
                actual_matches=None, applies_if=edit.applies_if,
                rationale=edit.rationale,
            ))
            continue
        if edit_result.status == "failed":
            reason_code, actual = _reason_for(edit_result.detail)
            context = _diff_context(previous_revision, revision, root, patch.path)
            if context is None:
                # Fall back to a bounded snippet around the FIRST line of the
                # target file's noise-stripped text -- not authoritative,
                # just orientation for a human who has no old-pin diff to go on.
                context = _context_snippet(text, 0, context_lines=context_lines)
            probe.edits.append(EditProbe(
                edit_id=edit.id, status=EDIT_FAILED, reason_code=reason_code,
                anchor=edit.anchor, expect_matches=edit.expect_matches,
                actual_matches=actual, applies_if=edit.applies_if,
                rationale=edit.rationale, context=context,
            ))
            continue
        # applied
        probe.edits.append(EditProbe(
            edit_id=edit.id, status=EDIT_APPLIED, reason_code=None,
            anchor=edit.anchor, expect_matches=edit.expect_matches,
            actual_matches=None, applies_if=edit.applies_if,
            rationale=edit.rationale,
        ))
        # Commit this edit's effect into the shared text so later edits in the
        # same file (and later patches) see it, matching apply_all()'s own
        # threaded-``texts`` semantics.
        merged = patcher.apply_patch(
            patcher.FilePatch(path=patch.path, edits=(edit,), language=patch.language),
            root, dry_run=True, texts=texts,
        )
        if not merged.ok:  # pragma: no cover - contradicts the single-edit probe above
            raise RebaseCheckError(
                f"probe/commit divergence on {patch.path}::{edit.id} -- "
                "this indicates a non-deterministic anchor or a bug in the probe"
            )

    return probe


@dataclass
class PatchProbe:
    patch_id: str
    implementation_digest: str
    state: str
    requires: tuple[str, ...]
    conflicts: tuple[str, ...]
    files: list[FileProbe] = field(default_factory=list)
    status: str = STATUS_CLEAN

    def to_dict(self) -> dict[str, Any]:
        return {
            "patch_id": self.patch_id,
            "implementation_digest": self.implementation_digest,
            "state": self.state,
            "requires": list(self.requires),
            "conflicts": list(self.conflicts),
            "status": self.status,
            "files": [f.to_dict() for f in self.files],
        }


def _classify_files(files: list[FileProbe]) -> str:
    statuses = {f.status() for f in files}
    if STATUS_FAILED in statuses:
        return STATUS_FAILED
    if statuses <= {STATUS_NOT_APPLICABLE}:
        return STATUS_NOT_APPLICABLE
    if statuses <= {STATUS_CLEAN_NOOP}:
        return STATUS_CLEAN_NOOP
    return STATUS_CLEAN


def probe_patch(
    module: patchset.PatchModule,
    root: Path,
    texts: dict[str, str],
    *,
    context_lines: int,
    previous_revision: str | None,
    revision: str,
) -> PatchProbe:
    file_patches = _load_module_patches(module)
    files = [
        _probe_file_patch(
            fp, root, texts,
            context_lines=context_lines,
            previous_revision=previous_revision, revision=revision,
        )
        for fp in file_patches
    ]
    probe = PatchProbe(
        patch_id=module.patch_id,
        implementation_digest=module.content_hash,
        state=module.state,
        requires=module.requires,
        conflicts=module.conflicts,
        files=files,
    )
    probe.status = _classify_files(files) if files else STATUS_CLEAN_NOOP
    return probe


# ------------------------------------------------------- quarantine fixed point


def quarantine_fixed_point(
    selection: patchset.ResolvedPatchSet,
    root: Path,
    *,
    context_lines: int,
    previous_revision: str | None,
    revision: str,
    overlay_texts: dict[str, str],
) -> tuple[dict[str, PatchProbe], tuple[str, ...]]:
    """Compute the stable, dependency-closed known-good subset.

    Returns (probes-by-id keyed with FINAL status, known_good_ids in
    selection order). A patch that fails on round 0 keeps
    FAILED_NEEDS_RECONCILIATION; one that only fails after an earlier
    patch's contribution is removed from the shared text (i.e. it depended,
    undeclared, on that patch's edit) is reported QUARANTINED so a human can
    tell the two apart -- the second case usually means a missing REQUIRES,
    not a broken anchor.
    """
    by_id = {m.patch_id: m for m in selection.modules}
    ordered_ids = [m.patch_id for m in selection.modules]
    quarantined: set[str] = set()
    blocked: set[str] = set()
    direct_failures: set[str] | None = None
    final_probes: dict[str, PatchProbe] = {}

    while True:
        texts = dict(overlay_texts)
        round_probes: dict[str, PatchProbe] = {}
        round_failed: set[str] = set()
        for patch_id in ordered_ids:
            if patch_id in quarantined or patch_id in blocked:
                continue
            probe = probe_patch(
                by_id[patch_id], root, texts,
                context_lines=context_lines,
                previous_revision=previous_revision, revision=revision,
            )
            round_probes[patch_id] = probe
            if probe.status == STATUS_FAILED:
                round_failed.add(patch_id)

        if direct_failures is None:
            direct_failures = set(round_failed)

        newly_failed = round_failed - quarantined
        if not newly_failed:
            final_probes.update(round_probes)
            break

        for patch_id in newly_failed:
            probe = round_probes[patch_id]
            if patch_id not in direct_failures:
                probe.status = STATUS_QUARANTINED
            final_probes[patch_id] = probe
        quarantined |= newly_failed

        newly_blocked = _propagate_dependency_blocks(ordered_ids, by_id, quarantined | blocked)
        newly_blocked -= quarantined | blocked
        for patch_id in newly_blocked:
            final_probes[patch_id] = PatchProbe(
                patch_id=patch_id,
                implementation_digest=by_id[patch_id].content_hash,
                state=by_id[patch_id].state,
                requires=by_id[patch_id].requires,
                conflicts=by_id[patch_id].conflicts,
                files=[],
                status=STATUS_BLOCKED,
            )
        blocked |= newly_blocked

    known_good = tuple(
        pid for pid in ordered_ids if pid not in quarantined and pid not in blocked
    )
    return final_probes, known_good


def _propagate_dependency_blocks(
    ordered_ids: list[str],
    by_id: dict[str, patchset.PatchModule],
    unavailable: set[str],
) -> set[str]:
    """Fixed point over ``requires``: any patch whose REQUIRES intersects an
    unavailable (quarantined or already-blocked) set is itself unavailable,
    and this can cascade (A requires B requires C)."""
    blocked = set()
    changed = True
    while changed:
        changed = False
        for patch_id in ordered_ids:
            if patch_id in unavailable or patch_id in blocked:
                continue
            if set(by_id[patch_id].requires) & (unavailable | blocked):
                blocked.add(patch_id)
                changed = True
    return blocked


# --------------------------------------------------------------- top level


def _previous_upstream_revision(revision: str) -> str | None:
    try:
        marker = pin_transition.load()
    except pin_transition.MarkerError:
        return None
    if marker is not None and marker.to_sha == revision:
        return marker.from_sha
    return None


def _bigcherry_revision() -> str:
    return _git(paths.REPO_ROOT, "rev-parse", "HEAD")


def _probe_group(
    group_ids: tuple[str, ...], *, context_ids: frozenset[str],
    repository: "UpstreamRepository", revision: str, context_lines: int,
    overlay_snapshot: dict[str, str],
) -> tuple[list["patchset.PatchModule"], dict[str, PatchProbe], tuple[str, ...]]:
    """One isolated-worktree probe pass for one conflict-free group. Returns
    (this group's modules in resolved order, its probes, its known_good ids)."""
    try:
        selection = patchset.resolve_exact(
            group_ids, allow_rejected=False, context_ids=context_ids,
        )
    except ValueError as exc:
        raise RebaseCheckError(f"patch-rebase-check: {exc}") from exc
    with tempfile.TemporaryDirectory(prefix="bigcherry-rebase-check-") as tmp:
        worktree = Path(tmp) / "worktree"
        try:
            repository.add_detached_worktree(revision, worktree)
            probes, known_good = quarantine_fixed_point(
                selection, worktree,
                context_lines=context_lines,
                previous_revision=_previous_upstream_revision(revision),
                revision=revision,
                overlay_texts=overlay_snapshot,
            )
        except WorkspaceError as exc:
            raise RebaseCheckError(f"isolated worktree probe failed: {exc}") from exc
        finally:
            try:
                repository.remove_worktree(worktree)
            except WorkspaceError:
                pass
    return list(selection.modules), probes, known_good


def run_rebase_check(
    root: Path,
    *,
    recipe_name: str | None = None,
    all_patches: bool = False,
    context_lines: int = 3,
) -> dict[str, Any]:
    """The full PA16 probe: an isolated detached worktree at the CURRENT
    vendor revision, every selected patch probed against it, quarantined to
    a stable known-good fixed point. Returns the JSON-serializable report;
    never mutates ``root``.

    ``all_patches=True`` probes every non-rejected registry patch, but NOT
    as one joint composition: mutually-exclusive alternative candidates
    (declared via `conflicts`) are split into separate conflict-free groups,
    each probed in its own isolated worktree, then merged. A patch's
    `requires` on a patch in another group is satisfied via `context_ids`
    (identity-only, never re-probed as part of the wrong group)."""
    repository = UpstreamRepository(root)
    revision = _git(root, "rev-parse", "HEAD")
    # Snapshot the overlay ONCE, up front: every probe round and the report's
    # own overlay_digest are computed from these exact same in-memory bytes,
    # rather than each re-reading src/ at its own moment -- a concurrent
    # overlay edit on this shared, multi-agent working tree could otherwise
    # make different rounds see different input and then publish a digest
    # describing only whatever state disk happened to be in last.
    overlay_snapshot = _overlay_texts()

    all_modules: list["patchset.PatchModule"] = []
    probes: dict[str, PatchProbe] = {}
    known_good: tuple[str, ...] = ()

    if all_patches:
        ids = _selection_patch_ids(recipe_name=None, all_patches=True)
        modules_by_id = {m.patch_id: m for m in patchset.catalog()}
        for group_ids in _partition_conflict_free(ids, modules_by_id):
            context = frozenset(
                requirement
                for pid in group_ids
                for requirement in modules_by_id[pid].requires
                if requirement not in group_ids
            )
            group_modules, group_probes, group_known_good = _probe_group(
                group_ids, context_ids=context, repository=repository,
                revision=revision, context_lines=context_lines,
                overlay_snapshot=overlay_snapshot,
            )
            all_modules.extend(group_modules)
            probes.update(group_probes)
            known_good += group_known_good
    else:
        selection = resolve_selection(recipe_name=recipe_name, all_patches=False)
        all_modules, group_probes, known_good = _probe_group(
            tuple(m.patch_id for m in selection.modules), context_ids=frozenset(),
            repository=repository, revision=revision, context_lines=context_lines,
            overlay_snapshot=overlay_snapshot,
        )
        probes.update(group_probes)

    ordered = [probes[m.patch_id] for m in all_modules if m.patch_id in probes]
    summary = {
        "total": len(ordered),
        "clean": sum(1 for p in ordered if p.status in (STATUS_CLEAN, STATUS_CLEAN_NOOP)),
        "not_applicable": sum(1 for p in ordered if p.status == STATUS_NOT_APPLICABLE),
        "failed": sum(1 for p in ordered if p.status == STATUS_FAILED),
        "blocked_by_dependency": sum(1 for p in ordered if p.status == STATUS_BLOCKED),
        "quarantined": sum(1 for p in ordered if p.status == STATUS_QUARANTINED),
    }
    summary["reconciliation_required"] = bool(
        summary["failed"] or summary["blocked_by_dependency"] or summary["quarantined"]
    )

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "upstream_revision": revision,
        "previous_upstream_revision": _previous_upstream_revision(revision),
        "bigcherry_revision": _bigcherry_revision(),
        "patch_application_semantics_version": PATCH_APPLICATION_SEMANTICS_VERSION,
        "overlay_digest": _digest_from_texts(overlay_snapshot),
        "selection": {
            "patch_ids": [m.patch_id for m in all_modules],
            # Adversarial-review follow-up: binds the report to the SELECTOR
            # that produced patch_ids, not just the resulting id list --
            # otherwise a concurrent, uncommitted edit to config/recipes.toml
            # that widens a recipe's groups/states (e.g. [A,B] -> [A,B,C])
            # leaves every other bound identity unchanged (no patch bytes,
            # overlay, or revision moved) and a stale report naming only
            # [A,B] would still pass every other freshness check.
            "recipe": recipe_name,
            "all_patches": bool(all_patches),
        },
        "known_good_patch_ids": list(known_good),
        "summary": summary,
        "patches": [probes[m.patch_id].to_dict() for m in all_modules if m.patch_id in probes],
    }


def render_report(report: dict[str, Any]) -> str:
    """Human-readable table, derived from the exact same report dict the
    JSON output serializes -- one model, two renderings."""
    lines: list[str] = []
    lines.append(f"upstream revision: {report['upstream_revision']}")
    if report.get("previous_upstream_revision"):
        lines.append(f"  previous:        {report['previous_upstream_revision']}")
    lines.append("")
    lines.append(f"{'PATCH':<45} STATUS")
    for patch in report["patches"]:
        lines.append(f"{patch['patch_id']:<45} {patch['status']}")
        if patch["status"] == STATUS_BLOCKED:
            blockers = [r for r in patch["requires"]]
            lines.append(f"  BLOCKED (requires {', '.join(blockers)})")
            continue
        for file_probe in patch["files"]:
            for edit in file_probe["edits"]:
                if edit["status"] != EDIT_FAILED:
                    continue
                lines.append(f"  {file_probe['path']} :: edit {edit['edit_id']}")
                detail = edit["reason_code"] or "failed"
                if edit["actual_matches"] is not None:
                    detail += f": expected {edit['expect_matches']}, found {edit['actual_matches']}"
                lines.append(f"    {detail}")
                if edit["rationale"]:
                    lines.append(f"    rationale: {edit['rationale']}")
    summary = report["summary"]
    lines.append("")
    lines.append(
        f"summary: {summary['total']} total, {summary['clean']} clean, "
        f"{summary['not_applicable']} not-applicable, {summary['failed']} failed, "
        f"{summary['blocked_by_dependency']} blocked, {summary['quarantined']} quarantined"
    )
    if summary["reconciliation_required"]:
        lines.append("RECONCILIATION REQUIRED")
    return "\n".join(lines)


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Atomic write (temp file + rename) -- a report is evidence a later
    ``apply --known-good`` trusts; a torn write must never look valid."""
    import os
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    json.loads(encoded)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_report(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RebaseCheckError(f"unreadable rebase report: {path}: {exc}") from exc


# ---------------------------------------------------- known-good apply


def _require_fresh(
    report: dict[str, Any], root: Path, *,
    overlay_snapshot_digest: str | None = None,
) -> tuple[str, ...]:
    """Every bound identity must match the LIVE tree exactly. Fails closed
    (StaleRebaseReportError) on the first mismatch found, naming it.

    ``overlay_snapshot_digest``: when the caller has already taken its own
    overlay snapshot (as ``apply_known_good`` does, so it can write from
    those exact bytes afterward), pass its digest here instead of letting
    this function call :func:`overlay_digest` a second time -- two separate
    disk reads of ``src/`` is exactly the TOCTOU window a snapshot-once
    caller is trying to close. Defaults to a live read for callers (and
    tests) that only want the freshness check itself."""
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise StaleRebaseReportError(
            f"report schema_version {report.get('schema_version')!r} != "
            f"current {REPORT_SCHEMA_VERSION!r}"
        )
    if report.get("patch_application_semantics_version") != PATCH_APPLICATION_SEMANTICS_VERSION:
        raise StaleRebaseReportError(
            "report's patch_application_semantics_version no longer matches "
            "the live apply.py -- re-run patch-rebase-check"
        )
    current_revision = _git(root, "rev-parse", "HEAD")
    if report.get("upstream_revision") != current_revision:
        raise StaleRebaseReportError(
            f"report upstream_revision {report.get('upstream_revision')!r} != "
            f"live vendor HEAD {current_revision!r}"
        )
    current_bigcherry = _bigcherry_revision()
    if report.get("bigcherry_revision") != current_bigcherry:
        raise StaleRebaseReportError(
            f"report bigcherry_revision {report.get('bigcherry_revision')!r} != "
            f"live HEAD {current_bigcherry!r}"
        )
    live_overlay_digest = (
        overlay_snapshot_digest if overlay_snapshot_digest is not None else overlay_digest()
    )
    if report.get("overlay_digest") != live_overlay_digest:
        raise StaleRebaseReportError(
            "report overlay_digest no longer matches src/ -- re-run patch-rebase-check"
        )

    selection_input = report.get("selection", {})
    selected = tuple(selection_input.get("patch_ids", ()))
    # Adversarial-review follow-up: none of the checks above notice a
    # SELECTOR change (e.g. an uncommitted edit to config/recipes.toml that
    # widens the recipe this report was generated for) -- no patch bytes,
    # overlay, or revision moved, so re-deriving the id set from the same
    # selector and requiring it to match exactly closes that gap.
    try:
        current_ids = set(_selection_patch_ids(
            recipe_name=selection_input.get("recipe"),
            all_patches=bool(selection_input.get("all_patches", False)),
        ))
    except RebaseCheckError as exc:
        raise StaleRebaseReportError(
            f"report's selector no longer resolves: {exc}"
        ) from exc
    if current_ids != set(selected):
        raise StaleRebaseReportError(
            "report's selection.patch_ids no longer matches what its own "
            "recipe/--all selector currently resolves to "
            f"(only in report: {sorted(set(selected) - current_ids)}, "
            f"only live: {sorted(current_ids - set(selected))}) -- "
            "re-run patch-rebase-check"
        )
    patch_entries = tuple(report.get("patches", ()))
    # Adversarial-review follow-up: a report edited (by hand, or a bug) to
    # drop entries from `patches[]` while leaving `selection.patch_ids`
    # untouched would otherwise pass every digest check below trivially --
    # there would just be fewer entries to check. The two lists are
    # required to name EXACTLY the same set, closing that gap before any
    # per-entry check runs.
    reported_ids = {entry.get("patch_id") for entry in patch_entries}
    if reported_ids != set(selected):
        raise StaleRebaseReportError(
            "report's patches[] does not name exactly its own selection.patch_ids "
            f"(only in patches[]: {sorted(reported_ids - set(selected))}, "
            f"only in selection: {sorted(set(selected) - reported_ids)}) -- "
            "the report is internally inconsistent, re-run patch-rebase-check"
        )

    catalog = {m.patch_id: m for m in patchset.catalog()}
    for patch_entry in patch_entries:
        patch_id = patch_entry["patch_id"]
        current = catalog.get(patch_id)
        if current is None:
            raise StaleRebaseReportError(f"patch {patch_id!r} no longer exists in the registry")
        if current.content_hash != patch_entry.get("implementation_digest"):
            raise StaleRebaseReportError(
                f"patch {patch_id!r} implementation changed since the report was written"
            )
        # Adversarial-review follow-up: a packaged patch's REQUIRES/CONFLICTS
        # live in patch.toml, a file implementation_digest never covers (that
        # only hashes patch.py). An uncommitted patch.toml edit can change
        # the dependency graph -- and therefore resolve_exact()'s topological
        # order, and therefore apply_all()'s actual apply sequence -- while
        # every digest above stays identical. The report already recorded
        # what requires/conflicts the probe actually resolved against; require
        # it still matches.
        if tuple(current.requires) != tuple(patch_entry.get("requires", ())):
            raise StaleRebaseReportError(
                f"patch {patch_id!r} REQUIRES changed since the report was written"
            )
        if tuple(current.conflicts) != tuple(patch_entry.get("conflicts", ())):
            raise StaleRebaseReportError(
                f"patch {patch_id!r} CONFLICTS changed since the report was written"
            )
    known_good = tuple(report.get("known_good_patch_ids", ()))
    unknown = set(known_good) - set(selected)
    if unknown:
        raise StaleRebaseReportError(
            f"report's known_good_patch_ids contains id(s) outside its own selection: "
            f"{sorted(unknown)}"
        )
    return known_good


def _write_overlay_snapshot(
    root: Path, texts: dict[str, str], *, dry_run: bool,
    backup: dict[str, str | None] | None = None,
) -> list[str]:
    """Write exactly the overlay bytes already captured in ``texts`` --
    never re-reads ``src/`` from disk.

    Adversarial-review follow-up: ``legacy._copy_overlay()`` (used by a
    plain ``apply``) reads ``src/`` itself at write time, which is exactly
    right for that caller -- it never claimed anything about an earlier,
    separately-timed digest. ``apply_known_good`` does make that claim
    (its report's ``overlay_digest`` must still match ``src/``), so it must
    write from the SAME bytes it just verified, not take a second,
    independently-timed look at disk that a concurrent overlay edit could
    answer differently.
    """
    written: list[str] = []
    for relative_str in sorted(texts):
        text = texts[relative_str]
        target = patcher.resolve_contained_target(root, relative_str)
        # `text` is always LF-only (read from src via universal-newline
        # translation) and gets written verbatim (newline=""). Comparing
        # against a UNIVERSAL-NEWLINE read of `target` would translate a
        # stale CRLF target to LF before the comparison, making it read as
        # "already matches" and skip the write forever -- leaving the CRLF
        # bytes on disk even though a real write would have normalized them.
        # Decode raw bytes directly (no translation) so a lingering CRLF
        # target is correctly seen as needing a rewrite. Found live during
        # the b10502->b10680 bump: two overlay files stayed CRLF across
        # repeated known-good applies for exactly this reason.
        # (Path.read_text(newline=...) is Python 3.13+ only -- Brutus runs
        # 3.12 -- so this reads bytes and decodes rather than using it.)
        if target.is_file() and target.read_bytes().decode("utf-8") == text:
            continue
        if backup is not None and relative_str not in backup:
            backup[relative_str] = target.read_text(encoding="utf-8") if target.is_file() else None
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8", newline="")
        written.append(relative_str)
    return written


@dataclass
class ApplyKnownGoodResult:
    ok: bool
    selected_patch_ids: tuple[str, ...]
    known_good_patch_ids: tuple[str, ...]
    partial: bool


def apply_known_good(
    root: Path,
    report_path: Path,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> ApplyKnownGoodResult:
    """Apply exactly the dependency-closed known-good subset a fresh
    ``patch-rebase-check`` report proved clean. Fails closed on any staleness
    (``_require_fresh``) and goes through the SAME ``apply_all`` transaction
    a plain ``apply`` uses -- no separate, less-audited write path.

    A *full* known-good apply (every selected patch reproved clean) records
    exactly what a plain ``apply`` records and can advance the release stage
    to ``patched``. A *partial* known-good set never advances the stage --
    it is explicit reconciliation progress, not a completed tree -- and is
    only accepted while the record is at stage ``audited`` or ``broken``
    (unless ``force``): applying a partial subset onto an already-later-stage
    tree (generated/built/tested/validated) would leave that later-stage
    evidence describing a composition that was never actually completed.
    """
    report = load_report(report_path)
    # Snapshot the overlay ONCE, verify it, then write from these exact
    # bytes below -- never a second, independently-timed disk read that a
    # concurrent overlay edit on this shared, multi-agent working tree
    # could answer differently than the one just verified fresh.
    overlay_snapshot = _overlay_texts()
    known_good = _require_fresh(
        report, root, overlay_snapshot_digest=_digest_from_texts(overlay_snapshot),
    )
    selected = tuple(report.get("selection", {}).get("patch_ids", ()))
    partial = set(known_good) != set(selected)

    from .. import __main__ as legacy  # noqa: PLC0415 (leaf import, avoids a cycle)

    record = legacy._record_for(root)
    original_stage = record.stage
    if not force and not record.audit.get("passed"):
        raise RebaseCheckError(
            "apply --known-good: tree has not passed a strict audit; run "
            "`bigcherry audit` first, or pass --force"
        )
    if partial and not force and original_stage not in ("audited", "broken"):
        raise RebaseCheckError(
            "apply --known-good: a PARTIAL known-good subset is only accepted "
            f"on a tree at stage 'audited' or 'broken' (current: {original_stage!r}) "
            "-- applying it now would leave later-stage evidence describing a "
            "composition that was never fully applied; pass --force only if "
            "you accept invalidating that evidence yourself"
        )

    # Bind each implementation load to the exact digest the REPORT recorded
    # (not merely whatever the registry resolves to right now) -- closes the
    # TOCTOU window between _require_fresh()'s digest check above and the
    # actual load below, on this shared, multi-agent working tree.
    report_digests = {p["patch_id"]: p["implementation_digest"] for p in report.get("patches", ())}
    resolved = patchset.resolve_exact(known_good, allow_rejected=True) if known_good else \
        patchset.ResolvedPatchSet((), None)
    file_patches: list[patcher.FilePatch] = []
    for module in resolved.modules:
        module_root = module.catalog_root or paths.PATCHES
        module_registry = patch_registry.load_registry(module_root)
        descriptor = module_registry.get(module.patch_id)
        file_patches.extend(patch_registry.load_implementation(
            descriptor, root=module_root, expected_digest=report_digests[module.patch_id],
        ))

    overlay_backup: dict[str, str | None] = {}
    written = _write_overlay_snapshot(root, overlay_snapshot, dry_run=dry_run, backup=overlay_backup)
    results = patcher.apply_all(file_patches, root, dry_run=dry_run, initial_texts=dict(overlay_snapshot))
    ok = all(r.ok for r in results)
    if not ok and not dry_run and overlay_backup:
        legacy._restore_overlay(root, overlay_backup)

    if not dry_run:
        record.patches = releases.summarise_patches(results)
        if partial:
            # Reconciliation progress only: never advance the stage, and a
            # successful partial apply must not even look like the failure
            # path (which explicitly demotes to `broken`) -- it *did* apply
            # exactly what it claimed to. A failed partial apply still means
            # the tree may be half-mutated in ways record_apply_result's
            # `broken` transition exists to name.
            if ok:
                if original_stage not in ("audited", "broken"):
                    # Adversarial-review follow-up: this only happens when
                    # --force overrode the eligibility guard above. The tree
                    # just changed to an incomplete composition, but
                    # `original_stage`'s generated/built/tested/validated
                    # evidence (and any promotion pointer) still describes
                    # the PREVIOUS, different tree -- leaving it in place
                    # would be silently stale evidence, not just an
                    # accepted override. There is no stage that means
                    # "partially patched", so explicitly invalidate to
                    # `broken` rather than pretend nothing changed.
                    record.promotion = None
                    record.manifest_hash = ""
                    record.advance_to("broken")
                    record.notes = (
                        "partial known-good apply (--force) replaced a "
                        f"'{original_stage}' tree with an incomplete "
                        "composition -- prior stage evidence invalidated"
                    )
                elif record.notes.startswith("patches failed:"):
                    record.notes = ""
            else:
                releases.record_apply_result(record, False)
                record.notes = "patches failed: " + ", ".join(
                    record.patches.get("failed_edits", [])
                )
            record.save()
        else:
            tree_mutated = bool(written) or any(result.changed for result in results)
            releases.record_apply_result(record, ok, mutated=tree_mutated)
            if not ok:
                record.notes = "patches failed: " + ", ".join(
                    record.patches.get("failed_edits", [])
                )
            elif record.notes.startswith("patches failed:"):
                record.notes = ""
            record.save()

    return ApplyKnownGoodResult(
        ok=ok,
        selected_patch_ids=selected,
        known_good_patch_ids=known_good,
        partial=partial,
    )

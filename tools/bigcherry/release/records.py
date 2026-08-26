"""Per-release compatibility records.

The question this answers is "does bigcherry work on llama.cpp b1234, and how
do we know?" -- without checking anything out.

The deliberate choice here is that there is **one** patch set, not a snapshot
per release. Snapshotting would give us N copies of every edit, all needing the
same correction when one turns out to be wrong; that is the fork problem we are
avoiding, moved up a level. Anchored edits are built to span releases, so they
evolve in place and git history carries the past.

What *is* recorded per release is the evidence:

* the audit result for that revision,
* which edits applied, were already applied, or were not applicable,
* the candidate manifest hash the build was generated from,
* how far validation got (built / tested / tuned / production).

Records are tracked in git under ``releases/``. The bulky outputs they refer to
-- manifests, databases, replay caches, build logs -- stay in ``artifacts/``
and are not tracked. When a release passes validation, tag the bigcherry repo
``supports/<release>``; checking out that tag reproduces the exact patch set
that worked.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal

from .. import ARTIFACT_VERSION
from ..core import paths
from ..tuning.promotion import PromotionPointer

RELEASES_DIR = paths.REPO_ROOT / "releases"
INDEX_PATH = RELEASES_DIR / "index.json"


def _atomic_write_json(path: Path, document: dict[str, Any]) -> None:
    """Validate and atomically publish one JSON document.

    The temporary file is created beside the destination so ``os.replace`` is
    atomic on the target filesystem.  Flushes make the file durable before it
    becomes visible; the directory flush is best-effort for platforms which
    do not allow directory handles to be opened (notably Windows).
    """
    try:
        encoded = json.dumps(document, indent=2, sort_keys=True,
                             allow_nan=False) + "\n"
        parsed = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"refusing to publish invalid JSON for {path}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"refusing to publish non-object JSON for {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

# How far a release has been taken. Each stage implies the ones before it.
Stage = Literal["pulled", "audited", "patched", "generated", "built",
                "tested", "tuned", "validated", "broken"]

STAGE_ORDER: tuple[Stage, ...] = (
    "pulled", "audited", "patched", "generated", "built",
    "tested", "tuned", "validated",
)
VALID_STAGES = frozenset((*STAGE_ORDER, "broken"))


def _validate_stage(stage: str) -> None:
    if stage not in VALID_STAGES:
        raise ValueError(f"unknown release stage: {stage!r}")


def _validate_transition(current: str, proposed: str) -> None:
    _validate_stage(current)
    _validate_stage(proposed)
    if current == proposed or current == "broken":
        return
    if proposed == "broken":
        return
    if STAGE_ORDER.index(proposed) < STAGE_ORDER.index(current):
        raise ValueError(
            f"release stage cannot move backwards: {current!r} -> {proposed!r}")


def record_apply_result(
        record: "ReleaseRecord", succeeded: bool, *, mutated: bool = False) -> None:
    """Record an apply result while respecting evidence invalidation.

    Applying an unchanged tree is idempotent and preserves later evidence. A
    successful apply that writes an overlay, changes a patch, or selects a
    different tree invalidates later evidence and returns the record to
    ``patched``. A failed apply remains an explicit transition to ``broken``.
    """
    if not succeeded:
        # RE13 (GPT-auto-agent review, 2026-08-17): a validated record
        # whose apply just failed had its evidence invalidated -- clearing
        # promotion here (not just deferring to advance_to's own guard)
        # means a record.promotion left stale can never be reused by ANY
        # future direct stage assignment, only by a fresh promote() call.
        record.promotion = None
        record.advance_to("broken")
        return
    if record.stage == "broken":
        record.advance_to("patched")
    elif mutated and STAGE_ORDER.index(record.stage) >= STAGE_ORDER.index("patched"):
        # This is an intentional evidence invalidation, not a general
        # relaxation of the monotonic lifecycle transition rule. The caller
        # has observed a real tree mutation, so generated/build/test evidence
        # no longer describes the checkout.
        # RE13 (GPT-auto-agent review): this is exactly the stale-evidence
        # bypass the review found -- demoting to `patched` used to clear
        # manifest_hash but leave `promotion` in place, so a later
        # advance_to("validated") (now impossible directly, but even a
        # future promote() call checking `if already validated, skip`
        # style logic would have been fooled) could see pre-mutation
        # evidence as still valid for the post-mutation tree.
        record.stage = "patched"
        record.manifest_hash = ""
        record.promotion = None
    elif STAGE_ORDER.index(record.stage) < STAGE_ORDER.index("patched"):
        record.advance_to("patched")


@dataclass
class ReleaseRecord:
    """Everything known about one llama.cpp revision."""

    revision: str
    #: Upstream release tag when the revision is one (``b1234``), else "".
    release_tag: str = ""
    stage: Stage = "pulled"
    #: bigcherry commit that produced this record, so a past result can be
    #: reproduced even after the patch set moves on.
    bigcherry_revision: str = ""
    artifact_version: int = ARTIFACT_VERSION
    audit: dict[str, Any] = field(default_factory=dict)
    patches: dict[str, Any] = field(default_factory=dict)
    manifest_hash: str = ""
    #: Fingerprint of the patch selection currently applied to the checkout,
    #: from ``recipes.tree_state_key``. Lets a build tell whether the tree is
    #: already what it needs, instead of re-applying and forcing a rebuild.
    #: RE13/RE23: legacy mutable-checkout state -- only the legacy
    #: cmd_build/legacy-build path still writes this. It stays populated for
    #: as long as that path exists (RE23 retires it); a `validated` record
    #: no longer needs it as evidence -- ``promotion`` below is what a
    #: canonical campaign-backed release actually points at.
    tree_state: str = ""
    #: A persisted ``PromotionPointer.document()`` (RE13) -- immutable
    #: source/build/runtime/campaign/report identities from a real
    #: ``execute_campaign_lane`` run, not a mutable checkout snapshot. Set
    #: only by ``promote()``, which is also what actually advances a
    #: record's stage to ``validated`` -- a record cannot reach ``validated``
    #: any other way, so ``validated`` implies a real, verifiable promotion
    #: pointer exists.
    promotion: dict[str, Any] | None = None
    #: Free-form notes -- in practice, why a release is `broken`.
    notes: str = ""
    first_seen: str = ""
    updated_at: str = ""

    def path(self) -> Path:
        return RELEASES_DIR / f"{self.slug()}.json"

    def slug(self) -> str:
        return self.release_tag or self.revision[:12]

    def advance_to(self, stage: Stage) -> None:
        """Move the stage forward, never backwards.

        A record that has been `validated` should not be demoted to `built`
        because someone re-ran `apply`. Only an explicit `broken` moves down.

        RE13: `validated` can ONLY be reached via ``releases.promote()`` --
        this method unconditionally rejects it, even when ``promotion`` is
        already set (GPT-auto-agent review, 2026-08-17: the original guard
        only checked ``promotion is None``, which meant any code path that
        left a stale pointer in place -- e.g. an evidence-invalidating
        mutation that forgot to clear it -- could still walk back to
        `validated` through this method directly, reusing pre-mutation
        evidence for a post-mutation tree). ``promote()`` performs the
        transition itself via ``_set_validated_via_promotion()``.
        """
        if stage == "validated":
            raise ValueError(
                "cannot advance to 'validated' directly -- call "
                "releases.promote() instead")
        _validate_transition(self.stage, stage)
        if stage == self.stage:
            return
        self.stage = stage

    def _set_validated_via_promotion(self) -> None:
        """Only called by releases.promote(), after it has set
        ``self.promotion`` to a freshly validated pointer document. Not
        part of advance_to()'s public contract on purpose -- see its
        docstring."""
        _validate_transition(self.stage, "validated")
        self.stage = "validated"

    def validate(self) -> None:
        """Reject malformed persisted records before they become evidence."""
        _validate_stage(self.stage)
        if not isinstance(self.revision, str) or not self.revision.strip():
            raise ValueError("release record has no source revision")
        if not isinstance(self.release_tag, str):
            raise ValueError("release record has an invalid release tag")
        # RE13 (GPT-auto-agent review, 2026-08-17): re-check on every
        # load/save, not only at the advance_to() transition -- a record
        # read back from disk could have `validated` and a missing/edited
        # promotion pointer with no transition ever having happened in
        # this process. Strict PromotionPointer.from_document() validation
        # (schema_version, every identity field non-empty) replaces the
        # old loose isinstance(dict) + bare revision-match check, which a
        # record carrying only {"schema_version": 2, "revision": "..."}
        # used to pass. release_tag must also agree, not just revision --
        # revision alone does not disambiguate a re-tagged release.
        if self.stage == "validated":
            from ..tuning.promotion import PromotionError, PromotionPointer
            try:
                pointer = PromotionPointer.from_document(self.promotion)
            except PromotionError as exc:
                raise ValueError(f"validated release record has an invalid promotion pointer: {exc}") from exc
            if pointer.revision != self.revision:
                raise ValueError(
                    "promotion pointer revision disagrees with release record")
            # PromotionPointer.release_tag is required non-empty (make_pointer
            # rejects ""), but ReleaseRecord.release_tag is legitimately ""
            # for an untagged revision (see its own docstring) -- the same
            # fallback slug() already uses (release_tag or revision) is the
            # correct comparison, not a bare equality that would reject
            # every untagged release's own real promotion pointer.
            if pointer.release_tag != (self.release_tag or self.revision):
                raise ValueError(
                    "promotion pointer release_tag disagrees with release record")

    def save(self) -> Path:
        now = _dt.datetime.now(_dt.timezone.utc).isoformat()
        self.first_seen = self.first_seen or now
        self.updated_at = now
        RELEASES_DIR.mkdir(parents=True, exist_ok=True)
        self.validate()
        target = self.path()
        _atomic_write_json(target, asdict(self))
        _rebuild_index()
        validate_index_consistency()
        return target


def promote(record: ReleaseRecord, pointer: PromotionPointer) -> None:
    """Advance ``record`` to ``validated`` via a real campaign-backed
    :class:`~bigcherry.tuning.promotion.PromotionPointer` (RE13) -- the only
    supported way to reach ``validated``. Does not save; the caller decides
    when to persist, same as every other stage transition in this module.
    """
    # GPT-auto-agent review (RE13 follow-up, 2026-08-17): re-parse through
    # PromotionPointer.from_document() rather than trusting the caller's
    # already-constructed object -- a caller could hand-build a
    # PromotionPointer with a bypassed/patched __init__ (frozen dataclasses
    # do not prevent object.__setattr__) or a subtly malformed instance;
    # this is the same strict validation validate() applies to a persisted
    # document, applied here too so the mutation API itself is fail-closed,
    # not just save()/validate() catching it later.
    pointer = PromotionPointer.from_document(pointer.document())
    if pointer.revision != record.revision:
        raise ValueError(
            f"promotion pointer revision {pointer.revision!r} does not "
            f"match release record revision {record.revision!r}")
    if pointer.release_tag != (record.release_tag or record.revision):
        raise ValueError(
            f"promotion pointer release_tag {pointer.release_tag!r} does not "
            f"match release record release_tag {record.release_tag!r}")
    record.promotion = pointer.document()
    record._set_validated_via_promotion()


def load(revision: str, release_tag: str = "") -> ReleaseRecord:
    """Load the record for a revision, or return a fresh one."""
    slug = release_tag or revision[:12]
    path = RELEASES_DIR / f"{slug}.json"
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {f for f in ReleaseRecord.__dataclass_fields__}
        record = ReleaseRecord(**{k: v for k, v in data.items() if k in known})
        record.validate()
        return record
    return ReleaseRecord(revision=revision, release_tag=release_tag)


def all_records() -> list[ReleaseRecord]:
    if not RELEASES_DIR.is_dir():
        return []
    records = []
    for path in sorted(RELEASES_DIR.glob("*.json")):
        if path.name == "index.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {f for f in ReleaseRecord.__dataclass_fields__}
        record = ReleaseRecord(**{k: v for k, v in data.items() if k in known})
        record.validate()
        if record.slug() != path.stem:
            raise ValueError(f"release filename does not match slug: {path.name}")
        records.append(record)
    return sorted(records, key=lambda r: r.updated_at, reverse=True)


def _rebuild_index() -> None:
    """Regenerate the tracked summary index.

    Derived from the individual records rather than maintained alongside them,
    so it cannot drift out of agreement with them.
    """
    records = all_records()
    index = {
        "artifact_version": ARTIFACT_VERSION,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "releases": [
            {
                "slug": r.slug(),
                "revision": r.revision,
                "release_tag": r.release_tag,
                "stage": r.stage,
                "manifest_hash": r.manifest_hash,
                "audit_passed": bool(r.audit.get("passed")),
                "updated_at": r.updated_at,
            }
            for r in records
        ],
    }
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(INDEX_PATH, index)


def validate_index_consistency(
    *, index_path: Path | None = None,
    records: list[ReleaseRecord] | None = None,
) -> None:
    """Fail closed when the tracked index disagrees with source records."""
    index_path = INDEX_PATH if index_path is None else index_path
    if not index_path.is_file():
        return
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid release index: {index_path}") from exc
    if not isinstance(index, dict) or not isinstance(index.get("releases"), list):
        raise ValueError("release index has no releases list")
    records = all_records() if records is None else records
    expected = {r.slug(): {
        "slug": r.slug(), "revision": r.revision,
        "release_tag": r.release_tag, "stage": r.stage,
        "manifest_hash": r.manifest_hash,
        "audit_passed": bool(r.audit.get("passed")),
        "updated_at": r.updated_at,
    } for r in records}
    actual: dict[str, dict[str, Any]] = {}
    for entry in index["releases"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("slug"), str):
            raise ValueError("release index contains an invalid entry")
        slug = entry["slug"]
        if slug in actual:
            raise ValueError(f"release index contains duplicate slug: {slug}")
        actual[slug] = entry
    if set(actual) != set(expected):
        raise ValueError("release index does not match source records")
    fields = ("slug", "revision", "release_tag", "stage", "manifest_hash",
              "audit_passed", "updated_at")
    for slug, source in expected.items():
        if any(actual[slug].get(field) != source[field] for field in fields):
            raise ValueError(f"release index disagrees with source record: {slug}")


def summarise_audit(report: dict[str, Any], *, strict: bool) -> dict[str, Any]:
    """Condense an audit report for storage in a release record.

    The full report goes to ``artifacts/``; what belongs in the tracked record
    is the verdict plus the identity of anything that failed.
    """
    from ..source.audit import passed as audit_passed
    failures = [c["id"] for c in report["checks"] if not c["ok"]]
    return {
        "passed": audit_passed(report, strict=strict),
        "strict": strict,
        "summary": report["summary"],
        "failed_checks": failures,
    }


def summarise_patches(results: list[Any]) -> dict[str, Any]:
    """Condense patch application results for storage."""
    per_file = {}
    for result in results:
        per_file[result.path] = {
            edit.edit_id: edit.status for edit in result.results
        }
    failed = [
        f"{result.path}:{edit.edit_id}"
        for result in results for edit in result.failed
    ]
    return {
        "applied_cleanly": not failed,
        "failed_edits": failed,
        "files": per_file,
    }

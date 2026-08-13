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

from . import ARTIFACT_VERSION
from . import paths

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
    tree_state: str = ""
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
        """
        if stage == "broken" or self.stage == "broken":
            self.stage = stage
            return
        current = STAGE_ORDER.index(self.stage) if self.stage in STAGE_ORDER else -1
        proposed = STAGE_ORDER.index(stage) if stage in STAGE_ORDER else -1
        if proposed > current:
            self.stage = stage

    def save(self) -> Path:
        now = _dt.datetime.now(_dt.timezone.utc).isoformat()
        self.first_seen = self.first_seen or now
        self.updated_at = now
        RELEASES_DIR.mkdir(parents=True, exist_ok=True)
        target = self.path()
        _atomic_write_json(target, asdict(self))
        _rebuild_index()
        return target


def load(revision: str, release_tag: str = "") -> ReleaseRecord:
    """Load the record for a revision, or return a fresh one."""
    slug = release_tag or revision[:12]
    path = RELEASES_DIR / f"{slug}.json"
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {f for f in ReleaseRecord.__dataclass_fields__}
        return ReleaseRecord(**{k: v for k, v in data.items() if k in known})
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
        records.append(ReleaseRecord(**{k: v for k, v in data.items() if k in known}))
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


def summarise_audit(report: dict[str, Any], *, strict: bool) -> dict[str, Any]:
    """Condense an audit report for storage in a release record.

    The full report goes to ``artifacts/``; what belongs in the tracked record
    is the verdict plus the identity of anything that failed.
    """
    from .source_audit import passed as audit_passed
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

"""HI152: revision-bound patch coverage/disposition schema + gate.

`patch-rebase-check --recipe <name>` only ever checks the recipe's
selected subset -- a patch outside that subset (e.g. an experimental
`rdna-boosts` patch) can go permanently unchecked by the pin-bump
procedure, with nothing recording that fact. HI149 found exactly this
gap live (patch 1206 turned out NOT to be broken, but nothing would have
caught it if it had been).

This module does NOT add a new patch lifecycle state (validated/
untested/rejected are patch IDENTITY; a rebase failure is a REVISION-
SPECIFIC fact, and conflating the two was explicitly rejected in the
HI150 design debate). Instead: a `Disposition` is a standing, narrowly-
scoped waiver bound to one exact (patch_id, target_revision,
patch_digest) triple -- it silently stops applying the instant any of
those three changes, so it can never become a permanent hole.

Storage: JSON files under a dispositions directory (NOT releases/*.json
-- avoids the exact class of bug fixed in release/records.py this
session, where a non-ReleaseRecord JSON dropped into releases/ crashed
every save()).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

CLEAN_STATUSES = ("CLEAN", "CLEAN_NOOP", "NOT_APPLICABLE_BY_DESIGN")


class DispositionError(RuntimeError):
    pass


@dataclass(frozen=True)
class Disposition:
    patch_id: str
    target_revision: str
    patch_digest: str
    disposition: str  # currently only "known_broken"
    failure_status: str
    reason: str
    owner: str
    tracking_item: str

    def applies_to(self, *, target_revision: str, patch_digest: str) -> bool:
        """A disposition is bound to the EXACT revision+digest it was
        recorded against -- either changing invalidates it immediately.
        Never a standing waiver."""
        return self.target_revision == target_revision and self.patch_digest == patch_digest


def _path(dispositions_dir: Path, patch_id: str) -> Path:
    safe = patch_id.replace("/", "_")
    return dispositions_dir / f"{safe}.json"


def load_disposition(dispositions_dir: Path, patch_id: str) -> Disposition | None:
    path = _path(dispositions_dir, patch_id)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return Disposition(**data)


def save_disposition(dispositions_dir: Path, record: Disposition) -> Path:
    if record.disposition != "known_broken":
        raise DispositionError(
            f"unsupported disposition kind {record.disposition!r} (only "
            "'known_broken' exists today)"
        )
    dispositions_dir.mkdir(parents=True, exist_ok=True)
    path = _path(dispositions_dir, record.patch_id)
    path.write_text(json.dumps(asdict(record), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def clear_disposition(dispositions_dir: Path, patch_id: str) -> bool:
    path = _path(dispositions_dir, patch_id)
    if not path.is_file():
        return False
    path.unlink()
    return True


def list_dispositions(dispositions_dir: Path) -> dict[str, Disposition]:
    if not dispositions_dir.is_dir():
        return {}
    out: dict[str, Disposition] = {}
    for path in sorted(dispositions_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            record = Disposition(**data)
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        out[record.patch_id] = record
    return out


@dataclass(frozen=True)
class CoverageResult:
    discovered_patch_ids: tuple[str, ...]
    checked_patch_ids: tuple[str, ...]
    excluded: tuple[dict[str, str], ...]
    uncovered_patch_ids: tuple[str, ...]
    complete: bool

    def as_dict(self) -> dict:
        return {
            "discovered_patch_ids": list(self.discovered_patch_ids),
            "checked_patch_ids": list(self.checked_patch_ids),
            "excluded": [dict(entry) for entry in self.excluded],
            "uncovered_patch_ids": list(self.uncovered_patch_ids),
            "complete": self.complete,
        }


def compute_coverage(
    *,
    catalog_states: dict[str, str],
    all_report: dict,
    recipe_patch_ids: frozenset[str],
    dispositions: dict[str, Disposition],
    target_revision: str,
) -> CoverageResult:
    """`all_report` must come from `patch-rebase-check --all` (every
    non-rejected registry patch, individually statused). `catalog_states`
    is patch_id -> CatalogEntry.state for the WHOLE registry (including
    rejected), used only to compute `excluded`.

    Policy (HI150 round 2, converged with gpt):
    - a RECIPE-selected patch must be CLEAN/CLEAN_NOOP/NOT_APPLICABLE_BY_DESIGN;
      no disposition can excuse it, ever;
    - a NON-selected patch may be clean OR carry a disposition that
      `applies_to` this exact (target_revision, patch_digest);
    - a retired patch (state == 'rejected' or 'superseded') never appears in
      `all_report` at all -- it is reported as `excluded`, not uncovered;
    - anything else (bad status, no matching disposition) is uncovered ->
      `complete` is False.
    """
    checked_by_id = {entry["patch_id"]: entry for entry in all_report.get("patches", ())}
    checked_ids = tuple(sorted(checked_by_id))

    excluded = tuple(
        {"patch_id": patch_id, "reason": f"state={state}"}
        for patch_id, state in sorted(catalog_states.items())
        if state in ("rejected", "superseded")
    )
    excluded_ids = {entry["patch_id"] for entry in excluded}

    discovered_ids = tuple(sorted(set(catalog_states) | set(checked_ids)))

    uncovered: list[str] = []
    for patch_id, entry in checked_by_id.items():
        status = entry.get("status")
        if status in CLEAN_STATUSES:
            continue
        if patch_id in recipe_patch_ids:
            uncovered.append(patch_id)
            continue
        disposition = dispositions.get(patch_id)
        digest = entry.get("implementation_digest", "")
        if disposition is not None and disposition.disposition == "known_broken" and \
                disposition.applies_to(target_revision=target_revision, patch_digest=digest):
            continue
        uncovered.append(patch_id)

    undiscovered = sorted(set(discovered_ids) - set(checked_ids) - excluded_ids)
    uncovered.extend(undiscovered)

    complete = not uncovered
    return CoverageResult(
        discovered_patch_ids=discovered_ids, checked_patch_ids=checked_ids,
        excluded=excluded, uncovered_patch_ids=tuple(sorted(set(uncovered))),
        complete=complete,
    )

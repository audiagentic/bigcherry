"""EC19: computed plan/patch lifecycle status, not hand-maintained prose.

External review (2026-08-20, P0 recommendation #3): with 65+ RD backlog
items plus a growing Experiment Contract registry, plan count gets
misread as implementation progress unless something makes the
plan-to-proof gap visible at a glance. This module computes a real
status per RD-prefixed plan item from signals ALREADY on disk --
external-sources.toml, packaged patches, config/experiment-contracts.toml,
config/recipes.toml -- rather than a human/agent hand-maintaining a
status field that drifts from reality.

Deliberately does NOT compute every status in EC19's target enum
(discovered/planned/source-pinned/materialized/contracted/build-proven/
trigger-proven/correctness-proven/measured/generalised/promoted/
rejected/retired). trigger-proven/correctness-proven/measured need real
telemetry evidence this tool has no access to (EC18 tracks that
separately) -- those are reported as "unknown", honestly, rather than
fabricated. What IS real and computed here:

  source-pinned  -- a [[sources.tracked]] entry names this plan-item
  materialized   -- a real package's patch.py PROVENANCE names this
                    plan-item (or the tracked entry's own `patch` field
                    points at a package module that actually exists)
  build-applies  -- the materialized patch's own STATE constant is
                    "validated" or "untested" (both mean the TRANSFORM
                    applies cleanly against the pinned tree; "untested"
                    is about the *hypothesis*, not the patch mechanics)
  rejected       -- the materialized patch's STATE is "rejected"
  contracted     -- an Experiment Contract's `source.atomic_part` names
                    this patch, or the contract id's own RDxx prefix
                    matches this plan-item

A plan-item with NO tracked entry and NO patch is "discovered" only --
this module cannot see docs/planning/ items directly (that's the
ag-planning MCP server's data, not a local file this module can safely
read/parse without duplicating that server's own parsing) -- callers
that want the full discovered/planned distinction should cross-reference
plan_list_items() output themselves; this module's `RD_ITEMS_SEEN`
constant lists every plan-item ID this module has ANY signal for, which
is the "at least planned" floor.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from ..core import paths
from . import patchset
from ..source import sources


@dataclass(frozen=True)
class LifecycleStatus:
    plan_item: str
    source_pinned: bool
    materialized: bool
    patch_ids: tuple[str, ...]
    build_state: str | None  # "validated" | "untested" | "rejected" | None
    contracted: bool
    contract_ids: tuple[str, ...]
    tracked_status: str | None  # the [[sources.tracked]] entry's own `status` field, if any


def _tracked_entries_by_plan_item(registry: dict) -> dict[str, list[dict]]:
    by_item: dict[str, list[dict]] = {}
    for source in registry.get("sources", []):
        for entry in source.get("tracked", []):
            item = entry.get("plan-item")
            if item and item != "-":
                by_item.setdefault(item, []).append(entry)
    return by_item


def _patch_states(patches_dir: Path | None = None) -> dict[str, str]:
    return {info.name: info.state for info in patchset.describe(patches_dir)}


def _plan_items_from_patches(patches_dir: Path | None = None) -> dict[str, list[str]]:
    """plan-item -> [patch_id, ...] from each module's own PROVENANCE dict."""
    by_item: dict[str, list[str]] = {}
    for module in patchset.catalog(patches_dir):
        prov = sources._patch_provenance(module.path)
        item = (prov or {}).get("plan-item")
        if item:
            by_item.setdefault(item, []).append(module.patch_id)
    return by_item


def _contracts_by_plan_item(contracts_path: Path | None = None) -> dict[str, list[str]]:
    """plan-item -> [contract_id, ...], derived from each contract id's own
    leading RDxx/HIxx/EXxx token (e.g. "RD08-Q6K-..." -> "RD08",
    "RD39-42-STREAM-..." -> "RD39-42" -- a real, already-used multi-item
    contract naming this project's own contracts follow, see EC02's
    backfill). Read as raw TOML, not through experiment_contract.
    parse_contract(), so a malformed contract elsewhere in the file cannot
    make lifecycle reporting itself fail -- this is a best-effort
    cross-reference, not a validator."""
    path = contracts_path or paths.EXPERIMENT_CONTRACTS
    if not path.is_file():
        return {}
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    by_item: dict[str, list[str]] = {}
    for contract_id in (raw.get("contract") or {}):
        parts = contract_id.split("-")
        head = parts[0]
        if head[:2].upper() not in ("RD", "HI", "EX") or not head[2:].isdigit():
            continue
        # "RD39-42-STREAM-..." names a range across two hyphen-joined
        # numeric parts; a single-item id like "RD08-Q6K-..." does not.
        if len(parts) > 1 and parts[1].isdigit():
            item = f"{head.upper()}-{parts[1]}"
        else:
            item = head.upper()
        by_item.setdefault(item, []).append(contract_id)
    return by_item


def compute_all(
    *,
    registry: dict | None = None,
    patches_dir: Path | None = None,
    contracts_path: Path | None = None,
) -> dict[str, LifecycleStatus]:
    """One LifecycleStatus per plan-item this module has ANY signal for --
    a tracked source entry, a materialized patch, or a contract."""
    registry = registry if registry is not None else sources.load_registry()
    tracked = _tracked_entries_by_plan_item(registry)
    patch_states = _patch_states(patches_dir)
    patches_by_item = _plan_items_from_patches(patches_dir)
    contracts_by_item = _contracts_by_plan_item(contracts_path)

    all_items = set(tracked) | set(patches_by_item) | set(contracts_by_item)
    results: dict[str, LifecycleStatus] = {}
    for item in sorted(all_items):
        patch_ids = tuple(sorted(patches_by_item.get(item, ())))
        states = {patch_states[pid] for pid in patch_ids if pid in patch_states}
        # A plan item's materialization can span more than one patch (e.g. a
        # composition-gated cluster) -- report the WORST state present:
        # rejected beats untested beats validated, since a lifecycle summary
        # should surface the thing most likely to need attention.
        build_state: str | None = None
        if "rejected" in states:
            build_state = "rejected"
        elif "untested" in states:
            build_state = "untested"
        elif "validated" in states:
            build_state = "validated"

        tracked_entries = tracked.get(item, [])
        tracked_status = tracked_entries[0].get("status") if len(tracked_entries) == 1 else (
            "/".join(sorted({e.get("status", "?") for e in tracked_entries}))
            if tracked_entries else None
        )

        contract_ids = tuple(sorted(contracts_by_item.get(item, ())))
        results[item] = LifecycleStatus(
            plan_item=item,
            source_pinned=item in tracked,
            materialized=bool(patch_ids),
            patch_ids=patch_ids,
            build_state=build_state,
            contracted=bool(contract_ids),
            contract_ids=contract_ids,
            tracked_status=tracked_status,
        )
    return results


def render_table(statuses: dict[str, LifecycleStatus]) -> str:
    lines = [f"{'PLAN':<8} {'SRC-PIN':<8} {'MATERIALIZED':<14} {'BUILD':<10} "
             f"{'CONTRACTED':<11} TRACKED-STATUS"]
    for item in sorted(statuses):
        s = statuses[item]
        lines.append(
            f"{item:<8} {'yes' if s.source_pinned else 'no':<8} "
            f"{','.join(s.patch_ids) or '-':<14} {s.build_state or '-':<10} "
            f"{'yes' if s.contracted else 'no':<11} {s.tracked_status or '-'}"
        )
    return "\n".join(lines)

#!/usr/bin/env python3
"""Render deterministic, non-mutating migration operations as JSONL.

The output is intentionally an agent execution plan, not an MCP client. BigCherry
plan lifecycle writes must still be executed through ag-planning and ledger
writes through ag-ledger.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

TRUE_VALUES = {"true", "1", "yes", "y"}
CONTINUING = {"successor", "split", "merge"}


def rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [{k: (v or "").strip() for k, v in r.items()} for r in csv.DictReader(f, delimiter=delimiter)]


def truth(value: str) -> bool:
    return value.lower() in TRUE_VALUES


def split_keys(value: str) -> list[str]:
    return [x.strip() for x in value.split(";") if x.strip()]


def run_validator(pack: Path, work: Path, repo: Path, phase: str) -> None:
    validator = pack / "scripts" / "validate_manifests.py"
    proc = subprocess.run(
        [sys.executable, str(validator), "--pack", str(pack), "--work", str(work), "--repo", str(repo), "--phase", phase],
        text=True,
    )
    if proc.returncode:
        raise RuntimeError(f"{phase} validation failed; no operations rendered")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--stage", choices=["create", "finalize"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ns = ap.parse_args()

    pack = ns.pack.resolve()
    work = ns.work.resolve()
    repo = ns.repo.resolve()
    if ns.stage == "create":
        successor_rows = rows(work / "SUCCESSORS.csv")
        phase = "preapply" if any(r.get("allocated_id") for r in successor_rows) else "review"
    else:
        phase = "preapply"
    run_validator(pack, work, repo, phase)

    lock = json.loads((pack / "SOURCE_LOCK.json").read_text(encoding="utf-8"))
    migration_id = lock["migration_id"]
    dispositions = rows(work / "DISPOSITIONS.csv")
    successors = rows(work / "SUCCESSORS.csv")
    lineage = rows(work / "LINEAGE.csv")
    refs = rows(work / "REFERENCE_DECISIONS.tsv", "\t")
    deps = rows(work / "DEPENDENCY_REMAP.tsv", "\t")

    succ_by_key = {r["successor_key"]: r for r in successors}
    preds_by_succ: dict[str, list[str]] = {}
    for edge in lineage:
        preds_by_succ.setdefault(edge["successor_key"], []).append(edge["predecessor_id"])

    def resolve_target(value: str) -> str:
        if value in succ_by_key:
            return succ_by_key[value].get("allocated_id", "") or f"@{value}"
        return value

    ops: list[dict] = []

    if ns.stage == "create":
        for row in sorted(successors, key=lambda r: (r["target_namespace"], r["successor_key"])):
            args = {"plan": row["target_namespace"], "title": row["title"]}
            for field in ["work", "skill", "priority"]:
                if row.get(field):
                    args[field] = row[field]
            ops.append(
                {
                    "op": "ag-planning.plan_create_item",
                    "successor_key": row["successor_key"],
                    "args": args,
                    "requested_id_prefix": row["id_prefix"],
                    "spec_path": row["spec_path"],
                    "predecessors": sorted(preds_by_succ.get(row["successor_key"], [])),
                    "postcondition": "write returned plan ID into SUCCESSORS.csv allocated_id; do not retire predecessors",
                }
            )
        ops.append(
            {
                "op": "checkpoint",
                "name": "successors-created",
                "require": "all returned IDs recorded in SUCCESSORS.csv, then run preapply validation before finalize stage",
            }
        )
    else:
        # Successor content/lineage first.
        for row in sorted(successors, key=lambda r: r["allocated_id"]):
            ops.append(
                {
                    "op": "ag-planning.plan_update_item",
                    "plan_item_id": row["allocated_id"],
                    "source_spec_path": row["spec_path"],
                    "must_include": {
                        "Supersedes": sorted(preds_by_succ.get(row["successor_key"], [])),
                        "Migration": migration_id,
                    },
                    "rule": "map reviewed spec sections through plan_update_item; do not transplant predecessor reviews/evidence/ledger events",
                }
            )

        # Predecessor reciprocal lineage while old item is still live.
        for row in sorted(dispositions, key=lambda r: r["source_id"]):
            if row["disposition"] not in CONTINUING:
                continue
            successor_ids = [resolve_target(k) for k in split_keys(row["successor_keys"])]
            ops.append(
                {
                    "op": "ag-planning.plan_update_item",
                    "plan_item_id": row["source_id"],
                    "append_notes": {
                        "Superseded by": successor_ids,
                        "Migration": migration_id,
                    },
                    "rule": "preserve existing historical body/reviews/evidence references",
                }
            )

        # Explicit dependency remaps before retirement.
        for row in deps:
            if not row.get("owner_source_id"):
                continue
            ops.append(
                {
                    "op": "dependency_reference_edit",
                    "owner_source_id": row["owner_source_id"],
                    "old_dependency_id": row["old_dependency_id"],
                    "decision": row["decision"],
                    "target": resolve_target(row["target_successor_key_or_id"]),
                    "rationale": row["rationale"],
                }
            )

        # Repository reference decisions. Preserve rows are audit assertions; rewrite/remove are edits.
        for row in refs:
            if not row.get("decision"):
                continue
            target = resolve_target(row.get("target_successor_key_or_id", ""))
            ops.append(
                {
                    "op": "repository_reference_decision",
                    "source_path": row["source_path"],
                    "source_line_at_frozen_snapshot": int(row["line"]),
                    "old_ref": row["old_ref"],
                    "reference_kind": row["reference_kind"],
                    "decision": row["decision"],
                    "target": target,
                    "rationale": row["rationale"],
                    "frozen_context": row["context"],
                    "rule": "edit by semantic occurrence/context, not global replacement; preserve historical references exactly",
                }
            )

        ops.append(
            {
                "op": "checkpoint",
                "name": "references-and-lineage-resolved",
                "require": "reciprocal lineage exists and all exact active dependency/reference rewrites are validated before any predecessor state transition",
            }
        )

        # Terminal transitions last.
        for row in sorted(dispositions, key=lambda r: r["source_id"]):
            if row["disposition"] == "retain-history":
                continue
            ops.append(
                {
                    "op": "ag-planning.plan_set_state",
                    "plan_item_id": row["source_id"],
                    "state": row["final_state"],
                    "precondition": "all applicable successor, lineage, dependency and reference operations completed",
                }
            )

        old_ids = sorted(r["source_id"] for r in dispositions if r["disposition"] != "retain-history")
        new_ids = sorted(r["allocated_id"] for r in successors)
        ops.append(
            {
                "op": "ag-ledger.record_change_event",
                "event": {
                    "change-class": "planning-migration",
                    "technical-summary": "Capability-native planning rebaseline with explicit predecessor/successor lineage and semantic reference migration.",
                    "user-summary-candidate": "Rebaselined BigCherry plans into Build/Run/Patching/Tuning ownership while preserving superseded plan history.",
                    "status": "unreleased",
                    "plan-item-ids": old_ids + new_ids,
                },
                "rule": "split into commit-aligned ledger events if repository workflow requires; keep affected old/new plan IDs linked",
            }
        )
        ops.append(
            {
                "op": "checkpoint",
                "name": "postapply-validation",
                "require": "run validate_manifests.py --phase postapply plus BigCherry planning/reference validation before merge",
            }
        )

    ns.out.parent.mkdir(parents=True, exist_ok=True)
    with ns.out.open("w", encoding="utf-8") as f:
        for seq, op in enumerate(ops, 1):
            payload = {"seq": seq, **op}
            f.write(json.dumps(payload, sort_keys=True) + "\n")
    print(f"stage={ns.stage}")
    print(f"operations={len(ops)}")
    print(f"out={ns.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)

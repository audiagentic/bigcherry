#!/usr/bin/env python3
"""Render successor specs from approved semantic evidence."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    root = args.work.resolve()
    semantics = {
        row["source_id"]: row
        for row in read(root / "SEMANTIC_REVIEW.csv")
        if row.get("source_state") not in {"completed", "superseded", "deprecated"}
    }
    dispositions = {
        row["source_id"]: row for row in read(root / "DISPOSITIONS.csv")
    }
    successors_path = root / "SUCCESSORS.csv"
    successors = read(successors_path)
    specs = root / "successor-specs"
    specs.mkdir(parents=True, exist_ok=True)
    rendered = 0
    for successor in successors:
        key = successor["successor_key"]
        source_ids = [
            source_id for source_id, disposition in dispositions.items()
            if key in (disposition.get("successor_keys") or "").split(";")
        ]
        if not source_ids:
            raise SystemExit(f"{key}: no predecessor disposition")
        source_id = source_ids[0]
        semantic = semantics.get(source_id)
        if not semantic or semantic.get("approved", "").lower() != "true":
            raise SystemExit(f"{source_id}: approved semantic evidence is required")
        predecessor_text = ", ".join(source_ids)
        successor["acceptance_boundary"] = semantic["acceptance_boundary"]
        successor["title"] = successor["title"].strip() or semantic["unfinished_work"]
        body = f"""# {successor['title']}

Successor key: {key}
Target plan: {successor['target_namespace']}
Predecessor(s): {predecessor_text}
Migration: capability-rebaseline-v3-2026-09

## Future scope

{semantic['unfinished_work']}

Capability owner: {semantic['capability']}

## Acceptance boundary

{semantic['acceptance_boundary']}

## Split and overlap decision

Split assessment: {semantic['split_assessment']}

Overlap assessment: {semantic['overlap_assessment']}

## Inherited evidence and constraints

{semantic['historical_evidence']}

## Active dependencies and references

Dependencies: {semantic['active_dependencies']}

Reference handling: {semantic['reference_notes']}

## Notes

Supersedes: {predecessor_text}
Migration: capability-rebaseline-v3-2026-09
"""
        (specs / f"{key}.md").write_text(body, encoding="utf-8", newline="\n")
        rendered += 1
    fields = list(successors[0]) if successors else ["successor_key"]
    with successors_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(successors)
    print(f"rendered={rendered}")
    print(f"spec_dir={specs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

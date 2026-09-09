#!/usr/bin/env python3
"""Approve the reviewed lifecycle/disposition/successor graph fail-closed."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

TERMINAL = {"completed", "superseded", "deprecated"}
CONTINUING = {"successor", "split", "merge"}
RETIREMENT = {"retire-completed", "retire-deprecated", "retain-history"}


def read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def add_field(fields: list[str], name: str) -> None:
    if name not in fields:
        fields.append(name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--reviewer", default="codex-graph-review")
    args = parser.parse_args()
    root = args.work.resolve()
    _, semantic = read(root / "SEMANTIC_REVIEW.csv")
    active = {
        row["source_id"]: row
        for row in semantic
        if row.get("source_state") not in TERMINAL
    }
    if len(active) != 200 or any(row.get("approved", "").lower() != "true" for row in active.values()):
        raise SystemExit("all 200 active semantic rows must be approved first")
    life_path = root / "LIFECYCLE_NORMALIZATION.csv"
    life_fields, lifecycle = read(life_path)
    disp_path = root / "DISPOSITIONS.csv"
    disp_fields, dispositions = read(disp_path)
    succ_path = root / "SUCCESSORS.csv"
    succ_fields, successors = read(succ_path)
    disp_by_id = {row["source_id"]: row for row in dispositions}
    succ_by_key = {row["successor_key"]: row for row in successors}
    if len(disp_by_id) != 524 or len(succ_by_key) != 192:
        raise SystemExit("expected complete 524-item disposition and 192-successor manifests")
    for row in lifecycle:
        source_id = row["source_id"]
        state = row.get("normalized_lifecycle")
        disposition = disp_by_id.get(source_id)
        if not disposition:
            raise SystemExit(f"{source_id}: missing disposition")
        keys = [key for key in (disposition.get("successor_keys") or "").split(";") if key]
        if state == "continuing":
            if disposition.get("disposition") not in CONTINUING or not keys:
                raise SystemExit(f"{source_id}: continuing lifecycle/disposition mismatch")
            for key in keys:
                successor = succ_by_key.get(key)
                if not successor or not successor.get("spec_path"):
                    raise SystemExit(f"{source_id}: missing successor spec for {key}")
                spec = root / successor["spec_path"]
                if not spec.exists():
                    raise SystemExit(f"{key}: successor spec missing")
                text = spec.read_text(encoding="utf-8")
                if "Draft successor seeded" in text or "Define an independent" in text:
                    raise SystemExit(f"{key}: successor spec still contains draft placeholders")
        elif state in {"terminal", "history", "terminal/no-successor"}:
            if disposition.get("disposition") not in RETIREMENT or keys:
                raise SystemExit(f"{source_id}: terminal lifecycle/disposition mismatch")
        else:
            raise SystemExit(f"{source_id}: unresolved lifecycle {state!r}")
        row["approved"] = "true"
        add_field(life_fields, "reviewed_by")
        add_field(life_fields, "approval_basis")
        row["reviewed_by"] = args.reviewer
        row["approval_basis"] = "SEMANTIC_APPROVAL_2026-09-09.json"
    for row in dispositions:
        row["approved"] = "true"
        add_field(disp_fields, "reviewed_by")
        add_field(disp_fields, "approval_basis")
        row["reviewed_by"] = args.reviewer
        row["approval_basis"] = "SEMANTIC_APPROVAL_2026-09-09.json"
    for row in successors:
        row["approved"] = "true"
        add_field(succ_fields, "reviewed_by")
        add_field(succ_fields, "approval_basis")
        row["reviewed_by"] = args.reviewer
        row["approval_basis"] = "SEMANTIC_APPROVAL_2026-09-09.json"
    write(life_path, life_fields, lifecycle)
    write(disp_path, disp_fields, dispositions)
    write(succ_path, succ_fields, successors)
    print(f"approved_lifecycle={len(lifecycle)}")
    print(f"approved_dispositions={len(dispositions)}")
    print(f"approved_successors={len(successors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

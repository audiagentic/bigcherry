#!/usr/bin/env python3
"""Apply an advisory seven-field semantic review without approving it."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

FIELDS = [
    "unfinished_work", "acceptance_boundary", "capability", "split_assessment",
    "overlap_assessment", "historical_evidence", "active_dependencies", "reference_notes",
]
PLACEHOLDERS = {"", "tbd", "todo", "adjudicate", "unreviewed"}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        out = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        out.writeheader()
        out.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--review", type=Path, required=True)
    ap.add_argument("--reviewer", required=True)
    ns = ap.parse_args()
    review = read(ns.review.resolve())
    by_id = {row.get("source_id", ""): row for row in review}
    if len(by_id) != len(review):
        raise SystemExit("semantic review contains duplicate or empty source IDs")
    for source_id, row in by_id.items():
        malformed = [field for field in FIELDS if row.get(field) is None]
        if malformed:
            raise SystemExit(
                f"{source_id}: malformed CSV row; missing columns {malformed}"
            )
        missing = [field for field in FIELDS if row.get(field, "").strip().lower() in PLACEHOLDERS]
        if missing:
            raise SystemExit(f"{source_id}: missing semantic fields {missing}")
    path = ns.work.resolve() / "SEMANTIC_REVIEW.csv"
    rows = read(path)
    active_ids = {row["source_id"] for row in rows if row.get("source_state") not in {"completed", "superseded", "deprecated"}}
    if set(by_id) != active_ids:
        raise SystemExit(f"review coverage mismatch: expected {len(active_ids)} active IDs, got {len(by_id)}")
    fields = list(rows[0]) if rows else ["source_id"]
    for row in rows:
        incoming = by_id.get(row["source_id"])
        if not incoming:
            continue
        for field in FIELDS:
            row[field] = incoming[field].strip()
        row["reviewed_by"] = ns.reviewer
        row["approved"] = "false"
    write(path, fields, rows)
    print(f"applied={len(by_id)}")
    print(f"semantic_review={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

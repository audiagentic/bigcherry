#!/usr/bin/env python3
"""Approve a complete semantic review through an explicit evidence gate."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FIELDS = (
    "unfinished_work",
    "acceptance_boundary",
    "capability",
    "split_assessment",
    "overlap_assessment",
    "historical_evidence",
    "active_dependencies",
    "reference_notes",
)
PLACEHOLDERS = {"", "tbd", "todo", "adjudicate", "unreviewed"}
TERMINAL = {"completed", "superseded", "deprecated"}


def read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--basis", type=Path, required=True)
    args = parser.parse_args()
    basis = json.loads(args.basis.resolve().read_text(encoding="utf-8"))
    if basis.get("decision") != "accept_as_primary":
        raise SystemExit("approval basis decision must be accept_as_primary")
    reviewer = str(basis.get("reviewer", "")).strip()
    if not reviewer:
        raise SystemExit("approval basis reviewer is required")
    path = args.work.resolve() / "SEMANTIC_REVIEW.csv"
    fields, rows = read(path)
    if "approved" not in fields:
        raise SystemExit("SEMANTIC_REVIEW.csv is missing approved")
    active = [row for row in rows if row.get("source_state") not in TERMINAL]
    if len(active) != 200:
        raise SystemExit(f"expected 200 active rows, found {len(active)}")
    for row in active:
        if row.get("reviewed_by", "").strip() != reviewer:
            raise SystemExit(
                f"{row.get('source_id')}: reviewer mismatch "
                f"{row.get('reviewed_by')!r} != {reviewer!r}"
            )
        malformed = [field for field in FIELDS if row.get(field) is None]
        missing = [
            field for field in FIELDS
            if (row.get(field) or "").strip().lower() in PLACEHOLDERS
        ]
        if malformed or missing:
            raise SystemExit(
                f"{row.get('source_id')}: incomplete semantic evidence "
                f"malformed={malformed} missing={missing}"
            )
        row["approved"] = "true"
    if "approval_basis" not in fields:
        fields.append("approval_basis")
    for row in active:
        row["approval_basis"] = args.basis.resolve().name
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"approved_active={len(active)}")
    print(f"semantic_review={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

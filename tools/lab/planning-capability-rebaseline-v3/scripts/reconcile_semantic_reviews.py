#!/usr/bin/env python3
"""Compare independent semantic review CSVs without selecting a winner."""
from __future__ import annotations

import argparse
import csv
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


def read(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        required = {"source_id", *FIELDS}
        missing = required - set(rows.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        result: dict[str, dict[str, str]] = {}
        for row in rows:
            source_id = (row.get("source_id") or "").strip()
            if not source_id or source_id in result:
                raise ValueError(f"{path}: duplicate or empty source_id: {source_id!r}")
            result[source_id] = row
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--independent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    primary = read(args.primary.resolve())
    independent = read(args.independent.resolve())
    rows: list[dict[str, str]] = []
    for source_id in sorted(set(primary) | set(independent)):
        left = primary.get(source_id, {})
        right = independent.get(source_id, {})
        paired = bool(left and right)
        row = {
            "source_id": source_id,
            "paired": "yes" if paired else "no",
        }
        for field in FIELDS:
            lv = left.get(field, "")
            rv = right.get(field, "")
            row[f"primary_{field}"] = lv
            row[f"independent_{field}"] = rv
            row[f"{field}_agreement"] = (
                "yes" if paired and lv == rv else "no" if paired else "unpaired"
            )
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["source_id", "paired"]
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    paired_rows = [row for row in rows if row["paired"] == "yes"]
    print(f"compared={len(rows)}")
    print(f"primary_rows={len(primary)}")
    print(f"independent_rows={len(independent)}")
    print(f"paired_rows={len(paired_rows)}")
    for field in FIELDS:
        key = f"{field}_agreement"
        print(f"{field}_agreements={sum(row[key] == 'yes' for row in paired_rows)}")
        print(f"{field}_conflicts={sum(row[key] == 'no' for row in paired_rows)}")
    print(f"unpaired={sum(row['paired'] == 'no' for row in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

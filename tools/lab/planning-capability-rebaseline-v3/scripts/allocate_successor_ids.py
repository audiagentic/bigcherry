#!/usr/bin/env python3
"""Allocate deterministic IDs for the approved successor manifest."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    path = args.work.resolve() / "SUCCESSORS.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if any(row.get("allocated_id") for row in rows):
        raise SystemExit("successor IDs already contain allocations")
    counts: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        prefix = row["id_prefix"]
        counts[prefix] += 1
        if row.get("work") == "XL":
            row["work"] = "L"
        row["allocated_id"] = f"{prefix}{counts[prefix]:02d}"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"allocated={len(rows)}")
    for prefix in sorted(counts):
        print(f"{prefix}={counts[prefix]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

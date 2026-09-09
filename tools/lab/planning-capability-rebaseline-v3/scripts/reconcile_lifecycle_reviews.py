#!/usr/bin/env python3
"""Compare independent lifecycle review evidence without choosing silently."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return {row["source_id"]: row for row in csv.DictReader(f)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", type=Path, required=True)
    ap.add_argument("--independent", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ns = ap.parse_args()
    primary = read(ns.primary.resolve())
    independent = read(ns.independent.resolve())
    rows: list[dict[str, str]] = []
    for source_id in sorted(set(primary) | set(independent)):
        p = primary.get(source_id, {})
        i = independent.get(source_id, {})
        pc = p.get("classification", "")
        ic = i.get("classification", "")
        rows.append({
            "source_id": source_id,
            "primary_classification": pc,
            "independent_classification": ic,
            "agreement": "yes" if pc and ic and pc == ic else "no" if pc and ic else "unpaired",
            "primary_evidence": p.get("evidence", ""),
            "independent_evidence": i.get("evidence", ""),
        })
    ns.output.parent.mkdir(parents=True, exist_ok=True)
    with ns.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["source_id"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"compared={len(rows)}")
    print(f"agreements={sum(r['agreement'] == 'yes' for r in rows)}")
    print(f"conflicts={sum(r['agreement'] == 'no' for r in rows)}")
    print(f"unpaired={sum(r['agreement'] == 'unpaired' for r in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

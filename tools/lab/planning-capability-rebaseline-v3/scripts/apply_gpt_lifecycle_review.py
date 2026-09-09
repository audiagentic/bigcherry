#!/usr/bin/env python3
"""Apply a dated GPT lifecycle classification to the working review manifests.

This only records evidence and normalizes the lifecycle hint. It never approves
semantic review, changes dispositions, allocates IDs, or mutates plan files.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

CLASSIFICATIONS = {"CONTINUING", "TERMINAL-COMPLETE", "TERMINAL-DEPRECATED", "AMBIGUOUS"}


def read(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        out = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        out.writeheader()
        out.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--evidence", type=Path, required=True)
    ns = ap.parse_args()
    work = ns.work.resolve()
    evidence = {r["source_id"]: r for r in read(ns.evidence.resolve())}
    if len(evidence) != len(read(ns.evidence.resolve())):
        raise ValueError("GPT evidence contains duplicate source_id values")
    for source_id, row in evidence.items():
        if row.get("classification") not in CLASSIFICATIONS:
            raise ValueError(f"{source_id}: invalid classification {row.get('classification')!r}")
        if not row.get("evidence", "").strip():
            raise ValueError(f"{source_id}: evidence is required")

    lifecycle_path = work / "LIFECYCLE_NORMALIZATION.csv"
    lifecycle = read(lifecycle_path)
    lifecycle_ids = {r["source_id"] for r in lifecycle}
    unknown = sorted(set(evidence) - lifecycle_ids)
    if unknown:
        raise ValueError(f"GPT evidence references unknown source IDs: {unknown}")
    lifecycle_fields = list(lifecycle[0]) if lifecycle else ["source_id"]
    for row in lifecycle:
        review = evidence.get(row["source_id"])
        if not review:
            continue
        classification = review["classification"]
        normalized = {
            "CONTINUING": "continuing",
            "TERMINAL-COMPLETE": "terminal/no-successor",
            "TERMINAL-DEPRECATED": "terminal/no-successor",
            "AMBIGUOUS": "adjudicate",
        }.get(classification, "adjudicate")
        row["normalized_lifecycle"] = normalized
        row["reason"] = f"GPT lifecycle review ({classification}): {review['evidence']}"
        # Classification evidence is advisory until all seven semantic review
        # fields and the resulting disposition are approved by the migration.
        row["approved"] = "false"
    write(lifecycle_path, lifecycle_fields, lifecycle)

    semantic_path = work / "SEMANTIC_REVIEW.csv"
    semantic = read(semantic_path)
    semantic_fields = list(semantic[0]) if semantic else ["source_id"]
    for row in semantic:
        review = evidence.get(row["source_id"])
        if not review:
            continue
        row["unfinished_work"] = review["evidence"]
        row["reviewed_by"] = "dev-gpt-agent"
        row["approved"] = "false"
    write(semantic_path, semantic_fields, semantic)
    print(f"applied={len(evidence)}")
    print(f"lifecycle={lifecycle_path}")
    print(f"semantic_review={semantic_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

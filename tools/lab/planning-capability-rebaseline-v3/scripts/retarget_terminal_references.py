#!/usr/bin/env python3
"""Remove invalid successor targets after terminal lifecycle alignment."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

REMOVED = {
    "patching-rdna-boost-experiments-rd39",
    "patching-rdna-boost-experiments-rd89",
    "patching-hip-autotune-hi145",
    "build-hip-autotune-hi150",
    "tuning-hip-autotune-hi143",
    "run-hip-autotune-hi132",
    "run-hip-autotune-hi166",
    "patching-hip-autotune-hi154",
}


def read(path: Path, delimiter: str) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=delimiter)
        return list(reader.fieldnames or []), list(reader)


def write(path: Path, fields: list[str], rows: list[dict[str, str]], delimiter: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, delimiter=delimiter, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    root = args.work.resolve()
    ref_path = root / "REFERENCE_DECISIONS.tsv"
    ref_fields, refs = read(ref_path, "\t")
    ref_changed = 0
    for row in refs:
        target = row.get("target_successor_key_or_id")
        if row.get("decision") == "preserve" and row.get("semantic_class") == "historical_reference":
            row["semantic_class"] = "historical_provenance"
            ref_changed += 1
        if target not in REMOVED and not (
            row.get("old_ref") == "RD39" and row.get("decision") == "preserve"
        ):
            continue
        row["target_successor_key_or_id"] = ""
        if row.get("old_ref") == "RD39":
            row["decision"] = "remove"
            row["rationale"] = (
                "RD39 is deprecated by the frozen lifecycle review; remove the "
                "active forward edge rather than preserve a dead execution target."
            )
            row["action"] = "remove"
            row["semantic_class"] = "active_scope"
        else:
            row["decision"] = "preserve"
            row["rationale"] = (
                "Referenced predecessor is terminal/no-successor; preserve historical "
                "provenance rather than rewrite to an unallocated successor."
            )
            row["action"] = "preserve_predecessor"
            row["semantic_class"] = "historical_provenance"
        ref_changed += 1
    dep_path = root / "DEPENDENCY_REMAP.tsv"
    dep_fields, deps = read(dep_path, "\t")
    dep_changed = 0
    for row in deps:
        if row.get("target_successor_key_or_id") not in REMOVED:
            continue
        row["target_successor_key_or_id"] = ""
        if row.get("old_dependency_id") == "RD39":
            row["decision"] = "remove"
            row["rationale"] = (
                "RD39 is deprecated; remove the active dependency edge without "
                "creating an unallocated successor target."
            )
        else:
            row["decision"] = "preserve"
            row["rationale"] = (
                "Dependency points to a terminal/no-successor predecessor; retain the "
                "historical dependency record without creating an active edge."
            )
        dep_changed += 1
    write(ref_path, ref_fields, refs, "\t")
    write(dep_path, dep_fields, deps, "\t")
    print(f"reference_rows_reclassified={ref_changed}")
    print(f"dependency_rows_reclassified={dep_changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

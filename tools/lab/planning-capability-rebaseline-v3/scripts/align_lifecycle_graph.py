#!/usr/bin/env python3
"""Align seeded dispositions/successors with the reviewed lifecycle gate."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    root = args.work.resolve()
    life_fields, lifecycle = read(root / "LIFECYCLE_NORMALIZATION.csv")
    disp_path = root / "DISPOSITIONS.csv"
    disp_fields, dispositions = read(disp_path)
    succ_path = root / "SUCCESSORS.csv"
    succ_fields, successors = read(succ_path)
    lineage_path = root / "LINEAGE.csv"
    lineage_fields, lineage = read(lineage_path)
    life_by_id = {row["source_id"]: row for row in lifecycle}
    removed_keys: set[str] = set()
    for disposition in dispositions:
        life = life_by_id[disposition["source_id"]]
        if life.get("normalized_lifecycle") != "terminal/no-successor":
            continue
        old_keys = [key for key in (disposition.get("successor_keys") or "").split(";") if key]
        removed_keys.update(old_keys)
        disposition["successor_keys"] = ""
        disposition["capability"] = ""
        disposition["target_namespace"] = ""
        if life.get("source_state") in {"completed", "superseded", "deprecated"}:
            disposition["disposition"] = "retain-history"
            disposition["final_state"] = life["source_state"]
            disposition["reason"] = "Frozen terminal state; retain predecessor history."
            disposition["scope_carry_forward"] = ""
            disposition["scope_retired"] = "Historical predecessor retained; no active successor."
        elif "TERMINAL-DEPRECATED" in life.get("reason", ""):
            disposition["disposition"] = "retire-deprecated"
            disposition["final_state"] = "deprecated"
            disposition["reason"] = life["reason"]
            disposition["scope_carry_forward"] = ""
            disposition["scope_retired"] = "Frozen scope is intentionally deprecated."
        else:
            disposition["disposition"] = "retire-completed"
            disposition["final_state"] = "completed"
            disposition["reason"] = life["reason"]
            disposition["scope_carry_forward"] = ""
            disposition["scope_retired"] = "Frozen acceptance boundary is complete; no successor."
        disposition["approved"] = "false"
    kept_successors = [
        row for row in successors if row["successor_key"] not in removed_keys
    ]
    kept_lineage = [
        row for row in lineage if row["successor_key"] not in removed_keys
    ]
    write(disp_path, disp_fields, dispositions)
    write(succ_path, succ_fields, kept_successors)
    write(lineage_path, lineage_fields, kept_lineage)
    print(f"terminal_aligned={len(removed_keys)}")
    print(f"successors_remaining={len(kept_successors)}")
    print(f"lineage_edges_remaining={len(kept_lineage)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

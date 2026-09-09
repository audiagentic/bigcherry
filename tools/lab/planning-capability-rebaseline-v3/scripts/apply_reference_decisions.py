#!/usr/bin/env python3
"""Apply exact occurrence-based reference/dependency decisions.

The default mode is dry-run. Edits are made only when the frozen source line,
old reference, and context identify one unique current occurrence.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def current_path(root: Path, value: str) -> Path | None:
    direct = root / value
    if direct.exists():
        return direct
    if value.startswith("docs/planning/active/"):
        completed = root / value.replace("docs/planning/active/", "docs/planning/completed/", 1)
        if completed.exists():
            return completed
    return None


def candidates(lines: list[str], old_ref: str, context: str, frozen_line: int) -> list[int]:
    hits = [index for index, line in enumerate(lines) if old_ref and old_ref in line]
    if len(hits) <= 1:
        return hits
    context_key = (context or "").strip()
    if context_key:
        # Inventory contexts are whitespace-normalized; current files retain
        # their original wrapping/spacing.  Compare the same normalized form.
        normalized = lambda value: " ".join(value.strip().split())
        contextual = [index for index in hits if context_key in normalized(lines[index])]
        if contextual:
            # The frozen line is the occurrence identity.  Prefer it even
            # when an identical sentence appears elsewhere in the item.
            frozen_index = frozen_line - 1
            if frozen_index in contextual:
                return [frozen_index]
            return contextual
    return sorted(hits, key=lambda index: abs((index + 1) - frozen_line))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo.resolve()
    work = args.work.resolve()
    successors = {
        row["successor_key"]: row.get("allocated_id", "")
        for row in read(work / "SUCCESSORS.csv")
    }
    operations: list[dict[str, str]] = []
    operations.extend(read(work / "REFERENCE_DECISIONS.tsv", "\t"))
    dependency_rows = read(work / "DEPENDENCY_REMAP.tsv", "\t")
    inventory = {
        row["source_id"]: row["source_path"]
        for row in read(work / "PLAN_INVENTORY.csv")
    }
    for row in dependency_rows:
        row = dict(row)
        row["_source_path"] = inventory.get(row.get("owner_source_id", ""), "")
        row["_kind"] = "dependency"
        operations.append(row)
    report: list[dict[str, object]] = []
    counters = Counter()
    pending: dict[Path, list[str]] = {}
    for row in operations:
        kind = row.get("_kind", "reference")
        source_path = row.get("_source_path") or row.get("source_path", "")
        path = current_path(root, source_path)
        old_ref = row.get("old_ref") or row.get("old_dependency_id", "")
        decision = row.get("decision", "")
        target_key = row.get("target_successor_key_or_id", "")
        target = successors.get(target_key, "") or target_key
        if decision not in {"rewrite", "remove"}:
            counters["preserved"] += 1
            continue
        if not path:
            counters["missing_path"] += 1
            report.append({"status": "missing_path", "kind": kind, "source_path": source_path, "old_ref": old_ref})
            continue
        lines = pending.setdefault(path, path.read_text(encoding="utf-8").splitlines(keepends=True))
        frozen_line = int(row.get("line", "0") or "0")
        # Front matter identity is canonical plan metadata, not a semantic
        # reference.  Never rewrite an item's own id/plan fields.
        if frozen_line and frozen_line <= len(lines):
            identity_line = lines[frozen_line - 1].lstrip()
            if identity_line.startswith(("id:", "plan:")):
                counters["preserved"] += 1
                continue
        hits = candidates(lines, old_ref, row.get("context", ""), frozen_line)
        if len(hits) != 1:
            counters["ambiguous" if hits else "not_found"] += 1
            report.append({
                "status": "ambiguous" if hits else "not_found",
                "kind": kind,
                "source_path": str(path.relative_to(root)),
                "old_ref": old_ref,
                "hits": [index + 1 for index in hits],
            })
            continue
        index = hits[0]
        replacement = target if decision == "rewrite" else ""
        lines[index] = lines[index].replace(old_ref, replacement, 1)
        counters["applied"] += 1
        report.append({
            "status": "applied",
            "kind": kind,
            "source_path": str(path.relative_to(root)),
            "line": index + 1,
            "old_ref": old_ref,
            "replacement": replacement,
        })
    if args.apply:
        for path, lines in pending.items():
            path.write_text("".join(lines), encoding="utf-8", newline="\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"apply": args.apply, "counts": counters, "rows": report}, indent=2), encoding="utf-8")
    print(" ".join(f"{key}={value}" for key, value in sorted(counters.items())))
    print(f"report={args.report}")
    return 0 if not counters["ambiguous"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

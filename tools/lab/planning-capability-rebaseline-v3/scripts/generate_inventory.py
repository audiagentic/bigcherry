#!/usr/bin/env python3
"""Generate a frozen, non-mutating planning inventory for capability rebaseline v3.

Reads plan files directly from the Git object named by SOURCE_LOCK.json using
`git ls-tree` and `git show`. It never checks out or edits the source snapshot.
Only generated review files under --output are written.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

TERMINAL_STATES = {"completed", "superseded", "deprecated"}
PLAN_PATH_RE = re.compile(r"docs/planning/(?:active|completed)/[A-Za-z0-9_.\-/]+\.md")
LIFECYCLE_PLAN_RE = re.compile(
    r"^docs/planning/(?:active|completed)/[^/]+/[A-Z]+[0-9]+\.md$"
)


@dataclass(frozen=True)
class PlanItem:
    item_id: str
    plan: str
    state: str
    path: str
    title: str
    work: str
    skill: str
    priority: str
    content_hash: str
    content: str


def run_git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        encoding="utf-8",
        errors="strict",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def find_repo(start: Path) -> Path:
    out = run_git(start, "rev-parse", "--show-toplevel").strip()
    return Path(out).resolve()


def parse_frontmatter(text: str) -> tuple[dict[str, str], int]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, 0
    data: dict[str, str] = {}
    end = 0
    for idx in range(1, len(lines)):
        line = lines[idx]
        if line.strip() == "---":
            end = idx + 1
            break
        if ":" not in line or line[:1].isspace():
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        data[key.strip()] = value
    return data, end


def first_heading(text: str, start_line: int) -> str:
    for line in text.splitlines()[start_line:]:
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def load_items(repo: Path, source_commit: str) -> list[PlanItem]:
    names = run_git(
        repo,
        "ls-tree",
        "-r",
        "--name-only",
        source_commit,
        "--",
        "docs/planning/active",
        "docs/planning/completed",
    ).splitlines()
    items: list[PlanItem] = []
    # Only a direct historical-topic/<PLAN_ID>.md entry is a lifecycle plan.
    # Nested reviews/deep-dives/support markdown may carry frontmatter IDs but
    # remain supporting material and must not become migration predecessors.
    for path in sorted(
        p for p in names if p.endswith(".md") and LIFECYCLE_PLAN_RE.match(p)
    ):
        text = run_git(repo, "show", f"{source_commit}:{path}")
        fm, end = parse_frontmatter(text)
        item_id = fm.get("id", "").strip()
        if not item_id:
            continue
        items.append(
            PlanItem(
                item_id=item_id,
                plan=fm.get("plan", "").strip(),
                state=fm.get("state", "").strip(),
                path=path,
                title=first_heading(text, end),
                work=fm.get("work", "").strip(),
                skill=fm.get("skill", "").strip(),
                priority=fm.get("priority", "").strip(),
                content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                content=text,
            )
        )
    return items


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]], delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def context(line: str, limit: int = 360) -> str:
    compact = " ".join(line.strip().split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--force", action="store_true", help="overwrite review manifests as well as generated inventories")
    ns = ap.parse_args()

    pack = ns.pack.resolve()
    output = ns.output.resolve()
    lock = json.loads((pack / "SOURCE_LOCK.json").read_text(encoding="utf-8"))
    source_commit = lock["source_commit"]
    migration_id = lock["migration_id"]
    repo = find_repo(ns.repo.resolve())

    resolved = run_git(repo, "rev-parse", f"{source_commit}^{{commit}}").strip()
    if resolved != source_commit:
        raise RuntimeError(f"source lock resolved to unexpected commit: {resolved}")

    items = load_items(repo, source_commit)
    if not items:
        raise RuntimeError("no lifecycle plan items found")

    ids = [item.item_id for item in items]
    if len(ids) != len(set(ids)):
        dups = sorted({x for x in ids if ids.count(x) > 1})
        raise RuntimeError(f"duplicate plan IDs in frozen source: {dups}")

    item_by_id = {item.item_id: item for item in items}
    id_pattern = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(re.escape(i) for i in sorted(ids, key=len, reverse=True)) + r")(?![A-Za-z0-9_])")

    inventory_rows: list[dict[str, str]] = []
    queue_rows: list[dict[str, str]] = []
    dispositions: list[dict[str, str]] = []
    reference_rows: list[dict[str, str]] = []

    for item in items:
        refs = 0
        for line_no, line in enumerate(item.content.splitlines(), 1):
            seen: set[tuple[str, str]] = set()
            for match in id_pattern.finditer(line):
                old_ref = match.group(1)
                key = ("plan_id", old_ref)
                if key in seen:
                    continue
                seen.add(key)
                refs += 1
                reference_rows.append(
                    {
                        "occurrence_id": hashlib.sha256(
                            f"{source_commit}:{item.path}:{line_no}:{match.start()}:{old_ref}".encode("utf-8")
                        ).hexdigest()[:16],
                        "source_path": item.path,
                        "source_id": item.item_id,
                        "line": str(line_no),
                        "ref_type": "plan_id",
                        "old_ref": old_ref,
                        "target_source_path": item_by_id[old_ref].path,
                        "context_hash": hashlib.sha256(context(line).encode("utf-8")).hexdigest(),
                        "context": context(line),
                    }
                )
            for match in PLAN_PATH_RE.finditer(line):
                old_ref = match.group(0)
                key = ("plan_path", old_ref)
                if key in seen:
                    continue
                seen.add(key)
                refs += 1
                reference_rows.append(
                    {
                        "occurrence_id": hashlib.sha256(
                            f"{source_commit}:{item.path}:{line_no}:{match.start()}:{old_ref}".encode("utf-8")
                        ).hexdigest()[:16],
                        "source_path": item.path,
                        "source_id": item.item_id,
                        "line": str(line_no),
                        "ref_type": "plan_path",
                        "old_ref": old_ref,
                        "target_source_path": old_ref,
                        "context_hash": hashlib.sha256(context(line).encode("utf-8")).hexdigest(),
                        "context": context(line),
                    }
                )

        inventory_rows.append(
            {
                "source_id": item.item_id,
                "source_plan": item.plan,
                "source_path": item.path,
                "source_state": item.state,
                "title": item.title,
                "work": item.work,
                "skill": item.skill,
                "priority": item.priority,
                "source_hash": item.content_hash,
                "reference_count": str(refs),
            }
        )
        queue_rows.append(
            {
                "source_id": item.item_id,
                "source_state": item.state,
                "source_plan": item.plan,
                "title": item.title,
                "source_path": item.path,
                "review_status": "UNREVIEWED",
            }
        )
        dispositions.append(
            {
                "source_id": item.item_id,
                "source_plan": item.plan,
                "source_path": item.path,
                "source_state": item.state,
                "source_hash": item.content_hash,
                "disposition": "UNREVIEWED",
                "final_state": "",
                "capability": "",
                "target_namespace": "",
                "successor_keys": "",
                "reason": "",
                "scope_carry_forward": "",
                "scope_retired": "",
                "reviewed_by": "",
                "approved": "false",
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    (output / "SOURCE_LOCK.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    write_csv(
        output / "PLAN_INVENTORY.csv",
        ["source_id", "source_plan", "source_path", "source_state", "title", "work", "skill", "priority", "source_hash", "reference_count"],
        inventory_rows,
    )
    write_csv(
        output / "LIFECYCLE_NORMALIZATION.csv",
        ["source_id", "source_state", "normalized_lifecycle", "reason", "approved"],
        [
            {
                "source_id": item.item_id,
                "source_state": item.state,
                "normalized_lifecycle": "terminal/no-successor" if item.state in TERMINAL_STATES else "adjudicate",
                "reason": "Frozen terminal state; retain predecessor history." if item.state in TERMINAL_STATES else "Active metadata requires explicit semantic completion/continuation review before successor allocation.",
                "approved": "true" if item.state in TERMINAL_STATES else "false",
            }
            for item in items
        ],
    )
    queue_rows.sort(key=lambda r: (r["source_state"] in TERMINAL_STATES, r["source_plan"], r["source_id"]))
    write_csv(
        output / "SEMANTIC_REVIEW_QUEUE.csv",
        ["source_id", "source_state", "source_plan", "title", "source_path", "review_status"],
        queue_rows,
    )
    write_csv(
        output / "PLAN_REFERENCES.tsv",
        ["occurrence_id", "source_path", "source_id", "line", "ref_type", "old_ref", "target_source_path", "context_hash", "context"],
        reference_rows,
        delimiter="\t",
    )

    def copy_or_write(name: str, writer) -> None:
        dst = output / name
        if dst.exists() and not ns.force:
            return
        writer(dst)

    copy_or_write(
        "DISPOSITIONS.csv",
        lambda dst: write_csv(
            dst,
            ["source_id", "source_plan", "source_path", "source_state", "source_hash", "disposition", "final_state", "capability", "target_namespace", "successor_keys", "reason", "scope_carry_forward", "scope_retired", "reviewed_by", "approved"],
            dispositions,
        ),
    )

    copy_or_write(
        "SEMANTIC_REVIEW.csv",
        lambda dst: write_csv(
            dst,
            [
                "source_id", "unfinished_work", "acceptance_boundary", "capability",
                "split_assessment", "overlap_assessment", "historical_evidence",
                "active_dependencies", "reference_notes", "reviewed_by", "approved",
            ],
            [
                {
                    "source_id": item.item_id,
                    "unfinished_work": "",
                    "acceptance_boundary": "",
                    "capability": "",
                    "split_assessment": "",
                    "overlap_assessment": "",
                    "historical_evidence": "",
                    "active_dependencies": "",
                    "reference_notes": "",
                    "reviewed_by": "",
                    "approved": "false",
                }
                for item in items
            ],
        ),
    )

    for filename in ["NAMESPACES.csv", "SUCCESSORS.csv", "LINEAGE.csv", "DEPENDENCY_REMAP.tsv"]:
        src = pack / "templates" / filename
        copy_or_write(filename, lambda dst, src=src: shutil.copyfile(src, dst))

    ref_decision_path = output / "REFERENCE_DECISIONS.tsv"
    if not ref_decision_path.exists() or ns.force:
        decision_rows = [
            {
                "source_path": r["source_path"],
                "occurrence_id": r["occurrence_id"],
                "line": r["line"],
                "source_id": r["source_id"],
                "old_ref": r["old_ref"],
                "reference_kind": "ambiguous",
                "decision": "",
                "target_successor_key_or_id": "",
                "rationale": "",
                "semantic_class": "unclassified",
                "action": "unclassified",
                "context_hash": r["context_hash"],
                "context": r["context"],
            }
            for r in reference_rows
        ]
        write_csv(
            ref_decision_path,
            ["occurrence_id", "source_path", "source_id", "line", "old_ref", "reference_kind", "decision", "target_successor_key_or_id", "rationale", "semantic_class", "action", "context_hash", "context"],
            decision_rows,
            delimiter="\t",
        )

    successor_specs = output / "successor-specs"
    successor_specs.mkdir(exist_ok=True)
    (output / "GENERATED.json").write_text(
        json.dumps(
            {
                "migration_id": migration_id,
                "source_commit": source_commit,
                "plan_item_count": len(items),
                "reference_occurrence_count": len(reference_rows),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"source_commit={source_commit}")
    print(f"plan_items={len(items)}")
    print(f"reference_occurrences={len(reference_rows)}")
    print(f"output={output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)

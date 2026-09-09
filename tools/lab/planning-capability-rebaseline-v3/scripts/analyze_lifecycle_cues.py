#!/usr/bin/env python3
"""Produce review hints for active items whose prose may be terminal."""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path


TERMINAL = {"completed", "superseded", "deprecated"}
CUES = re.compile(
    r"\b(?:objectives?\s+(?:are|is)\s+(?:satisfied|complete)|"
    r"all\s+(?:work|steps|objectives).*\b(?:done|complete)|"
    r"no\s+(?:remaining|further)\s+(?:work|scope)|"
    r"already\s+(?:done|complete|implemented)|"
    r"\b(?:retired|closed|superseded)\b)",
    re.IGNORECASE,
)


def git_show(repo: Path, commit: str, path: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), "show", f"{commit}:{path}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="strict",
        check=False,
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip())
    return proc.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    with args.inventory.open(encoding="utf-8", newline="") as handle:
        items = list(csv.DictReader(handle))
    for item in items:
        if item["source_state"] in TERMINAL:
            continue
        text = git_show(args.repo.resolve(), args.commit, item["source_path"])
        matches = [line.strip() for line in text.splitlines() if CUES.search(line)]
        rows.append({
            "source_id": item["source_id"],
            "source_state": item["source_state"],
            "title": item["title"],
            "cue_count": str(len(matches)),
            "cues": " | ".join(matches[:5]),
            "review_recommendation": "adjudicate" if matches else "review-continuing",
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["source_id"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"active_items={len(rows)} cue_items={sum(row['review_recommendation'] == 'adjudicate' for row in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fail-closed audit for semantic carry-forward into live successor plans.

The structural rebaseline validator proves identity and graph integrity.  This
audit checks the complementary invariant that a live successor is actionable
without opening its terminal predecessor: generic migration scaffolding and
``successor-specs/...`` placeholders are findings unless the item explicitly
records a disposition or is marked as reference-only.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

TERMINAL = {"completed", "superseded", "deprecated"}
ACTIVE = {"pending", "in_progress"}
GENERIC_STEPS = "Implement the still-valid future scope."
GENERIC_ACCEPTANCE = "Close the recorded gap and pass the frozen scope's stated implementation, correctness, performance, or evidence gate."
PLACEHOLDER = re.compile(r"(?:^|\s)successor-specs/[^\s]+")
FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
FIELD = re.compile(r"^(?P<key>[A-Za-z][A-Za-z0-9_-]*):\s*(?P<value>.*)$")


@dataclass
class Finding:
    item_id: str
    path: str
    kind: str
    detail: str


def frontmatter(text: str) -> dict[str, str]:
    match = FRONTMATTER.match(text)
    if not match:
        return {}
    values: dict[str, str] = {}
    for line in match.group(1).splitlines():
        field = FIELD.match(line)
        if field:
            values[field.group("key")] = field.group("value").strip().strip("'")
    return values


def section(text: str, heading: str) -> str:
    marker = f"## {heading}"
    start = text.find(marker)
    if start < 0:
        return ""
    body = text[start + len(marker):]
    next_heading = body.find("\n## ")
    return body if next_heading < 0 else body[:next_heading]


def explicit_exemption(text: str, fm: dict[str, str]) -> bool:
    # Exemptions are deliberate and local, so they survive regeneration and
    # are visible to reviewers.  They never apply to pending implementation
    # items unless the reason is written in Notes.
    notes = section(text, "Notes").lower()
    return "semantic-carryforward: reference-only" in notes or "semantic-carryforward: dispositioned" in notes


def audit(repo: Path) -> list[Finding]:
    findings: list[Finding] = []
    root = repo / "docs" / "planning" / "active"
    for path in sorted(root.glob("*/*.md")):
        if "reviews" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        fm = frontmatter(text)
        state = fm.get("state", "")
        item_id = fm.get("id", path.stem)
        if state not in ACTIVE or explicit_exemption(text, fm):
            continue
        steps = section(text, "Steps")
        acceptance = section(text, "Acceptance Criteria")
        files = section(text, "Files")
        if GENERIC_STEPS in steps:
            findings.append(Finding(item_id, str(path.relative_to(repo)), "generic-steps", GENERIC_STEPS))
        if GENERIC_ACCEPTANCE in acceptance:
            findings.append(Finding(item_id, str(path.relative_to(repo)), "generic-acceptance", GENERIC_ACCEPTANCE))
        if PLACEHOLDER.search(files):
            findings.append(Finding(item_id, str(path.relative_to(repo)), "placeholder-files", files.strip()))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--report", type=Path)
    parser.add_argument("--allow-findings", action="store_true", help="emit findings but return success")
    args = parser.parse_args()
    findings = audit(args.repo.resolve())
    payload = {"repository": str(args.repo.resolve()), "findings": [asdict(item) for item in findings], "count": len(findings)}
    if args.report:
        args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for item in findings:
        print(f"ERROR: {item.item_id} {item.kind}: {item.detail}")
    print(f"semantic_findings={len(findings)}")
    return 0 if args.allow_findings or not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())

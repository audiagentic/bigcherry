#!/usr/bin/env python3
"""Rewrite known-bad lines in the open release-please PRs (body and changelog file).

Rules (.github/release-line-rewrites.toml) are keyed by commit sha. A release-notes bullet looks like
    * <subject> ([#N](issue-url)) ([abcdef0](https://github.com/OWNER/REPO/commit/<full sha>))
and a rule replaces the <subject> of the bullet whose commit link carries the rule's sha. Idempotent: a bullet already
carrying the rule's text is left alone; a PR with nothing to rewrite is not touched.

Usage (CI): GH_TOKEN=<pat> GITHUB_REPOSITORY=owner/repo rewrite_release_lines.py [rules.toml]
The token must be able to edit pull requests and push to the release branches (the same PAT release-please uses).
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

LABEL = "autorelease: pending"
DEFAULT_RULES = Path(__file__).resolve().parents[1] / "release-line-rewrites.toml"


def load_rules(path: Path) -> dict[str, str]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    rules: dict[str, str] = {}
    for item in data.get("line", []):
        sha, text = item.get("sha", ""), item.get("text", "")
        if len(sha) != 40 or not text.strip() or sha in rules:
            raise ValueError(f"{path}: every [[line]] needs a unique 40-character sha and a non-empty text")
        rules[sha] = text
    return rules


def rewrite(text: str, rules: dict[str, str]) -> str:
    """Apply the rules to every bullet of `text`; lines without a matching commit link are returned unchanged."""
    out = []
    for line in text.split("\n"):
        if line.startswith("* "):
            for sha, replacement in rules.items():
                if f"/commit/{sha}" in line:
                    cut = line.find(" ([#")
                    if cut > 2:
                        line = "* " + replacement + line[cut:]
                    break
        out.append(line)
    return "\n".join(out)


def gh(*args: str, payload: dict | None = None) -> object:
    cmd = ["gh", "api", *args] + (["--input", "-"] if payload is not None else [])
    res = subprocess.run(cmd, input=json.dumps(payload) if payload is not None else None, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"gh api {' '.join(args)} failed: {res.stderr.strip()}")
    return json.loads(res.stdout) if res.stdout.strip() else {}


def main(argv: list[str]) -> int:
    repo = os.environ["GITHUB_REPOSITORY"]
    rules = load_rules(Path(argv[1]) if len(argv) > 1 else DEFAULT_RULES)
    changed = 0
    for pr in gh(f"repos/{repo}/pulls?state=open&per_page=100"):  # type: ignore[union-attr]
        if LABEL not in [lab["name"] for lab in pr["labels"]]:
            continue
        number, ref = pr["number"], pr["head"]["ref"]
        body = pr.get("body") or ""
        new_body = rewrite(body, rules)
        if new_body != body:
            gh(f"repos/{repo}/pulls/{number}", "-X", "PATCH", payload={"body": new_body})
            print(f"#{number}: body rewritten")
            changed += 1
        for f in gh(f"repos/{repo}/pulls/{number}/files?per_page=100"):  # type: ignore[union-attr]
            name = f["filename"]
            if not name.endswith(".md"):
                continue
            blob = gh(f"repos/{repo}/contents/{name}?ref={ref}")
            old = base64.b64decode(blob["content"]).decode("utf-8")  # type: ignore[index]
            new = rewrite(old, rules)
            if new != old:
                gh(f"repos/{repo}/contents/{name}", "-X", "PUT", payload={
                    "message": "chore: rewrite release-note lines listed in .github/release-line-rewrites.toml",
                    "content": base64.b64encode(new.encode("utf-8")).decode("ascii"),
                    "sha": blob["sha"], "branch": ref})  # type: ignore[index]
                print(f"#{number}: {name} rewritten")
                changed += 1
    print(f"rewrite_release_lines: {changed} change(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

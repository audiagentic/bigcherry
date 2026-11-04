#!/usr/bin/env python3
"""Whether a pull request touches any of the given path globs (GitHub `paths` semantics).

Prints `run=true|false` for $GITHUB_OUTPUT. Used by each workflow's `changes` job so path filtering happens
inside the workflow: every job is always created and a job skipped by its `if:` still reports, which a
required status check accepts (a workflow-level `paths:` filter creates no check at all and blocks a merge).
Fails safe: any error prints run=true, so the real job runs.

A pattern line starting with `!` excludes: a changed file counts only if some include pattern matches it and no
exclude pattern does (an exclude always wins, wherever it is listed). A list of only excludes matches nothing.

Usage: changed_paths.py BASE_SHA HEAD_SHA < patterns (one glob a line; `**` crosses directories, `*` does not)
"""
import re
import subprocess
import sys


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    out, i = "", 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out, i = out + "(?:.*/)?", i + 3
        elif pattern.startswith("**", i):
            out, i = out + ".*", i + 2
        elif pattern[i] == "*":
            out, i = out + "[^/]*", i + 1
        elif pattern[i] == "?":
            out, i = out + "[^/]", i + 1
        else:
            out, i = out + re.escape(pattern[i]), i + 1
    return re.compile(out + r"\Z")


def first_match(changed: list[str], patterns: list[str]) -> str | None:
    """The first changed file some include pattern matches and no `!` exclude pattern does."""
    include = [glob_to_regex(p) for p in patterns if not p.startswith("!")]
    exclude = [glob_to_regex(p[1:]) for p in patterns if p.startswith("!")]
    return next((f for f in changed
                 if any(p.match(f) for p in include) and not any(p.match(f) for p in exclude)), None)


def main() -> int:
    try:
        base, head = sys.argv[1], sys.argv[2]
        patterns = [p.strip() for p in sys.stdin if p.strip() and not p.strip().startswith("#")]
        changed = subprocess.run(["git", "diff", "--name-only", f"{base}...{head}"], check=True,
                                 capture_output=True, text=True).stdout.split()
        hit = first_match(changed, patterns)
        print(f"run={'true' if hit else 'false'}")
        print(f"changed {len(changed)} file(s); first match: {hit or '-'}", file=sys.stderr)
    except Exception as exc:  # fail safe: run the checks
        print(f"changed_paths: {exc}; running the checks", file=sys.stderr)
        print("run=true")
    return 0


if __name__ == "__main__":
    sys.exit(main())

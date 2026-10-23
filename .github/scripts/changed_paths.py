#!/usr/bin/env python3
"""Whether a pull request touches any of the given path globs (GitHub `paths` semantics).

Prints `run=true|false` for $GITHUB_OUTPUT. Used by each workflow's `changes` job so path filtering happens
inside the workflow: every job is always created and a job skipped by its `if:` still reports, which a
required status check accepts (a workflow-level `paths:` filter creates no check at all and blocks a merge).
Fails safe: any error prints run=true, so the real job runs.

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


def main() -> int:
    try:
        base, head = sys.argv[1], sys.argv[2]
        patterns = [glob_to_regex(p.strip()) for p in sys.stdin if p.strip() and not p.strip().startswith("#")]
        changed = subprocess.run(["git", "diff", "--name-only", f"{base}...{head}"], check=True,
                                 capture_output=True, text=True).stdout.split()
        hit = next((f for f in changed if any(p.match(f) for p in patterns)), None)
        print(f"run={'true' if hit else 'false'}")
        print(f"changed {len(changed)} file(s); first match: {hit or '-'}", file=sys.stderr)
    except Exception as exc:  # fail safe: run the checks
        print(f"changed_paths: {exc}; running the checks", file=sys.stderr)
        print("run=true")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Replace machine-specific paths with the environment variable that names them.

The single source of the substitution rules. Variable names are expected wherever a
value is needed; this tool only rewrites literal host paths into those names.

Usage:
    substitute_paths.py [--check] PATH [PATH ...]    # PATH may be a file or a directory

--check reports the files that would change and exits 1 if any would change.
JSON and JSONL files are re-parsed after substitution; a file that stops parsing is
left unchanged and reported.
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Each host setting names a directory; its literal value is replaced by the variable that names it.
# The rules come from the local config (and the home directory), so this file holds no host paths.
# rocm-root is deliberately absent: /opt/rocm is the standard system location, not a local folder.
HOST_SETTINGS = (
    ("repo-root", "BC_REPO_ROOT"),
    ("cache-root", "BC_CACHE_ROOT"),
    ("model-root", "BC_MODEL_ROOT"),
    ("work-root", "BC_WORK_ROOT"),
    ("tmp-root", "BC_TMP_ROOT"),
    ("dev-root", "BC_DEV_ROOT"),
    ("data-root", "BC_DATA_ROOT"),
    ("vault-root", "BC_VAULT_ROOT"),
)


def _rules() -> list[tuple[str, str]]:
    import os
    import tomllib

    rules = []
    local = Path(__file__).resolve().parents[2] / "config" / "environment.local.toml"
    if local.is_file():
        body = tomllib.loads(local.read_text(encoding="utf-8")).get("host", {})
        for host in body.values():
            for key, variable in HOST_SETTINGS:
                if isinstance(host.get(key), str) and host[key]:
                    rules.append((host[key].rstrip("/"), "${" + variable + "}"))
    home = os.path.expanduser("~")
    rules.append((home, "${HOME}"))
    # Longest first, so a more specific directory wins over its parent.
    return sorted(set(rules), key=lambda rule: len(rule[0]), reverse=True)


RULES = _rules()

TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".txt", ".toml", ".yaml", ".yml", ".py", ".sh", ".cfg",
                 ".ini", ".env", ".example", ".ndjson", ".log", ".csv", ".cpp", ".h", ".hpp", ".c", ""}


def substitute(text: str) -> str:
    for literal, variable in RULES:
        text = text.replace(literal, variable)
    return text


def _valid_structured(path: Path, text: str) -> bool:
    try:
        if path.suffix == ".json":
            json.loads(text)
        elif path.suffix in {".jsonl", ".ndjson"}:
            for line in text.splitlines():
                if line.strip():
                    json.loads(line)
    except ValueError:
        return False
    return True


def files_under(paths: list[str]):
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            yield from (f for f in sorted(p.rglob("*")) if f.is_file() and f.suffix in TEXT_SUFFIXES)
        elif p.is_file():
            yield p


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="report files that would change; write nothing")
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args(argv)
    changed, refused = [], []
    for path in files_under(args.paths):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        new = substitute(text)
        if new == text:
            continue
        # Only refuse a change that breaks a file which parsed before. A file whose suffix
        # misdescribes its format (e.g. a flat-text hot list named .json) was never structured.
        if _valid_structured(path, text) and not _valid_structured(path, new):
            refused.append(path)
            continue
        changed.append(path)
        if not args.check:
            path.write_text(new, encoding="utf-8", newline="")
    for path in changed:
        print(("would change: " if args.check else "changed: ") + str(path))
    for path in refused:
        print(f"left unchanged (would break parsing): {path}", file=sys.stderr)
    return 1 if (args.check and changed) or refused else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Copy libr4d's sources into the build tree with libr3's textual rewrites applied (run at configure time).

Most of what gfx11 lacks is supplied by r3_compat.h as functions. What cannot be supplied that way is text inside a
string, such as an inline-assembly mnemonic, so those are rewritten in a copy. libr4d's sources are not edited and
nothing is copied into this repository; the copy lives in the build directory.

rewrites.txt holds one rule per line:   <text to find> ==> <replacement>
Blank lines and lines starting with '#' are ignored. A rule that matches nowhere is an error: it means libr4d
changed and the rule has to be looked at again.

--native names a directory of block rules, one file per libr4d source: <source name>.rw. They replace a
kernel's own lines with a native gfx11 form (see native/). A block is

    @@ find <count>
    <lines to find, exactly>
    @@ replace
    <lines to put there>
    @@ end

and must occur exactly <count> times in that source, or the configure step fails: the kernel changed upstream and
the native form has to be looked at again. Lines outside blocks are comments.

Usage: r3_rewrite.py --source <libr4d dir> --rules <rewrites.txt> [--native <native dir>] (--out <out dir> | --check)

--check writes nothing: it applies every rule and block to each source in memory and prints one JSON document on
stdout, {"failures": {<file>: [<what failed>, ...]}}, listing every failing block per file (the build stops at the
first; the check lists them all, so a bump sees the whole port at once). Exit 0 when nothing drifted, 1 when
something did. bigcherry pin-bump runs it right after pulling a bumped radiance.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def rules(path: Path) -> list[tuple[str, str]]:
    out = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if " ==> " not in line:
            raise SystemExit(f"{path}:{number}: expected '<find> ==> <replace>'")
        find, replace = line.split(" ==> ", 1)
        out.append((find, replace))
    return out


def blocks(path: Path) -> list[tuple[int, str, str]]:
    out, state, count, find, replace = [], "", 0, [], []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith("@@ find "):
            if state:
                raise SystemExit(f"{path}:{number}: '@@ find' inside a block")
            state, count, find, replace = "find", int(line.split()[2]), [], []
        elif line.strip() == "@@ replace" and state == "find":
            state = "replace"
        elif line.strip() == "@@ end" and state == "replace":
            out.append((count, "\n".join(find) + "\n", "\n".join(replace) + "\n"))
            state = ""
        elif line.startswith("@@"):
            raise SystemExit(f"{path}:{number}: unexpected {line.strip()!r}")
        elif state == "find":
            find.append(line)
        elif state == "replace":
            replace.append(line)
    if state:
        raise SystemExit(f"{path}: unterminated block")
    return out


def rewrite(source: Path, rule_file: Path, native: Path | None) -> tuple[dict[str, str], dict[str, list[str]], list[str]]:
    """Apply the rules and native blocks to every libr4d source in memory.

    Returns (rewritten text by file name, failures by file name, report lines). A failure names the rule file and the
    block, so one run lists everything that no longer matches."""
    native_rules = {p.name[:-3]: blocks(p) for p in sorted(native.glob("*.rw"))} if native else {}
    failures: dict[str, list[str]] = {}
    for name in native_rules:
        if not (source / name).is_file():
            failures.setdefault(f"native/{name}.rw", []).append(f"libr4d has no source {name}")
    todo = rules(rule_file)
    hits = {find: 0 for find, _ in todo}
    texts: dict[str, str] = {}
    report: list[str] = []
    for path in sorted(source.iterdir()):
        if path.is_dir() or path.suffix not in (".hip", ".h", ".cpp"):
            continue
        text = path.read_text(encoding="utf-8")
        for find, replace in todo:
            count = text.count(find)
            if count:
                hits[find] += count
                text = text.replace(find, replace)
        for index, (count, find, replace) in enumerate(native_rules.get(path.name, []), 1):
            found = text.count(find)
            if found != count:
                failures.setdefault(f"native/{path.name}.rw", []).append(
                    f"block {index} expected {count} time(s) occurs {found} time(s): {find.splitlines()[0].strip()!r}")
                continue
            text = text.replace(find, replace)
        if path.name in native_rules:
            report.append(f"r3_rewrite: native form of {path.name}: {len(native_rules[path.name])} block(s)")
        texts[path.name] = text
    for find, count in hits.items():
        report.append(f"r3_rewrite: {count} x {find!r}")
        if count == 0:
            failures.setdefault(rule_file.name, []).append(f"rule matched nothing: {find!r}")
    return texts, failures, report


def main() -> int:
    parser = argparse.ArgumentParser(description="libr3's textual and native rewrites of libr4d's sources")
    parser.add_argument("--source", type=Path, required=True, help="libr4d source directory")
    parser.add_argument("--rules", type=Path, required=True, help="rewrites.txt")
    parser.add_argument("--native", type=Path, help="directory of <source>.rw block rules")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", type=Path, help="write the rewritten copy here")
    mode.add_argument("--check", action="store_true", help="write nothing; print the failing rules and blocks as JSON")
    args = parser.parse_args()
    texts, failures, report = rewrite(args.source, args.rules, args.native)
    if args.check:
        print(json.dumps({"failures": failures}, indent=2, sort_keys=True))
        return 1 if failures else 0
    for line in report:
        print(line)
    if failures:
        for name, problems in sorted(failures.items()):
            for problem in problems:
                print(f"r3_rewrite: {name}: {problem}", file=sys.stderr)
        return 1
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    for name, text in texts.items():
        target = out / name
        # leave an unchanged file alone so the build does not recompile it
        if not target.is_file() or target.read_text(encoding="utf-8") != text:
            target.write_text(text, encoding="utf-8", newline="\n")
    for stale in out.iterdir():
        if stale.is_file() and not (args.source / stale.name).is_file():
            stale.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())

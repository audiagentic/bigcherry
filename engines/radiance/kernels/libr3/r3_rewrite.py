#!/usr/bin/env python3
"""Copy libr4d's sources into the build tree with libr3's textual rewrites applied (run at configure time).

Most of what gfx11 lacks is supplied by r3_compat.h as functions. What cannot be supplied that way is text inside a
string, such as an inline-assembly mnemonic, so those are rewritten in a copy. libr4d's sources are not edited and
nothing is copied into this repository; the copy lives in the build directory.

rewrites.txt holds one rule per line:   <text to find> ==> <replacement>
Blank lines and lines starting with '#' are ignored. A rule that matches nowhere is an error: it means libr4d
changed and the rule has to be looked at again.

Usage: r3_rewrite.py <libr4d dir> <out dir> <rewrites.txt>
"""
from __future__ import annotations

import shutil
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


def main() -> int:
    source, out, rule_file = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    todo = rules(rule_file)
    hits = {find: 0 for find, _ in todo}
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.iterdir()):
        if path.is_dir():
            continue
        target = out / path.name
        if path.suffix not in (".hip", ".h", ".cpp"):
            continue
        text = path.read_text(encoding="utf-8")
        for find, replace in todo:
            count = text.count(find)
            if count:
                hits[find] += count
                text = text.replace(find, replace)
        # leave an unchanged file alone so the build does not recompile it
        if not target.is_file() or target.read_text(encoding="utf-8") != text:
            target.write_text(text, encoding="utf-8", newline="\n")
    for stale in out.iterdir():
        if stale.is_file() and not (source / stale.name).is_file():
            stale.unlink()
    missing = [find for find, count in hits.items() if count == 0]
    for find, count in hits.items():
        print(f"r3_rewrite: {count} x {find!r}")
    if missing:
        print("r3_rewrite: rule(s) matched nothing: " + "; ".join(repr(m) for m in missing), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

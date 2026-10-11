#!/usr/bin/env python3
"""Take a legacy part out of a merged patch package, or put an edited one back.

A merged package (PA44 slice E, e.g. 1307_q81_activation_cache_mmvq) keeps each former patch's patch.py as one
escaped string argument of _load_legacy_part(name, "<source>"), which cannot be edited by hand. This writes that
source to a file to edit normally and re-encodes it into the package exactly as it was written (json.dumps), so
the diff of the package is the diff of the part.

Usage: legacy-part.py list    <patch-id>
       legacy-part.py extract <patch-id> <part-name> <file>
       legacy-part.py inject  <patch-id> <part-name> <file>         (run from the repository root)
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


def parts(source: str):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_load_legacy_part":
            yield node.args[0].value, node.args[1].value, ast.get_source_segment(source, node.args[1])


def main() -> int:
    action, patch_id = sys.argv[1], sys.argv[2]
    path = Path("engines/llamacpp/patches") / patch_id / "patch.py"
    source = path.read_text(encoding="utf-8")
    found = {name: (text, segment) for name, text, segment in parts(source)}
    if action == "list":
        for name, (text, segment) in found.items():
            state = "ok" if json.dumps(text) == segment else "DOES NOT ROUND-TRIP"
            print(f"{name}: {len(text)} characters, {state}")
        return 0
    name, file = sys.argv[3], Path(sys.argv[4])
    if name not in found:
        print(f"no legacy part {name!r} in {patch_id}; have {sorted(found)}")
        return 1
    text, segment = found[name]
    if json.dumps(text) != segment:
        print(f"{name} does not round-trip through json.dumps; refusing to rewrite it")
        return 1
    if action == "extract":
        file.write_text(text, encoding="utf-8", newline="\n")
        return 0
    if action == "inject":
        new = file.read_text(encoding="utf-8")
        compile(new, f"{patch_id}:{name}", "exec")  # an edited part that does not parse never reaches the package
        if source.count(segment) != 1:
            print(f"{name}'s encoded source occurs {source.count(segment)} times; refusing")
            return 1
        path.write_text(source.replace(segment, json.dumps(new)), encoding="utf-8", newline="")
        print(f"{name}: {len(text)} -> {len(new)} characters")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())

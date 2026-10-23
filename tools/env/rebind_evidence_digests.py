#!/usr/bin/env python3
"""Recompute evidence record digests after a content-only rewrite (for example substitute_paths.py).

An evidence record carries ``record_digest`` over its own payload, so rewriting a host path inside it
breaks the record. This tool re-derives the digest of each record that was valid at the base revision
and changes nothing else; a record that did not verify at the base is left exactly as it was. Every
reference to a re-derived digest elsewhere in the same file (withdrawn lists, links) follows it.

Usage:
    rebind_evidence_digests.py [--base REV] [--check] FILE [FILE ...]

--check reports the files whose digests would change and exits 1 if any would.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry.patch.evidence import _record_digest  # noqa: E402


def _records(node: object, out: list[dict]) -> None:
    """Every dict carrying record_digest, innermost first (a parent's digest covers its children)."""
    if isinstance(node, dict):
        for value in node.values():
            _records(value, out)
        if isinstance(node.get("record_digest"), str):
            out.append(node)
    elif isinstance(node, list):
        for value in node:
            _records(value, out)


def _replace(node: object, mapping: dict[str, str]) -> object:
    if isinstance(node, dict):
        return {k: _replace(v, mapping) for k, v in node.items()}
    if isinstance(node, list):
        return [_replace(v, mapping) for v in node]
    if isinstance(node, str):
        return mapping.get(node, node)
    return node


def rebind(head: object, base: object) -> tuple[object, dict[str, str]]:
    head_records: list[dict] = []
    base_records: list[dict] = []
    _records(head, head_records)
    _records(base, base_records)
    if len(head_records) != len(base_records):
        raise ValueError(f"record count changed ({len(base_records)} -> {len(head_records)}); not a content-only rewrite")
    mapping: dict[str, str] = {}
    for record, original in zip(head_records, base_records):
        if original.get("record_digest") != _record_digest(original):
            continue  # did not verify at the base: left as it was
        # children were re-derived first; their new digests are already in place inside this record
        for key, value in list(record.items()):
            record[key] = _replace(value, mapping) if key != "record_digest" else value
        old = record["record_digest"]
        new = _record_digest(record)
        record["record_digest"] = new
        if new != old:
            mapping[old] = new
    return _replace(head, mapping), mapping


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("files", nargs="+")
    args = parser.parse_args(argv)
    changed = 0
    for name in args.files:
        path = Path(name)
        text = path.read_text(encoding="utf-8")
        base_text = subprocess.run(["git", "show", f"{args.base}:{path.as_posix()}"], capture_output=True,
                                   text=True, encoding="utf-8", check=True).stdout
        new, mapping = rebind(json.loads(text), json.loads(base_text))
        if not mapping:
            continue
        changed += 1
        print(f"{'would rebind' if args.check else 'rebound'} {len(mapping)} record(s): {path}")
        if not args.check:
            indent = 2 if text.lstrip().startswith("{\n") or "\n  " in text[:200] else None
            out = json.dumps(new, indent=indent, ensure_ascii=False)
            path.write_text(out + ("\n" if text.endswith("\n") else ""), encoding="utf-8", newline="\n")
    return 1 if args.check and changed else 0


if __name__ == "__main__":
    sys.exit(main())

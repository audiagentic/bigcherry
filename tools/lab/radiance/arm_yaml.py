#!/usr/bin/env python3
"""Write one lab arm of a serving config: a copy of a native YAML with a few values changed.

A managed start takes no --set: a lab arm authors its own YAML file and passes it with --native-config, where it is sealed
like any config. Each --set is a YAML path in the file, spelled as in it:

    models.<id>.<key>=VALUE           a Radiance model option (kv-cache-dtype, max-num-seqs, ...)
    server.<key>=VALUE                port, host, kernels, ...
    x-bigcherry.<table>.<key>=VALUE   a BigCherry-owned setting (handoff.heat, helper.combine, ...)
    x-bigcherry.<key>=VALUE           description, ...

Each set is a targeted text edit of its one line (a leaf of a one-line flow table `{a: 1, b: 2}` edits only that entry);
every other byte of the base file, comments included, is unchanged, so an arm cannot rewrite the author's scalars (no YAML
library is involved: the engine reads YAML 1.2, and a load-and-dump would not). The path must already name a table of the
file (a typo is an error, not a new key); the leaf may be new, and an absent x-bigcherry table is added whole. VALUE is YAML
1.2 core typed: true/false, an integer or float, null, otherwise a string, which is written double-quoted.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

TABLES = {"power", "service", "memory", "handoff", "helper", "kernels"}
KEY_LINE = re.compile(r"^(?P<indent> *)(?P<key>[A-Za-z0-9_.-]+):(?P<rest>.*)$")


class ArmError(Exception):
    pass


def typed(text: str):
    if text in ("true", "false"):
        return text == "true"
    if text in ("null", "~", ""):
        return None
    if re.fullmatch(r"[+-]?[0-9]+", text):
        return int(text)
    if re.fullmatch(r"[+-]?([0-9]+\.[0-9]*|\.[0-9]+|[0-9]+)([eE][+-]?[0-9]+)?", text):
        return float(text)
    return text


def render(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return json.dumps(value, ensure_ascii=False)   # a YAML double-quoted scalar


def significant(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("#")


def child_range(lines: list[str], start: int, indent: int) -> tuple[int, int]:
    """Lines (start, end) holding the block below lines[start]: deeper than `indent`, comments and blanks inside."""
    end = start + 1
    last = start + 1
    while end < len(lines):
        if significant(lines[end]):
            if len(lines[end]) - len(lines[end].lstrip(" ")) <= indent:
                break
            last = end + 1
        end += 1
    return start + 1, last


def find_key(lines: list[str], lo: int, hi: int, indent: int, key: str) -> int | None:
    for index in range(lo, hi):
        match = KEY_LINE.match(lines[index])
        if match and len(match["indent"]) == indent and match["key"] == key and significant(lines[index]):
            return index
    return None


def split_flow(body: str) -> list[tuple[str, str]]:
    """Entries of the inside of a one-line flow mapping, respecting quotes and nesting."""
    entries, depth, quote, start = [], 0, "", 0
    for position, char in enumerate(body + ","):
        if quote:
            if char == quote and body[position - 1] != "\\":
                quote = ""
        elif char in "\"'":
            quote = char
        elif char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
        elif char == "," and depth == 0:
            piece = body[start:position].strip()
            start = position + 1
            if piece:
                key, separator, value = piece.partition(":")
                if not separator:
                    raise ArmError(f"cannot read the flow entry {piece!r}")
                entries.append((key.strip(), value.strip()))
    return entries


def edit_flow_leaf(line: str, leaf: str, value_text: str, path: str) -> str:
    match = re.match(r"^(?P<head> *[A-Za-z0-9_.-]+: *)\{(?P<body>.*)\}(?P<tail> *(#.*)?)$", line)
    if not match:
        raise ArmError(f"{path}: {line.strip()!r} is not a one-line table")
    entries = split_flow(match["body"])
    for position, (key, _) in enumerate(entries):
        if key == leaf:
            entries[position] = (key, value_text)
            break
    else:
        entries.append((leaf, value_text))
    return match["head"] + "{" + ", ".join(f"{k}: {v}" for k, v in entries) + "}" + match["tail"]


def apply(lines: list[str], path: str, value) -> None:
    parts = path.split(".")
    text = render(value)
    top = parts[0]
    if top == "cards" and len(parts) == 1:
        raise ArmError("cards is a list: author a variant file instead")
    lo, hi, indent = 0, len(lines), 0
    at = find_key(lines, lo, hi, 0, top)
    if at is None:
        raise ArmError(f"{path}: {top} is not a top-level key of the file")
    walked = [top]
    for depth, part in enumerate(parts[1:], start=1):
        lo, hi = child_range(lines, at, indent)
        child_indent = indent + 2
        last = depth == len(parts) - 1
        if top == "x-bigcherry" and depth == 1 and not last and part not in TABLES:
            raise ArmError(f"{path}: x-bigcherry.{part} is not a table (ranks and helpers are lists: author a variant file)")
        found = find_key(lines, lo, hi, child_indent, part)
        flow = found is not None and re.match(r"^ *[A-Za-z0-9_.-]+: *\{.*\} *(#.*)?$", lines[found]) is not None
        if flow and not last:
            if depth != len(parts) - 2:
                raise ArmError(f"{path}: {'.'.join(walked + [part])} is a one-line table; only its entries can be set")
            lines[found] = edit_flow_leaf(lines[found], parts[-1], text, path)
            return
        if last:
            if found is not None:
                match = KEY_LINE.match(lines[found])
                tail = re.search(r"(\s+#.*)$", match["rest"])
                lines[found] = f"{match['indent']}{part}: {text}" + (tail.group(1) if tail else "")
            else:
                lines.insert(hi, f"{' ' * child_indent}{part}: {text}")
            return
        if found is None:
            if top == "x-bigcherry" and depth == 1 and part in TABLES and len(parts) == 3:
                lines.insert(hi, f"{' ' * child_indent}{part}: {{{parts[2]}: {text}}}")
                return
            raise ArmError(f"{path}: {'.'.join(walked + [part])} is not a table of the file")
        at, indent = found, child_indent
        walked.append(part)
    raise ArmError(f"{path}: not a settable leaf")


def write_arm(base: Path, out: Path, sets: list[str]) -> list[str]:
    text = base.read_text(encoding="utf-8")
    lines = text.split("\n")
    if not any(line.startswith("x-bigcherry:") for line in lines):
        raise ArmError(f"{base}: not a serving config (no x-bigcherry section)")
    touched = []
    for setting in sets:
        path, separator, raw = setting.partition("=")
        if not separator or not path:
            raise ArmError(f"{setting}: expected PATH=VALUE")
        apply(lines, path, typed(raw))
        touched.append(path)
    out.write_text("\n".join(lines), encoding="utf-8")
    return touched


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--set", action="append", default=[], metavar="PATH=VALUE")
    args = parser.parse_args(argv)
    try:
        touched = write_arm(args.base, args.out, args.set)
    except ArmError as error:
        print(f"arm_yaml: {error}", file=sys.stderr)
        return 1
    print(f"arm_yaml: {args.out} from {args.base} ({len(touched)} change(s))", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

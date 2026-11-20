#!/usr/bin/env python3
"""Write one lab arm of a serving config: a copy of a native YAML with a few values changed.

A managed start takes no --set: a lab arm authors its own YAML file and passes it with --native-config, where it is sealed
like any config. Each --set is a YAML path in the file, spelled as in it:

    models.<id>.<key>=VALUE       a Radiance model option (kv-cache-dtype, max-num-seqs, ...)
    server.<key>=VALUE            port, host, kernels, ...
    x-bigcherry.<table>.<key>=VALUE   a BigCherry-owned setting (handoff.heat, helper.combine, ...)

The path must already name a table of the file (a typo is an error, not a new key); the leaf may be new. VALUE is YAML 1.2
core typed: true/false, an integer or float, null, otherwise a string (quote nothing needed).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

TABLES = {"service", "memory", "handoff", "helper", "kernels"}


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


def apply(document: dict, path: str, value) -> None:
    parts = path.split(".")
    node = document
    for depth, part in enumerate(parts[:-1]):
        if not isinstance(node, dict) or part not in node:
            if (parts[:depth + 1] and parts[0] == "x-bigcherry" and depth == 1 and part in TABLES
                    and isinstance(node, dict)):
                node[part] = {}   # a table the closed x-bigcherry schema defines, absent from this file
            else:
                raise SystemExit(f"arm_yaml: {path}: {'.'.join(parts[:depth + 1])} is not a table of the file")
        node = node[part]
    if not isinstance(node, dict):
        raise SystemExit(f"arm_yaml: {path}: {'.'.join(parts[:-1])} is not a table")
    node[parts[-1]] = value


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--set", action="append", default=[], metavar="PATH=VALUE")
    args = parser.parse_args(argv)
    document = yaml.safe_load(args.base.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or "x-bigcherry" not in document:
        raise SystemExit(f"arm_yaml: {args.base}: not a serving config (no x-bigcherry section)")
    models = list(document.get("models", {}))
    for setting in args.set:
        path, separator, text = setting.partition("=")
        if not separator or not path:
            raise SystemExit(f"arm_yaml: {setting}: expected PATH=VALUE")
        apply(document, path, typed(text))
    args.out.write_text(yaml.safe_dump(document, sort_keys=False, default_flow_style=False, width=1000),
                        encoding="utf-8")
    print(f"arm_yaml: {args.out} from {args.base} ({len(args.set)} change(s), model {models[0] if models else '?'})",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

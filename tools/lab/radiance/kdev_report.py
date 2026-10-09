#!/usr/bin/env python3
"""Turn a rad-kbench report into a side-by-side table of kernel candidates (RR01 dev harness).

rad-kbench writes one markdown report with a Correctness table (one row per op, kernel and geometry: verdict and
rel_l2 against the reference) and a Speed table (median microseconds and spread). This reads both and prints, per op,
one line per geometry with every kernel that ran there next to each other, then a summary per kernel. The same data
is written as JSON so two runs can be compared.

Usage: kdev_report.py <kbench.md> [--json out.json] [--against earlier.json]
  --against: also print, per kernel and geometry present in both runs, the change in time.
"""
from __future__ import annotations

import json
import math
import re
import sys


def tables(text: str) -> dict[str, list[dict[str, str]]]:
    """Every markdown table of the report, keyed by the '## ' section it sits in."""
    out: dict[str, list[dict[str, str]]] = {}
    section, header = "", None
    for line in text.splitlines():
        if line.startswith("## "):
            section, header = line[3:].strip(), None
            continue
        if not line.startswith("|"):
            header = None
            continue
        cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
        if header is None:
            header = cells
            continue
        if all(re.fullmatch(r":?-+:?", c) for c in cells):
            continue
        out.setdefault(section, []).append(dict(zip(header, cells)))
    return out


def number(text: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", text or "")
    return float(match.group(0)) if match else None


def geometry_key(geometry: str) -> tuple:
    """Sort geometries by their numeric parameters, largest extents last."""
    return tuple(sorted((k, float(v)) for k, v in re.findall(r"(\w+)=(\d+)", geometry)))


def load(path: str) -> dict:
    found = tables(open(path, encoding="utf-8", errors="replace").read())
    cases: dict[tuple[str, str, str], dict] = {}
    for row in found.get("Correctness", []):
        key = (row.get("op", ""), row.get("geometry", ""), row.get("kernel", ""))
        cases[key] = {"op": key[0], "geometry": key[1], "kernel": key[2], "library": row.get("library", ""),
                      "domain": row.get("domain", ""), "verdict": row.get("verdict", ""),
                      "rel_l2": number(row.get("rel_l2", "")), "tol": number(row.get("tol", "")),
                      "us": None, "spread": None}
    for row in found.get("Speed", []):
        key = (row.get("op", ""), row.get("geometry", ""), row.get("kernel", ""))
        case = cases.setdefault(key, {"op": key[0], "geometry": key[1], "kernel": key[2],
                                      "library": row.get("library", ""), "domain": "", "verdict": "",
                                      "rel_l2": None, "tol": None, "us": None, "spread": None})
        case["us"] = number(row.get("us", ""))
        case["spread"] = number(row.get("spread", ""))
        case["ref_us"] = number(row.get("libref us", ""))
    return {"report": path, "cases": sorted(cases.values(), key=lambda c: (c["op"], geometry_key(c["geometry"]), c["kernel"]))}


def geomean(values: list[float]) -> float | None:
    values = [v for v in values if v and v > 0]
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else None


def show(data: dict) -> None:
    cases = data["cases"]
    for op in sorted({c["op"] for c in cases}):
        mine = [c for c in cases if c["op"] == op]
        kernels = sorted({c["kernel"] for c in mine})
        print(f"== {op}: time in us per launch (! = failed against the reference, ~ = spread over 20%)")
        print("  " + "geometry".ljust(34) + "".join(k[:18].rjust(20) for k in kernels) + "   fastest")
        for geometry in sorted({c["geometry"] for c in mine}, key=geometry_key):
            cells, timed = [], {}
            for kernel in kernels:
                case = next((c for c in mine if c["geometry"] == geometry and c["kernel"] == kernel), None)
                if case is None:
                    cells.append("-".rjust(20))
                    continue
                flag = "" if case["verdict"] in ("ok", "") else "!"
                flag += "~" if (case["spread"] or 0) > 20 else ""
                value = f"{case['us']:.1f}" if case["us"] is not None else (case["verdict"] or "?")
                cells.append((value + flag).rjust(20))
                if case["us"] is not None and not flag.startswith("!"):
                    timed[kernel] = case["us"]
            fastest = min(timed, key=timed.get) if timed else "-"
            print("  " + geometry[:34].ljust(34) + "".join(cells) + "   " + fastest)
    print("== per kernel")
    for kernel in sorted({c["kernel"] for c in cases}):
        mine = [c for c in cases if c["kernel"] == kernel]
        failed = [c for c in mine if c["verdict"] not in ("ok", "")]
        worst = max((c["rel_l2"] or 0.0) for c in mine)
        mean = geomean([c["us"] for c in mine if c["us"] is not None])
        print(f"  {kernel} ({mine[0]['library']}, {mine[0]['domain'] or '?'}): {len(mine)} case(s), "
              f"{len(failed)} failed, worst rel_l2 {worst:.2e}, "
              f"geometric-mean time {'%.1f us' % mean if mean else 'not timed'}")
        for case in failed[:6]:
            print(f"    FAILED {case['op']} {case['geometry']}: {case['verdict']} rel_l2 {case['rel_l2']} tol {case['tol']}")


def compare(now: dict, before: dict) -> None:
    earlier = {(c["op"], c["geometry"], c["kernel"]): c for c in before["cases"]}
    ratios: dict[str, list[float]] = {}
    for case in now["cases"]:
        old = earlier.get((case["op"], case["geometry"], case["kernel"]))
        if old and old.get("us") and case.get("us"):
            ratios.setdefault(case["kernel"], []).append(case["us"] / old["us"])
    print(f"== against {before.get('report', 'the earlier run')} (time now / time before; below 1 is faster)")
    for kernel, values in sorted(ratios.items()):
        print(f"  {kernel}: {len(values)} shared case(s), geometric mean {geomean(values):.3f}, "
              f"best {min(values):.3f}, worst {max(values):.3f}")
    if not ratios:
        print("  no timed case in common")


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0].startswith("--"):
        print(__doc__)
        return 2
    data = load(args[0])
    if not data["cases"]:
        print("no case in the report's Correctness or Speed table")
        return 1
    show(data)
    if "--json" in args:
        json.dump(data, open(args[args.index("--json") + 1], "w", encoding="utf-8"), indent=1)
    if "--against" in args:
        compare(data, json.load(open(args[args.index("--against") + 1], encoding="utf-8")))
    return 1 if any(c["verdict"] not in ("ok", "") for c in data["cases"]) else 0


if __name__ == "__main__":
    sys.exit(main())

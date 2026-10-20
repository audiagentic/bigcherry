#!/usr/bin/env python3
"""Per-step kernel time of a model on one card, from radiance's own tools (shape harness, RAD15 step 1).

Inputs, both produced by the pinned radiance build with no server:
  GRAPH   `radiance --model M ... --debug-graph --debug-selection > GRAPH`: the declared graph, one block an op instance
          (op name and parameters at column 0, then its selection bands, `weights` and `writes` lines).
  REPORT  `rad-kbench -m M --kernels <libs> --max-tok T --timing-only --report REPORT`: device time of every kernel at the
          model's real shapes (weights read from the container in the layout each row asked for).

Each graph instance is joined to the report row with the same op whose geometry carries every one of the instance's
parameters with the same value (the report may add tuning keys; the step-size key, M or q_len, is compared apart), at the largest timed step size not above --rows. Instances are classed by the weights they read:
vision (`v.` weights), MTP (`mtp`/`nextn` weights, or a block index at or past --layers) and trunk; vision is left out of
the step, MTP is reported on its own and multiplied by --draft-passes (the drafter runs that many serial passes a step).
Instances with no matching timed row are listed, never filled in from another geometry. Ops whose cost depends on routing
(moe_*) are timed by rad-kbench on Zipf-skewed routing, not real traffic; use replay-kernel-time.sh on captured requests.

usage: kbench_step.py GRAPH REPORT [--rows 4] [--layers N] [--draft-passes 3] [--top 25]
"""
import argparse
import collections
import re
import sys

SIZE_KEYS = ("M", "q_len")


def params_of(text):
    out = {}
    for tok in text.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def graph_instances(path):
    """[(op, params without the size key, weight names)] for every op block before the selection table."""
    insts = []
    cur = None
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("selection table"):
            break
        m = re.match(r"^([a-z][a-z0-9_]*)\s+(\S+=.*)$", line.rstrip("\n"))
        if m:
            cur = [m.group(1), {k: v for k, v in params_of(m.group(2)).items() if k not in SIZE_KEYS}, []]
            insts.append(cur)
            continue
        w = re.match(r"^\s+weights\s+(.*)$", line)
        if w and cur is not None:
            cur[2].extend(x.strip() for x in w.group(1).split(","))
    return insts


def report_rows(path):
    rows = []
    in_speed = False
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("## Speed"):
            in_speed = True
            continue
        if in_speed and line.startswith("## "):
            break
        if not (in_speed and line.startswith("| `")):
            continue
        cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
        try:
            us = float(cells[4])
        except (ValueError, IndexError):
            continue
        p = params_of(cells[3])
        size = next((int(p[k]) for k in SIZE_KEYS if k in p), None)
        rows.append((cells[0], cells[1], {k: v for k, v in p.items() if k not in SIZE_KEYS}, size, us))
    return rows


def classify(weights, layers):
    names = " ".join(weights)
    if any(w.startswith("v.") for w in weights):
        return "vision"
    if "mtp" in names or "nextn" in names:
        return "mtp"
    if layers:
        for b in re.findall(r"blk\.(\d+)\.", names):
            if int(b) >= layers:
                return "mtp"
    return "trunk"


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graph")
    ap.add_argument("report")
    ap.add_argument("--rows", type=int, default=4, help="the step size to read times at (M or q_len)")
    ap.add_argument("--layers", type=int, default=0, help="trunk layer count; a block index at or past it is the MTP layer")
    ap.add_argument("--draft-passes", type=int, default=3, help="serial MTP passes a step (speculative tokens)")
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args(argv)
    rows = report_rows(a.report)
    per = collections.defaultdict(lambda: [0, 0.0, set()])     # (class, op) -> [calls, us, rows]
    missing = collections.Counter()
    for op, params, weights in graph_instances(a.graph):
        cls = classify(weights, a.layers)
        if cls == "vision":
            continue
        cands = [r for r in rows if r[0] == op and all(r[2].get(k) == v for k, v in params.items())
                 and r[3] is not None and r[3] <= a.rows]
        if not cands:
            missing[(cls, op)] += 1
            continue
        best = max(cands, key=lambda r: r[3])
        mult = a.draft_passes if cls == "mtp" else 1
        e = per[(cls, op)]
        e[0] += mult
        e[1] += best[4] * mult
        e[2].add(best[1])
    total = sum(v[1] for v in per.values())
    by_cls = collections.defaultdict(float)
    for (cls, _), v in per.items():
        by_cls[cls] += v[1]
    print("step size %d rows: %.0f us of matched kernels a step (%s)" % (
        a.rows, total, ", ".join("%s %.0f us" % (k, v) for k, v in sorted(by_cls.items()))))
    print("%-6s %-28s %6s %10s %10s %6s  %s" % ("class", "op", "calls", "us/call", "us/step", "share", "row(s)"))
    for (cls, op), (n, us, rs) in sorted(per.items(), key=lambda kv: -kv[1][1])[: a.top]:
        print("%-6s %-28s %6d %10.1f %10.1f %5.1f%%  %s" % (cls, op, n, us / n, us, 100 * us / total if total else 0,
                                                           ", ".join(sorted(rs))))
    if missing:
        print("instances with no timed row at <= %d rows (not counted): %s" % (
            a.rows, ", ".join("%s/%s x%d" % (c, o, n) for (c, o), n in sorted(missing.items()))))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

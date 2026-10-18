#!/usr/bin/env python3
"""RHP07-01: route-trace format (v1 JSONL) and deterministic synthetic traces for the placement simulator.

One line per event. `meta` carries run constants; `route` is one layer of one step:

  {"v":1,"type":"meta","layers":8,"experts":64,"top_k":10,"marks":{"shift":4000}}
  {"v":1,"type":"route","ts_ns":0,"session_id":1,"step_id":17,"phase":"decode","layer":3,
   "experts":[5,9,11],"expert_rows":[1,1,1]}

`marks` names step ids the simulator reports separately (e.g. the step a session mix changes). Timing fields measured on
real runs (helper_us, queue_us, merge_us, exposed_wait_us) may be present and are ignored by the policy; the latency model
is the simulator's, and is calibrated against measured traces in RHP07-02, so a policy is never judged on them here.

The synthetic generators are seeded and pure: the same arguments give the same bytes. They are synthetic by name and by
construction (Zipf-like routing per regime); real traces replace them, they do not extend them.
"""
import bisect
import json
import random
import sys

VERSION = 1
PHASES = ("decode", "prefill")


class TraceError(ValueError):
    pass


def parse(lines):
    """Events from JSONL text lines, validated. Returns (meta, routes)."""
    meta = {"layers": 0, "experts": 0, "top_k": 0, "marks": {}}
    routes = []
    for n, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except ValueError as exc:
            raise TraceError(f"line {n}: not JSON ({exc})") from exc
        if e.get("v") != VERSION:
            raise TraceError(f"line {n}: version {e.get('v')!r}, this reader takes {VERSION}")
        kind = e.get("type")
        if kind == "meta":
            meta.update({k: e[k] for k in ("layers", "experts", "top_k") if k in e})
            meta["marks"] = dict(e.get("marks", {}))
        elif kind == "route":
            routes.append(_route(n, e, meta))
        else:
            raise TraceError(f"line {n}: unknown event type {kind!r}")
    if not meta["layers"] or not meta["experts"]:
        raise TraceError("a trace needs a meta event with layers and experts before its routes")
    return meta, routes


def _route(n, e, meta):
    for key in ("session_id", "step_id", "phase", "layer", "experts", "expert_rows"):
        if key not in e:
            raise TraceError(f"line {n}: route without {key}")
    if e["phase"] not in PHASES:
        raise TraceError(f"line {n}: phase {e['phase']!r}")
    experts, rows = e["experts"], e["expert_rows"]
    if len(experts) != len(rows) or not experts:
        raise TraceError(f"line {n}: experts and expert_rows differ in length or are empty")
    if len(set(experts)) != len(experts):
        raise TraceError(f"line {n}: an expert appears twice in one route")
    if any(r <= 0 for r in rows):
        raise TraceError(f"line {n}: expert_rows must be positive")
    if not 0 <= e["layer"] < meta["layers"] or any(not 0 <= x < meta["experts"] for x in experts):
        raise TraceError(f"line {n}: layer or expert outside the meta shape")
    return e


def dumps(meta, routes):
    out = [json.dumps({"v": VERSION, "type": "meta", **meta}, sort_keys=True)]
    out += [json.dumps(r, sort_keys=True) for r in routes]
    return "\n".join(out) + "\n"


# ---- synthetic traces ----------------------------------------------------------------------------------------------

def _regime(rng, layers, experts, skew):
    """A routing regime: per layer a random expert order and Zipf weights over it."""
    weights = [1.0 / (i + 1) ** skew for i in range(experts)]
    total = sum(weights)
    cum, run = [], 0.0
    for w in weights:
        run += w / total
        cum.append(run)
    orders = []
    for _ in range(layers):
        order = list(range(experts))
        rng.shuffle(order)
        orders.append(order)
    return orders, cum


def _draw(rng, regime, layer, k):
    orders, cum = regime
    chosen = []
    while len(chosen) < k:
        e = orders[layer][min(bisect.bisect_left(cum, rng.random()), len(cum) - 1)]
        if e not in chosen:
            chosen.append(e)
    return sorted(chosen)


def synth(kind, seed=7, layers=8, experts=64, top_k=6, steps=6000, skew=1.1):
    """Deterministic synthetic trace. kind: stationary | shift | flipflop | concurrent | prefill_burst."""
    rng = random.Random(seed)
    a, b = _regime(rng, layers, experts, skew), _regime(rng, layers, experts, skew)
    marks = {}
    routes = []

    def regime_at(step, session):
        if kind == "shift":
            return a if step < steps // 2 else b
        if kind == "flipflop":
            return a if (step // 150) % 2 == 0 else b      # flips every 150 steps: should not be chased
        if kind == "concurrent":
            return a if session % 2 == 0 else b             # two session populations at once
        return a

    if kind == "shift":
        marks["shift"] = steps // 2
    ts = 0
    for step in range(steps):
        sessions = (0, 1, 2) if kind == "concurrent" else (0,)
        for session in sessions:
            prefill = kind == "prefill_burst" and step % 400 == 0
            phase = "prefill" if prefill else "decode"
            rows = 128 if prefill else 1
            for layer in range(layers):
                chosen = _draw(rng, regime_at(step, session), layer, top_k)
                routes.append({
                    "v": VERSION, "type": "route", "ts_ns": ts, "session_id": session, "step_id": step,
                    "phase": phase, "layer": layer, "experts": chosen, "expert_rows": [rows] * len(chosen)})
                ts += 1000
    return {"layers": layers, "experts": experts, "top_k": top_k, "marks": marks}, routes


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "shift"
    sys.stdout.write(dumps(*synth(kind)))

#!/usr/bin/env python3
"""RHP07-01: host-only, trace-driven simulator for cost-aware, hysteretic expert placement on the helpers.

It replays a route trace (placement_trace.py) against a placement of (layer, expert) pairs on two helpers and scores
modelled critical-path latency. It compares

  static   hot-first ownership from a training window, never changed (today's behaviour, R3_MOE_HELPER_HEAT)
  dynamic  the same start, re-evaluated every interval from decayed heat, with hysteresis (RHP07)

and reports modelled latency overall and after each named mark, moves and bytes moved, capacity violations and the
duplicate-contribution check. The latency model and its constants are PLACEHOLDERS: RHP07-02 calibrates them against
measured traces (static placement first), and nothing here is evidence of a real speedup. Nothing touches r3_moe_helper,
the mailbox, radiance's mover or live ownership.

  python3 placement_sim.py TRACE.jsonl [--json]
  python3 placement_sim.py --synth shift [--json]
"""
import argparse
import json
import math
import sys
from dataclasses import dataclass, field

import placement_trace as pt

MIB = 1024 * 1024


@dataclass
class Model:
    """Placeholder cost model (microseconds). Calibrate before believing any result."""
    expert_bytes: int = int(2.4 * MIB)
    helper_fixed_us: float = 40.0        # launch + mailbox, paid when a helper has any rows
    helper_per_expert_us: float = 6.0
    helper_per_row_us: float = 8.0
    cold_per_expert_us: float = 20.0     # experts not on a helper run on the engine side
    cold_per_row_us: float = 30.0
    merge_us: float = 10.0
    link_bytes_per_s: float = 12e9       # host to helper, effective
    move_setup_us: float = 50.0
    move_stall_fraction: float = 0.10    # share of a move's time that lands on the critical path
    helper_fraction: float = 0.35        # of all (layer, expert) pairs, split over the two helpers


@dataclass
class Policy:
    interval_ticks: int = 2048           # re-evaluate every this many routed tokens
    short_ticks: float = 2048.0          # decay half-lives, in routed tokens
    long_ticks: float = 32768.0
    prefill_weight: float = 0.25
    mix_short: float = 0.65
    min_residency_epochs: int = 4
    margin: float = 1.0                  # incoming priority must beat the victim's by this fraction
    gain_factor: float = 2.0             # predicted saving must be at least this multiple of the move cost
    horizon_ticks: float = 32768.0       # how far ahead a saving is counted
    budget_fraction_per_epoch: float = 0.01   # of helper-resident bytes
    session_cap: float = 0.5             # no session may contribute more than this share of the mass
    train_fraction: float = 0.15         # of the trace, used to build the starting hot-first placement


class Heat:
    """Exact decayed per-expert counters on two timescales, decode and prefill apart, lazily decayed."""

    def __init__(self, policy):
        self.p = policy
        self.v = {}                      # key -> [short_d, long_d, short_p, long_p, last_tick]
        self.g = [0.0, 0.0, 0.0, 0.0]    # global mass per channel, same decay
        self.g_last = 0

    def _decay(self, vals, dt):
        if dt <= 0:
            return
        ds, dl = 2.0 ** (-dt / self.p.short_ticks), 2.0 ** (-dt / self.p.long_ticks)
        vals[0] *= ds; vals[2] *= ds; vals[1] *= dl; vals[3] *= dl

    def add(self, key, phase, mass, tick):
        e = self.v.setdefault(key, [0.0, 0.0, 0.0, 0.0, tick])
        self._decay(e, tick - e[4]); e[4] = tick
        self._decay_global(tick)
        i = 0 if phase == "decode" else 2
        e[i] += mass; e[i + 1] += mass
        self.g[i] += mass; self.g[i + 1] += mass

    def _decay_global(self, tick):
        g = self.g
        ds, dl = 2.0 ** (-(tick - self.g_last) / self.p.short_ticks), 2.0 ** (-(tick - self.g_last) / self.p.long_ticks)
        g[0] *= ds; g[2] *= ds; g[1] *= dl; g[3] *= dl
        self.g_last = tick

    def priority(self, key, tick):
        e = self.v.get(key)
        if e is None:
            return 0.0
        vals = e[:4]
        self._decay(vals, tick - e[4])
        g = list(self.g)
        dt = tick - self.g_last
        ds, dl = 2.0 ** (-dt / self.p.short_ticks), 2.0 ** (-dt / self.p.long_ticks)
        g[0] *= ds; g[2] *= ds; g[1] *= dl; g[3] *= dl
        n = [vals[i] / g[i] if g[i] > 1e-12 else 0.0 for i in range(4)]
        short = n[0] + self.p.prefill_weight * n[2]
        long_ = n[1] + self.p.prefill_weight * n[3]
        return self.p.mix_short * short + (1.0 - self.p.mix_short) * long_


class Placement:
    def __init__(self, model, n_keys):
        self.model = model
        total = int(n_keys * model.helper_fraction)
        self.cap_experts = [total // 2, total - total // 2]
        self.cap_bytes = [c * model.expert_bytes for c in self.cap_experts]
        self.owner = {}                  # key -> helper index
        self.placed_epoch = {}
        self.used = [0, 0]

    def put(self, key, helper, epoch):
        self.owner[key] = helper
        self.placed_epoch[key] = epoch
        self.used[helper] += self.model.expert_bytes

    def drop(self, key):
        helper = self.owner.pop(key)
        self.placed_epoch.pop(key, None)
        self.used[helper] -= self.model.expert_bytes
        return helper

    def violations(self):
        return sum(1 for h in (0, 1) if self.used[h] > self.cap_bytes[h])


def key_of(layer, expert, experts):
    return layer * experts + expert


def hot_first(routes, meta, model, count):
    """Alternating hot-first ownership from a histogram: the current static policy."""
    hist = {}
    for r in routes[:count]:
        for x, n in zip(r["experts"], r["expert_rows"]):
            k = key_of(r["layer"], x, meta["experts"])
            hist[k] = hist.get(k, 0) + n
    place = Placement(model, meta["layers"] * meta["experts"])
    ranked = sorted(hist, key=lambda k: (-hist[k], k))
    i = 0
    for k in ranked:
        h = i % 2
        if place.used[h] + model.expert_bytes > place.cap_bytes[h]:
            h ^= 1
            if place.used[h] + model.expert_bytes > place.cap_bytes[h]:
                break
        place.put(k, h, 0)
        i += 1
    return place


def step_latency(route, place, model, meta):
    """Modelled critical path of one layer-step: the slowest of the two helpers and the engine's cold path."""
    t = [0.0, 0.0]
    cold = 0.0
    used = [False, False]
    rows_seen = 0
    for x, n in zip(route["experts"], route["expert_rows"]):
        k = key_of(route["layer"], x, meta["experts"])
        h = place.owner.get(k)
        if h is None:
            cold += model.cold_per_expert_us + model.cold_per_row_us * n
        else:
            used[h] = True
            t[h] += model.helper_per_expert_us + model.helper_per_row_us * n
        rows_seen += n
    assert rows_seen == sum(route["expert_rows"])            # every routed row is computed exactly once
    for h in (0, 1):
        if used[h]:
            t[h] += model.helper_fixed_us
    return max(t[0], t[1], cold) + model.merge_us, rows_seen


@dataclass
class Result:
    name: str
    latency_us: float = 0.0
    steps: int = 0
    moves: int = 0
    bytes_moved: int = 0
    moves_after_warmup: int = 0
    capacity_violations: int = 0
    rows_routed: int = 0
    rows_computed: int = 0
    epochs: int = 0
    by_mark: dict = field(default_factory=dict)      # mark -> [latency_us, steps] at and after that step id

    def mean_us(self):
        return self.latency_us / self.steps if self.steps else 0.0

    def mark_mean_us(self, mark):
        lat, n = self.by_mark.get(mark, [0.0, 0])
        return lat / n if n else 0.0

    def as_dict(self):
        return {"name": self.name, "mean_step_us": round(self.mean_us(), 3), "steps": self.steps, "moves": self.moves,
                "bytes_moved": self.bytes_moved, "moves_after_warmup": self.moves_after_warmup,
                "capacity_violations": self.capacity_violations,
                "no_duplicate_contributions": self.rows_routed == self.rows_computed, "epochs": self.epochs,
                "after_mark_mean_us": {m: round(self.mark_mean_us(m), 3) for m in self.by_mark}}


def move_cost_us(model):
    return model.move_setup_us + model.expert_bytes / model.link_bytes_per_s * 1e6


def propose(heat, place, policy, model, meta, tick, epoch, move_budget_bytes):
    """One evaluation: promote the best outsiders over the weakest eligible incumbents, if worth it. Returns moves."""
    n_keys = meta["layers"] * meta["experts"]
    prio = {k: heat.priority(k, tick) for k in range(n_keys)}
    outsiders = sorted((k for k in range(n_keys) if k not in place.owner and prio[k] > 0), key=lambda k: (-prio[k], k))
    eligible = sorted((k for k in place.owner
                       if epoch - place.placed_epoch[k] >= policy.min_residency_epochs), key=lambda k: (prio[k], k))
    save_per_row = max(0.0, model.cold_per_row_us - model.helper_per_row_us)
    cost = move_cost_us(model)
    moves, spent, vi = [], 0, 0
    for k in outsiders:
        if spent + model.expert_bytes > move_budget_bytes:
            break
        free = [h for h in (0, 1) if place.used[h] + model.expert_bytes <= place.cap_bytes[h]]
        if free:                                         # free capacity: no victim, only the cost test
            gain = prio[k] * policy.horizon_ticks * save_per_row
            if gain >= policy.gain_factor * cost:
                moves.append((k, None, min(free, key=lambda h: place.used[h])))
                spent += model.expert_bytes
            continue
        if vi >= len(eligible):
            break
        victim = eligible[vi]
        if prio[k] <= (1.0 + policy.margin) * prio[victim]:
            break                                        # outsiders only get weaker from here
        gain = (prio[k] - prio[victim]) * policy.horizon_ticks * save_per_row
        if gain < policy.gain_factor * cost:
            break
        moves.append((k, victim, place.owner[victim]))
        spent += model.expert_bytes
        vi += 1
    return moves


def run(routes, meta, model, policy, dynamic, name):
    train = int(len(routes) * policy.train_fraction)
    place = hot_first(routes, meta, model, train)
    res = Result(name)
    heat = Heat(policy)
    tick, next_eval, epoch, warm_epochs = 0, policy.interval_ticks, 0, 3
    sess_rows, prev_share = {}, {}
    eval_routes = routes[train:]
    cost = move_cost_us(model)
    marks = meta.get("marks", {})
    for r in eval_routes:
        tokens = max(r["expert_rows"])
        tick += tokens
        sid = r["session_id"]
        sess_rows[sid] = sess_rows.get(sid, 0) + tokens
        lat, rows = step_latency(r, place, model, meta)
        res.latency_us += lat
        res.steps += 1
        res.rows_routed += sum(r["expert_rows"])
        res.rows_computed += rows
        for m, at in marks.items():
            if r["step_id"] >= at:
                e = res.by_mark.setdefault(m, [0.0, 0])
                e[0] += lat; e[1] += 1
        if dynamic:
            share = min(1.0, policy.session_cap / prev_share[sid]) if prev_share.get(sid, 0) > 0 else 1.0
            for x, n in zip(r["experts"], r["expert_rows"]):
                heat.add(key_of(r["layer"], x, meta["experts"]), r["phase"], n * share, tick)
            if tick >= next_eval:
                epoch += 1
                next_eval += policy.interval_ticks
                total = sum(sess_rows.values()) or 1
                prev_share = {s: n / total for s, n in sess_rows.items()}
                sess_rows = {}
                budget = int(policy.budget_fraction_per_epoch * sum(place.cap_bytes))
                budget = max(budget, model.expert_bytes)
                for k, victim, helper in propose(heat, place, policy, model, meta, tick, epoch, budget):
                    if victim is not None:
                        place.drop(victim)
                    place.put(k, helper, epoch)
                    res.moves += 1
                    res.bytes_moved += model.expert_bytes
                    res.latency_us += model.move_stall_fraction * cost
                    if epoch > warm_epochs:
                        res.moves_after_warmup += 1
                res.capacity_violations += place.violations()
    res.epochs = epoch
    return res


def simulate(meta, routes, model=None, policy=None):
    model, policy = model or Model(), policy or Policy()
    return {"static": run(routes, meta, model, policy, False, "static"),
            "dynamic": run(routes, meta, model, policy, True, "dynamic")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", nargs="?")
    ap.add_argument("--synth")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.synth:
        meta, routes = pt.synth(a.synth)
    elif a.trace:
        with open(a.trace) as f:
            meta, routes = pt.parse(f)
    else:
        ap.error("a trace file or --synth KIND")
    out = {k: v.as_dict() for k, v in simulate(meta, routes).items()}
    if a.json:
        json.dump(out, sys.stdout, indent=1, sort_keys=True)
        print()
    else:
        for r in out.values():
            print(f"{r['name']:8} mean step {r['mean_step_us']:9.2f} us  moves {r['moves']:5d}  "
                  f"after-warmup {r['moves_after_warmup']:4d}  GiB moved {r['bytes_moved'] / 2**30:7.3f}  "
                  f"cap-violations {r['capacity_violations']}  marks {r['after_mark_mean_us']}")


if __name__ == "__main__":
    main()

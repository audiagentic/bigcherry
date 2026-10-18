#!/usr/bin/env python3
"""RHP07-01: host-only, trace-driven simulator for cost-aware, hysteretic expert placement on the helpers.

It replays a route trace (placement_trace.py) against a placement of (layer, expert) pairs on two helpers and scores
modelled critical-path latency. It compares

  static   hot-first ownership from a training window, never changed (today's behaviour, R3_MOE_HELPER_HEAT)
  dynamic  the same start, re-evaluated every interval from decayed heat, with hysteresis (RHP07)

A candidate move is scored by replaying a recent causal window of routes with and without it (a counterfactual on the
critical path), not by a scalar heat difference, and must beat gain_factor x its move cost over a horizon. Moves are applied
to a shadow placement as they are accepted, so capacity and interactions stay consistent. Migration time is reported apart
from inference latency (a zero-stall and a with-stall figure), never mixed into it.

The latency model and its constants are PLACEHOLDERS: RHP07-02 calibrates them against measured traces (static placement
first), and nothing here is evidence of a real speedup. Nothing touches r3_moe_helper, the mailbox, radiance's mover or
live ownership.

  python3 placement_sim.py TRACE.jsonl [--json]
  python3 placement_sim.py --synth shift [--json]
"""
import argparse
import collections
import json
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
    move_stall_fraction: float = 0.10    # share of a move's time that lands on the critical path (a bound)
    helper_fraction: float = 0.35        # of all (layer, expert) pairs, split over the two helpers


@dataclass
class Policy:
    interval_ticks: int = 2048           # re-evaluate every this many routed tokens
    short_ticks: float = 2048.0          # decay half-lives, in routed tokens
    long_ticks: float = 32768.0
    prefill_weight: float = 0.25
    mix_short: float = 0.65
    min_residency_epochs: int = 4
    margin: float = 1.0                  # incoming heat must beat the victim's by this fraction
    gain_factor: float = 2.0             # predicted saving must be at least this multiple of the move cost
    horizon_ticks: float = 32768.0       # how far ahead the window's saving is projected
    budget_fraction_per_epoch: float = 0.01   # of helper-resident bytes; strict, no floor
    session_cap: float = 0.5             # no session may contribute more than this share of the epoch's mass
    train_fraction: float = 0.15         # of the trace, used to build the starting hot-first placement
    window_ticks: int = 8192             # recent causal routes a candidate move is replayed against
    shortlist: int = 32                  # outsiders (by heat) whose moves are scored each epoch


class Heat:
    """Exact decayed per-expert counters on two timescales, decode and prefill apart, lazily decayed.

    Heat ranks and shortlists; it is not a cost predictor. The economics come from the window replay."""

    def __init__(self, policy):
        self.p = policy
        self.v = {}                      # key -> [short_d, long_d, short_p, long_p, last_tick]
        self.g = [0.0, 0.0, 0.0, 0.0]    # global mass per channel, same decay
        self.g_last = 0

    def _decayed(self, vals, dt):
        if dt <= 0:
            return list(vals)
        ds, dl = 2.0 ** (-dt / self.p.short_ticks), 2.0 ** (-dt / self.p.long_ticks)
        return [vals[0] * ds, vals[1] * dl, vals[2] * ds, vals[3] * dl]

    def add(self, key, phase, mass, tick):
        e = self.v.get(key)
        cur = self._decayed(e[:4], tick - e[4]) if e else [0.0, 0.0, 0.0, 0.0]
        g = self._decayed(self.g, tick - self.g_last)
        i = 0 if phase == "decode" else 2
        cur[i] += mass
        cur[i + 1] += mass
        g[i] += mass
        g[i + 1] += mass
        self.v[key] = cur + [tick]
        self.g, self.g_last = g, tick

    def priority(self, key, tick):
        e = self.v.get(key)
        if e is None:
            return 0.0
        vals = self._decayed(e[:4], tick - e[4])
        g = self._decayed(self.g, tick - self.g_last)
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
        if key in self.owner:
            raise ValueError(f"expert {key} is already placed")
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
    """Alternating hot-first ownership from a histogram over the first `count` routes. A stand-in for the production
    histogram and ownership bitmap: RHP07-02 loads the exact ones used by the measured run."""
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


def step_latency(route, owner_get, model, meta):
    """Modelled critical path of one layer-step: the slowest of the two helpers and the engine's cold path.

    A replaceable placeholder. RHP07-02 fits the per-helper terms from measured service times, measures the cold path and
    the merge separately, and calibrates the overlap from timestamps. `owner_get(key)` is a helper index or None."""
    t = [0.0, 0.0]
    cold = 0.0
    used = [False, False]
    for x, n in zip(route["experts"], route["expert_rows"]):
        h = owner_get(key_of(route["layer"], x, meta["experts"]))
        if h is None:
            cold += model.cold_per_expert_us + model.cold_per_row_us * n
        else:
            used[h] = True
            t[h] += model.helper_per_expert_us + model.helper_per_row_us * n
    for h in (0, 1):
        if used[h]:
            t[h] += model.helper_fixed_us
    return max(t[0], t[1], cold) + model.merge_us


@dataclass
class Result:
    name: str
    inference_us: float = 0.0             # modelled critical path of the routed steps only
    migration_exposed_us: float = 0.0     # moves' share of time landing on the critical path (a bound, not a trace)
    steps: int = 0
    moves: int = 0
    bytes_moved: int = 0
    moves_after_warmup: int = 0
    capacity_violations: int = 0
    epochs: int = 0
    history: list = field(default_factory=list)       # (epoch, key, victim or None, helper)
    by_mark: dict = field(default_factory=dict)       # mark -> [inference_us, steps] at and after that step id

    def mean_us(self, stall=False):
        total = self.inference_us + (self.migration_exposed_us if stall else 0.0)
        return total / self.steps if self.steps else 0.0

    def mark_mean_us(self, mark):
        lat, n = self.by_mark.get(mark, [0.0, 0])
        return lat / n if n else 0.0

    def as_dict(self):
        return {"name": self.name, "mean_step_us": round(self.mean_us(), 3),
                "mean_step_us_with_migration_stall": round(self.mean_us(True), 3), "steps": self.steps,
                "moves": self.moves, "bytes_moved": self.bytes_moved, "moves_after_warmup": self.moves_after_warmup,
                "capacity_violations": self.capacity_violations, "epochs": self.epochs,
                "after_mark_mean_us": {m: round(self.mark_mean_us(m), 3) for m in self.by_mark}}


def move_cost_us(model):
    return model.move_setup_us + model.expert_bytes / model.link_bytes_per_s * 1e6


def propose(heat, place, policy, model, meta, tick, epoch, move_budget_bytes, window):
    """One evaluation. Shortlist outsiders by heat, then score each (outsider, helper, victim) candidate by replaying the
    recent causal window of routes with and without it, and accept those whose predicted saving over the horizon beats
    gain_factor x the move cost. Accepted moves go onto a shadow placement before the next candidate is scored.
    Returns [(key, victim or None, helper)]."""
    n_keys = meta["layers"] * meta["experts"]
    prio = {k: heat.priority(k, tick) for k in range(n_keys)}
    outsiders = sorted((k for k in range(n_keys) if k not in place.owner and prio[k] > 0),
                       key=lambda k: (-prio[k], k))[:policy.shortlist]
    shadow = dict(place.owner)
    used = list(place.used)
    placed = dict(place.placed_epoch)
    index = {}
    for i, (_, r) in enumerate(window):
        for x in r["experts"]:
            index.setdefault(key_of(r["layer"], x, meta["experts"]), []).append(i)
    span = max(1, window[-1][0] - window[0][0]) if window else 1
    scale = policy.horizon_ticks / span
    cost = move_cost_us(model)
    size = model.expert_bytes
    moves, spent = [], 0

    def window_gain(k, victim, helper):
        touched = set(index.get(k, ()))
        if victim is not None:
            touched |= set(index.get(victim, ()))
        after = dict(shadow)
        if victim is not None:
            del after[victim]
        after[k] = helper
        gain = 0.0
        for i in touched:
            r = window[i][1]
            gain += step_latency(r, shadow.get, model, meta) - step_latency(r, after.get, model, meta)
        return gain * scale

    for k in outsiders:
        if spent + size > move_budget_bytes:
            break
        best = None
        for h in (0, 1):
            if used[h] + size <= place.cap_bytes[h]:
                victim = None
            else:
                pool = [v for v, owner in shadow.items()
                        if owner == h and epoch - placed.get(v, -10 ** 9) >= policy.min_residency_epochs]
                if not pool:
                    continue
                victim = min(pool, key=lambda v: (prio[v], v))
                if prio[k] <= (1.0 + policy.margin) * prio[victim]:
                    continue
            net = window_gain(k, victim, h) - policy.gain_factor * cost
            if net > 0 and (best is None or net > best[0]):
                best = (net, victim, h)
        if best is None:
            continue
        _, victim, h = best
        if victim is not None:
            del shadow[victim]
            placed.pop(victim, None)
            used[h] -= size
        shadow[k] = h
        placed[k] = epoch
        used[h] += size
        spent += size
        moves.append((k, victim, h))
    return moves


def run(routes, meta, model, policy, dynamic, name, train_count=None):
    """Replay the trace after the training window. `train_count` fixes the window outside the slice under test, so a shorter
    slice starts from the identical placement (the no-lookahead test relies on it)."""
    train = int(len(routes) * policy.train_fraction) if train_count is None else train_count
    place = hot_first(routes, meta, model, train)
    res = Result(name)
    heat = Heat(policy)
    tick, next_eval, epoch, warm_epochs = 0, policy.interval_ticks, 0, 3
    sess_rows, window = {}, collections.deque()
    cost = move_cost_us(model)
    marks = meta.get("marks", {})
    for r in routes[train:]:
        tokens = max(r["expert_rows"])
        tick += tokens
        sid = r["session_id"]
        sess_rows[sid] = sess_rows.get(sid, 0) + tokens
        lat = step_latency(r, place.owner.get, model, meta)
        res.inference_us += lat
        res.steps += 1
        for m, at in marks.items():
            if r["step_id"] >= at:
                e = res.by_mark.setdefault(m, [0.0, 0])
                e[0] += lat
                e[1] += 1
        if not dynamic:
            continue
        window.append((tick, r))
        while window and tick - window[0][0] > policy.window_ticks:
            window.popleft()
        total = sum(sess_rows.values())
        share = 1.0
        if total >= 256 and sess_rows[sid] > 0:
            share = min(1.0, policy.session_cap * total / sess_rows[sid])     # running share inside this epoch
        for x, n in zip(r["experts"], r["expert_rows"]):
            heat.add(key_of(r["layer"], x, meta["experts"]), r["phase"], n * share, tick)
        if tick >= next_eval:
            while next_eval <= tick:                       # one evaluation however far a large step jumped
                next_eval += policy.interval_ticks
            epoch += 1
            sess_rows = {}
            budget = int(policy.budget_fraction_per_epoch * sum(place.cap_bytes))
            for k, victim, helper in propose(heat, place, policy, model, meta, tick, epoch, budget, list(window)):
                if victim is not None:
                    place.drop(victim)
                place.put(k, helper, epoch)
                res.moves += 1
                res.bytes_moved += model.expert_bytes
                res.migration_exposed_us += model.move_stall_fraction * cost
                res.history.append((epoch, k, victim, helper))
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
            print(f"{r['name']:8} mean step {r['mean_step_us']:9.2f} us ({r['mean_step_us_with_migration_stall']:9.2f} "
                  f"with migration stall)  moves {r['moves']:4d}  after-warmup {r['moves_after_warmup']:4d}  "
                  f"GiB moved {r['bytes_moved'] / 2 ** 30:6.3f}  cap-violations {r['capacity_violations']}  "
                  f"marks {r['after_mark_mean_us']}")


if __name__ == "__main__":
    main()

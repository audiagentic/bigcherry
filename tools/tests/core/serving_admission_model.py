"""The serving manifest v1 admission rule (spec §5.2, amendment 5) and a model of the engine's VRAM split, used to generate
the shared admission vectors (docs/design/serving-manifest-v1/admission-vectors.json).

Two consumers read the same vectors, so they cannot drift apart:
- the contract test (test_serving_manifest_v1_contract.py) regenerates them from this module and checks every verdict;
- rad0011's check test (engines/radiance/tests/plan_memory/plan_check_test.cpp) runs bc_plan_memory_check on each pair
  and requires the same verdict and refusal text.

`resolve` follows Radiance's vram_budget_resolve (core/mem/vram_budget.cpp, from the claimable figure on: the cache and
expert split, the prefill staging region and who pays for it, and the MiB rounding of the cache) closely enough that it
reproduces the production example's measured plan byte for byte; the contract test checks that.

Run as a script to rewrite the vectors file: python3 tools/tests/core/serving_admission_model.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

MIB = 1024 * 1024
ALIGN = 4096                        # RAD_ALIGN_UNIT (abi/rad_format.h)
MAX_BYTES = 2**53 - 1               # I-JSON: the v1 byte-count range
PPM = 1_000_000
STEP = 8192                         # the claimable reading's step between starts (RSC03 A3)
VECTORS = Path(__file__).resolve().parents[3] / "docs/design/serving-manifest-v1/admission-vectors.json"


def to_mib(b: int) -> int:
    return b // MIB if b > 0 else 0


def to_mib_up(b: int) -> int:
    return (b + MIB - 1) // MIB if b > 0 else 0


def resolve(*, claimable: int, statics: int, kv_ceiling: int, kv_one: int, expert_ceiling: int, expert_floor: int,
            lends: bool, layer_max: int) -> dict:
    """The split vram_budget_resolve makes of `claimable` (placement neither all_vram nor layer_offload, no KV streams)."""
    statics = min(statics, claimable)
    elastic = max(claimable - statics, 0)
    kv_cap = kv_ceiling if kv_ceiling > 0 else elastic
    if kv_cap + expert_ceiling <= elastic:
        kv, exp = kv_cap, expert_ceiling
    elif lends:
        exp = min(expert_floor, elastic)
        kv = min(kv_cap, max(elastic - exp, kv_one))
        exp = min(max(expert_ceiling, expert_floor), elastic - kv)
    else:
        kv = min(kv_cap, max(elastic - expert_ceiling, kv_one))
        exp = min(expert_ceiling, elastic - kv)
    kv = min(kv, elastic)
    exp = max(exp, 0)
    weights = max(statics + exp, MIB)
    stage = 0
    if layer_max > 0 and expert_ceiling > exp:
        f = (expert_ceiling - exp) / expert_ceiling          # a C++ double, as upstream
        one = (math.ceil(f * layer_max) + ALIGN - 1) // ALIGN * ALIGN
        if lends:
            if exp - 2 * one >= expert_floor:
                stage, exp = 2 * one, exp - 2 * one
                weights = statics + exp
            elif kv - 2 * one >= kv_one:
                stage, kv = 2 * one, kv - 2 * one
        elif 2 * one < exp:
            stage, exp = 2 * one, exp - 2 * one
            weights = statics + exp
    weights_mib = to_mib_up(weights)
    kv_mib = to_mib(max(kv - (weights_mib * MIB - weights), 0))
    if kv_ceiling > 0 and kv >= kv_ceiling and kv_mib * MIB < kv_ceiling:
        kv_mib = to_mib_up(kv_ceiling)
    return {"claimable": claimable, "statics": statics, "elastic": elastic, "experts": exp, "kv": kv, "stage": stage,
            "kv_resolved": kv_mib * MIB, "expert_floor": expert_floor, "kv_ceiling": kv_ceiling,
            "expert_ceiling": expert_ceiling}


def plan(vram: dict, *, kv_one: int, capacity: int, unit_max: int) -> dict:
    """A complete rad0011 plan document around a VRAM split (host and checkpoints fixed: not what these vectors test)."""
    return {
        "plan_version": 1, "producer": "radiance --plan-memory (rad0011)",
        "host": [{"name": "host_arenas", "bytes": 11665408, "residence": "pinned",
                  "from": "engine plan (rad0011: arena_host_bytes)", "remedy": ""}],
        "checkpoints": {"slot_bytes": 117938964, "floor_slots": 8, "want_slots": 72, "from": "engine plan (vectors)"},
        "kv_one_session": {"bytes": kv_one, "from": "engine plan (vectors)"},
        "vram": {**vram, "from": "engine plan (vectors)",
                 "cards": [{"capacity": capacity, "free": capacity - 64 * MIB,
                            "engine_expert_plane": vram["expert_ceiling"], "expert_unit_max": unit_max}]},
        "host_pool_mib": 9216,
    }


def kv_required(kv_ceiling: int, kv_one: int, ppm: int) -> int:
    """max(one full-length session's KV, ceil(kv_ceiling * ppm / 10^6)), without a product beyond 2^63."""
    q, r = divmod(kv_ceiling, PPM)
    return max(kv_one, q * ppm + (r * ppm + PPM - 1) // PPM)


def _short(what: str, live: int, verb: str, bound: int) -> str:
    return f"{what}: live {live} bytes {verb} {bound} bytes"


_STABLE = ("statics", "expert_floor", "expert_ceiling", "kv_ceiling")
_EXPECTED_TERMS = ("statics", "experts", "stage", "kv_ceiling", "expert_floor", "expert_ceiling")
_LIVE_TERMS = ("claimable", "statics", "experts", "stage", "kv_resolved", "kv_ceiling", "expert_floor", "expert_ceiling")


def admit(expected: dict, live: dict, ppm: int) -> dict:
    """The VRAM part of bc_plan_memory_check, in its order. Returns {"admit": True, "kv_required": n} or a refusal:
    {"admit": False, "class": "input" | "plan" | "admission", "needle": text the C++ refusal contains}."""
    ev, lv = expected["vram"], live["vram"]
    if not 0 <= ppm < PPM:
        return {"admit": False, "class": "input", "needle": "plan check: kv_backing_floor_ppm must be in [0, 1000000)"}
    terms = [ev.get(k, -1) for k in _EXPECTED_TERMS] + [expected["kv_one_session"]["bytes"]]
    terms += [lv.get(k, -1) for k in _LIVE_TERMS]
    if any(not isinstance(v, int) or not 0 <= v <= MAX_BYTES for v in terms):
        return {"admit": False, "class": "input",
                "needle": "plan check: the VRAM terms must be byte counts in [0, 2^53 - 1]"}
    for k in _STABLE:
        if lv[k] != ev[k]:
            return {"admit": False, "class": "plan", "needle": _short(f"VRAM {k}", lv[k], "differ from the planned", ev[k])}
    lc, ec = lv["cards"], ev["cards"]
    if len(lc) != len(ec):
        return {"admit": False, "class": "plan",
                "needle": f"plan check: the live plan has {len(lc)} card(s), the manifest's {len(ec)}"}
    for i, (l, e) in enumerate(zip(lc, ec)):
        if l["capacity"] < e["capacity"]:
            return {"admit": False, "class": "plan",
                    "needle": _short(f"card {i} capacity", l["capacity"], "fall short of the planned", e["capacity"])}
        for k in ("engine_expert_plane", "expert_unit_max"):
            if l[k] != e[k]:
                return {"admit": False, "class": "plan",
                        "needle": _short(f"card {i} {k}", l[k], "differ from the planned", e[k])}
    need = kv_required(ev["kv_ceiling"], expected["kv_one_session"]["bytes"], ppm)
    if lv["experts"] < ev["expert_floor"]:
        return {"admit": False, "class": "admission", "kv_required": need,
                "needle": _short("admission: expert residency", lv["experts"], "fall short of the required",
                                 ev["expert_floor"])}
    total = ev["statics"] + lv["experts"] + lv["stage"] + need
    if lv["claimable"] < total:
        return {"admit": False, "class": "admission", "kv_required": need,
                "needle": _short("admission: claimable VRAM", lv["claimable"], "fall short of the required", total)}
    if lv["kv_resolved"] < need:
        return {"admit": False, "class": "admission", "kv_required": need,
                "needle": _short("admission: resolved KV", lv["kv_resolved"], "fall short of the required", need)}
    return {"admit": True, "kv_required": need}


# ---- the vectors ---------------------------------------------------------------------------------------------------

# The production example's geometry (example.flashnext-4gpu.json): the cache lends, no expert floor, every expert on
# the host tier, so the staging region comes out of the KV.
PROD = dict(statics=5800083456, kv_ceiling=31511797760, kv_one=5557452800, expert_ceiling=6351237120, expert_floor=0,
            lends=True, layer_max=1293066240)
PROD_CARD = dict(capacity=34208743424, unit_max=2535424)
PROD_CLAIMABLE = 31198251000
A3_READINGS = (31198251000, 31198242808, 31198234616, 31198242808)

# A lab geometry with resident experts above a host-pool floor: the cache is capped at its ceiling, so expert
# residency takes what is left and moves with claimable, and the staging region is paid by the experts while they
# stay above their floor, by the KV once they would not.
LAB = dict(statics=2048 * MIB, kv_ceiling=8192 * MIB, kv_one=512 * MIB, expert_ceiling=4096 * MIB,
           expert_floor=1024 * MIB, lends=True, layer_max=512 * MIB)
LAB_CARD = dict(capacity=16384 * MIB, unit_max=4 * MIB)


def _pair(geo: dict, card: dict, expected_claimable: int, live_claimable: int) -> tuple[dict, dict]:
    kv_one = geo["kv_one"]
    return (plan(resolve(claimable=expected_claimable, **geo), kv_one=kv_one, **card),
            plan(resolve(claimable=live_claimable, **geo), kv_one=kv_one, **card))


def _experts_pay(geo: dict, claimable: int) -> bool:
    """In the LAB range the cache sits at its ceiling unless it paid for the staging region."""
    v = resolve(claimable=claimable, **geo)
    return v["stage"] > 0 and v["kv"] == geo["kv_ceiling"]


def _lowest(pred, lo: int, hi: int) -> int:
    """The lowest claimable in [lo, hi] where pred holds (pred monotone: false below, true from there on)."""
    assert not pred(lo) and pred(hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if pred(mid) else (mid, hi)
    return hi


def resolver_vectors() -> list[dict]:
    out = []

    def add(name, why, geo, card, exp_c, live_c, ppm):
        expected, live = _pair(geo, card, exp_c, live_c)
        out.append({"name": name, "kind": "resolver", "why": why, "ppm": ppm, "expected": expected, "live": live,
                    "verdict": admit(expected, live, ppm)})

    for i, reading in enumerate(A3_READINGS):
        add(f"prod-a3-reading-{i + 1}", "an RSC03 A3 claimable reading against the measured plan: admitted",
            PROD, PROD_CARD, PROD_CLAIMABLE, reading, 600000)
    for k in range(1, 4):
        add(f"prod-minus-{8 * k}k", f"{8 * k} KiB less claimable than measured: the measured split no longer fits, "
            "the required one does", PROD, PROD_CARD, PROD_CLAIMABLE, PROD_CLAIMABLE - k * STEP, 600000)

    # The MiB truncation edge: the lowest claimable whose resolved (MiB-rounded) KV still covers kv_required, and the
    # reading one step below it, whose unrounded KV share may still cover it while the pool the engine gets does not.
    need = kv_required(PROD["kv_ceiling"], PROD["kv_one"], 600000)
    edge = _lowest(lambda c: resolve(claimable=c, **PROD)["kv_resolved"] >= need, 1, PROD_CLAIMABLE)
    add("prod-kv-truncation-edge", "the lowest claimable whose resolved KV covers kv_required: admitted",
        PROD, PROD_CARD, PROD_CLAIMABLE, edge, 600000)
    below = resolve(claimable=edge - 1, **PROD)
    assert below["kv"] >= need > below["kv_resolved"], "the edge must be one the MiB truncation decides"
    add("prod-kv-truncation-below", "one byte below the edge the KV share still covers kv_required but the MiB-rounded "
        "pool does not: refused naming the resolved KV", PROD, PROD_CARD, PROD_CLAIMABLE, edge - 1, 600000)

    # The staging payer switch (review B1): at the measured claimable the experts pay; one step lower they would fall
    # below their floor, so the KV pays and loses the whole region at once.
    switch = _lowest(lambda c: _experts_pay(LAB, c), 10000 * MIB, 14000 * MIB)
    assert not _experts_pay(LAB, switch - STEP)
    for ppm, verdict in ((600000, "admitted: the KV left after paying still covers the floor"),
                         (950000, "refused: the KV left after paying is below the floor")):
        add(f"lab-stage-payer-switch-{ppm}", f"the staging region moves from the experts to the KV: {verdict}",
            LAB, LAB_CARD, switch, switch - STEP, ppm)

    # Expert residency moving with claimable in 8 KiB steps, above its floor, with a lender (review B3: the live stage
    # moves too, and must not be compared as a stable consumer).
    base = switch + 64 * MIB
    for k in range(1, 5):
        add(f"lab-experts-minus-{8 * k}k", f"{8 * k} KiB less claimable: expert residency and staging move, admitted",
            LAB, LAB_CARD, base, base - k * STEP, 600000)

    moved = [v for v in out if v["name"].startswith("lab-experts-minus-")]
    assert len({v["live"]["vram"]["experts"] for v in moved}) == len(moved), "residency must move at every step"
    assert any(v["live"]["vram"]["stage"] > v["expected"]["vram"]["stage"] for v in moved), "a step must grow the stage"

    # Expert residency below its host-pool floor: one session's KV wins and pushes residency under the floor.
    low = LAB["statics"] + LAB["expert_floor"] + LAB["kv_one"] - 64 * MIB
    add("lab-experts-below-floor", "one session's KV pushes expert residency below the floor: refused",
        LAB, LAB_CARD, base, low, 0)

    # No lender: experts first, the cache keeps the rest.
    nolend = {**LAB, "lends": False}
    add("lab-no-lender-minus-8k", "no lender, 8 KiB less claimable: admitted", nolend, LAB_CARD, base, base - STEP, 600000)
    return out


def _small(**vram) -> dict:
    """The boundary fixture: statics 3000, kv_ceiling 7000, kv_one 500, no expert floor (ppm 600000: kv_required 4200,
    required total 3000 + 0 + 200 + 4200 = 7400). Live plans built by hand stand in for a resolver that over-allocates."""
    v = {"claimable": 9000, "statics": 3000, "elastic": 6000, "experts": 0, "kv": 5000, "stage": 200,
         "kv_resolved": 5000, "expert_floor": 0, "kv_ceiling": 7000, "expert_ceiling": 100}
    kv_one = vram.pop("kv_one", 500)
    v.update(vram)
    p = plan(v, kv_one=kv_one, capacity=10000, unit_max=10)
    p["vram"]["cards"][0]["free"] = 9900
    return p


def boundary_vectors() -> list[dict]:
    out = []
    base = _small()

    def add(name, why, live, ppm=600000, expected=None):
        expected = expected or base
        out.append({"name": name, "kind": "boundary", "why": why, "ppm": ppm, "expected": expected, "live": live,
                    "verdict": admit(expected, live, ppm)})

    add("capacity-equal", "claimable equals the required total: admitted", _small(claimable=7400, kv_resolved=4200))
    add("capacity-one-short", "one byte short of the required total: refused", _small(claimable=7399, kv_resolved=4200))
    add("below-old-measured-split", "below the old measured split (8200), above the required total: admitted",
        _small(claimable=8199))
    add("kv-resolved-equal", "resolved KV equals kv_required: admitted", _small(kv_resolved=4200))
    add("kv-resolved-one-short", "resolved KV one byte short of kv_required: refused", _small(kv_resolved=4199))
    add("measured-kv-not-an-input", "the live kv share is not an input", _small(kv=0))
    add("no-floor-selected", "ppm 0: kv_required is one session (500)", _small(kv_resolved=500), ppm=0)
    add("no-floor-one-short", "ppm 0, resolved KV 499: refused", _small(kv_resolved=499), ppm=0)
    big = _small(kv_one=4500)
    add("session-above-floor", "one session (4500) above the floor (4200) is the requirement: refused at 4499",
        _small(kv_one=4500, kv_resolved=4499), expected=big)
    odd = _small(kv_ceiling=7001)
    add("floor-rounds-up", "ceil(7001 * 0.6) = 4201: 4200 refused", _small(kv_ceiling=7001, kv_resolved=4200),
        expected=odd)
    add("floor-rounds-up-met", "ceil(7001 * 0.6) = 4201: 4201 admitted", _small(kv_ceiling=7001, kv_resolved=4201),
        expected=odd)
    add("ppm-max", "ppm 999999, the largest valid floor: kv_required ceil(7000 * 0.999999) = 7000, admitted",
        _small(claimable=10200, kv_resolved=7000), ppm=999999)
    add("ppm-negative", "ppm -1: refused as invalid input", base, ppm=-1)
    add("ppm-million", "ppm 1000000 is outside the schema (exclusiveMaximum): refused", base, ppm=1000000)
    huge_need = kv_required(MAX_BYTES, 500, 999999)
    huge = _small(kv_ceiling=MAX_BYTES)
    add("ceiling-at-schema-max", "kv_ceiling 2^53 - 1 at ppm 999999: no overflow, admitted at the exact requirement",
        _small(kv_ceiling=MAX_BYTES, claimable=3200 + huge_need, kv_resolved=huge_need), ppm=999999, expected=huge)
    add("ceiling-at-schema-max-short", "the same, one byte short: refused",
        _small(kv_ceiling=MAX_BYTES, claimable=3200 + huge_need, kv_resolved=huge_need - 1), ppm=999999, expected=huge)
    add("ceiling-past-schema-max", "kv_ceiling 2^53: refused as invalid input",
        _small(kv_ceiling=MAX_BYTES + 1), ppm=999999, expected=_small(kv_ceiling=MAX_BYTES + 1))
    missing = _small()
    del missing["vram"]["kv_resolved"]
    add("kv-resolved-missing", "a live plan without kv_resolved: refused as invalid input", missing)
    add("live-stage-grew", "a live stage above the measured one is not plan drift (review B3): admitted",
        _small(stage=201))
    add("statics-differ", "statics are a stable fact: a different value is plan drift", _small(statics=3001))
    add("expert-ceiling-differs", "the expert ceiling is a stable fact", _small(expert_ceiling=101))
    add("kv-ceiling-differs", "the KV ceiling is a stable fact", _small(kv_ceiling=7001))
    floor_plan = _small(expert_floor=100, experts=100)
    add("experts-at-floor", "expert residency equal to its floor: admitted", _small(expert_floor=100, experts=100),
        expected=floor_plan)
    add("experts-below-floor", "expert residency one byte below its floor: refused",
        _small(expert_floor=100, experts=99), expected=floor_plan)
    plane = _small()
    plane["vram"]["cards"][0]["engine_expert_plane"] = 101
    add("card-plane-differs", "the card's expert plane is a stable fact", plane)
    return out


def vectors() -> dict:
    return {
        "vectors_version": 1,
        "about": "Shared admission vectors for serving manifest v1 spec §5.2 (amendment 5). Generated by "
                 "tools/tests/core/serving_admission_model.py; read by the v1 contract test and rad0011's check test.",
        "vectors": resolver_vectors() + boundary_vectors(),
    }


def render() -> str:
    return json.dumps(vectors(), indent=1, sort_keys=True) + "\n"


if __name__ == "__main__":
    VECTORS.write_text(render(), encoding="utf-8")
    print(f"wrote {VECTORS}")

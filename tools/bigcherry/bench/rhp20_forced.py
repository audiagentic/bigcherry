"""RHP20 card-free eligibility and fail-closed acceptance receipts.

Auto-swap's a_layer/a_expert fields name helper IDs, NOT expert IDs. The
runtime resolves a same-layer pair at proposal time; its receiver remains
authoritative for incoming shadow slots and format support.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from pathlib import Path

TARGET = 1000
PAIRS = ((0, 1), (0, 2), (1, 2))
SUMMARY = re.compile(r"forced placement helper=(\d+) attempted=(\d+) committed=(\d+) aborted=(\d+) inflight=(\d+)")
RESOLVED = re.compile(r"forced resolved transaction=(\d+) epoch=(\d+) swap=(\d+):(\d+):(\d+):(\d+) helpers=(\d+):(\d+)")
PROTECTED = re.compile(r"^blk\.(\d+)\.ffn_\*_exps\.(\d+)\.weight\s+cast dtype=bf16\s*$", re.MULTILINE)
W4 = re.compile(r"^blk\.\*\.ffn_\*_exps\.\*\.weight\s+gptq table=w4nl group=64\b", re.MULTILINE)


class GateError(ValueError):
    pass


def recipe_evidence(model: Path) -> tuple[set[tuple[int, int]], str]:
    recipes = list(model.parent.glob("*.recipe"))
    if len(recipes) != 1:
        raise GateError("one model-side quantisation recipe is required")
    raw = recipes[0].read_text(encoding="utf-8")
    if not W4.search(raw):
        raise GateError("recipe does not establish w4nl64 for other routed experts")
    protected = {(int(a), int(b)) for a, b in PROTECTED.findall(raw)}
    specific = re.findall(r"^blk\.(\d+)\.ffn_\*_exps\.(\d+)\.weight\s+(.+)$", raw, re.MULTILINE)
    if any((int(a), int(b)) not in protected or rule.strip() != "cast dtype=bf16"
           for a, b, rule in specific):
        raise GateError("unrecognised format override in recipe")
    return protected, hashlib.sha256(raw.encode()).hexdigest()


def manifest_evidence(manifest: dict, layout: str, protected: set[tuple[int, int]]) -> dict:
    spec = manifest.get("spec", {})
    if spec.get("intent", {}).get("layout") != layout:
        raise GateError("manifest is from a different layout")
    if spec.get("plan", {}).get("state") != "measured":
        raise GateError("requires measured manifest, not a dry owner map")
    helpers = spec.get("processes", {}).get("helpers", [])
    groups = spec.get("plan", {}).get("owner_map", {}).get("helpers", [])
    facts = {x["bdf"]: x["arch"] for x in spec.get("facts", {}).get("gpus", [])}
    if len(helpers) != 3 or len(groups) != 3:
        raise GateError("requires two head helpers and one RDNA2 tail")
    seen: set[tuple[int, int]] = set()
    owners = []
    stride = None
    for i, (helper, group) in enumerate(zip(helpers, groups)):
        if facts.get(helper["gpu"]) != ("gfx1100" if i < 2 else "gfx1030"):
            raise GateError(f"helper {i} architecture mismatch")
        if helper["tier"] != ("head" if i < 2 else "tail"):
            raise GateError(f"helper {i} placement role mismatch")
        allowed = set(helper["formats"])
        if "w4nl64" not in allowed or i == 2 and allowed != {"w4nl64"}:
            raise GateError(f"helper {i} format incompatible")
        owned = [tuple(expert) for expert in group["experts"]]
        if not owned or len(owned) != group["count"] or len(set(owned)) != len(owned):
            raise GateError(f"helper {i} zero-owner/invalid count")
        if set(owned) & seen:
            raise GateError(f"helper {i} duplicate ownership")
        seen.update(owned)
        if group["bytes"] % len(owned):
            raise GateError("nonintegral expert payload")
        size = group["bytes"] // len(owned)
        if stride is not None and stride != size:
            raise GateError("mixed expert payloads: format proof insufficient")
        stride = size
        if group["bytes"] > helper["vram_budget_mib"] * 1048576:
            raise GateError("ownership exceeds manifest VRAM budget")
        owners.append(set(owned) - protected)
    if spec.get("handoff", {}).get("placement", {}).get("strategy") != "tail":
        raise GateError("not a tail strategy")
    witness = {}
    for a, b in PAIRS:
        # A valid two-way swap preserves postcommit usage on equal-stride
        # helpers; runtime must still reserve incoming staging/retirement.
        by_layer = {}
        for layer, expert in owners[b]:
            by_layer[layer] = min(by_layer.get(layer, expert), expert)
        both = sorted((l, e, by_layer[l]) for l, e in owners[a] if l in by_layer)
        if not both:
            raise GateError(f"no same-layer pair for helpers {a},{b}")
        witness[f"{a}:{b}"] = both[0]
    return {"layout": layout, "owner_map_sha256": spec["plan"]["owner_map"]["owner_map_sha256"],
            "counts": [len(x) for x in owners], "expert_bytes": stride,
            "pair_witnesses": witness, "model": spec["intent"]["model"],
            "mtp_depth": spec["processes"]["engine"]["execution"]["num_speculative_tokens"]}


def experiment_evidence(experiment: dict) -> dict:
    placement = experiment.get("placement", {})
    forced = placement.get("forced", {})
    steps = forced.get("steps", [])
    expected = [{"kind": "auto_swap", "a_layer": a, "a_expert": b} for a, b in PAIRS]
    if placement.get("mode") != "forced_churn" or steps != expected:
        raise GateError("experiment must auto-swap 0:1,0:2,1:2 in order")
    repeat, pace = forced.get("repeat"), forced.get("pace_requests")
    if type(repeat) is not int or not math.ceil(TARGET / len(steps)) <= repeat <= 1000000:
        raise GateError("repeat cannot issue at least 1000 proposals")
    if type(pace) is not int or not 1 <= pace <= 4294967295:
        raise GateError("invalid pace_requests")
    return {"scheduled": repeat * len(steps), "pace_requests": pace,
            "minimum_completed_lane_requests": pace * TARGET}


def receipt_evidence(logs: dict[int, str], *, combine: bool, identity: dict | None = None) -> dict:
    if set(logs) != {0, 1, 2}:
        raise GateError("require helper0/helper1/helper2 logs")
    totals = {}
    for i, body in logs.items():
        matches = list(SUMMARY.finditer(body))
        if not matches:
            raise GateError(f"helper{i} lacks terminal summary")
        helper, attempted, committed, aborted, inflight = map(int, matches[-1].groups())
        if helper != i or inflight:
            raise GateError(f"helper{i} identity/inflight mismatch")
        totals[i] = {"attempted": attempted, "committed": committed, "aborted": aborted}
    primary = totals[0]
    if primary["committed"] < TARGET or primary["aborted"] or primary["attempted"] != primary["committed"]:
        raise GateError(f"committed={primary['committed']} aborted={primary['aborted']} (need {TARGET}/0)")
    events = [tuple(map(int, m.groups())) for m in RESOLVED.finditer(logs[0])]
    if len(events) < primary["committed"]:
        raise GateError("per-transition resolved log truncated")
    ids = [event[0] for event in events]
    if len(set(ids)) != len(ids) or ids != sorted(ids):
        raise GateError("duplicate or unordered transaction IDs")
    pairs = Counter(tuple(sorted(event[-2:])) for event in events)
    if any(pairs[p] == 0 for p in PAIRS):
        raise GateError(f"missing same-ISA/cross-ISA pair coverage: {pairs}")
    if combine and not any(int(n) > 0 for n in re.findall(r"combine steps (\d+)", logs[0])):
        raise GateError("CPU combine did not serve any steps")
    if not identity or not identity.get("records"):
        raise GateError("no per-expert same-ISA bit-identity evidence (token-level match is insufficient)")
    by_id = {row[0]: row for row in events}
    verified = set()
    for record in identity["records"]:
        txn = record.get("transaction_id")
        row = by_id.get(txn) if type(txn) is int else None
        if (row is None or type(record.get("before_epoch")) is not int or
            type(record.get("after_epoch")) is not int or
            record["after_epoch"] != row[1] or
            record["before_epoch"] + 1 != row[1] or
            record.get("source_isa") != "gfx1100" or
            record.get("target_isa") != "gfx1100" or
            not re.fullmatch(r"[0-9a-f]{64}", record.get("before_sha256", "")) or
            record["before_sha256"] != record.get("after_sha256")):
            raise GateError("per-expert proof not bound to a same-ISA committed epoch or hashes differ")
        expert = record.get("expert")
        moved = {f"{row[2]}:{row[3]}", f"{row[4]}:{row[5]}"}
        if tuple(sorted(row[-2:])) != (0, 1) or expert not in moved:
            raise GateError("identity proof must name an actual gfx1100-to-gfx1100 moved expert")
        verified.add(txn)
    if not verified:
        raise GateError("no correlated same-ISA identity proof")
    return {"totals": totals, "resolved": len(events),
            "pairs": {f"{a}:{b}": pairs[(a, b)] for a, b in PAIRS},
            "combine_steps_present": combine, "expert_identity_records": len(identity["records"])}

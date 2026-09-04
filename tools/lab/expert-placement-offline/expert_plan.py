#!/usr/bin/env python3
"""Offline MoE expert placement compiler for BigCherry experiments."""
from __future__ import annotations
import argparse, hashlib, json, re, sys
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1
EXPERT_RE = re.compile(r"^blk\.(?P<layer>\d+)\.ffn_(?P<component>gate|up|down|gate_up)_exps(?:\.weight)?$")

def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        v = json.load(f)
    if not isinstance(v, dict):
        raise ValueError(f"{path}: expected JSON object")
    return v

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True); f.write("\n")

def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()

def jsonify(v: Any) -> Any:
    if hasattr(v, "item"):
        try: return jsonify(v.item())
        except (ValueError, TypeError): pass
    if isinstance(v, bytes): return v.decode("utf-8", errors="replace")
    if isinstance(v, (str, int, float, bool)) or v is None: return v
    if isinstance(v, dict): return {str(k): jsonify(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [jsonify(x) for x in v]
    if hasattr(v, "tolist"): return jsonify(v.tolist())
    return str(v)

def metadata_value(reader: Any, key: str) -> Any:
    field = reader.fields.get(key)
    if field is None: return None
    try: return jsonify(field.contents())
    except Exception: return None

def tensor_class(name: str) -> str:
    if EXPERT_RE.match(name): return "routed_expert"
    lower = name.lower()
    if "per_layer_token_embd" in lower or ".ple_" in lower or lower.startswith("ple_"): return "ple_engram"
    if "mtp" in lower: return "mtp"
    return "other"

def inventory_gguf(path: Path, full_sha256: bool = False) -> dict[str, Any]:
    try:
        from gguf import GGUFReader  # type: ignore
    except ImportError as exc:
        raise RuntimeError("gguf-py required; set PYTHONPATH=<llama.cpp>/gguf-py") from exc
    reader = GGUFReader(str(path))
    tensors, layers = [], {}
    for tensor in reader.tensors:
        name = str(tensor.name); shape = [int(x) for x in tensor.shape]; n_bytes = int(tensor.n_bytes)
        rec = {"name": name, "type": str(tensor.tensor_type.name), "shape": shape,
               "n_elements": int(tensor.n_elements), "n_bytes": n_bytes,
               "data_offset": int(tensor.data_offset), "class": tensor_class(name)}
        tensors.append(rec)
        m = EXPERT_RE.match(name)
        if not m: continue
        if len(shape) < 3: raise ValueError(f"{name}: expected >=3 dimensions, got {shape}")
        # gguf-py reverses GGUF dimensions; expert ne[2] is shape[0].
        n_experts = int(shape[0])
        if n_experts <= 0 or n_bytes % n_experts:
            raise ValueError(f"{name}: invalid/non-divisible expert storage")
        il = int(m.group("layer")); comp = m.group("component")
        info = layers.setdefault(il, {"layer": il, "n_experts": n_experts, "components": {}})
        if info["n_experts"] != n_experts: raise ValueError(f"layer {il}: inconsistent expert counts")
        info["components"][comp] = {"name": name, "type": rec["type"], "shape": shape,
                                    "n_bytes": n_bytes, "bytes_per_expert": n_bytes // n_experts}
    expert_layers = []
    for il in sorted(layers):
        info = layers[il]
        bpe = sum(c["bytes_per_expert"] for c in info["components"].values())
        info["bytes_per_expert"] = bpe; info["total_expert_bytes"] = bpe * info["n_experts"]
        expert_layers.append(info)
    totals = {
        "tensor_bytes": sum(t["n_bytes"] for t in tensors),
        "routed_expert_bytes": sum(t["n_bytes"] for t in tensors if t["class"] == "routed_expert"),
        "ple_engram_bytes": sum(t["n_bytes"] for t in tensors if t["class"] == "ple_engram"),
        "mtp_named_bytes": sum(t["n_bytes"] for t in tensors if t["class"] == "mtp"),
    }
    totals["other_bytes"] = totals["tensor_bytes"] - totals["routed_expert_bytes"] - totals["ple_engram_bytes"] - totals["mtp_named_bytes"]
    fp_input = {"file_size": path.stat().st_size,
                "tensors": [{k: t[k] for k in ("name", "type", "shape", "n_bytes", "data_offset")} for t in tensors]}
    out = {"schema": 1, "kind": "bigcherry.expert_inventory",
           "source": {"path": str(path), "file_size": path.stat().st_size, "layout_sha256": canonical_sha256(fp_input)},
           "model": {"architecture": metadata_value(reader, "general.architecture"), "name": metadata_value(reader, "general.name")},
           "totals": totals, "expert_layers": expert_layers, "tensors": tensors}
    if full_sha256:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""): h.update(chunk)
        out["source"]["sha256"] = h.hexdigest()
    return out

def validate_inventory(inv: dict[str, Any]) -> None:
    if inv.get("schema") != 1 or inv.get("kind") != "bigcherry.expert_inventory": raise ValueError("unsupported inventory schema/kind")
    layers = inv.get("expert_layers")
    if not isinstance(layers, list) or not layers: raise ValueError("inventory contains no routed expert layers")
    for layer in layers:
        if int(layer.get("n_experts", 0)) <= 0 or int(layer.get("bytes_per_expert", 0)) <= 0: raise ValueError(f"invalid expert layer: {layer}")

def normalize_topology(top: dict[str, Any]) -> dict[str, Any]:
    if top.get("schema") != 1: raise ValueError("unsupported topology schema")
    devices = top.get("devices")
    if not isinstance(devices, list) or not devices: raise ValueError("topology.devices must be non-empty")
    seen, norm = set(), []
    for raw in devices:
        did = str(raw.get("id", "")).strip(); budget = int(raw.get("expert_budget_bytes", 0)); weight = float(raw.get("placement_weight", 1.0)); roles = [str(x) for x in raw.get("roles", ["expert"])]
        if not did or did in seen: raise ValueError(f"invalid/duplicate device id {did!r}")
        if budget < 0 or weight <= 0: raise ValueError(f"{did}: invalid budget/weight")
        seen.add(did); norm.append({**raw, "id": did, "expert_budget_bytes": budget, "placement_weight": weight, "roles": roles})
    primary = top.get("primary_device")
    if primary is not None and primary not in seen: raise ValueError("primary_device not in devices")
    groups = top.get("tensor_groups", [])
    if not isinstance(groups, list): raise ValueError("tensor_groups must be an array")
    group_ids = set()
    for group in groups:
        gid = str(group.get("id", "")).strip()
        members = [str(x) for x in group.get("members", [])]
        if not gid or gid in group_ids: raise ValueError(f"invalid/duplicate tensor group {gid!r}")
        if not members or any(m not in seen for m in members): raise ValueError(f"tensor group {gid}: invalid members {members}")
        group_ids.add(gid)
    primary_group = top.get("primary_tensor_group")
    if primary_group is not None and primary_group not in group_ids: raise ValueError("primary_tensor_group not in tensor_groups")
    return {**top, "devices": norm}

def compile_plan(inv: dict[str, Any], top: dict[str, Any]) -> dict[str, Any]:
    validate_inventory(inv); top = normalize_topology(top)
    devices = [d for d in top["devices"] if "expert" in d["roles"] and d["expert_budget_bytes"] > 0]
    if not devices: raise ValueError("no expert-capable device with positive budget")
    budgets = {d["id"]: int(d["expert_budget_bytes"]) for d in devices}; remaining = dict(budgets); weights = {d["id"]: float(d["placement_weight"]) for d in devices}
    required = sum(int(l["n_experts"]) * int(l["bytes_per_expert"]) for l in inv["expert_layers"])
    if sum(budgets.values()) < required: raise ValueError(f"budgets total {sum(budgets.values())}, require {required}")
    layer_by_id = {int(l["layer"]): l for l in inv["expert_layers"]}
    units = [(int(l["bytes_per_expert"]), int(l["layer"]), e) for l in inv["expert_layers"] for e in range(int(l["n_experts"]))]
    units.sort(key=lambda x: (-x[0], x[1], x[2]))
    maps, next_slot = {}, {}; placed = {d["id"]: 0 for d in devices}; counts = {d["id"]: 0 for d in devices}
    for n_bytes, il, expert in units:
        candidates = [d for d in devices if remaining[d["id"]] >= n_bytes]
        if not candidates: raise ValueError(f"cannot place layer={il} expert={expert}; remaining={remaining}")
        def score(d: dict[str, Any]) -> tuple[float, int, str]:
            did = d["id"]; return ((remaining[did] / max(budgets[did], 1)) * weights[did], remaining[did], did)
        did = max(candidates, key=score)["id"]; src = layer_by_id[il]
        lm = maps.setdefault(il, {"layer": il, "n_experts": int(src["n_experts"]), "bytes_per_expert": int(src["bytes_per_expert"]), "owners": [None] * int(src["n_experts"]), "local_slots": [None] * int(src["n_experts"])})
        key = (did, il); slot = next_slot.get(key, 0); next_slot[key] = slot + 1
        lm["owners"][expert] = did; lm["local_slots"][expert] = slot
        remaining[did] -= n_bytes; placed[did] += n_bytes; counts[did] += 1
    plan = {"schema": 1, "kind": "bigcherry.expert_map", "model_layout_sha256": inv["source"].get("layout_sha256"), "model_sha256": inv["source"].get("sha256"),
            "policy": {"name": "static-budget-weighted", "version": 1}, "primary_device": top.get("primary_device"), "primary_tensor_group": top.get("primary_tensor_group"), "devices": top["devices"],
            "tensor_groups": top.get("tensor_groups", []), "transport": top.get("transport", []), "layers": [maps[k] for k in sorted(maps)],
            "summary": {"required_expert_bytes": required, "budget_bytes_by_device": budgets, "placed_bytes_by_device": placed, "remaining_bytes_by_device": remaining, "experts_by_device": counts}}
    validate_plan(inv, top, plan); return plan

def validate_plan(inv: dict[str, Any], top: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    validate_inventory(inv); top = normalize_topology(top)
    if plan.get("schema") != 1 or plan.get("kind") != "bigcherry.expert_map": raise ValueError("unsupported map schema/kind")
    if plan.get("model_layout_sha256") != inv.get("source", {}).get("layout_sha256"): raise ValueError("layout fingerprint mismatch")
    devices = {d["id"]: d for d in top["devices"]}; source_layers = {int(x["layer"]): x for x in inv["expert_layers"]}; seen = set(); bytes_by = {d: 0 for d in devices}; count_by = {d: 0 for d in devices}
    layers = plan.get("layers")
    if not isinstance(layers, list): raise ValueError("map layers must be array")
    for lm in layers:
        il = int(lm["layer"])
        if il in seen or il not in source_layers: raise ValueError(f"duplicate/unknown layer {il}")
        seen.add(il); src = source_layers[il]; n = int(src["n_experts"]); bpe = int(src["bytes_per_expert"]); owners = lm.get("owners"); slots = lm.get("local_slots")
        if not isinstance(owners, list) or len(owners) != n or not isinstance(slots, list) or len(slots) != n: raise ValueError(f"layer {il}: wrong map length")
        per_slots = {}
        for e, (owner, slot) in enumerate(zip(owners, slots)):
            if owner not in devices or "expert" not in devices[owner].get("roles", []): raise ValueError(f"layer {il} expert {e}: invalid owner {owner}")
            if not isinstance(slot, int) or slot < 0: raise ValueError(f"layer {il} expert {e}: invalid slot")
            per_slots.setdefault(owner, []).append(slot); bytes_by[owner] += bpe; count_by[owner] += 1
        for did, vals in per_slots.items():
            if sorted(vals) != list(range(len(vals))): raise ValueError(f"layer {il} {did}: non-contiguous local slots")
    if seen != set(source_layers): raise ValueError(f"missing layers {sorted(set(source_layers) - seen)}")
    for did, used in bytes_by.items():
        if used > int(devices[did].get("expert_budget_bytes", 0)): raise ValueError(f"{did}: budget exceeded")
    return {"valid": True, "layers": len(seen), "experts": sum(count_by.values()), "bytes_by_device": bytes_by, "experts_by_device": count_by}

def simulate_trace(plan: dict[str, Any], trace: Path, activation_bytes: int) -> dict[str, Any]:
    if activation_bytes <= 0: raise ValueError("activation_bytes must be >0")
    maps = {int(x["layer"]): x for x in plan.get("layers", [])}; primary = plan.get("primary_device")
    local_devices = {primary} if primary is not None else set()
    primary_group = plan.get("primary_tensor_group")
    if primary_group is not None:
        for group in plan.get("tensor_groups", []):
            if group.get("id") == primary_group:
                local_devices = set(group.get("members", [])); break
    selections, touches, transport = {}, {}, {}; records = remote_records = 0
    with trace.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw or raw.startswith("#"): continue
            rec = json.loads(raw); il = int(rec["layer"]); experts = [int(x) for x in rec["experts"]]; tokens = int(rec.get("tokens", 1)); lm = maps.get(il)
            if lm is None: raise ValueError(f"{trace}:{line_no}: unknown layer {il}")
            touched = set()
            for e in experts:
                if e < 0 or e >= len(lm["owners"]): raise ValueError(f"{trace}:{line_no}: expert {e} out of range")
                did = lm["owners"][e]; selections[did] = selections.get(did, 0) + tokens; touched.add(did)
            records += 1; remote = False
            for did in touched:
                touches[did] = touches.get(did, 0) + 1
                if did not in local_devices:
                    remote = True; transport[did] = transport.get(did, 0) + 2 * activation_bytes * tokens
            remote_records += int(remote)
    return {"records": records, "records_touching_remote_store": remote_records, "remote_record_fraction": remote_records / records if records else 0.0,
            "expert_selections_by_device": selections, "records_touched_by_device": touches, "estimated_transport_bytes_by_device": transport,
            "activation_bytes": activation_bytes, "transport_model": "one activation + one locally reduced output per touched remote store per token"}

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("inventory"); q.add_argument("--gguf", required=True, type=Path); q.add_argument("--out", required=True, type=Path); q.add_argument("--full-sha256", action="store_true")
    q = sub.add_parser("plan"); q.add_argument("--inventory", required=True, type=Path); q.add_argument("--topology", required=True, type=Path); q.add_argument("--out", required=True, type=Path)
    q = sub.add_parser("validate"); q.add_argument("--inventory", required=True, type=Path); q.add_argument("--topology", required=True, type=Path); q.add_argument("--map", required=True, type=Path)
    q = sub.add_parser("simulate"); q.add_argument("--map", required=True, type=Path); q.add_argument("--trace", required=True, type=Path); q.add_argument("--activation-bytes", required=True, type=int); q.add_argument("--out", type=Path)
    return p

def main(argv: Iterable[str] | None = None) -> int:
    a = parser().parse_args(argv)
    try:
        if a.command == "inventory": r = inventory_gguf(a.gguf, a.full_sha256); write_json(a.out, r); print(json.dumps(r["totals"], sort_keys=True))
        elif a.command == "plan": r = compile_plan(read_json(a.inventory), read_json(a.topology)); write_json(a.out, r); print(json.dumps(r["summary"], sort_keys=True))
        elif a.command == "validate": print(json.dumps(validate_plan(read_json(a.inventory), read_json(a.topology), read_json(a.map)), sort_keys=True))
        elif a.command == "simulate":
            r = simulate_trace(read_json(a.map), a.trace, a.activation_bytes)
            if a.out: write_json(a.out, r)
            print(json.dumps(r, sort_keys=True))
        else: raise AssertionError(a.command)
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr); return 2
    return 0

if __name__ == "__main__": raise SystemExit(main())

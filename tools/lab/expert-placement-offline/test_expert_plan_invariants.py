import importlib.util
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("expert_plan", HERE / "expert_plan.py")
EP = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(EP)


def inventory():
    return {
        "schema": 1,
        "kind": "bigcherry.expert_inventory",
        "source": {"layout_sha256": "layout-invariants"},
        "expert_layers": [
            {"layer": 0, "n_experts": 4, "bytes_per_expert": 100},
            {"layer": 1, "n_experts": 4, "bytes_per_expert": 100},
        ],
    }


def topology():
    return {
        "schema": 1,
        "primary_device": "xtx0",
        "primary_tensor_group": "target",
        "devices": [
            {"id": "xtx0", "roles": ["tensor", "expert"], "expert_budget_bytes": 200, "placement_weight": 1.0},
            {"id": "xtx1", "roles": ["tensor", "expert"], "expert_budget_bytes": 200, "placement_weight": 1.0},
            {"id": "r9700", "roles": ["expert"], "expert_budget_bytes": 200, "placement_weight": 1.0},
            {"id": "rx6900", "roles": ["expert"], "expert_budget_bytes": 200, "placement_weight": 1.0},
        ],
        "tensor_groups": [
            {"id": "target", "members": ["xtx0", "xtx1"], "collective": "rccl"},
        ],
        "transport": [
            {"from_group": "target", "to": "r9700", "mode": "backend-copy"},
            {"from_group": "target", "to": "rx6900", "mode": "host-staged"},
        ],
    }


class ExpertPlanInvariantTests(unittest.TestCase):
    def test_exact_total_budget_is_accepted(self):
        inv = inventory()
        top = topology()
        self.assertEqual(
            sum(d["expert_budget_bytes"] for d in top["devices"]),
            sum(l["n_experts"] * l["bytes_per_expert"] for l in inv["expert_layers"]),
        )
        plan = EP.compile_plan(inv, top)
        result = EP.validate_plan(inv, top, plan)
        self.assertTrue(result["valid"])
        self.assertEqual(sum(result["bytes_by_device"].values()), 800)

    def test_layout_fingerprint_mismatch_is_rejected(self):
        inv = inventory()
        plan = EP.compile_plan(inv, topology())
        stale = deepcopy(inv)
        stale["source"]["layout_sha256"] = "different-layout"
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            EP.validate_plan(stale, topology(), plan)

    def test_invalid_non_expert_owner_is_rejected(self):
        inv = inventory()
        top = topology()
        top["devices"].append({
            "id": "storage-only",
            "roles": ["tensor"],
            "expert_budget_bytes": 0,
            "placement_weight": 1.0,
        })
        plan = EP.compile_plan(inv, top)
        plan["layers"][0]["owners"][0] = "storage-only"
        with self.assertRaisesRegex(ValueError, "invalid owner"):
            EP.validate_plan(inv, top, plan)

    def test_noncontiguous_local_slots_are_rejected(self):
        inv = inventory()
        top = topology()
        plan = EP.compile_plan(inv, top)
        layer = plan["layers"][0]
        owner = layer["owners"][0]
        indices = [i for i, value in enumerate(layer["owners"]) if value == owner]
        if len(indices) == 1:
            layer["local_slots"][indices[0]] = 2
        else:
            layer["local_slots"][indices[-1]] = 99
        with self.assertRaisesRegex(ValueError, "non-contiguous local slots"):
            EP.validate_plan(inv, top, plan)

    def test_wrong_expert_map_length_is_rejected(self):
        inv = inventory()
        top = topology()
        plan = EP.compile_plan(inv, top)
        plan["layers"][0]["owners"].pop()
        plan["layers"][0]["local_slots"].pop()
        with self.assertRaisesRegex(ValueError, "wrong map length"):
            EP.validate_plan(inv, top, plan)

    def test_each_expert_has_exactly_one_home(self):
        inv = inventory()
        top = topology()
        plan = EP.compile_plan(inv, top)
        result = EP.validate_plan(inv, top, plan)
        expected = sum(l["n_experts"] for l in inv["expert_layers"])
        self.assertEqual(result["experts"], expected)
        self.assertEqual(sum(result["experts_by_device"].values()), expected)
        for layer in plan["layers"]:
            self.assertEqual(len(layer["owners"]), layer["n_experts"])
            self.assertTrue(all(owner is not None for owner in layer["owners"]))

    def test_mtp_width_scales_remote_transport_linearly(self):
        inv = inventory()
        top = topology()
        plan = EP.compile_plan(inv, top)
        layer = plan["layers"][0]
        remote_index = next(
            i for i, owner in enumerate(layer["owners"])
            if owner not in {"xtx0", "xtx1"}
        )
        remote_owner = layer["owners"][remote_index]
        with tempfile.TemporaryDirectory() as td:
            trace = Path(td) / "trace.jsonl"
            trace.write_text(
                json.dumps({"layer": 0, "experts": [remote_index], "tokens": 1}) + "\n" +
                json.dumps({"layer": 0, "experts": [remote_index], "tokens": 4}) + "\n",
                encoding="utf-8",
            )
            result = EP.simulate_trace(plan, trace, 5120)
        # One activation and one returned partial vector per participating remote store.
        self.assertEqual(result["estimated_transport_bytes_by_device"][remote_owner], 2 * 5120 * (1 + 4))
        self.assertEqual(result["records_touching_remote_store"], 2)

    def test_one_record_touching_two_remote_stores_counts_both_once(self):
        inv = inventory()
        top = topology()
        plan = EP.compile_plan(inv, top)
        pair = None
        for layer in plan["layers"]:
            by_owner = {}
            for expert, owner in enumerate(layer["owners"]):
                if owner not in {"xtx0", "xtx1"}:
                    by_owner.setdefault(owner, expert)
            if len(by_owner) >= 2:
                pair = (layer["layer"], list(by_owner.items())[:2])
                break
        self.assertIsNotNone(pair)
        il, owners = pair
        experts = [expert for _, expert in owners]
        with tempfile.TemporaryDirectory() as td:
            trace = Path(td) / "trace.jsonl"
            trace.write_text(json.dumps({"layer": il, "experts": experts, "tokens": 3}) + "\n", encoding="utf-8")
            result = EP.simulate_trace(plan, trace, 5120)
        for owner, _ in owners:
            self.assertEqual(result["estimated_transport_bytes_by_device"][owner], 2 * 5120 * 3)
        self.assertEqual(result["records_touching_remote_store"], 1)

    def test_out_of_range_trace_expert_is_rejected(self):
        plan = EP.compile_plan(inventory(), topology())
        with tempfile.TemporaryDirectory() as td:
            trace = Path(td) / "trace.jsonl"
            trace.write_text(json.dumps({"layer": 0, "experts": [999], "tokens": 1}) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                EP.simulate_trace(plan, trace, 5120)


if __name__ == "__main__":
    unittest.main()

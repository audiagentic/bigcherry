import importlib.util
import json
import tempfile
import unittest
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
        "source": {"layout_sha256": "layout-1"},
        "expert_layers": [
            {"layer": 0, "n_experts": 4, "bytes_per_expert": 100},
            {"layer": 1, "n_experts": 4, "bytes_per_expert": 200},
        ],
    }


def topology():
    return {
        "schema": 1,
        "primary_device": "xtx0",
        "primary_tensor_group": "target",
        "devices": [
            {"id": "xtx0", "roles": ["tensor", "expert"], "expert_budget_bytes": 600, "placement_weight": 1.0},
            {"id": "xtx1", "roles": ["tensor", "expert"], "expert_budget_bytes": 600, "placement_weight": 1.0},
            {"id": "r9700", "roles": ["expert"], "expert_budget_bytes": 600, "placement_weight": 0.8},
        ],
        "tensor_groups": [{"id": "target", "members": ["xtx0", "xtx1"], "collective": "rccl"}],
    }


class ExpertPlanTests(unittest.TestCase):
    def test_compile_is_deterministic_and_valid(self):
        a = EP.compile_plan(inventory(), topology())
        b = EP.compile_plan(inventory(), topology())
        self.assertEqual(a, b)
        result = EP.validate_plan(inventory(), topology(), a)
        self.assertTrue(result["valid"])
        self.assertEqual(result["experts"], 8)
        self.assertEqual(sum(result["bytes_by_device"].values()), 1200)

    def test_budget_failure(self):
        top = topology()
        for device in top["devices"]:
            device["expert_budget_bytes"] = 100
        with self.assertRaises(ValueError):
            EP.compile_plan(inventory(), top)

    def test_validate_rejects_missing_layer(self):
        plan = EP.compile_plan(inventory(), topology())
        plan["layers"].pop()
        with self.assertRaises(ValueError):
            EP.validate_plan(inventory(), topology(), plan)

    def test_simulation_treats_tensor_group_as_local(self):
        plan = EP.compile_plan(inventory(), topology())
        lm = plan["layers"][1]
        xtx1_expert = next((i for i, d in enumerate(lm["owners"]) if d == "xtx1"), None)
        remote_expert = next((i for i, d in enumerate(lm["owners"]) if d == "r9700"), None)
        self.assertIsNotNone(xtx1_expert)
        self.assertIsNotNone(remote_expert)
        with tempfile.TemporaryDirectory() as td:
            trace = Path(td) / "trace.jsonl"
            trace.write_text(
                json.dumps({"layer": 1, "experts": [xtx1_expert]}) + "\n" +
                json.dumps({"layer": 1, "experts": [remote_expert]}) + "\n",
                encoding="utf-8",
            )
            result = EP.simulate_trace(plan, trace, 5120)
        self.assertEqual(result["records"], 2)
        self.assertEqual(result["records_touching_remote_store"], 1)
        self.assertEqual(result["estimated_transport_bytes_by_device"]["r9700"], 10240)
        self.assertNotIn("xtx1", result["estimated_transport_bytes_by_device"])


if __name__ == "__main__":
    unittest.main()

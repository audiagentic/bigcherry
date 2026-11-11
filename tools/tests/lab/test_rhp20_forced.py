"""No-device RHP20 forced-churn acceptance tests."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from bigcherry.bench.rhp20_forced import (
    GateError,
    experiment_evidence,
    manifest_evidence,
    receipt_evidence,
    recipe_evidence,
)

TAIL = "flashnext-2xtx-rdna2-tail"
MODEL = str(Path(os.environ.get("BC_MODEL_ROOT", "models")) / "radiance/StillDeadcode/qwen3.8-next-flash-fp8-iq4r-moe/qwen3.8-next-flash-fp8-iq4r-moe.rad")
RECORD = Path(os.environ.get("BC_WORK_ROOT", "work")) / "flashnext/manifest-1111-131338-1271995.json"
REPO = Path(__file__).resolve().parents[3]


def fake_manifest():
    owned = [[(0, 1), (0, 3), (34, 407)], [(0, 0), (0, 2)], [(0, 6), (0, 8)]]
    bdfs = ["0000:03:00.0", "0000:06:00.0", "0000:17:00.0"]
    return {"spec": {
        "intent": {"layout": TAIL, "model": MODEL},
        "plan": {"state": "measured", "owner_map": {
            "owner_map_sha256": "a" * 64,
            "helpers": [{"count": len(items), "bytes": len(items) * 2534400,
                         "experts": [list(x) for x in items]} for items in owned]}},
        "facts": {"gpus": [{"bdf": b, "arch": "gfx1100" if i < 2 else "gfx1030"}
                           for i, b in enumerate(bdfs)]},
        "handoff": {"placement": {"strategy": "tail"}},
        "processes": {"engine": {"execution": {"num_speculative_tokens": 3}},
                      "helpers": [{"gpu": b, "tier": "head" if i < 2 else "tail",
                                   "formats": ["w4nl64", "fp8_e4m3_block"] if i < 2 else ["w4nl64"],
                                   "vram_budget_mib": 1024} for i, b in enumerate(bdfs)]}}}


def sample_logs(n=1000, *, combine=True):
    pairs = ((0, 1), (0, 2), (1, 2))
    events = []
    for k in range(n):
        a, b = pairs[k % len(pairs)]
        events.append(f"forced resolved transaction={k+1} epoch={k+2} swap=0:1:0:0 helpers={a}:{b}")
    return {i: ("\n".join(events) + "\n" if i == 0 else "") +
            f"forced placement helper={i} attempted={n} committed={n} aborted=0 inflight=0\n" +
            ("r3_moe_worker: combine steps 45, CPU 10.0 ms\n" if combine and i == 0 else "")
            for i in range(3)}


class RHP20Tests(unittest.TestCase):
    def test_recipe_and_protected_owner_not_swappable(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.rad"
            recipe = Path(tmp) / "qwen.recipe"
            recipe.write_text("blk.34.ffn_*_exps.407.weight cast dtype=bf16\n"
                              "blk.*.ffn_*_exps.*.weight gptq table=w4nl group=64 scale=fp8_e4m3\n")
            protected, digest = recipe_evidence(model)
            self.assertEqual(protected, {(34, 407)})
            self.assertEqual(len(digest), 64)
            witness = manifest_evidence(fake_manifest(), TAIL, protected)
            self.assertEqual(witness["pair_witnesses"]["0:2"], (0, 1, 6))
            self.assertEqual(witness["counts"], [2, 2, 2])

    def test_mismatched_arch_and_capacity_fail_closed(self):
        m = fake_manifest()
        m["spec"]["facts"]["gpus"][-1]["arch"] = "gfx1201"
        with self.assertRaisesRegex(GateError, "architecture"):
            manifest_evidence(m, TAIL, set())
        m = fake_manifest()
        m["spec"]["processes"]["helpers"][2]["vram_budget_mib"] = 0
        with self.assertRaisesRegex(GateError, "budget"):
            manifest_evidence(m, TAIL, set())
        m = fake_manifest()
        m["spec"]["plan"]["owner_map"]["helpers"][2]["experts"] = []
        m["spec"]["plan"]["owner_map"]["helpers"][2]["count"] = 0
        m["spec"]["plan"]["owner_map"]["helpers"][2]["bytes"] = 0
        with self.assertRaisesRegex(GateError, "zero-owner"):
            manifest_evidence(m, TAIL, set())
        m = fake_manifest()
        m["spec"]["processes"]["helpers"][2]["formats"] += ["fp8_e4m3_block"]
        with self.assertRaisesRegex(GateError, "format"):
            manifest_evidence(m, TAIL, set())
        with self.assertRaisesRegex(GateError, "different layout"):
            manifest_evidence(fake_manifest(), TAIL + "-combine", set())

    def test_experiment_counts_issued_not_committed(self):
        path = REPO / "tools/lab/radiance/rhp20/tail.experiment.json"
        exp = json.loads(path.read_text())
        evidence = experiment_evidence(exp)
        self.assertEqual(evidence["scheduled"], 1002)
        self.assertEqual(evidence["minimum_completed_lane_requests"], 104000)
        exp["placement"]["forced"]["repeat"] = 333
        with self.assertRaisesRegex(GateError, "1000"):
            experiment_evidence(exp)
        exp["placement"]["forced"]["repeat"] = 334
        exp["placement"]["forced"]["steps"][1]["a_expert"] = 3
        with self.assertRaisesRegex(GateError, "auto-swap"):
            experiment_evidence(exp)

    def test_receipts_require_real_commits_and_expert_equality(self):
        evidence = {"records": [{"expert": "0:1", "transaction_id": 1, "before_epoch": 1, "after_epoch": 2, "source_isa": "gfx1100",
                                 "target_isa": "gfx1100", "before_sha256": "b"*64,
                                 "after_sha256": "b"*64}]}
        self.assertEqual(receipt_evidence(sample_logs(), combine=True, identity=evidence)["resolved"], 1000)
        with self.assertRaisesRegex(GateError, "bit-identity"):
            receipt_evidence(sample_logs(), combine=True)
        with self.assertRaisesRegex(GateError, "committed"):
            receipt_evidence(sample_logs(999), combine=True, identity=evidence)
        with self.assertRaisesRegex(GateError, "combine"):
            receipt_evidence(sample_logs(combine=False), combine=True, identity=evidence)
        evidence["records"][0]["after_sha256"] = "c"*64
        with self.assertRaisesRegex(GateError, "hashes differ"):
            receipt_evidence(sample_logs(), combine=False, identity=evidence)

    @unittest.skipUnless(RECORD.is_file() and Path(MODEL).is_file(), "recorded manifest/model absent")
    def test_recorded_tail_owner_map_if_available(self):
        protected, _ = recipe_evidence(Path(MODEL))
        evidence = manifest_evidence(json.loads(RECORD.read_text()), TAIL, protected)
        self.assertEqual(evidence["owner_map_sha256"],
                         "28f43e45923dd6e0f78e9ad6e631c87af8675acf0a46b11223201a4bb09c1f4d")
        self.assertEqual(evidence["expert_bytes"], 2534400)
        self.assertEqual(set(evidence["pair_witnesses"]), {"0:1", "0:2", "1:2"})


if __name__ == "__main__":
    unittest.main()

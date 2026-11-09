"""Synthetic-only tests for the XDP03 conservative stage-aware quant selector."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lab.radiance import xtx27_quant_plan as qp
from lab.radiance import xtx27_quant_select as qs


def sample():
    config = {
        "model_type": "qwen3_5",
        "text_config": {
            "num_hidden_layers": 64,
            "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 16,
            "max_position_embeddings": 262144,
            "num_key_value_heads": 4,
            "head_dim": 256,
        },
    }
    measurements = {
        "data_kind": "synthetic",
        "evidence_id": "SYNTHETIC_DO_NOT_USE",
        "source_config_sha256": qp.sha256(json.dumps(config, sort_keys=True).encode()),
        "native_context": 262144,
        "budgets": {
            "xtx": {"vram_mib": 24576, "runtime_reserve_mib": 2048},
            "r9700": {"vram_mib": 32768, "runtime_reserve_mib": 2048},
        },
        "layers": [
            {
                "layer": index,
                "xtx": [
                    {"format": "mxfp4", "prefill_us": 10, "decode_us": 10, "weights_mib_per_card": 40},
                    {"format": "w4", "prefill_us": 8, "decode_us": 12, "weights_mib_per_card": 25},
                ],
                "r9700": [
                    {"format": "mxfp4", "prefill_us": 8, "decode_us": 8, "weights_mib_per_card": 60},
                    {"format": "fp8", "prefill_us": 5, "decode_us": 6, "weights_mib_per_card": 100},
                ],
            } for index in range(64)
        ],
    }
    return config, measurements


class QuantSelectTests(unittest.TestCase):
    def test_native_context_kv_fp8_is_stage_owned(self):
        cfg, _ = sample()
        got = qs.kv_residency_mib(cfg, cut=40, tokens=262144)
        self.assertEqual(got, {"xtx": 2560, "r9700": 3072})
        for cut in (32, 48):
            got = qs.kv_residency_mib(cfg, cut=cut, tokens=262144)
            self.assertEqual(got["xtx"] * 2 + got["r9700"], 8192)

    def test_prefill_selects_mixed_quant_using_metrics_not_gpu_label(self):
        cfg, data = sample()
        plan = qs.select(cfg, data, cut=40, tokens=262144)
        x, r = plan["stages"]
        self.assertEqual(plan["status"], "PREVIEW_ONLY_NOT_END_TO_END_QUALIFIED")
        self.assertEqual(len(x["chosen"]), 40)
        self.assertEqual(len(r["chosen"]), 24)
        self.assertEqual({p["format"] for p in x["chosen"]}, {"w4"})
        self.assertEqual({p["format"] for p in r["chosen"]}, {"fp8"})
        self.assertEqual(x["tp"], 2)
        self.assertEqual(r["tp"], 1)
        self.assertEqual(x["logical_fp8_kv_mib_per_card"], 2560)

    def test_decode_chooses_distinct_formats_from_prefill(self):
        cfg, data = sample()
        pre = qs.select(cfg, data, cut=40, tokens=131072)
        dec = qs.select(cfg, data, cut=40, tokens=131072, objective="decode")
        self.assertEqual({p["format"] for p in pre["stages"][0]["chosen"]}, {"w4"})
        self.assertEqual({p["format"] for p in dec["stages"][0]["chosen"]}, {"mxfp4"})
        self.assertNotEqual(pre["objective"], dec["objective"])
        self.assertEqual(pre["status"], dec["status"])

    def test_capacity_forces_more_compact_variant(self):
        cfg, data = sample()
        # At 262K the R9700 suffix KV is 3072 MiB; the runtime reserve
        # is 1000 MiB. A 6000 MiB device leaves 1928 MiB for weights.
        data["budgets"]["r9700"] = {"vram_mib": 6000, "runtime_reserve_mib": 1000}
        result = qs.select(cfg, data, cut=40, tokens=262144)
        choices = result["stages"][1]["chosen"]
        self.assertEqual(len(choices), 24)
        self.assertGreater(sum(p["format"] == "mxfp4" for p in choices), 0)
        self.assertGreater(sum(p["format"] == "fp8" for p in choices), 0)
        self.assertLessEqual(result["stages"][1]["conservative_weights_mib"], 1928)

    def test_unfittable_memory_fail_closed(self):
        cfg, data = sample()
        data["budgets"]["xtx"]["vram_mib"] = 2048
        with self.assertRaisesRegex(ValueError, "no weight space"):
            qs.select(cfg, data, cut=40, tokens=262144)
        data["budgets"]["xtx"]["vram_mib"] = 2048 + 2560 + 200
        with self.assertRaisesRegex(ValueError, "no valid weight"):
            qs.select(cfg, data, cut=40, tokens=262144)

    def test_no_1m_in_unqualified_262k_model(self):
        cfg, data = sample()
        with self.assertRaisesRegex(ValueError, "native limit"):
            qs.select(cfg, data, cut=40, tokens=1048576)

    def test_provenance_and_incomplete_metrics_rejected(self):
        cfg, data = sample()
        data["source_config_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "config"):
            qs.select(cfg, data, cut=40, tokens=131072)
        cfg, data = sample()
        data["layers"].pop()
        with self.assertRaisesRegex(ValueError, "all layer records"):
            qs.select(cfg, data, cut=40, tokens=131072)
        cfg, data = sample()
        data["data_kind"] = "measured"
        data.pop("evidence_id")
        with self.assertRaisesRegex(ValueError, "evidence"):
            qs.select(cfg, data, cut=40, tokens=131072)

    def test_bad_candidates_and_duplicates_rejected(self):
        cfg, data = sample()
        data["layers"][0]["xtx"][0]["prefill_us"] = float("nan")
        with self.assertRaises(ValueError):
            qs.select(cfg, data, cut=40, tokens=131072)
        cfg, data = sample()
        data["layers"][0]["xtx"].append(dict(data["layers"][0]["xtx"][0]))
        with self.assertRaisesRegex(ValueError, "repeated"):
            qs.select(cfg, data, cut=40, tokens=131072)

    def test_real_cli_read_only_synthetic_plan(self):
        cfg, data = sample()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            conf = root / "config.json"
            metrics = root / "metrics.json"
            conf.write_text(json.dumps(cfg))
            metrics.write_text(json.dumps(data))
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(qs.main([
                    "--config", str(conf), "--metrics", str(metrics),
                    "--cut", "40", "--tokens", "131072",
                ]), 0)
            report = json.loads(output.getvalue())
            self.assertEqual(report["data_kind"], "synthetic")
            self.assertEqual(report["cut"], 40)
            self.assertFalse((root / "selection.rad").exists())


if __name__ == "__main__":
    unittest.main()

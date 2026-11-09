"""Card-free quant recipe and stage-placement safety checks for XDP03."""
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


def fixture(root: Path, *, layers: int = 64, model_type: str = "qwen3_5") -> Path:
    source = root / "bf16"
    source.mkdir()
    config = {
        "model_type": model_type,
        "text_config": {
            "num_hidden_layers": layers,
            "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * (layers // 4),
            "max_position_embeddings": 262144,
        },
    }
    (source / "config.json").write_text(json.dumps(config))
    (source / "model.safetensors.index.json").write_text('{"weight_map": {"x": "model-00001-of-00018.safetensors"}}')
    (source / "model-00001-of-00018.safetensors").write_bytes(b"fixture")
    return source


class QuantPlanTests(unittest.TestCase):
    def test_mixed_has_compatible_int4_weights_and_explicit_fallback(self):
        recipe, q = qp.build_recipe(64, "mixed", 40, 8, False)
        self.assertEqual(q["w4_ffn_layer_indices"], list(range(8, 40)))
        self.assertEqual(len(q["ffn_overrides"]), 64)
        self.assertIn("blk.8.ffn_gate_up.weight  rtn  codes=u4 zero=u8 group=128 scale=f16", recipe)
        self.assertNotIn("blk.7.ffn_gate_up.weight  rtn  codes=u4", recipe)
        self.assertIn("blk.*.ssm_out.weight  rtn  codes=fp4_e2m1 group=32 scale=e8m0", recipe)
        self.assertIn("output.weight  rtn  codes=fp8_e4m3 block=128x128 scale=bf16", recipe)
        self.assertLess(recipe.index("blk.8.ffn_down.weight"), recipe.index("blk.*.ffn_down.weight"))

    def test_baseline_same_source_all_mxfp4_no_w4(self):
        recipe, q = qp.build_recipe(64, "baseline", 40, 8, False)
        self.assertFalse(q["ffn_overrides"])
        self.assertFalse(q["w4_ffn_layer_indices"])
        self.assertFalse(q["fp8_ffn_layer_indices"])
        self.assertNotIn("codes=u4", recipe)
        self.assertEqual(recipe.count("codes=fp4_e2m1"), len(qp.LINEARS))

    def test_optional_r9700_fp8_is_suffix_only(self):
        recipe, q = qp.build_recipe(64, "mixed", 40, 8, True)
        self.assertEqual(q["fp8_ffn_layer_indices"], list(range(40, 64)))
        self.assertIn("blk.40.ffn_down.weight  rtn  codes=fp8_e4m3", recipe)
        self.assertNotIn("blk.39.ffn_down.weight  rtn  codes=fp8_e4m3", recipe)
        self.assertEqual(len(q["ffn_overrides"]), 2 * (64 - 8))

    def test_invalid_settings_fail_closed(self):
        cases = [
            ("baseline", 0, 0, False), ("mixed", 64, 8, False),
            ("mixed", 40, -1, False), ("mixed", 40, 40, False),
            ("baseline", 40, 8, True), ("unknown", 40, 8, False),
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                qp.build_recipe(64, *case)

    def test_reject_non_dense_or_wrong_geometry(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            source = fixture(path, model_type="qwen4exp")
            with self.assertRaises(ValueError):
                qp.make_plan(source, path / "scratch")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            source = fixture(path, layers=60)
            with self.assertRaises(ValueError):
                qp.make_plan(source, path / "scratch")

    def test_missing_checkpoint_or_source_shards_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            source = fixture(path)
            (source / "model-00001-of-00018.safetensors").unlink()
            with self.assertRaisesRegex(ValueError, "shards"):
                qp.make_plan(source, path / "scratch")

    def test_emit_is_opt_in_and_never_calls_converter(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            source = fixture(path)
            out_dir = path / "scratch"
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(qp.main([
                    "--checkpoint", str(source), "--output-dir", str(out_dir),
                    "--convert-bin", "/nonexistent/DO-NOT-EXECUTE",
                ]), 0)
            plan = json.loads(output.getvalue())
            self.assertFalse(out_dir.exists())
            self.assertFalse(plan["target"]["runtime_full_layer_staging_implemented"])
            self.assertFalse(plan["target"]["runtime_gpu_quant_switching_implemented"])
            self.assertEqual(plan["next_command_requires_exclusive_gpu_lease_if_using_gpu_kernels"][0],
                             "/nonexistent/DO-NOT-EXECUTE")
            self.assertEqual(plan["model_native_context"], 262144)
            self.assertEqual(plan["checkpoint_config_sha256"],
                             qp.sha256((source / "config.json").read_bytes()))

    def test_write_manifest_and_recipe_are_reproducible(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            source = fixture(path)
            output = path / "scratch"
            manifest, recipe = qp.make_plan(source, output)
            qp.write_plan(output, manifest, recipe)
            self.assertEqual(Path(manifest["recipe_path"]).read_text(), recipe)
            self.assertEqual(json.loads(Path(manifest["recipe_path"]).with_suffix(".json").read_text()),
                             manifest)
            qp.write_plan(output, manifest, recipe)
            with self.assertRaises(FileExistsError):
                qp.write_plan(output, manifest, recipe + "\n# conflicting experiment")

    def test_plan_command_requests_no_speculation_and_no_conversion(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            source = fixture(path)
            manifest, _ = qp.make_plan(source, path / "scratch", profile="mixed",
                                       cut=32, preserve=4, suffix_fp8=True,
                                       home=path / "engine-home")
            command = manifest["next_command_requires_exclusive_gpu_lease_if_using_gpu_kernels"]
            self.assertIn("--plan-only", command)
            self.assertEqual(command[command.index("--max-spec") + 1], "0")
            self.assertIn("quantization_config.quant_method=fp8", command)
            self.assertIn("--home", command)
            self.assertEqual(manifest["target"]["xtx_gfx1100_tp2_layers"], [0, 32])
            self.assertEqual(manifest["target"]["r9700_gfx1201_tp1_layers"], [32, 64])
            self.assertEqual(len(manifest["quantization"]["w4_ffn_layer_indices"]), 28)

    def test_prefix_boundary_changes_only_w4_override_range(self):
        a, qa = qp.build_recipe(64, "mixed", 32, 8, False)
        b, qb = qp.build_recipe(64, "mixed", 40, 8, False)
        self.assertIn("blk.39.ffn_gate_up.weight  rtn  codes=u4", b)
        self.assertNotIn("blk.39.ffn_gate_up.weight  rtn  codes=u4", a)
        self.assertEqual(qa["w4_ffn_layer_indices"], list(range(8, 32)))
        self.assertEqual(qb["w4_ffn_layer_indices"], list(range(8, 40)))


    def test_selector_can_emit_provenance_checked_per_layer_recipe(self):
        from lab.radiance import xtx27_quant_select as qs

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = fixture(root)
            config = json.loads((source / "config.json").read_text())
            # A measured-metadata fixture to test wiring, never a claimed run.
            config["text_config"]["num_key_value_heads"] = 4
            config["text_config"]["head_dim"] = 256
            (source / "config.json").write_text(json.dumps(config))
            metrics = {
                "data_kind": "measured", "evidence_id": "TEST_FIXTURE_ONLY",
                "native_context": 262144,
                "source_config_sha256": qp.sha256(json.dumps(config, sort_keys=True).encode()),
                "budgets": {
                    "xtx": {"vram_mib": 24576, "runtime_reserve_mib": 2048},
                    "r9700": {"vram_mib": 32768, "runtime_reserve_mib": 2048},
                },
                "layers": [
                    {"layer": i,
                     "xtx": [{"format": "w4" if i % 2 else "mxfp4",
                              "prefill_us": 1, "decode_us": 1,
                              "weights_mib_per_card": 5}],
                     "r9700": [{"format": "fp8" if i % 2 else "mxfp4",
                                "prefill_us": 1, "decode_us": 1,
                                "weights_mib_per_card": 10}]}
                    for i in range(64)
                ],
            }
            selected = qs.select(config, metrics, cut=40, tokens=131072)
            manifest, recipe = qp.make_plan(
                source, root / "scratch", profile="selected", cut=40, selected=selected,
            )
            self.assertIn("blk.1.ffn_down.weight  rtn  codes=u4", recipe)
            self.assertIn("blk.41.ffn_down.weight  rtn  codes=fp8_e4m3", recipe)
            self.assertNotIn("blk.40.ffn_down.weight  rtn  codes=fp8_e4m3", recipe)
            self.assertTrue(manifest["selection_sha256"])
            self.assertEqual(manifest["selection_evidence_id"], "TEST_FIXTURE_ONLY")
            qp.write_plan(root / "scratch", manifest, recipe)
            self.assertTrue(Path(manifest["recipe_path"]).exists())
            self.assertFalse(Path(manifest["candidate_model_path"]).exists())

    def test_selector_rejects_synthetic_wrong_checkpoint_and_bad_layers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = fixture(root)
            config = json.loads((source / "config.json").read_text())
            dummy = {
                "schema": "xtx27.quant.selection.v1",
                "status": "PREVIEW_ONLY_NOT_END_TO_END_QUALIFIED",
                "data_kind": "synthetic", "evidence_id": "fixture", "cut": 40,
                "native_context_tokens": 262144,
                "source_config_canonical_sha256": qp.sha256(
                    json.dumps(config, sort_keys=True).encode()
                ),
                "stages": [
                    {"stage": "xtx", "chosen": [
                        {"layer": i, "format": "w4"} for i in range(40)]},
                    {"stage": "r9700", "chosen": [
                        {"layer": i, "format": "mxfp4"} for i in range(40, 64)]},
                ],
            }
            with self.assertRaisesRegex(ValueError, "measured"):
                qp.selected_formats(dummy, config, 40)
            dummy["data_kind"] = "measured"
            dummy["source_config_canonical_sha256"] = "invalid"
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                qp.selected_formats(dummy, config, 40)
            dummy["source_config_canonical_sha256"] = qp.sha256(
                json.dumps(config, sort_keys=True).encode()
            )
            dummy["stages"][1]["chosen"][0]["layer"] = 41
            with self.assertRaisesRegex(ValueError, "layer"):
                qp.selected_formats(dummy, config, 40)


if __name__ == "__main__":
    unittest.main()

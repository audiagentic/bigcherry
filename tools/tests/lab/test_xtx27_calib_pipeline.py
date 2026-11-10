"""Card-free VTP06 corpus and resumable Hessian campaign tests.

All fixtures are synthetic; no GPU, model, tokenizer, or network is used.
"""
from __future__ import annotations

import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[2] / "lab" / "radiance"


def load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


corpus = load("xtx27_calib_corpus")
ranges = load("xtx27_calib_ranges")


def fake_documents(mode="pilot"):
    def item(i, category, split):
        return {
            "id": f"{split}-{i}",
            "prompt": (f"{split} document {i}: " + "meaningful example input " * 5)
            + (" extra long-text " * 600 if category == "long_context" and i == 4 else ""),
            "source": f"synthetic-source-{i % 3}",
            "license": "TEST_SYNTHETIC_ONLY",
            "category": category,
            "split": split,
        }
    if mode == "pilot":
        return [item(0, "language", "calibration"),
                item(1, "code", "calibration")]
    cats = ["language", "code", "math", "multilingual", "long_context"]
    out = [item(i, cats[i % len(cats)], "calibration") for i in range(200)]
    out += [item(i, cats[i % len(cats)], "holdout") for i in range(50)]
    return out


def write_source(path, records):
    path.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in records))


def write_tensor(path, *, k=128, rows=7, neg=False, nan=False):
    elements = k * (k + 1) // 2
    data = [0.25] * elements
    if neg:
        data[0] = -2.0
    if nan:
        data[15] = float("nan")
    payload = b"".join(struct.pack("<f", x) for x in data)
    hdr = {
        "__metadata__": {
            "k": str(k), "rows": str(rows),
            "linears": "synthetic.0.test", "layout": "upper triangle, row-major, packed"
        },
        "hessian_upper": {
            "dtype": "F32", "shape": [elements],
            "data_offsets": [0, elements * 4]
        }
    }
    head = json.dumps(hdr, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(head)) + head + payload)


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.input = self.root / "candidate.jsonl"
        self.checkpoint = self.root / "model"
        self.checkpoint.mkdir()
        (self.checkpoint / "tokenizer.json").write_text('{"synthetic": true}')
        (self.checkpoint / "tokenizer_config.json").write_text('{"synthetic": true}')

    def test_pilot_is_non_quality_with_distinct_holdout(self):
        docs = fake_documents()
        docs.append({
            **docs[1], "id": "hold-0", "split": "holdout",
            "prompt": "Independently held out synthetic evaluation text " * 3
        })
        docs[-1]["score_from"] = 3  # native-token scoring begins after the prompt.
        write_source(self.input, docs)
        bundle = self.root / "bundle"
        result = corpus.prepare(self.input, self.checkpoint, bundle, "pilot")
        self.assertEqual(result["status"], "PILOT_ONLY_NOT_QUALITY_QUALIFIED")
        self.assertFalse(result["native_token_counts_verified"])
        self.assertFalse(result["iq6r_quality_approved"])
        self.assertEqual(result["overview"]["documents"], {"calibration": 2, "holdout": 1})
        calibration = (bundle / "calibration.jsonl").read_text().splitlines()
        holdout = (bundle / "holdout.jsonl").read_text().splitlines()
        self.assertEqual(len(calibration), 2)
        self.assertEqual(len(holdout), 1)
        self.assertNotIn("hold-0", "\n".join(calibration))
        self.assertEqual(json.loads(holdout[0])["score_from"], 3)
        self.assertEqual(corpus.inspect_bundle(bundle, self.checkpoint), result)
        with self.assertRaises(FileExistsError):
            corpus.prepare(self.input, self.checkpoint, bundle, "pilot")
        (bundle / "calibration.jsonl").write_text("drift")
        with self.assertRaisesRegex(ValueError, "digest drift"):
            corpus.inspect_bundle(bundle, self.checkpoint)

    def test_candidate_is_structural_only(self):
        write_source(self.input, fake_documents("candidate"))
        bundle = self.root / "bundle"
        result = corpus.prepare(self.input, self.checkpoint, bundle, "candidate")
        self.assertEqual(result["overview"]["documents"]["calibration"], 200)
        self.assertEqual(result["overview"]["documents"]["holdout"], 50)
        self.assertFalse(result["licensed_corpus_suitability_manually_approved"])
        self.assertFalse(result["iq6r_quality_approved"])
        self.assertFalse(result["native_token_counts_verified"])

    def test_candidate_rejects_short_nonrepresentative_source(self):
        write_source(self.input, fake_documents("pilot"))
        with self.assertRaisesRegex(ValueError, ">=200"):
            corpus.checked_records(self.input, mode="candidate")

    def test_deduplication_across_splits(self):
        docs = fake_documents()
        docs.append({**docs[0], "id": "distinct-id", "split": "holdout",
                     "prompt": docs[0]["prompt"].upper()})
        write_source(self.input, docs)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            corpus.checked_records(self.input, mode="pilot")

    def test_score_from_is_holdout_only(self):
        docs = fake_documents()
        docs[0]["score_from"] = 12
        write_source(self.input, docs)
        with self.assertRaisesRegex(ValueError, "held-out"):
            corpus.checked_records(self.input, mode="pilot")

    def test_missing_license_and_malformed_json_fail(self):
        docs = fake_documents()
        docs[0]["license"] = "unknown"
        write_source(self.input, docs)
        with self.assertRaisesRegex(ValueError, "license"):
            corpus.checked_records(self.input, mode="pilot")
        self.input.write_text("{this is not JSON}\n")
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            corpus.checked_records(self.input, mode="pilot")

    def test_invalid_type_and_source_output_collision(self):
        self.input.write_text("[]\n")
        with self.assertRaises(TypeError):
            corpus.checked_records(self.input, mode="pilot")
        write_source(self.input, fake_documents())
        with self.assertRaisesRegex(ValueError, "overwrite source"):
            corpus.prepare(self.input, self.checkpoint, self.input, "pilot")


class RangeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.checkpoint = self.root / "bf16"
        self.checkpoint.mkdir()
        cfg = {"model_type": "qwen3_5", "text_config": {
            "num_hidden_layers": 64,
            "layer_types": ["full_attention" if i % 4 == 3 else "linear_attention"
                            for i in range(64)],
            "max_position_embeddings": 262144
        }}
        (self.checkpoint / "config.json").write_text(json.dumps(cfg))
        (self.checkpoint / "model.safetensors.index.json").write_text('{"fake":"index"}')
        (self.checkpoint / "tokenizer.json").write_text('{"fake":"tokenizer"}')
        (self.checkpoint / "tokenizer_config.json").write_text('{"fake":"cfg"}')
        audit = {
            "status": "STRUCTURALLY_VALID_FULL_SHARDS_SHA256_FINGERPRINTED",
            "payload_integrity": "FULL_SHARD_SHA256_FINGERPRINTED_NOT_EXTERNALLY_VERIFIED",
            "shard_count": 18, "total_tensor_payload_bytes": 55562855904,
            "config_sha256": corpus.sha256_file(self.checkpoint / "config.json"),
            "index_sha256": corpus.sha256_file(
                self.checkpoint / "model.safetensors.index.json"),
            "shards": {}
        }
        for n in range(1, 19):
            name = f"model-{n:05d}-of-00018.safetensors"
            (self.checkpoint / name).write_bytes(b"fake shard bytes")
            audit["shards"][name] = {
                "bytes": len(b"fake shard bytes"), "full_shard_sha256": "a" * 64
            }
        self.audit_path = self.root / "audit.json"
        self.audit_path.write_text(json.dumps(audit))
        self.source = self.root / "input.jsonl"
        write_source(self.source, fake_documents())
        self.bundle_dir = self.root / "bundle"
        corpus.prepare(self.source, self.checkpoint, self.bundle_dir, "pilot")
        self.build = self.root / "build"
        (self.build / "bin").mkdir(parents=True)
        (self.build / "radiance_home").mkdir()
        (self.build / "bin" / "radiance").write_text("fake exe")
        self.reference = self.root / "reference"
        self.reference.mkdir()
        for i in range(4):
            write_tensor(self.reference /
                f"model.language_model.layers.0.input-{i}.safetensors")
        self.campaign_dir = self.root / "range-campaign"

    def make_plan(self, chunk_size=2):
        return ranges.plan(
            self.bundle_dir, self.checkpoint, self.audit_path, self.build,
            self.reference, self.campaign_dir, self.root,
            chunk_size=chunk_size, tp=1, devices="2", host_pool_mib=4096,
            reserve_gib=8,
        )

    def test_card_free_plan_and_no_overwrite(self):
        doc = self.make_plan()
        self.assertEqual(doc["ranges"][0], [0, 2])
        self.assertEqual(doc["ranges"][-1], [62, 64])
        self.assertEqual(len(doc["ranges"]), 32)
        self.assertFalse(doc["source_payload_rehashed_now"])
        self.assertFalse(doc["quality_approved"])
        self.assertEqual(ranges.campaign_status(self.campaign_dir)["validated_layers"], 0)
        with self.assertRaisesRegex(ValueError, "must not exist"):
            self.make_plan()

    def test_preflight_rejects_low_ram_before_gpu_check(self):
        self.make_plan()
        with (patch.object(ranges, "mem_available", return_value=1 * ranges.GIB),
              patch.object(ranges, "check_gpu_lease",
                           side_effect=AssertionError("GPU check must not execute")),
              self.assertRaisesRegex(ValueError, "RAM preflight BLOCKED")):
            ranges.run_range(
                self.campaign_dir, 0, 2, 1, lease_purpose="VTP06/unit-test-pilot")
        self.assertFalse((self.campaign_dir / "ranges").exists())

    def test_unplanned_range_rejected(self):
        self.make_plan()
        with self.assertRaisesRegex(ValueError, "not in"):
            ranges.run_range(
                self.campaign_dir, 1, 3, 1, lease_purpose="VTP06/unit-test-pilot")

    def create_run(self, layer_missing=None, nonfinite=False, negative=False,
                   lo=0, hi=2):
        self.make_plan()
        folder = self.campaign_dir / "ranges" / f"{lo:02d}-{hi:02d}" / "attempt-01"
        (folder / "hessians").mkdir(parents=True)
        for idx in range(lo, hi):
            if idx == layer_missing:
                continue
            base = f"model.language_model.layers.{idx}."
            kind = ("self_attn.o_proj.input" if idx % 4 == 3
                    else "linear_attn.out_proj.input")
            for suffix in ("input", kind, "post_attention.input", "mlp.down_proj.input"):
                write_tensor(
                    folder / "hessians" / f"{base}{suffix}.safetensors",
                    rows=7, nan=(idx == lo and suffix == "input" and nonfinite),
                    neg=(idx == lo and suffix == "input" and negative),
                )
        if hi == 64:
            write_tensor(folder / "hessians" / "lm_head.input.safetensors")
        manifest = ranges.manifest_at(self.campaign_dir)
        receipt = {
            "schema": "VTP06_RANGE_RUN_V1", "state": "EXIT_ZERO_UNVERIFIED",
            "exit_code": 0, "range": [lo, hi],
            "calibration_sha256": manifest["calibration_sha256"],
            "engine_sha256": manifest["engine_sha256"],
        }
        (folder / "receipt.json").write_text(json.dumps(receipt))
        (folder / "engine.log").write_text(
            "I calib: 2 documents, 7 tokens in 1 s, 0 preemption(s)\\n"
        )
        return folder

    def test_verified_range_does_not_claim_q6_approval(self):
        folder = self.create_run()
        result = ranges.verify_range(self.campaign_dir, 0, 2, 1)
        self.assertEqual(result["status"], "HESSIAN_PIPELINE_PASS_NOT_QUALITY_APPROVAL")
        self.assertEqual(set(result["per_layer_file_count"].values()), {4})
        self.assertTrue(all(d["nonfinite"] == 0 for d in result["hessians"]))
        status = ranges.campaign_status(self.campaign_dir)
        self.assertEqual(status["validated_layers"], 2)
        self.assertFalse(status["complete_hessian_pipeline"])
        self.assertFalse(status["iq6r_quality_approved"])
        with self.assertRaises(FileExistsError):
            ranges.verify_range(self.campaign_dir, 0, 2, 1)
        self.assertTrue((folder / "validation.json").is_file())

    def test_full_verification_detects_post_validation_byte_drift(self):
        folder = self.create_run()
        ranges.verify_range(self.campaign_dir, 0, 2, 1)
        checked = ranges.campaign_status(self.campaign_dir, full_verify=True)
        self.assertTrue(checked["output_bytes_reverified"])
        target = folder / "hessians" / (
            "model.language_model.layers.0.input.safetensors"
        )
        content = bytearray(target.read_bytes())
        content[-4:] = bytes(4)
        target.write_bytes(content)
        with self.assertRaisesRegex(ValueError, "digest changed"):
            ranges.campaign_status(self.campaign_dir, full_verify=True)

    def test_missing_layer_is_rejected(self):
        self.create_run(layer_missing=1)
        with self.assertRaisesRegex(ValueError, "missing or unexpected Hessian"):
            ranges.verify_range(self.campaign_dir, 0, 2, 1)

    def test_final_range_requires_lm_head_and_full_attention_taps(self):
        folder = self.create_run(lo=62, hi=64)
        out = ranges.verify_range(self.campaign_dir, 62, 64, 1)
        self.assertEqual(len(out["hessians"]), 9)
        self.assertIn("lm_head.input.safetensors",
                      {d["name"] for d in out["hessians"]})
        self.assertIn(
            "model.language_model.layers.63.self_attn.o_proj.input.safetensors",
            {d["name"] for d in out["hessians"]})
        self.assertTrue((folder / "validation.json").is_file())

    def test_final_range_rejects_missing_head(self):
        folder = self.create_run(lo=62, hi=64)
        (folder / "hessians" / "lm_head.input.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "missing or unexpected Hessian"):
            ranges.verify_range(self.campaign_dir, 62, 64, 1)

    def test_nan_and_negative_hessian_fail(self):
        for kw in ({"nonfinite": True}, {"negative": True}):
            with self.subTest(case=kw):
                # A separate temp run per case (never reuse partial outputs).
                root = self.root / ("corrupt-nan" if "nonfinite" in kw else "corrupt-neg")
                root.mkdir()
                save = self.campaign_dir
                self.campaign_dir = root / "campaign"
                self.create_run(**kw)
                with self.assertRaises(ValueError):
                    ranges.verify_range(self.campaign_dir, 0, 2, 1)
                self.campaign_dir = save

    def test_extraneous_unexpected_file_rejected(self):
        folder = self.create_run()
        (folder / "hessians" / "rogue.txt").write_text("untrusted")
        with self.assertRaisesRegex(ValueError, "unexpected files"):
            ranges.verify_range(self.campaign_dir, 0, 2, 1)

    def test_manifest_tampering_rejected(self):
        doc = self.make_plan()
        doc["ranges"][1] = [1, 4]
        (self.campaign_dir / "campaign.json").write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError, "gap or overlap"):
            ranges.manifest_at(self.campaign_dir)

    def test_lease_check_rejects_different_owner(self):
        with patch.object(ranges.subprocess, "run") as call:
            call.return_value.stdout = 'gpu: held by other for "other"\n'
            with self.assertRaisesRegex(ValueError, "lease purpose not matched"):
                ranges.check_gpu_lease("VTP06/unit-test-pilot")


if __name__ == "__main__":
    unittest.main()

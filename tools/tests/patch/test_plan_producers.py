"""Offline tests for the PRBE25 (1237) and PRBE35 (1216) patch-local producers."""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch import validation_producer as vp  # noqa: E402

_PATCHES = Path(__file__).resolve().parents[3] / "patches"


def _load(patch_id: str):
    path = _PATCHES / patch_id / "validation" / "producer.py"
    spec = importlib.util.spec_from_file_location(f"producer_{patch_id}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProducerResolutionTests(unittest.TestCase):
    def test_both_producers_resolve(self):
        for patch_id, producer_id, standard in (
            ("1216_rd43_concurrent_join_fusion_guard", "rd43", "run"),
            ("1237_rd30_moe_mmq_compact_grid", "rd30", "run"),
            ("1215_rd394041_amd_stream_moe_overlap", "rd3942", "run"),
            ("1241_rd33_mmvq_q8_0_f32_decode", "rd33", "run"),
            ("1207_rd17_moe_topk_down_fold", "rd17", "run"),
            ("1262_nro15_mmvdq", "nro15", "run"),
            ("1253_nro04_gfx1100_bf16_chunked_gdn", "nro04", "run"),
            ("1254_nro05_gdn_mtp_prefix_tail", "nro05", "run"),
        ):
            selection = vp.resolve_producer(patch_dir=_PATCHES / patch_id, producer_id=producer_id)
            self.assertEqual(selection.spec.standard_campaign, standard)
            self.assertTrue(callable(selection.producer))


class FullVocabParsingTests(unittest.TestCase):
    def setUp(self):
        from bigcherry.experiment import full_vocab

        self.m = full_vocab
        self.err = full_vocab.FullVocabError

    def _row(self, vocab, generated=1, drop=None, dup=False):
        top = [{"id": i, "logprob": -float(i)} for i in range(vocab) if i != drop]
        if dup:
            top[-1] = {"id": 0, "logprob": 0.0}
        return {"id": generated, "top_logprobs": top}

    def test_dense_row_round_trips(self):
        generated, values = self.m.dense_logprobs(self._row(4), vocab_size=4, arm="control", step=0)
        self.assertEqual(generated, 1)
        self.assertEqual(list(values), [0.0, -1.0, -2.0, -3.0])

    def test_partial_vocabulary_fails_closed(self):
        with self.assertRaises(self.err):
            self.m.dense_logprobs(self._row(4, drop=2), vocab_size=4, arm="control", step=0)

    def test_duplicate_token_fails_closed(self):
        with self.assertRaises(self.err):
            self.m.dense_logprobs(self._row(4, dup=True), vocab_size=4, arm="control", step=0)

    def test_out_of_range_generated_id_fails_closed(self):
        with self.assertRaises(self.err):
            self.m.dense_logprobs(self._row(4, generated=9), vocab_size=4, arm="subject", step=3)

    def test_non_finite_logprob_fails_closed(self):
        row = self._row(2)
        row["top_logprobs"][0]["logprob"] = math.inf
        with self.assertRaises(self.err):
            self.m.dense_logprobs(row, vocab_size=2, arm="subject", step=0)


class Rd30HelperTests(unittest.TestCase):
    def setUp(self):
        self.m = _load("1237_rd30_moe_mmq_compact_grid")

    def test_enum_lookup_is_case_insensitive_and_fails_closed(self):
        self.assertEqual(self.m._enum_id({7: "mul_mat_id"}, "MUL_MAT_ID", "ggml_op"), 7)
        with self.assertRaises(vp.ValidationProducerError):
            self.m._enum_id({7: "mul_mat"}, "MUL_MAT_ID", "ggml_op")

    def test_marker_regex_matches_patch_output(self):
        patch_text = (_PATCHES / "1237_rd30_moe_mmq_compact_grid" / "patch.py").read_text(encoding="utf-8")
        self.assertIn("BIGCHERRY_PATCH_HIT patch=1237_rd30 \"\n    \"path=moe_mmq_compact_grid", patch_text)


if __name__ == "__main__":
    unittest.main()


class ExecutionPackageTests(unittest.TestCase):
    """A package with a validation.toml must be startable: README, bound
    contract and adapter all present (1262 once reached hardware without a
    README and was refused at run time)."""

    def test_every_validation_package_passes_the_execution_gate(self):
        from bigcherry.patch import registry, validation_policy

        loaded = registry.load_registry()
        items = getattr(loaded, "descriptors", None) or getattr(loaded, "patches", None) or loaded
        descriptors = list(items.values()) if isinstance(items, dict) else list(items)
        for descriptor in descriptors:
            if descriptor.state in ("rejected", "superseded"):
                continue
            if not (_PATCHES / descriptor.patch_id / "validation.toml").is_file():
                continue
            with self.subTest(patch=descriptor.patch_id):
                validation_policy.require_execution_package(descriptor)

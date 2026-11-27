"""Card-free tests for pinned, head-first, W8A8-compatible staged IQ6R recipes."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[2] / "lab/radiance/iq6r_v4_stage_recipes.py"
spec = importlib.util.spec_from_file_location("iq6r_v4_stage_recipes", PATH)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class StageRecipeTests(unittest.TestCase):
    def setUp(self):
        head = "".join(f"output.k{i}.weight gptq codes=u{i}\n" for i in range(4, 8))
        self.data = ("# header\n" + "".join(
            f"blk.{i}.ffn_down.weight gptq codes=u6 calib=$CALIB\n"
            for i in range(288)
        ) + "token_embd.weight rtn codes=u6\n" + head).encode()
        self.fallback = "".join(
            f"{name} rtn  codes=i8 group=128 scale=bf16 transform=fwht128 clamp=sym rule=search\n"
            for name in sorted(module.EXPECTED_FALLBACK)
        ).encode()

    def staged(self, *, chunk=20, source=None, fallback=None):
        source = self.data if source is None else source
        fallback = self.fallback if fallback is None else fallback
        with (patch.object(module, "PINNED_RECIPE_SHA256", module.sha256(source)),
              patch.object(module, "PINNED_FALLBACK_SHA256", module.sha256(fallback))):
            return module.recipes(source, fallback, chunk=chunk)

    def test_head_first_then_cumulative_and_exact_final(self):
        stages = self.staged()
        self.assertEqual(len(stages), 16)
        self.assertEqual(stages[-1], self.data)
        for i, data in enumerate(stages[:-1], 1):
            self.assertEqual(data.count(b" gptq "), 4 + (i - 1) * 20)
            self.assertEqual(data.count(b"token_embd.weight rtn"), 1)
            self.assertEqual(data.count(b"transform=fwht128 clamp=sym"), 8)
            for k in (4, 5, 6, 7):
                self.assertIn(f"output.k{k}.weight".encode(), data)
        self.assertNotIn(b"Interim W8A8", stages[-1])

    def test_gptq_precedes_fallback_globs(self):
        stage2 = self.staged()[1]
        self.assertLess(stage2.index(b"blk.0.ffn_down.weight gptq"),
                        stage2.index(b"blk.*.ffn_down.weight rtn"))

    def test_rejects_unpinned_primary(self):
        with (patch.object(module, "PINNED_FALLBACK_SHA256", module.sha256(self.fallback)),
              self.assertRaisesRegex(ValueError, "source pin")):
            module.recipes(self.data, self.fallback, chunk=20)

    def test_rejects_unpinned_fallback(self):
        with (patch.object(module, "PINNED_RECIPE_SHA256", module.sha256(self.data)),
              self.assertRaisesRegex(ValueError, "source pin")):
            module.recipes(self.data, self.fallback, chunk=20)

    def test_rejects_bad_chunk(self):
        with self.assertRaisesRegex(ValueError, "chunk"):
            module.recipes(self.data, self.fallback, chunk=0)

    def test_rejects_missing_embedding_rule(self):
        incorrect = self.data.replace(b"token_embd.weight rtn codes=u6\n", b"")
        with self.assertRaisesRegex(ValueError, "schema"):
            self.staged(source=incorrect)

    def test_rejects_missing_head_class(self):
        incorrect = self.data.replace(b"output.k4.weight gptq codes=u4\n", b"")
        with self.assertRaisesRegex(ValueError, "schema"):
            self.staged(source=incorrect)

    def test_rejects_partial_baseline(self):
        incorrect = self.fallback.replace(b"blk.*.ssm_out.weight", b"absent.ssm_out.weight")
        with self.assertRaisesRegex(ValueError, "fallback"):
            self.staged(fallback=incorrect)


if __name__ == "__main__":
    unittest.main()

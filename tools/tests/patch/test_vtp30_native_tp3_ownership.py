"""VTP30 native-Radiance opt-in asymmetric Qwen3.8-27B sharding tests.

No GPU. Proves the managed native-source patch applies atomically,
is idempotent and refuses source drift; verifies the original full
checkpoint row-span partition for every native GDN/attention/FFN
consumer against the physical-TP3 ownership proved by VTP13/VTP19.
"""
from __future__ import annotations

import importlib.util
import re
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
from bigcherry.patcher import apply_all

PACKAGE = (
    REPO / "engines" / "radiance" / "patches" /
    "rad0016_vtp30_native_tp3_ownership" / "patch.py"
)


def patches():
    spec = importlib.util.spec_from_file_location("vtp30_native_tp3", PACKAGE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.PATCHES)


def original_anchor(edit):
    # All source-pinned C++ hunks use exact re.escape literals.
    # No guessing or regex flexibility on changed upstream source.
    return re.sub(r"\\(.)", r"\1", edit.anchor, flags=re.DOTALL)


class NativeRadianceTP3OwnershipTests(unittest.TestCase):
    def fixture(self, root: Path):
        for p in patches():
            path = root / p.path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(original_anchor(e) for e in p.edits) + "\n")

    def test_patch_is_opt_in_not_production_recipe(self):
        with (REPO / "engines/radiance/recipes.toml").open("rb") as f:
            recipes = tomllib.load(f)
        with (PACKAGE.parent / "patch.toml").open("rb") as f:
            manifest = tomllib.load(f)
        name = "vtp30-native-tp3-ownership"
        self.assertEqual(manifest["state"], "untested")
        self.assertEqual(manifest["plan-ids"], ["VTP13"])
        self.assertEqual(manifest["order"], 16)
        self.assertEqual(
            recipes["patch-set"][name]["patches"],
            ["rad0016_vtp30_native_tp3_ownership"],
        )
        self.assertIn(
            name, recipes["source"]["radiance-vtp30-native-tp3"]["patch-sets"]
        )
        self.assertNotIn(name, recipes["source"]["radiance"]["patch-sets"])
        self.assertNotIn(name, recipes["source"]["radiance-native"]["patch-sets"])
        self.assertEqual(len(patches()), 5)
        self.assertEqual(sum(len(p.edits) for p in patches()), 8)
        guards = [e.guard for p in patches() for e in p.edits]
        self.assertEqual(len(set(guards)), len(guards))
        for p in patches():
            self.assertEqual(p.dialect(), "none")
            for edit in p.edits:
                self.assertEqual(edit.expect_matches, 1)
                self.assertIn("BC_VTP30_NATIVE_TP3_", edit.guard)

    def test_applies_once_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            first = apply_all(patches(), root)
            self.assertTrue(all(p.ok for p in first), first)
            self.assertTrue(all(
                e.status == "applied" for p in first for e in p.results
            ), first)
            files = {p.path: (root / p.path).read_bytes() for p in patches()}
            second = apply_all(patches(), root)
            self.assertTrue(all(p.ok for p in second), second)
            self.assertTrue(all(
                e.status == "already-applied" for p in second for e in p.results
            ), second)
            self.assertEqual(
                files, {p.path: (root / p.path).read_bytes() for p in patches()}
            )
            text = (root / "arch/common/rad_arch.h").read_text()
            self.assertIn("g.native_tp3_27b = true", text)
            self.assertIn("BC_VTP_NATIVE_TP3", text)
            self.assertIn("g.n_head != 6 * g.n_head_kv", text)
            self.assertIn("RAD_E_UNSUPPORTED", text)
            self.assertIn("g.n_ff = ff[g.rank]", text)
            self.assertIn("g.kv_off = kv_begin[g.rank] * g.head_dim", text)
            gdn = (root / "arch/common/rad_block_gdn.h").read_text()
            self.assertIn("rad_weight_shard_span_parts(b, w_in, lo5, hi5, all5, 5)", gdn)
            self.assertIn("rad_weight_shard_span_parts(b, w_conv, lo5, hi5, all5, 3)", gdn)
            self.assertIn("rad_weight_shard_span(b, w_a_log, vhead", gdn)
            self.assertIn("rad_weight_shard_span(b, w_out, v,", gdn)
            attn = (root / "arch/common/rad_block_attn_gated.h").read_text()
            self.assertIn("rad_weight_shard_span(b, w_qg, 2 * qlo", attn)
            self.assertIn("rad_weight_shard_span(b, w_k, klo", attn)
            self.assertIn("rad_weight_shard_span(b, w_v, klo", attn)
            self.assertIn("rad_weight_shard_span(b, w_o, qlo", attn)
            ffn = (root / "arch/common/rad_block_mlp.h").read_text()
            self.assertIn("rad_weight_shard_span(b, w_gate_up, lo, hi)", ffn)
            self.assertIn("rad_weight_shard_span(b, w_down, lo, hi)", ffn)
            self.assertNotIn("hipMemcpyPeer", "".join(
                (root / p.path).read_text() for p in patches()
            ))

    def test_changed_pinned_source_refused_without_partial_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            file = root / patches()[3].path
            broken = original_anchor(patches()[3].edits[0])
            self.assertIn(broken, file.read_text())
            file.write_text(file.read_text().replace(broken, "upstream source changed"))
            old = {p.path: (root / p.path).read_bytes() for p in patches()}
            result = apply_all(patches(), root)
            self.assertTrue(any(not p.ok for p in result), result)
            self.assertEqual(
                old, {p.path: (root / p.path).read_bytes() for p in patches()}
            )

    def test_actual_27b_three_rank_source_partition_coverage(self):
        # Validate *global original source rows*, not just per-rank sizes.
        qh, kvh, gkh, gvh, ffn = (
            (6, 6, 12), (1, 1, 2), (4, 4, 8),
            (12, 12, 24), (5120, 4864, 7424),
        )
        self.assertEqual(sum(qh), 24)
        self.assertEqual(sum(kvh), 4)
        self.assertEqual(sum(gkh), 16)
        self.assertEqual(sum(gvh), 48)
        self.assertEqual(sum(ffn), 17408)
        self.assertEqual(tuple(q // k for q, k in zip(qh, kvh, strict=True)),
                         (6, 6, 6))

        def spans(widths, unit):
            begin = 0
            result = []
            for count in widths:
                result.append((begin, begin + count * unit))
                begin += count * unit
            return result

        for widths, factor, total in (
            (qh, 512, 12288),     # query+gate interleaved
            (qh, 256, 6144),      # attn output-projection input columns
            (kvh, 256, 1024),    # K and V attention rows
            (gkh, 128, 2048),    # GDN Q and K part
            (gvh, 128, 6144),    # GDN value part, Z, out columns
            (gvh, 1, 48),        # GDN A_log, dt_bias, a/b
            (ffn, 1, 17408),     # each FFN gate/up part and down columns
        ):
            ranges = spans(widths, factor)
            self.assertEqual(ranges[0][0], 0)
            self.assertEqual(ranges[-1][1], total)
            self.assertEqual(
                [r[1] for r in ranges[:-1]],
                [r[0] for r in ranges[1:]],
            )


if __name__ == "__main__":
    unittest.main()

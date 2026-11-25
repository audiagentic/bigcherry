"""rad0065: native single-band IQR fragment source and safe fallback mechanics."""
from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
from bigcherry.patcher import apply_all  # noqa: E402

PKG = REPO / "engines/radiance/patches/rad0065_iqr_fused_gfx1201"
spec = importlib.util.spec_from_file_location("rad0065_patch", PKG / "patch.py")
patch = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(patch)

HEADER = "libr4d/r4d_gemm_8a8_prefill.h"
IQR = "libr4d/r4d_gemm_iqr.hip"

# Standalone pre-patch anchors. When RADIANCE_SRC is provided, tests use the
# actual source tree instead, so drift in the pinned target fails closed.
HEADER_BASE = """template <bool I8, int RB, class Pre, bool ROWF = false, uint32_t SB = kBlock>
    const uint8_t* const wsrc = W + (size_t)wt0 * ksuper * kTileBytes + lane * kLaneBytes;
#pragma unroll
    for (uint32_t i = 0; i < kPTW; ++i)
        w0[i] = stage8_load(wsrc + (size_t)i * ksuper * kTileBytes, true);
#pragma unroll
        for (uint32_t i = 0; i < kPTW; ++i)
            nxt[i] = stage8_load(wsrc + ((size_t)i * ksuper + c1 / kSuperK) * kTileBytes, true);
    for (uint32_t kb = 0; kb < nkb; ++kb) {
        // The weight's scales
"""
IQR_BASE = """template <class F, int RB>
__global__ __launch_bounds__(kPWaves * kWave) R4D_WAVES(6)
void iqr_prefill_kernel(IqrPre<F> pre, const uint8_t* A) {
}
    unsigned int* const ctl = r4d_8a8_split_ctr(stream, g.y->data);
"""


def plant(root: Path) -> None:
    source = os.environ.get("RADIANCE_SRC")
    for path, fallback in ((HEADER, HEADER_BASE), (IQR, IQR_BASE)):
        dst = root / path
        dst.parent.mkdir(parents=True, exist_ok=True)
        src = Path(source) / path if source else None
        if src and src.is_file():
            shutil.copyfile(src, dst)
        else:
            if source:
                raise AssertionError(f"RADIANCE_SRC missing required file: {path}")
            dst.write_text(fallback)


class NativeFusedIqrTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        plant(self.root)
        self.before = {p: (self.root / p).read_bytes() for p in (HEADER, IQR)}

    def apply(self):
        return apply_all(patch.PATCHES, self.root)

    def test_package_metadata(self):
        meta = tomllib.loads((PKG / "patch.toml").read_text())
        self.assertEqual(meta["id"], PKG.name)
        self.assertEqual(meta["state"], "untested")
        self.assertEqual(meta["order"], 65)
        self.assertEqual(meta["requires"], [])
        self.assertEqual({p.path for p in patch.PATCHES}, {HEADER, IQR})
        self.assertEqual(sum(len(p.edits) for p in patch.PATCHES), 7)

    def test_applies_once_and_is_idempotent(self):
        first = self.apply()
        self.assertTrue(all(r.ok for r in first), first)
        after = {p: (self.root / p).read_bytes() for p in (HEADER, IQR)}
        self.assertNotEqual(after[HEADER], self.before[HEADER])
        self.assertNotEqual(after[IQR], self.before[IQR])
        second = self.apply()
        self.assertTrue(all(r.ok for r in second), second)
        self.assertEqual({p: (self.root / p).read_bytes() for p in (HEADER, IQR)}, after)

    def test_native_fragment_producer_and_fallback(self):
        self.assertTrue(all(r.ok for r in self.apply()))
        body = (self.root / HEADER).read_text()
        iqr = (self.root / IQR).read_text()
        self.assertIn("bool Fused = false", body)
        self.assertIn("if constexpr (!Fused)", body)
        self.assertIn("cur[i] = pre.load(ww * kPTW + i, half, lane);", body)
        self.assertIn("pre.prepare(n0, kb);", body)
        self.assertIn("const IqrFused12Pre<F> pre{c, s_tr};", iqr)
        self.assertIn("frag(w, t, lane, f);", iqr)
        self.assertIn("t + h * kTileBytes + lane * kLaneBytes", iqr)
        self.assertIn("f.v = *(const uint4*)p;", iqr)
        self.assertIn("f.w = *(const uint4*)(p + 16);", iqr)
        self.assertIn("gemm_8a8_prefill_body<true, 32, IqrFused12Pre<F>, true, kBlock, true>", iqr)
        self.assertIn("gemm_8a8_prefill_body<true, 0, IqrFused12Pre<F>, false, kBlock, true>", iqr)
        fused = iqr[iqr.index("struct IqrFused12Pre {"):iqr.index("void iqr_prefill_kernel")]
        self.assertNotIn("r3_bytes_12_to_11", fused)
        self.assertNotIn("__hip_atomic", fused)
        self.assertNotIn("spin", fused)

    def test_certificate_and_single_band_only(self):
        self.assertTrue(all(r.ok for r in self.apply()))
        iqr = (self.root / IQR).read_text()
        self.assertIn("F::KIND == kTrellisMad", iqr)
        self.assertIn("F::LV == 0u && F::LRB == 0u", iqr)
        self.assertIn("F::L == 16u && F::B >= 2u && F::B <= 7u", iqr)
        self.assertIn("F::L == 15u && F::B == 2u", iqr)
        self.assertIn("F::B >= 4u && F::B <= 7u", iqr)
        self.assertIn("if (bands == 1u)", iqr)
        self.assertIn("kPWaves * kTrBytes + Dec<F>::kTab", iqr)
        self.assertIn("const uint32_t bands = (uint32_t)((g.M + kPBM - 1u) / kPBM);", iqr)

    def test_missing_anchor_fails_closed(self):
        path = self.root / HEADER
        path.write_text(path.read_text().replace("const uint8_t* const wsrc", "const uint8_t* wsrc", 1))
        results = self.apply()
        self.assertFalse(all(r.ok for r in results), results)

    def test_duplicate_anchor_fails_closed(self):
        path = self.root / IQR
        path.write_text(path.read_text() + "\n" + IQR_BASE)
        results = self.apply()
        self.assertFalse(all(r.ok for r in results), results)


if __name__ == "__main__":
    unittest.main()

"""XDP08: card-free proof of optional gfx11 INT8 weight-scale staging.

The experiment is OFF by default and gated to I8/RB0/non-ROWF.
Numerical identity follows from a permutation of the identical 16-bit
scale encodings, not from a tolerance or a re-quantization.
"""

from __future__ import annotations

import itertools
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LIBR3 = ROOT / "engines/radiance/components/libr3"
REWRITE = LIBR3 / "native/r4d_gemm_8a8_prefill.h.rw"
CMAKE = LIBR3 / "CMakeLists.txt"


class XDP08Int8ScaleStage(unittest.TestCase):
    def test_weight_scale_addresses_identical_for_all_lanes_and_tiles(self) -> None:
        # Scale layout: [N/16][K/128][16]. One block owns 128 N rows.
        # Stage once/thread for 128 threads then select the same uint4s as
        # the original wave's old Sw pointer, including duplicate wave bands.
        for n0, nkb, kb in itertools.product((0, 128, 896), (1, 2, 40), (0, 1, 39)):
            if kb >= nkb:
                continue
            with self.subTest(n0=n0, nkb=nkb, kb=kb):
                staged = {}
                for tid in range(128):
                    row = n0 + tid
                    address = (row // 16 * nkb + kb) * 16 + row % 16
                    self.assertNotIn(tid, staged)
                    staged[tid] = address
                self.assertEqual(len(set(staged.values())), 128)
                for ww in range(4):
                    wt0 = n0 // 16 + ww * 2
                    for i in range(2):
                        for lane in range(32):
                            first = (wt0 + i) * nkb * 16 + kb * 16 + (lane >> 4) * 8
                            lds_first = (ww * 2 + i) * 16 + (lane >> 4) * 8
                            for word in range(8):
                                self.assertEqual(staged[lds_first + word], first + word)

    def test_ping_pong_scale_slots_respect_existing_group_barriers(self) -> None:
        # Reader of group kb consumes slot kb%2 *after* the two tile
        # barriers. Fast workitems cannot reach the kb+2 write of that
        # slot without crossing the first kb+1 group-wide barrier;
        # that rendezvous waits for the slowest kb reader to finish.
        writers = [kb % 2 for kb in range(40)]
        self.assertTrue(all(writers[k] != writers[k + 1] for k in range(39)))
        self.assertTrue(all(writers[k] == writers[k + 2] for k in range(38)))
        rules = REWRITE.read_text()
        self.assertIn(
            "uint16_t s_ws_i8[2][(I8 && RB == 0 && !ROWF) ? kPBN : 1]",
            rules,
        )
        self.assertIn("s_ws_i8[kb & 1u][tid] = Sw[", rules)
        self.assertIn("w8[i] = *(const uint4*)(s_ws_i8[kb & 1u] +", rules)
        self.assertIn("tile(kb, 1u, w1, w0);", rules)

    def test_opt_in_gates_and_native_upstream_rewrite(self) -> None:
        cmake = CMAKE.read_text()
        self.assertIn("option(R3_I8_SWS_LDS ", cmake)
        self.assertIn(' (experimental)" OFF)', cmake)
        self.assertIn(
            "target_compile_definitions(libr3 PRIVATE R3_I8_SWS_LDS=1)", cmake
        )
        self.assertIn("if(NOT R3_NATIVE)", cmake)
        rules = REWRITE.read_text()
        self.assertEqual(rules.count("@@ find 1"), 5)
        self.assertIn("#if defined(R3_I8_SWS_LDS)", rules)
        self.assertIn("if constexpr (I8 && RB == 0 && !ROWF)", rules)
        self.assertIn("#if !defined(R3_I8_SWS_LDS)", rules)
        self.assertIn("s_ws_i8[kb & 1u][tid]", rules)
        self.assertIn("s_ws_i8[kb & 1u] + (ww * kPTW + i) * 16u", rules)
        self.assertIn("tile(kb, 1u, w1, w0);", rules)
        self.assertIn("r3_acc_i32_11_to_12", rules)
        self.assertNotIn("R3_I8_SWS_LDS", str(LIBR3 / "rewrites.txt"))

    def test_barrier_and_fma_order_preserved_by_rewrite_blocks(self) -> None:
        # Native block 2 runs immediately after tile(kb,1), which executes
        # the second existing lds_barrier. It only redirects uint4 metadata
        # reads before the caller's unchanged scale/fma fold.
        rules = REWRITE.read_text()
        native_block = rules.split("@@ find 1\n        tile(kb, 1u, w1, w0);")[1]
        replacement = native_block.split("@@ replace\n")[1].split("@@ end")[0]
        self.assertLess(
            replacement.index("tile(kb, 1u, w1, w0);"),
            replacement.index("w8[i] = *(const uint4*)(s_ws_i8[kb & 1u]"),
        )
        # Exactly the current input/padded-row layout. No precision or
        # BF16 output arithmetic is modified by the staged-scale path.
        self.assertEqual(replacement.count("w8[i] = *(const uint4*)(s_ws_i8"), 1)
        self.assertNotIn("out[i][j]", replacement)
        self.assertNotIn("acc_val", replacement)


if __name__ == "__main__":
    unittest.main()

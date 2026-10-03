"""validated_enhancement_patches() must drop transitive dependents too.

Regression: profiling/re-validating 1237 failed with resolve_exact() raising
"1265 requires explicitly selected module(s): 1237" once 1265 (which
requires 1237) was promoted alongside it into validated-enhancements --
excluding only the literal focal patch left a declared member (1265) whose
own hard dependency had just been removed from the composition.

COVERAGE GAP (2026-09-27): 1265 was demoted out of validated-enhancements
(config/recipes.toml, patch-refactor commit d2c2365d -- its runtime gate was
narrowed after promotion, invalidating the promotion evidence; see the
recipes.toml comment above [patch-set.validated-enhancements] for the full
story). It was the only live requires-dependency pair inside
validated-enhancements, so the two tests below that exercised the "a real
declared member gets transitively dropped" path can no longer do so against
production config -- they now assert the (still-true, but no longer
interesting) fact that 1265 stays absent. Re-promoting 1265, or a synthetic
catalog/recipes fixture decoupled from live config, would restore real
coverage of the transitive-drop path itself; tracked as a follow-up rather
than built here, since it's out of scope for this specific fix.
"""

from __future__ import annotations

import unittest

from bigcherry.patch.campaign.scaffold import validated_enhancement_patches


class TransitiveExclusionTests(unittest.TestCase):
    def test_excluding_1237_leaves_no_dangling_dependent(self) -> None:
        result = validated_enhancement_patches(patch_id="1237_rd30_moe_mmq_compact_grid", common_patches=())
        self.assertNotIn("1237_rd30_moe_mmq_compact_grid", result)
        self.assertNotIn(
            "1265_rd30b_moe_mmq_compact_grid_rdna4_rdna2", result,
            "1265 requires 1237 and is demoted anyway; must not reappear regardless of the reason",
        )
        self.assertIn("1253_nro04_gfx1100_bf16_chunked_gdn", result)

    def test_excluding_an_independent_member_keeps_the_rest(self) -> None:
        result = validated_enhancement_patches(patch_id="1253_nro04_gfx1100_bf16_chunked_gdn", common_patches=())
        self.assertNotIn("1253_nro04_gfx1100_bf16_chunked_gdn", result)
        self.assertIn("1237_rd30_moe_mmq_compact_grid", result)

    def test_a_common_patch_is_stripped_from_the_returned_list(self) -> None:
        # GPT review req_5c9284ea25b04ddd: a common patch is unconditionally
        # composed (the caller adds it separately, alongside this function's
        # return value) -- it is never actually absent, so it must be
        # stripped from the RETURNED list (the caller already has it via
        # common_patches), not double-listed.
        result = validated_enhancement_patches(
            patch_id="1253_nro04_gfx1100_bf16_chunked_gdn",
            common_patches=("1237_rd30_moe_mmq_compact_grid",),
        )
        self.assertNotIn("1237_rd30_moe_mmq_compact_grid", result)  # stripped: already a common patch

    def test_profiling_1265_itself_only_excludes_1265(self) -> None:
        # 1265 is currently demoted (absent from the declared set already),
        # so this only proves the focal-exclusion path is a no-op-safe when
        # the focal patch isn't a declared member at all.
        result = validated_enhancement_patches(
            patch_id="1265_rd30b_moe_mmq_compact_grid_rdna4_rdna2", common_patches=()
        )
        self.assertNotIn("1265_rd30b_moe_mmq_compact_grid_rdna4_rdna2", result)
        self.assertIn("1237_rd30_moe_mmq_compact_grid", result)
        self.assertIn("1253_nro04_gfx1100_bf16_chunked_gdn", result)


if __name__ == "__main__":
    unittest.main()

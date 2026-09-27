"""validated_enhancement_patches() must drop transitive dependents too.

Regression: profiling/re-validating 1237 failed with resolve_exact() raising
"1265 requires explicitly selected module(s): 1237" once 1265 (which
requires 1237) was promoted alongside it into validated-enhancements --
excluding only the literal focal patch left a declared member (1265) whose
own hard dependency had just been removed from the composition.
"""

from __future__ import annotations

import unittest

from bigcherry.patch.campaign.scaffold import validated_enhancement_patches


class TransitiveExclusionTests(unittest.TestCase):
    def test_excluding_1237_also_excludes_its_declared_dependent_1265(self) -> None:
        result = validated_enhancement_patches(patch_id="1237_rd30_moe_mmq_compact_grid", common_patches=())
        self.assertNotIn("1237_rd30_moe_mmq_compact_grid", result)
        self.assertNotIn(
            "1265_rd30b_moe_mmq_compact_grid_rdna4_rdna2", result,
            "1265 requires 1237; it cannot remain in the composition once 1237 is excluded",
        )
        self.assertIn("1253_nro04_gfx1100_bf16_chunked_gdn", result)

    def test_excluding_an_independent_member_keeps_the_rest(self) -> None:
        result = validated_enhancement_patches(patch_id="1253_nro04_gfx1100_bf16_chunked_gdn", common_patches=())
        self.assertNotIn("1253_nro04_gfx1100_bf16_chunked_gdn", result)
        self.assertIn("1237_rd30_moe_mmq_compact_grid", result)
        self.assertIn("1265_rd30b_moe_mmq_compact_grid_rdna4_rdna2", result)

    def test_a_common_patch_does_not_trigger_exclusion_of_its_dependents(self) -> None:
        # GPT review req_5c9284ea25b04ddd: a common patch is unconditionally
        # composed (the caller adds it separately, alongside this function's
        # return value) -- it is never actually absent, so a declared member
        # requiring one must NOT be excluded on that basis. 1265 requires
        # 1237; naming 1237 as a COMMON patch for an unrelated focal (1253)
        # must not exclude 1265 (only strip 1237 itself from the list, since
        # the caller already has it via common_patches).
        result = validated_enhancement_patches(
            patch_id="1253_nro04_gfx1100_bf16_chunked_gdn",
            common_patches=("1237_rd30_moe_mmq_compact_grid",),
        )
        self.assertNotIn("1237_rd30_moe_mmq_compact_grid", result)  # stripped: already a common patch
        self.assertIn(
            "1265_rd30b_moe_mmq_compact_grid_rdna4_rdna2", result,
            "1237 (1265's dependency) is present via common_patches, so 1265 is not weakened",
        )

    def test_profiling_1265_itself_only_excludes_1265(self) -> None:
        # 1265 depends ON 1237, not the other way around -- nothing else in
        # the declared set requires 1265, so only 1265 itself is excluded.
        result = validated_enhancement_patches(
            patch_id="1265_rd30b_moe_mmq_compact_grid_rdna4_rdna2", common_patches=()
        )
        self.assertNotIn("1265_rd30b_moe_mmq_compact_grid_rdna4_rdna2", result)
        self.assertIn("1237_rd30_moe_mmq_compact_grid", result)
        self.assertIn("1253_nro04_gfx1100_bf16_chunked_gdn", result)


if __name__ == "__main__":
    unittest.main()

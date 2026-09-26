"""--allow-rejected admits only explicitly named retired patches."""

from __future__ import annotations

import unittest

from bigcherry.core import config, paths
from bigcherry.patch import patchset
from bigcherry.patch import source as psi
from bigcherry.patch.campaign.build import LLAMA_CPP_SRC

REJECTED = "1215_rd394041_amd_stream_moe_overlap"


@unittest.skipUnless(LLAMA_CPP_SRC.exists(), "upstream llama.cpp checkout not available")
class AllowRejectedCompositionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base_ref = config.load(paths.RECIPES).pinned
        state = {m.patch_id: m.state for m in patchset.catalog()}.get(REJECTED)
        if state not in patchset.RETIRED_STATES:
            self.skipTest(f"{REJECTED} is no longer retired")

    def _resolve(self, **kwargs):
        return psi.resolve_source_composition(
            "bigcherry-tuning", base_ref=self.base_ref, base_repo=LLAMA_CPP_SRC, **kwargs
        )

    def test_named_rejected_focal_needs_the_flag(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --allow-rejected"):
            self._resolve(focal=REJECTED)
        _, composition = self._resolve(focal=REJECTED, allow_rejected=True)
        self.assertEqual(composition[-1][0], REJECTED)

    def test_named_rejected_common_patch_is_admitted(self) -> None:
        _, composition = self._resolve(extra_patches=(REJECTED,), allow_rejected=True)
        self.assertIn(REJECTED, [pid for pid, _ in composition])


if __name__ == "__main__":
    unittest.main()

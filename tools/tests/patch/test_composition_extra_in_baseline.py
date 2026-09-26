"""A common extra patch already in the named composition is not added twice."""

from __future__ import annotations

import unittest

from bigcherry.core import config, paths
from bigcherry.patch import source as psi
from bigcherry.patch.campaign.build import LLAMA_CPP_SRC


@unittest.skipUnless(LLAMA_CPP_SRC.exists(), "upstream llama.cpp checkout not available")
class ExtraAlreadyInBaselineTests(unittest.TestCase):
    def test_promoted_common_patch_resolves_once(self) -> None:
        base_ref = config.load(paths.RECIPES).pinned
        _, named = psi.resolve_source_composition(
            "bigcherry-tuning", base_ref=base_ref, base_repo=LLAMA_CPP_SRC)
        member = named[-1][0]
        _, with_extra = psi.resolve_source_composition(
            "bigcherry-tuning", extra_patches=(member,), base_ref=base_ref, base_repo=LLAMA_CPP_SRC)
        self.assertEqual(with_extra, named)


if __name__ == "__main__":
    unittest.main()

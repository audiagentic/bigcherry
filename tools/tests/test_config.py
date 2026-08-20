"""v2 campaign config parsing (RE14): platform toolchain fields."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import config as campaign_config  # noqa: E402


_DOC = """
version = 2
pinned = "abc123"

[platform.with-toolchain]
targets = ["gfx1100"]
options = {}
c-compiler = "/opt/rocm/llvm/bin/clang"
cxx-compiler = "/opt/rocm/llvm/bin/clang++"

[platform.without-toolchain]
targets = ["gfx1100"]
options = {}
"""


class PlatformToolchainTests(unittest.TestCase):
    """A prior version of config.Platform silently dropped c-compiler/
    cxx-compiler from [platform.*] bodies: the dataclass had no fields for
    them and no unknown-key check caught the loss, so a campaign path
    reading only config.Platform could never reproduce the legacy build's
    toolchain -- it would silently fall back to $PATH.
    """

    def setUp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipes.toml"
            path.write_text(_DOC, encoding="utf-8")
            self.cfg = campaign_config.load(path)

    def test_compiler_fields_are_preserved(self):
        platform = self.cfg.platforms["with-toolchain"]
        self.assertEqual(platform.c_compiler, "/opt/rocm/llvm/bin/clang")
        self.assertEqual(platform.cxx_compiler, "/opt/rocm/llvm/bin/clang++")

    def test_compiler_fields_default_to_none(self):
        platform = self.cfg.platforms["without-toolchain"]
        self.assertIsNone(platform.c_compiler)
        self.assertIsNone(platform.cxx_compiler)

    def test_non_string_compiler_field_rejected(self):
        doc = _DOC + "\n[platform.bad]\ntargets = [\"gfx1100\"]\nc-compiler = 5\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipes.toml"
            path.write_text(doc, encoding="utf-8")
            with self.assertRaises(campaign_config.ConfigError):
                campaign_config.load(path)


_CAMPAIGN_DOC = """
version = 2
pinned = "abc123"

[source.a]
ref = "pinned"
overlay = false

[build.stock]
needs = []

[build.tune]
needs = ["inventory"]

[platform.linux-multi]
targets = ["gfx1100"]
options = {}

[campaign.standard]
lanes = [
    { source = "a", build = "stock", platform = "linux-multi" },
    { source = "a", build = "tune", platform = "linux-multi" },
]
"""


class CampaignProfileTests(unittest.TestCase):
    """RE19: the canonical standard/default campaign profile, expressed
    directly in v2 identities rather than derived from compat.recipe.*."""

    def _load(self, doc: str) -> campaign_config.Config:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipes.toml"
            path.write_text(doc, encoding="utf-8")
            return campaign_config.load(path)

    def test_parses_lanes_in_order(self):
        cfg = self._load(_CAMPAIGN_DOC)
        profile = cfg.campaigns["standard"]
        self.assertEqual(profile.name, "standard")
        self.assertEqual(
            profile.lanes,
            (campaign_config.CampaignLaneSelector("a", "stock", "linux-multi"),
             campaign_config.CampaignLaneSelector("a", "tune", "linux-multi")))

    def test_rejects_unknown_source(self):
        doc = _CAMPAIGN_DOC.replace('source = "a", build = "stock"',
                                     'source = "does-not-exist", build = "stock"')
        with self.assertRaises(campaign_config.ConfigError):
            self._load(doc)

    def test_rejects_unknown_build(self):
        doc = _CAMPAIGN_DOC.replace('build = "stock", platform',
                                     'build = "does-not-exist", platform')
        with self.assertRaises(campaign_config.ConfigError):
            self._load(doc)

    def test_rejects_unknown_platform(self):
        doc = _CAMPAIGN_DOC.replace('platform = "linux-multi" },\n    { source = "a", build = "tune"',
                                     'platform = "does-not-exist" },\n    { source = "a", build = "tune"')
        with self.assertRaises(campaign_config.ConfigError):
            self._load(doc)

    def test_rejects_empty_lanes(self):
        doc = _CAMPAIGN_DOC.replace(
            'lanes = [\n    { source = "a", build = "stock", platform = "linux-multi" },\n'
            '    { source = "a", build = "tune", platform = "linux-multi" },\n]',
            "lanes = []")
        with self.assertRaises(campaign_config.ConfigError):
            self._load(doc)

    def test_no_campaign_section_yields_empty_dict(self):
        doc = _CAMPAIGN_DOC.split("[campaign.standard]")[0]
        cfg = self._load(doc)
        self.assertEqual(cfg.campaigns, {})


class RealRecipesTomlCampaignStandardTests(unittest.TestCase):
    """Cross-checks the real recipes.toml campaign.standard profile added
    for RE19 against the actual current default=true recipe/build coverage
    (confirmed by direct inspection before writing it, not assumed) --
    upstream/stock, bigcherry-native/native (aliased to the real v2 build
    "control"), bigcherry/{record,tune,replay}."""

    def test_standard_profile_covers_the_same_roles_as_default_recipes(self):
        from bigcherry import paths
        cfg = campaign_config.load(paths.RECIPES)
        profile = cfg.campaigns["standard"]
        lane_pairs = {(lane.source, lane.build) for lane in profile.lanes}
        self.assertEqual(lane_pairs, {
            ("llama-native", "stock"),
            ("bigcherry-native", "control"),
            ("bigcherry", "record"),
            ("bigcherry", "tune"),
            ("bigcherry", "replay"),
        })
        self.assertTrue(all(lane.platform == "linux-multi" for lane in profile.lanes))


if __name__ == "__main__":
    unittest.main()

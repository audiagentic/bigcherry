"""RAD01 PR 5: libr3 is a declared radiance component whose runtime settings match what its source reads."""

import re
import unittest

from bigcherry.core import components, config, engines, paths

_GETENV = re.compile(r'getenv\("(R3_[A-Z0-9_]+)"\)')
#: settings the campaign harness reads (engine_adapters/radiance.py), not the plugin source
HARNESS_ONLY = {"R3_MOE_HELPER_DEVICES"}


class Libr3ComponentTests(unittest.TestCase):
    def setUp(self):
        self.libr3 = components.load_all(paths.REPO_ROOT, engines.RADIANCE)["libr3"]

    def test_declaration(self):
        self.assertEqual(self.libr3.engine, "radiance")
        self.assertEqual(self.libr3.targets, ("gfx1100",))
        recipes = config.load(engines.load(paths.REPO_ROOT, engines.RADIANCE).recipes_path(paths.REPO_ROOT))
        self.assertEqual(self.libr3.built_against, recipes.pinned)

    def test_every_variable_the_source_reads_is_declared_and_nothing_else(self):
        read = set()
        for path in self.libr3.root.rglob("*"):
            if path.is_file() and path.suffix in {".cpp", ".hip", ".h", ".rw"}:
                read |= set(_GETENV.findall(path.read_text(encoding="utf-8", errors="replace")))
        self.assertEqual(read, set(self.libr3.settings) - HARNESS_ONLY)

    def test_only_native_attention_is_production(self):
        production = {s.name for s in self.libr3.settings.values() if s.status == "production"}
        self.assertEqual(production, {"R3_ATTN_NATIVE"})

    def test_content_digest_is_stable(self):
        self.assertEqual(self.libr3.content_digest(), self.libr3.content_digest())

    def test_profiles_restate_nothing_and_set_only_production_settings(self):
        self.assertEqual(components.check_environment(self.libr3, {}, production=True), [])
        problems = components.check_environment(
            self.libr3, {"R3_ATTN_NATIVE": "1", "R3_GDN_NATIVE": "1", "R3_TYPO": "1"}, production=True)
        self.assertIn("R3_ATTN_NATIVE=1 restates libr3's default", problems)
        self.assertIn("R3_GDN_NATIVE is evaluated in libr3; a production profile may not set it", problems)
        self.assertIn("R3_TYPO is not a setting libr3 declares", problems)

    def test_a_profile_environment_is_checked_against_the_components(self):
        import tempfile
        from pathlib import Path
        text = "\n".join([
            "version = 2",
            'engine = "radiance"',
            'pinned = "p"',
            "[runtime-profile.x]",
            "server-args = []",
            "tune-context = 1",
            "production-context = 1",
            "min-free-vram-bytes-per-device = 0",
            'environment = { R3_ATTN_NATIVE = "1" }',
            "",
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "r.toml"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(config.ConfigError, "restates libr3's default"):
                config.load(path)
            path.write_text(text.replace('R3_ATTN_NATIVE = "1"', 'R3_HC_NATIVE = "1"'), encoding="utf-8")
            self.assertEqual(dict(config.load(path).runtime_profiles["x"].environment), {"R3_HC_NATIVE": "1"})

    def test_llamacpp_declares_no_components(self):
        self.assertEqual(components.load_all(paths.REPO_ROOT, engines.LLAMACPP), {})


if __name__ == "__main__":
    unittest.main()

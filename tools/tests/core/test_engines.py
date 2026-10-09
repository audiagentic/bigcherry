"""Engine pillar declarations (engines/<name>/engine.toml) and the paths derived from them."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.core import engines, paths  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]


def _declare(root: Path, name: str, body: str) -> None:
    path = engines.declaration_path(root, name)
    path.parent.mkdir(parents=True)
    path.write_text(body, encoding="utf-8")


_GOOD = 'schema = 1\nname = "{name}"\nupstream = "https://example.invalid/{name}"\n\n[layout]\npatches = "{p}"\noverlay = "{o}"\nvendor = "{v}"\n'


class EngineDeclarations(unittest.TestCase):
    def test_repository_declares_llamacpp_and_radiance_without_overlap(self):
        layouts = engines.load_all(_REPO)
        self.assertEqual(sorted(layouts), ["llamacpp", "radiance"])
        self.assertEqual(layouts["llamacpp"].upstream, "https://github.com/ggml-org/llama.cpp")
        self.assertEqual(layouts["radiance"].upstream, "https://codeberg.org/StillDeadcode/radiance")

    def test_paths_constants_follow_the_llamacpp_declaration(self):
        layout = engines.load(_REPO, engines.LLAMACPP)
        self.assertEqual(paths.PATCHES, _REPO / layout.patches)
        self.assertEqual(paths.SRC_OVERLAY, _REPO / layout.overlay)
        self.assertEqual(paths.llama_root(), paths.primary_root() / layout.vendor)
        # the declared locations are the ones in use: at least one patch package and the overlay exist
        self.assertTrue(any(paths.PATCHES.glob("*/patch.toml")))
        self.assertTrue(paths.SRC_OVERLAY.is_dir())

    def test_layout_resolves_against_any_project_root(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _declare(root, "alpha", _GOOD.format(name="alpha", p="engines/alpha/patches", o="engines/alpha/overlay", v="vendor/alpha"))
            layout = engines.load(root, "alpha")
            other = root / "elsewhere"
            self.assertEqual(layout.patches_root(other), other / "engines/alpha/patches")
            self.assertEqual(layout.overlay_root(other), other / "engines/alpha/overlay")
            self.assertEqual(layout.vendor_root(other), other / "vendor/alpha")
            self.assertEqual(layout.patch_path_prefix(), "engines/alpha/patches/")

    def test_malformed_declarations_fail_closed(self):
        cases = {
            "missing": None,
            "wrong-name": _GOOD.format(name="other", p="a", o="b", v="c"),
            "absolute": _GOOD.format(name="absolute", p="/abs", o="b", v="c"),
            "escapes": _GOOD.format(name="escapes", p="../x", o="b", v="c"),
            "backslash": _GOOD.format(name="backslash", p="a\\\\b", o="b", v="c"),
            "schema": _GOOD.format(name="schema", p="a", o="b", v="c").replace("schema = 1", "schema = 2"),
            "extra-key": _GOOD.format(name="extra-key", p="a", o="b", v="c") + 'tests = "t"\n',
            "no-layout": 'schema = 1\nname = "no-layout"\nupstream = "u"\n',
        }
        for name, body in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                if body is not None:
                    _declare(root, name, body)
                with self.assertRaises(engines.EngineError):
                    engines.load(root, name)

    def test_two_engines_may_not_share_a_location(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _declare(root, "one", _GOOD.format(name="one", p="patches", o="o1", v="v1"))
            _declare(root, "two", _GOOD.format(name="two", p="patches", o="o2", v="v2"))
            with self.assertRaises(engines.EngineError):
                engines.load_all(root)


if __name__ == "__main__":
    unittest.main()

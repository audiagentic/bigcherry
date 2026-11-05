"""A bump moves every component's built-against with the pin."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from bigcherry.core import components, engines, paths


class RepinComponentsTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))
        layout = engines.load(paths.REPO_ROOT, engines.RADIANCE)
        for relative in ("engines/radiance/engine.toml", *(f"{rel}/component.toml" for _, rel in layout.components)):
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(paths.REPO_ROOT / relative, target)
        self.names = [name for name, _ in layout.components]

    def test_every_component_is_declared_against_the_new_ref_and_nothing_else_changes(self):
        before = {p: p.read_text() for p in self.root.rglob("component.toml")}
        changed = components.repin_components(self.root, engines.RADIANCE, "v9.9.9")
        self.assertEqual({p.parent.name for p in changed}, {p.parent.name for p in before})
        for path, old in before.items():
            new = path.read_text()
            self.assertIn('built-against = "v9.9.9"', new)
            self.assertEqual(len(new.splitlines()), len(old.splitlines()))
        declared = components.load_all(self.root, engines.RADIANCE)
        self.assertEqual({c.built_against for c in declared.values()}, {"v9.9.9"})
        self.assertEqual(sorted(declared), sorted(self.names))

    def test_moving_to_the_ref_already_declared_changes_nothing(self):
        components.repin_components(self.root, engines.RADIANCE, "v9.9.9")
        self.assertEqual(components.repin_components(self.root, engines.RADIANCE, "v9.9.9"), [])

    def test_a_component_with_no_built_against_line_is_an_error(self):
        victim = next(self.root.rglob("component.toml"))
        victim.write_text(victim.read_text().replace("built-against", "built_against_typo"))
        with self.assertRaises(components.ComponentError):
            components.repin_components(self.root, engines.RADIANCE, "v9.9.9")


if __name__ == "__main__":
    unittest.main()

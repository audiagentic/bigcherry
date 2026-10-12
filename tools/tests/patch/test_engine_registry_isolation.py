"""RAD01: engine-qualified registry identity and namespace isolation."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from bigcherry.core import engines, paths
from bigcherry.patch import registry


class EngineRegistryIsolationTests(unittest.TestCase):
    def test_real_radiance_packages_live_only_in_radiance(self) -> None:
        llama_root = engines.load(paths.REPO_ROOT, "llamacpp").patches_root(paths.REPO_ROOT)
        radiance_root = engines.load(paths.REPO_ROOT, "radiance").patches_root(paths.REPO_ROOT)
        llama = registry.load_registry("llamacpp", llama_root)
        radiance = registry.load_registry("radiance", radiance_root)
        expected = {
            "0001_drain_mover_before_arena_free",
            "0002_moe_remote_handoff",
            "0003_mtp_full_head_switch",
        }
        self.assertTrue(expected <= set(radiance.by_id))
        self.assertFalse(expected & set(llama.by_id))
        self.assertEqual({d.engine for d in radiance.descriptors}, {"radiance"})
        self.assertEqual({d.engine for d in llama.descriptors}, {"llamacpp"})
        for patch_id in sorted(expected):
            implementation = registry.load_implementation(
                "radiance", radiance.get(patch_id), root=radiance.root,
            )
            self.assertTrue(implementation)
        with self.assertRaises(registry.PatchRegistryError):
            registry.load_implementation(
                "llamacpp", radiance.get("0001_drain_mover_before_arena_free"),
                root=radiance.root,
            )

    def test_same_patch_id_in_two_engine_roots_has_separate_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            roots = {engine: base / engine for engine in ("llamacpp", "radiance")}
            for engine, root in roots.items():
                package = root / "0001_same_id"
                package.mkdir(parents=True)
                (package / "patch.toml").write_text(
                    'schema = 1\nid = "0001_same_id"\norder = 1\n'
                    'state = "validated"\nkind = "framework"\n',
                    encoding="utf-8",
                )
                (package / "patch.py").write_text(
                    "from bigcherry.patcher import FilePatch, Edit\n"
                    'PATCHES = [FilePatch(path="x.txt", edits=('
                    'Edit(id="marker", anchor="old", text="new", rationale="test"),))]\n',
                    encoding="utf-8",
                )
            a = registry.load_registry("llamacpp", roots["llamacpp"])
            b = registry.load_registry("radiance", roots["radiance"])
            self.assertEqual(a.get("0001_same_id").engine, "llamacpp")
            self.assertEqual(b.get("0001_same_id").engine, "radiance")
            self.assertNotEqual(a.get("0001_same_id"), b.get("0001_same_id"))
            registry.load_implementation("llamacpp", a.get("0001_same_id"), root=a.root)
            registry.load_implementation("radiance", b.get("0001_same_id"), root=b.root)
            self.assertIn("bigcherry._patches.llamacpp.0001_same_id", sys.modules)
            self.assertIn("bigcherry._patches.radiance.0001_same_id", sys.modules)


if __name__ == "__main__":
    unittest.main()

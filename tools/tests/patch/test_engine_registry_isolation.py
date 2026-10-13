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
            "rad0001_drain_mover_before_arena_free",
            "rad0002_moe_remote_handoff",
            "rad0003_mtp_full_head_switch",
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
                "llamacpp", radiance.get("rad0001_drain_mover_before_arena_free"),
                root=radiance.root,
            )

    def test_each_engine_accepts_only_its_own_patch_id_prefix(self) -> None:
        """IDs never collide across engines: radiance IDs carry "rad", llama.cpp IDs none."""
        def package(root: Path, patch_id: str) -> None:
            pkg = root / patch_id
            pkg.mkdir(parents=True)
            (pkg / "patch.toml").write_text(
                f'schema = 1\nid = "{patch_id}"\norder = 1\n'
                'state = "validated"\nkind = "framework"\n',
                encoding="utf-8",
            )
            (pkg / "patch.py").write_text(
                "from bigcherry.patcher import FilePatch, Edit\n"
                'PATCHES = [FilePatch(path="x.txt", edits=('
                'Edit(id="marker", anchor="old", text="new", rationale="test"),))]\n',
                encoding="utf-8",
            )

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            package(base / "llama-ok", "0001_same_name")
            package(base / "radiance-ok", "rad0001_same_name")
            a = registry.load_registry("llamacpp", base / "llama-ok")
            b = registry.load_registry("radiance", base / "radiance-ok")
            self.assertEqual(a.get("0001_same_name").engine, "llamacpp")
            self.assertEqual(b.get("rad0001_same_name").engine, "radiance")
            registry.load_implementation("llamacpp", a.get("0001_same_name"), root=a.root)
            registry.load_implementation("radiance", b.get("rad0001_same_name"), root=b.root)
            self.assertIn("bigcherry._patches.llamacpp.0001_same_name", sys.modules)
            self.assertIn("bigcherry._patches.radiance.rad0001_same_name", sys.modules)

            package(base / "radiance-bare", "0001_same_name")
            package(base / "llama-prefixed", "rad0001_same_name")
            with self.assertRaisesRegex(registry.PatchRegistryError, "engine 'radiance'"):
                registry.load_registry("radiance", base / "radiance-bare")
            with self.assertRaisesRegex(registry.PatchRegistryError, "engine 'llamacpp'"):
                registry.load_registry("llamacpp", base / "llama-prefixed")

if __name__ == "__main__":
    unittest.main()

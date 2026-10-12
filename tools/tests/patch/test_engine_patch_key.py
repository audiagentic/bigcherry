"""RAD01 PR 2 groundwork: canonical engine-qualified patch identity."""

from __future__ import annotations

import unittest

from bigcherry.core import engines, paths
from bigcherry.patch.registry import PatchKey, PatchRegistryError, load_implementation, load_registry


class EnginePatchKeyTests(unittest.TestCase):
    def test_real_engine_packages_have_disjoint_qualified_keys(self) -> None:
        llama = engines.load(paths.REPO_ROOT, "llamacpp")
        radiance = engines.load(paths.REPO_ROOT, "radiance")
        llama_root = llama.patches_root(paths.REPO_ROOT)
        radiance_root = radiance.patches_root(paths.REPO_ROOT)
        llama_registry = load_registry(llama_root)
        radiance_registry = load_registry(radiance_root)
        radiance_ids = {
            "0001_drain_mover_before_arena_free",
            "0002_moe_remote_handoff",
            "0003_mtp_full_head_switch",
        }
        self.assertEqual(set(radiance_registry.by_id), radiance_ids)
        self.assertFalse(radiance_ids & set(llama_registry.by_id))
        keys = [
            *(PatchKey("llamacpp", patch.patch_id) for patch in llama_registry.descriptors),
            *(PatchKey("radiance", patch.patch_id) for patch in radiance_registry.descriptors),
        ]
        self.assertEqual(len(keys), len(set(keys)))
        for patch_id in sorted(radiance_ids):
            self.assertTrue(
                load_implementation(radiance_registry.get(patch_id), root=radiance_root)
            )

    def test_same_bare_id_is_distinct_across_engines(self) -> None:
        llama = PatchKey("llamacpp", "0001_example")
        radiance = PatchKey("radiance", "0001_example")
        self.assertNotEqual(llama, radiance)
        self.assertEqual(len({llama, radiance}), 2)
        self.assertEqual((radiance.engine, radiance.patch_id), ("radiance", "0001_example"))

    def test_unknown_engine_and_bad_ids_fail_closed(self) -> None:
        with self.assertRaises(PatchRegistryError):
            PatchKey("not-declared", "0001_example")
        for invalid in ("", "../0001_example", "0001-example"):
            with self.subTest(invalid=invalid), self.assertRaises(PatchRegistryError):
                PatchKey("radiance", invalid)


if __name__ == "__main__":
    unittest.main()

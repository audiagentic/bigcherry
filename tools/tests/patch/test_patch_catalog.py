"""RE30 phase 1: declarative patch-catalog coverage/consistency tests."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.core import paths # noqa: E402
from bigcherry.patch import catalog as patch_catalog # noqa: E402


class TestPatchCatalogLoads(unittest.TestCase):
    def test_loads_the_real_catalog(self):
        entries = patch_catalog.load_catalog()
        self.assertGreater(len(entries), 0)
        for entry in entries.values():
            self.assertIn(entry.kind, patch_catalog.KINDS)
            self.assertIn(entry.origin, patch_catalog.ORIGINS)
            self.assertIn(entry.backend, patch_catalog.BACKENDS)

    def test_1000_is_classified_as_upstream_backport(self):
        """RE30's own investigation finding: 1000 declares GROUP='upstream-fixes'
        in its Python module but is compiled into the standard framework
        patch-set -- the catalog should classify it by what it actually IS."""
        entries = patch_catalog.load_catalog()
        self.assertEqual(
            entries["1000_rdna4_mmq_q2k_q6k_fix"].kind, "upstream-backport"
        )

    def test_no_vulkan_patches_exist_yet(self):
        entries = patch_catalog.load_catalog()
        vulkan = [e for e in entries.values() if e.backend == "vulkan"]
        self.assertEqual(
            vulkan, [], "no Vulkan patches should exist before RE30 phase 2+"
        )


class TestPatchCatalogCrossCheck(unittest.TestCase):
    def test_cross_check_is_clean_on_the_real_catalog(self):
        problems = patch_catalog.cross_check()
        self.assertEqual(problems, [])

    def test_cross_check_detects_orphan_catalog_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog_path = Path(tmp) / "catalog.toml"
            catalog_path.write_text(
                "version = 1\n\n"
                "[[patch]]\n"
                'id = "9999_does_not_exist"\n'
                'kind = "framework"\n'
                'origin = "local"\n'
                'backend = "hip"\n'
                'state = "validated"\n',
                encoding="utf-8",
            )
            problems = patch_catalog.cross_check(catalog_path=catalog_path)
            self.assertTrue(any("9999_does_not_exist" in p for p in problems))

    def test_cross_check_detects_dangling_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            patches_dir = Path(tmp) / "patches"
            patches_dir.mkdir()
            (patches_dir / "0001_untracked.py").write_text(
                'GROUP = "core"\nSTATE = "validated"\n',
                encoding="utf-8",
            )
            catalog_path = Path(tmp) / "catalog.toml"
            catalog_path.write_text("version = 1\n", encoding="utf-8")
            problems = patch_catalog.cross_check(
                catalog_path=catalog_path, patches_dir=patches_dir
            )
            self.assertTrue(any("0001_untracked" in p for p in problems))

    def test_cross_check_detects_state_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            patches_dir = Path(tmp) / "patches"
            patches_dir.mkdir()
            (patches_dir / "0001_module.py").write_text(
                'GROUP = "core"\nSTATE = "validated"\n',
                encoding="utf-8",
            )
            catalog_path = Path(tmp) / "catalog.toml"
            catalog_path.write_text(
                "version = 1\n\n"
                "[[patch]]\n"
                'id = "0001_module"\n'
                'kind = "framework"\n'
                'origin = "local"\n'
                'backend = "hip"\n'
                'state = "untested"\n',
                encoding="utf-8",
            )
            problems = patch_catalog.cross_check(
                catalog_path=catalog_path, patches_dir=patches_dir
            )
            self.assertTrue(any("state" in p and "0001_module" in p for p in problems))


class TestPatchCatalogValidation(unittest.TestCase):
    def test_rejects_duplicate_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog_path = Path(tmp) / "catalog.toml"
            catalog_path.write_text(
                "version = 1\n\n"
                "[[patch]]\n"
                'id = "dup"\nkind = "framework"\norigin = "local"\nbackend = "hip"\nstate = "validated"\n\n'
                "[[patch]]\n"
                'id = "dup"\nkind = "framework"\norigin = "local"\nbackend = "hip"\nstate = "validated"\n',
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                patch_catalog.load_catalog(catalog_path)

    def test_rejects_unknown_kind(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog_path = Path(tmp) / "catalog.toml"
            catalog_path.write_text(
                "version = 1\n\n"
                "[[patch]]\n"
                'id = "x"\nkind = "not-a-real-kind"\norigin = "local"\nbackend = "hip"\nstate = "validated"\n',
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                patch_catalog.load_catalog(catalog_path)


class TestPatchContext(unittest.TestCase):
    def _catalog(self, tmp: str, records: str) -> Path:
        catalog_path = Path(tmp) / "catalog.toml"
        catalog_path.write_text(f"version = 1\n\n{records}", encoding="utf-8")
        return catalog_path

    def test_backend_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog_path = self._catalog(
                tmp,
                (
                    '[[patch]]\nid = "vk_only"\nkind = "framework"\norigin = "local"\n'
                    'backend = "vulkan"\nstate = "validated"\n'
                ),
            )
            ctx = patch_catalog.PatchContext(backend="hip")
            with self.assertRaises(ValueError):
                patch_catalog.resolve_for_context(
                    ["vk_only"], ctx, catalog_path=catalog_path
                )

    def test_agnostic_backend_is_always_applicable(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog_path = self._catalog(
                tmp,
                (
                    '[[patch]]\nid = "any"\nkind = "framework"\norigin = "local"\n'
                    'backend = "agnostic"\nstate = "validated"\n'
                ),
            )
            for backend in ("hip", "vulkan"):
                ctx = patch_catalog.PatchContext(backend=backend)
                result = patch_catalog.resolve_for_context(
                    ["any"], ctx, catalog_path=catalog_path
                )
                self.assertEqual(result, ("any",))

    def test_missing_required_option_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog_path = self._catalog(
                tmp,
                (
                    '[[patch]]\nid = "needs_flag"\nkind = "framework"\norigin = "local"\n'
                    'backend = "hip"\nstate = "validated"\nrequires-options = ["GGML_HIP_AUTOTUNE"]\n'
                ),
            )
            ctx = patch_catalog.PatchContext(backend="hip", options=())
            with self.assertRaises(ValueError):
                patch_catalog.resolve_for_context(
                    ["needs_flag"], ctx, catalog_path=catalog_path
                )
            ctx_ok = patch_catalog.PatchContext(
                backend="hip", options=("GGML_HIP_AUTOTUNE",)
            )
            result = patch_catalog.resolve_for_context(
                ["needs_flag"], ctx_ok, catalog_path=catalog_path
            )
            self.assertEqual(result, ("needs_flag",))

    def test_forbidden_option_present_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog_path = self._catalog(
                tmp,
                (
                    '[[patch]]\nid = "clashes"\nkind = "framework"\norigin = "local"\n'
                    'backend = "hip"\nstate = "validated"\nforbids-options = ["GGML_HIP_DISPATCH_REPLAY"]\n'
                ),
            )
            ctx = patch_catalog.PatchContext(
                backend="hip", options=("GGML_HIP_DISPATCH_REPLAY",)
            )
            with self.assertRaises(ValueError):
                patch_catalog.resolve_for_context(
                    ["clashes"], ctx, catalog_path=catalog_path
                )

    def test_resolved_base_revision_is_forwarded_to_the_admission_gate(self):
        """Adversarial-review follow-up (HI102): resolve_for_context() must
        actually thread a caller-supplied resolved_base_revision through to
        patch_admission.require_admission() at the real production seam --
        a prior pass added the parameter but never wired a real call site
        to actually populate it, so live-revision enforcement was
        documented as done while never actually invoked in production."""
        with tempfile.TemporaryDirectory() as tmp:
            catalog_path = self._catalog(
                tmp,
                (
                    '[[patch]]\nid = "any"\nkind = "framework"\norigin = "local"\n'
                    'backend = "agnostic"\nstate = "validated"\n'
                ),
            )
            ctx = patch_catalog.PatchContext(backend="hip")
            with mock.patch.object(paths, "PATCH_CATALOG", catalog_path), \
                 mock.patch("bigcherry.patch_admission.require_admission") as mocked:
                patch_catalog.resolve_for_context(
                    ["any"], ctx, catalog_path=catalog_path,
                    resolved_base_revision="deadbeef" * 5,
                )
            mocked.assert_called_once()
            self.assertEqual(mocked.call_args.kwargs.get("resolved_base_revision"), "deadbeef" * 5)

    def test_patches_for_backend_on_the_real_catalog_is_empty_for_vulkan(self):
        """No Vulkan patches exist yet -- an empty result is the CORRECT
        answer (RE30 phases 2+ need real Vulkan hardware evidence first),
        not a bug."""
        result = patch_catalog.patches_for_backend("vulkan")
        self.assertEqual(result, ())

    def test_patches_for_backend_on_the_real_catalog_returns_all_hip_patches(self):
        result = patch_catalog.patches_for_backend("hip")
        entries = patch_catalog.load_catalog()
        expected = tuple(
            sorted(
                pid for pid, e in entries.items() if e.backend in ("hip", "agnostic")
            )
        )
        self.assertEqual(result, expected)
        self.assertGreater(len(result), 0)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------- RS03: packaged
# patch-system PA03/RS03: packaged patches carry their metadata in
# patch.toml (no duplicate catalog.toml entry); legacy patches keep
# catalog.toml as their authority.


PACKED_TOML = """\
schema = 1
id = "1204_focal"
order = 1204
group = "rdna-boosts"
state = "untested"
kind = "enhancement"
origin = "external-fork"
backend = "hip"
external-source = "stew675-rdna-boosts"
plan-ids = ["RD12"]
"""

LEGACY_CATALOG_TOML = """\
version = 1

[[patch]]
id = "0100_dep"
kind = "framework"
origin = "local"
backend = "hip"
state = "validated"
"""


class TestPackagedCatalogIntegration(unittest.TestCase):
    def _tree(self, tmp: str, *, with_catalog_entry: bool = False):
        root = Path(tmp) / "patches"
        (root / "rd/1204_focal").mkdir(parents=True)
        (root / "0100_dep.py").write_text(
            "GROUP = 'core'\nSTATE = 'validated'\nPATCHES = []\n", encoding="utf-8"
        )
        (root / "rd/1204_focal/patch.toml").write_text(PACKED_TOML, encoding="utf-8")
        (root / "rd/1204_focal/patch.py").write_text("PATCHES = []\n", encoding="utf-8")
        catalog_body = LEGACY_CATALOG_TOML
        if with_catalog_entry:
            catalog_body += (
                "\n[[patch]]\n"
                'id = "1204_focal"\n'
                'kind = "enhancement"\n'
                'origin = "external-fork"\n'
                'backend = "hip"\n'
                'state = "untested"\n'
            )
        (root / "catalog.toml").write_text(catalog_body, encoding="utf-8")
        return root

    def test_build_snapshot_merges_packaged_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._tree(tmp)
            snapshot = patch_catalog.build_snapshot(
                patches_dir=root, catalog_path=root / "catalog.toml"
            )
            self.assertEqual(
                [m.patch_id for m in snapshot.modules], ["0100_dep", "1204_focal"]
            )
            legacy = snapshot.entry_for("0100_dep")
            self.assertIsNotNone(legacy)
            self.assertEqual(legacy.kind, "framework")
            self.assertEqual(legacy.origin, "local")
            packaged = snapshot.entry_for("1204_focal")
            self.assertIsNotNone(packaged)
            self.assertEqual(packaged.kind, "enhancement")
            self.assertEqual(packaged.origin, "external-fork")
            self.assertEqual(packaged.backend, "hip")
            self.assertEqual(packaged.external_source, "stew675-rdna-boosts")
            self.assertEqual(packaged.plan_ids, ("RD12",))
            self.assertEqual(packaged.state, "untested")

    def test_build_snapshot_deterministic_for_mixed_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._tree(tmp)
            first = patch_catalog.build_snapshot(
                patches_dir=root, catalog_path=root / "catalog.toml"
            )
            second = patch_catalog.build_snapshot(
                patches_dir=root, catalog_path=root / "catalog.toml"
            )
            self.assertEqual(first.digest, second.digest)
            self.assertEqual(first.modules, second.modules)
            self.assertEqual(first.metadata, second.metadata)

    def test_packaged_patch_with_catalog_entry_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._tree(tmp, with_catalog_entry=True)
            with self.assertRaisesRegex(ValueError, "duplicate metadata authority"):
                patch_catalog.build_snapshot(
                    patches_dir=root, catalog_path=root / "catalog.toml"
                )

    def test_cross_check_exempts_packaged_patches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._tree(tmp)
            problems = patch_catalog.cross_check(
                catalog_path=root / "catalog.toml", patches_dir=root
            )
            self.assertEqual(problems, [])

    def test_cross_check_still_flags_uncataloged_legacy_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._tree(tmp)
            (root / "0200_rogue.py").write_text(
                "GROUP = 'core'\nSTATE = 'untested'\nPATCHES = []\n", encoding="utf-8"
            )
            problems = patch_catalog.cross_check(
                catalog_path=root / "catalog.toml", patches_dir=root
            )
            self.assertIn("patch module '0200_rogue' has no catalog entry", problems)

    def test_explain_renders_packaged_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._tree(tmp)
            snapshot = patch_catalog.build_snapshot(
                patches_dir=root, catalog_path=root / "catalog.toml"
            )
            info = patch_catalog.explain("1204_focal", snapshot, cfg=None)
            self.assertEqual(info.kind, "enhancement")
            self.assertEqual(info.origin, "external-fork")
            self.assertEqual(info.backend, "hip")
            self.assertEqual(info.plan_ids, ("RD12",))
            rendered = patch_catalog.render_explanation(info)
            self.assertIn("kind:           enhancement", rendered)
            self.assertIn("origin:         external-fork", rendered)

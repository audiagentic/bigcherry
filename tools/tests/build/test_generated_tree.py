"""RE14 generated-tree hash verification (generated_tree.py)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.build.generated_tree import (GeneratedTreeError, build_manifest,  # noqa: E402
                                      file_sha256, verify_tree)


def _write(root: Path, relative: str, content: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


class BuildManifestTests(unittest.TestCase):
    def test_records_every_file_and_the_compile_input_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = _write(root, "hip-autotune-registry.inc", "registry")
            _write(root, "hip-autotune-manifest.json", '{"generated_at": "t1"}')

            tree = build_manifest(root, compile_inputs=(registry,))
            self.assertEqual(set(tree["files"]), {
                "hip-autotune-registry.inc", "hip-autotune-manifest.json"})
            self.assertEqual(tree["compile_inputs"], ["hip-autotune-registry.inc"])
            self.assertEqual(
                tree["files"]["hip-autotune-registry.inc"], file_sha256(registry))

    def test_compile_inputs_hash_ignores_files_outside_the_subset(self):
        # The exact scenario this module exists to solve: a manifest with a
        # different generated_at timestamp must NOT change
        # compile_inputs_hash, since it isn't a compiler input.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = _write(root, "hip-autotune-registry.inc", "same-bytes")
            _write(root, "hip-autotune-manifest.json", '{"generated_at": "t1"}')
            first = build_manifest(root, compile_inputs=(registry,))

            _write(root, "hip-autotune-manifest.json", '{"generated_at": "t2-different"}')
            second = build_manifest(root, compile_inputs=(registry,))

            self.assertEqual(first["compile_inputs_hash"], second["compile_inputs_hash"])
            self.assertNotEqual(
                first["files"]["hip-autotune-manifest.json"],
                second["files"]["hip-autotune-manifest.json"])

    def test_compile_inputs_hash_changes_when_a_compile_input_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = _write(root, "hip-autotune-registry.inc", "v1")
            first = build_manifest(root, compile_inputs=(registry,))

            _write(root, "hip-autotune-registry.inc", "v2-changed")
            second = build_manifest(root, compile_inputs=(registry,))

            self.assertNotEqual(first["compile_inputs_hash"], second["compile_inputs_hash"])

    def test_missing_declared_compile_input_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "does-not-exist.inc"
            _write(root, "other.txt", "x")
            with self.assertRaises(GeneratedTreeError):
                build_manifest(root, compile_inputs=(missing,))

    def test_symlink_in_tree_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = _write(root, "real.txt", "x")
            link = root / "link.txt"
            try:
                link.symlink_to(target)
            except OSError as exc:
                self.skipTest(
                    "generated-tree symlink rejection requires symlink creation "
                    f"privilege, unavailable in this environment: {exc}"
                )
            with self.assertRaises(GeneratedTreeError):
                build_manifest(root, compile_inputs=(target,))


class VerifyTreeTests(unittest.TestCase):
    def test_unmodified_tree_verifies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = _write(root, "hip-autotune-registry.inc", "registry")
            tree = build_manifest(root, compile_inputs=(registry,))
            verify_tree(root, tree)  # must not raise

    def test_modified_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = _write(root, "hip-autotune-registry.inc", "original")
            tree = build_manifest(root, compile_inputs=(registry,))
            _write(root, "hip-autotune-registry.inc", "tampered")
            with self.assertRaises(GeneratedTreeError):
                verify_tree(root, tree)

    def test_missing_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = _write(root, "hip-autotune-registry.inc", "x")
            tree = build_manifest(root, compile_inputs=(registry,))
            registry.unlink()
            with self.assertRaises(GeneratedTreeError):
                verify_tree(root, tree)

    def test_unexpected_extra_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = _write(root, "hip-autotune-registry.inc", "x")
            tree = build_manifest(root, compile_inputs=(registry,))
            _write(root, "sneaked-in.inc", "surprise")
            with self.assertRaises(GeneratedTreeError):
                verify_tree(root, tree)

    def test_unknown_schema_version_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(GeneratedTreeError):
                verify_tree(root, {"schema_version": 999, "files": {}})


if __name__ == "__main__":
    unittest.main()

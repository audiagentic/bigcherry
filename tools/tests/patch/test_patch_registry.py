"""RS01 tests for tools/bigcherry/patch_registry.py (patch-system PA02).

Covers the runbook RS01 required test list (section 54) plus digest
determinism, the byte-compile loader, the v1 dependency-DAG pin, and the
path-safety rule.
"""

from __future__ import annotations

import ast
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch import registry as patch_registry # noqa: E402

# RV80: the registry now resolves linked contracts through the canonical
# ExperimentContract.contract_hash (experiment_contract.load_contracts), so
# any fixture contract must be SCHEMA-VALID (strict parse), not a bare table.
CONTRACTS_TOML = """\
[contract.TEST-CONTRACT]
title = "test contract"

[contract.TEST-CONTRACT.source]
source_id = "stew675-rdna-boosts"
commits = ["abc123def456"]
atomic_part = "test"

[contract.TEST-CONTRACT.hypothesis]
family = "mmvq"
expected_effect = "performance"
rationale = "test hypothesis"

[contract.TEST-CONTRACT.scope]
backend = "hip"
architectures = ["gfx1100"]

[contract.TEST-CONTRACT.positive]
models = ["m1"]
workloads = ["decode"]

[contract.TEST-CONTRACT.controls]
models = ["m1"]
workloads = ["prefill"]

[contract.TEST-CONTRACT.acceptance]
target_kernel_gain_pct = 1
max_control_regression_pct = 1
"""

SIMPLE_PY = '''\
GROUP = "core"
STATE = "validated"
from bigcherry.patcher import Edit, FilePatch
PATCHES = [FilePatch(path="a.txt", edits=(Edit(id="e1", anchor="old", text="new"),))]
'''

PACKAGE_TOML = """\
schema = 1
id = "{patch_id}"
order = {order}
group = "core"
state = "validated"
kind = "framework"
origin = "local"
backend = "hip"
{extra}
"""

NO_EDIT_PY = "STATE = 'validated'\n"
NO_CONTRACT_PY = "GROUP = 'core'\nSTATE = 'validated'\nPATCHES = []\n"


def write(root: Path, rel: str, content: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def make_contracts(root: Path) -> Path:
    return write(root, "experiment-contracts.toml", CONTRACTS_TOML)


class DiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="bigcherry-registry-")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "patches"
        self.root.mkdir()

    def test_root_legacy_patch_discovered(self) -> None:
        write(self.root, "0100_foo.py", SIMPLE_PY)
        registry = patch_registry.load_registry(self.root)
        self.assertEqual([d.patch_id for d in registry.descriptors], ["0100_foo"])
        descriptor = registry.get("0100_foo")
        self.assertEqual(descriptor.representation, patch_registry.REPRESENTATION_SIMPLE)
        self.assertIsNone(descriptor.package_root)
        self.assertIsNone(descriptor.metadata_path)
        self.assertEqual(descriptor.state, "validated")
        self.assertEqual(descriptor.group, "core")

    def test_nested_patch_toml_discovered(self) -> None:
        package = "rd/1204_rd08_test"
        write(self.root, f"{package}/patch.toml",
              PACKAGE_TOML.format(patch_id="1204_rd08_test", order=1204, extra=""))
        write(self.root, f"{package}/patch.py", NO_CONTRACT_PY)
        registry = patch_registry.load_registry(self.root)
        descriptor = registry.get("1204_rd08_test")
        self.assertEqual(descriptor.representation, patch_registry.REPRESENTATION_PACKAGED)
        self.assertEqual(descriptor.package_root, Path("rd/1204_rd08_test"))
        self.assertEqual(descriptor.metadata_path, Path("rd/1204_rd08_test/patch.toml"))
        self.assertEqual(descriptor.implementation_path, Path("rd/1204_rd08_test/patch.py"))

    def test_nested_patch_py_not_separately_discovered(self) -> None:
        package = "rd/1204_rd08_test"
        write(self.root, f"{package}/patch.toml",
              PACKAGE_TOML.format(patch_id="1204_rd08_test", order=1204, extra=""))
        write(self.root, f"{package}/patch.py", NO_CONTRACT_PY)
        # A nested .py that is NOT a package's patch.py: never discovered.
        write(self.root, f"{package}/implementation/helpers.py", NO_CONTRACT_PY)
        registry = patch_registry.load_registry(self.root)
        self.assertEqual(len(registry.descriptors), 1)

    def test_nested_validation_checks_not_discovered(self) -> None:
        package = "rd/1204_rd08_test"
        write(self.root, f"{package}/patch.toml",
              PACKAGE_TOML.format(patch_id="1204_rd08_test", order=1204, extra=""))
        write(self.root, f"{package}/patch.py", NO_CONTRACT_PY)
        write(self.root, f"{package}/validation/checks.py", "def check(ctx):\n    pass\n")
        registry = patch_registry.load_registry(self.root)
        ids = [d.patch_id for d in registry.descriptors]
        self.assertEqual(ids, ["1204_rd08_test"])
        self.assertNotIn("checks", ids)

    def test_private_legacy_patch_ignored(self) -> None:
        write(self.root, "_private_helper.py", NO_CONTRACT_PY)
        write(self.root, "0100_foo.py", SIMPLE_PY)
        registry = patch_registry.load_registry(self.root)
        self.assertEqual([d.patch_id for d in registry.descriptors], ["0100_foo"])

    def test_template_package_ignored(self) -> None:
        write(self.root, "_template/patch.toml",
              PACKAGE_TOML.format(patch_id="_template", order=1, extra=""))
        write(self.root, "_template/patch.py", NO_CONTRACT_PY)
        write(self.root, "0100_foo.py", SIMPLE_PY)
        registry = patch_registry.load_registry(self.root)
        self.assertEqual([d.patch_id for d in registry.descriptors], ["0100_foo"])

    def test_missing_root_is_empty_registry(self) -> None:
        registry = patch_registry.load_registry(self.root / "nope")
        self.assertEqual(registry.descriptors, ())


class DuplicateIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="bigcherry-registry-")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "patches"
        self.root.mkdir()

    def _packaged(self, patch_id: str, order: int, directory: str) -> None:
        package = f"grp/{directory}"
        write(self.root, f"{package}/patch.toml",
              PACKAGE_TOML.format(patch_id=patch_id, order=order, extra=""))
        write(self.root, f"{package}/patch.py", NO_CONTRACT_PY)

    def test_duplicate_id_simple_and_packaged_rejected(self) -> None:
        # Runbook section 5: same canonical ID, different physical
        # representations, must fail.
        write(self.root, "1204_rd08_test.py", NO_CONTRACT_PY)
        self._packaged("1204_rd08_test", 1204, "1204_rd08_test")
        with self.assertRaisesRegex(patch_registry.PatchRegistryError, "duplicate patch ID"):
            patch_registry.load_registry(self.root)

    def test_duplicate_id_two_packages_rejected(self) -> None:
        # Runbook section 5: two different package directories declaring the
        # same ID must fail, even though the directories differ.
        write(self.root, "rd/1204_rd08_test/patch.toml",
              PACKAGE_TOML.format(patch_id="1204_rd08_test", order=1204, extra=""))
        write(self.root, "rd/1204_rd08_test/patch.py", NO_CONTRACT_PY)
        write(self.root, "core/1204_rd08_test/patch.toml",
              PACKAGE_TOML.format(patch_id="1204_rd08_test", order=1204, extra=""))
        write(self.root, "core/1204_rd08_test/patch.py", NO_CONTRACT_PY)
        with self.assertRaisesRegex(patch_registry.PatchRegistryError, "duplicate patch ID"):
            patch_registry.load_registry(self.root)


class PackagedSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="bigcherry-registry-")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "patches"
        self.root.mkdir()
        self.contracts = make_contracts(self.root.parent)

    def _load(self) -> None:
        patch_registry.load_registry(self.root, contracts_path=self.contracts)

    def _write_package(self, toml_body: str) -> None:
        write(self.root, "rd/1204_rd08_test/patch.toml", toml_body)
        write(self.root, "rd/1204_rd08_test/patch.py", NO_CONTRACT_PY)

    def test_directory_id_mismatch_rejected(self) -> None:
        self._write_package(
            PACKAGE_TOML.format(patch_id="1204_rd08", order=1204, extra="")
        )
        with self.assertRaisesRegex(patch_registry.PatchRegistryError, "must equal id"):
            self._load()

    def test_missing_patch_py_rejected(self) -> None:
        write(self.root, "rd/1204_rd08_test/patch.toml",
              PACKAGE_TOML.format(patch_id="1204_rd08_test", order=1204, extra=""))
        with self.assertRaisesRegex(patch_registry.PatchRegistryError, "missing patch.py"):
            self._load()

    def test_bad_schema_rejected(self) -> None:
        self._write_package(
            PACKAGE_TOML.format(patch_id="1204_rd08_test", order=1204, extra="").replace(
                "schema = 1", "schema = 2"
            )
        )
        with self.assertRaisesRegex(patch_registry.PatchRegistryError, "unsupported schema"):
            self._load()

    def test_unknown_key_rejected(self) -> None:
        self._write_package(
            PACKAGE_TOML.format(patch_id="1204_rd08_test", order=1204, extra="bogus-key = 1")
        )
        with self.assertRaisesRegex(patch_registry.PatchRegistryError, "unknown key"):
            self._load()

    def test_missing_required_key_rejected(self) -> None:
        body = PACKAGE_TOML.format(patch_id="1204_rd08_test", order=1204, extra="")
        self._write_package(body.replace('group = "core"\n', ""))
        with self.assertRaisesRegex(patch_registry.PatchRegistryError, "missing required"):
            self._load()

    def test_bad_state_rejected(self) -> None:
        self._write_package(
            PACKAGE_TOML.format(patch_id="1204_rd08_test", order=1204, extra="").replace(
                'state = "validated"', 'state = "shipped"'
            )
        )
        with self.assertRaisesRegex(patch_registry.PatchRegistryError, "state must be one of"):
            self._load()

    def test_bad_dependency_metadata_rejected(self) -> None:
        # A non-string dependency: [42] is valid TOML (homogeneous int array)
        # but fails the string-list schema check.
        self._write_package(
            PACKAGE_TOML.format(
                patch_id="1204_rd08_test", order=1204, extra="requires = [42]"
            )
        )
        with self.assertRaisesRegex(patch_registry.PatchRegistryError,
                                    "requires' must be a list of non-empty strings"):
            self._load()

    def test_order_must_match_id_prefix(self) -> None:
        self._write_package(
            PACKAGE_TOML.format(patch_id="1204_rd08_test", order=1205, extra="")
        )
        with self.assertRaisesRegex(patch_registry.PatchRegistryError,
                                    "does not match id prefix"):
            self._load()

    def test_unknown_experiment_contract_rejected(self) -> None:
        self._write_package(
            PACKAGE_TOML.format(
                patch_id="1204_rd08_test", order=1204,
                extra='experiment-contract = "NO-SUCH-CONTRACT"',
            )
        )
        with self.assertRaisesRegex(patch_registry.PatchRegistryError,
                                    "not found in experiment-contracts"):
            self._load()

    def test_non_canonical_id_pattern_rejected(self) -> None:
        write(self.root, "rd/rd08_test/patch.toml",
              PACKAGE_TOML.format(patch_id="rd08_test", order=1204, extra=""))
        write(self.root, "rd/rd08_test/patch.py", NO_CONTRACT_PY)
        with self.assertRaisesRegex(patch_registry.PatchRegistryError,
                                    "canonical ID pattern"):
            self._load()


class LegacyValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="bigcherry-registry-")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "patches"
        self.root.mkdir()

    def test_bad_state_rejected(self) -> None:
        write(self.root, "0100_foo.py", "GROUP = 'core'\nSTATE = 'shipped'\nPATCHES = []\n")
        with self.assertRaisesRegex(patch_registry.PatchRegistryError, "invalid STATE"):
            patch_registry.load_registry(self.root)

    def test_bad_dependency_constant_rejected(self) -> None:
        write(self.root, "0100_foo.py",
              "STATE = 'validated'\nREQUIRES = 'not-a-list-and-nope'\nPATCHES = []\n")
        # A plain string IS legal (single dependency); exercise the real bad
        # shape: a list containing a non-string.
        write(self.root, "0100_foo.py",
              "STATE = 'validated'\nREQUIRES = ['0100_foo', 3]\nPATCHES = []\n")
        with self.assertRaisesRegex(patch_registry.PatchRegistryError,
                                    "must be a string or list/tuple of strings"):
            patch_registry.load_registry(self.root)


class OrderingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="bigcherry-registry-")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "patches"
        self.root.mkdir()

    def test_deterministic_ordering_mixed_representations(self) -> None:
        write(self.root, "1200_b.py", NO_CONTRACT_PY)
        write(self.root, "1000_a.py", NO_CONTRACT_PY)
        for patch_id in ("1100_c_pkg", "1300_d_pkg"):
            package = f"rd/{patch_id}"
            write(self.root, f"{package}/patch.toml",
                  PACKAGE_TOML.format(patch_id=patch_id,
                                      order=int(patch_id.split("_")[0]), extra=""))
            write(self.root, f"{package}/patch.py", NO_CONTRACT_PY)
        first = patch_registry.load_registry(self.root)
        second = patch_registry.load_registry(self.root)
        self.assertEqual(
            [d.patch_id for d in first.descriptors],
            ["1000_a", "1100_c_pkg", "1200_b", "1300_d_pkg"],
        )
        self.assertEqual(first.descriptors, second.descriptors)


class DigestTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="bigcherry-registry-")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "patches"
        self.root.mkdir()
        self.contracts = make_contracts(self.root.parent)

    def _package(self) -> Path:
        package = self.root / "rd/1204_rd08_test"
        package.mkdir(parents=True, exist_ok=True)
        (package / "patch.toml").write_text(
            PACKAGE_TOML.format(
                patch_id="1204_rd08_test", order=1204,
                extra='experiment-contract = "TEST-CONTRACT"',
            ),
            encoding="utf-8",
        )
        (package / "patch.py").write_text(NO_CONTRACT_PY, encoding="utf-8")
        return package

    def test_no_validation_toml_gives_none(self) -> None:
        self._package()
        descriptor = patch_registry.load_registry(
            self.root, contracts_path=self.contracts
        ).get("1204_rd08_test")
        self.assertIsNone(descriptor.validation_path)
        self.assertIsNone(descriptor.validation_digest)

    def test_validation_digest_deterministic_and_sensitive(self) -> None:
        self._package()
        package = self.root / "rd/1204_rd08_test"
        (package / "validation").mkdir()
        (package / "validation/checks.py").write_text("def check(ctx):\n    pass\n", encoding="utf-8")
        (package / "validation.toml").write_text("schema = 1\n", encoding="utf-8")
        first = patch_registry.load_registry(self.root, contracts_path=self.contracts)
        second = patch_registry.load_registry(self.root, contracts_path=self.contracts)
        digest_a = first.get("1204_rd08_test").validation_digest
        digest_b = second.get("1204_rd08_test").validation_digest
        self.assertIsNotNone(digest_a)
        self.assertEqual(digest_a, digest_b)
        self.assertEqual(
            first.get("1204_rd08_test").validation_path,
            Path("rd/1204_rd08_test/validation.toml"),
        )

        # Any content change under validation/ (or the manifest) invalidates.
        (package / "validation/checks.py").write_text("def check(ctx):\n    pass\n# edit\n", encoding="utf-8")
        third = patch_registry.load_registry(self.root, contracts_path=self.contracts)
        self.assertNotEqual(third.get("1204_rd08_test").validation_digest, digest_a)

    def test_validation_digest_includes_framework_version(self) -> None:
        self._package()
        package = self.root / "rd/1204_rd08_test"
        (package / "validation.toml").write_text("schema = 1\n", encoding="utf-8")
        baseline = patch_registry.load_registry(self.root, contracts_path=self.contracts)
        digest = baseline.get("1204_rd08_test").validation_digest
        self.assertEqual(patch_registry.VALIDATION_FRAMEWORK_VERSION, "2")
        # The digest is stable absent semantic change...
        second = patch_registry.load_registry(self.root, contracts_path=self.contracts)
        self.assertEqual(
            second.get("1204_rd08_test").validation_digest, digest,
            "digest must be stable absent semantic change",
        )
        # ...and a semantic-version bump MUST invalidate it (runbook 14.2/15).
        original = patch_registry.VALIDATION_FRAMEWORK_VERSION
        try:
            patch_registry.VALIDATION_FRAMEWORK_VERSION = "3"
            bumped = patch_registry.load_registry(self.root, contracts_path=self.contracts)
            self.assertNotEqual(
                bumped.get("1204_rd08_test").validation_digest, digest,
                "framework version bump must change the validation digest",
            )
        finally:
            patch_registry.VALIDATION_FRAMEWORK_VERSION = original

    def test_implementation_digest_is_sha256_of_bytes(self) -> None:
        write(self.root, "0100_foo.py", SIMPLE_PY)
        descriptor = patch_registry.load_registry(self.root).get("0100_foo")
        expected = hashlib.sha256(
            (self.root / "0100_foo.py").read_bytes()
        ).hexdigest()
        self.assertEqual(descriptor.implementation_digest, expected)


class LoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="bigcherry-registry-")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "patches"
        self.root.mkdir()

    def test_load_simple_patch(self) -> None:
        write(self.root, "0100_foo.py", SIMPLE_PY)
        registry = patch_registry.load_registry(self.root)
        patches = patch_registry.load_implementation(registry.get("0100_foo"), root=self.root)
        self.assertEqual(len(patches), 1)
        self.assertEqual(patches[0].path, "a.txt")

    def test_load_packaged_patch(self) -> None:
        package = "rd/1204_rd08_test"
        write(self.root, f"{package}/patch.toml",
              PACKAGE_TOML.format(patch_id="1204_rd08_test", order=1204, extra=""))
        write(self.root, f"{package}/patch.py", SIMPLE_PY)
        registry = patch_registry.load_registry(self.root)
        patches = patch_registry.load_implementation(registry.get("1204_rd08_test"), root=self.root)
        self.assertEqual(patches[0].path, "a.txt")

    def test_module_defining_no_patches_rejected(self) -> None:
        write(self.root, "0100_foo.py", NO_EDIT_PY)
        registry = patch_registry.load_registry(self.root)
        with self.assertRaisesRegex(patch_registry.PatchRegistryError,
                                    "defines neither PATCH nor PATCHES"):
            patch_registry.load_implementation(registry.get("0100_foo"), root=self.root)

    def test_loader_uses_byte_compilation_not_importlib(self) -> None:
        source = (
            Path(__file__).resolve().parents[2] / "bigcherry" / "patch" / "registry.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertNotIn("importlib", [alias.name for alias in node.names])
            if isinstance(node, ast.ImportFrom) and node.module:
                self.assertNotEqual(node.module.split(".")[0], "importlib")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotEqual(node.func.attr, "import_module")


class ArchitectureTests(unittest.TestCase):
    """Runbook section 11 dependency-DAG pin: the registry imports only
    {paths, apply} (+ stdlib); never patchset/check/catalog/validation/source."""

    _FORBIDDEN = {"patchset", "check", "catalog", "validation", "source"}

    def test_registry_import_boundary(self) -> None:
        source = (
            Path(__file__).resolve().parents[2] / "bigcherry" / "patch" / "registry.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        local_modules = {name for name in imported if not Path(name).is_absolute()
                         and name not in sys.stdlib_module_names}
        self.assertFalse(self._FORBIDDEN & local_modules,
                         f"registry imports forbidden local modules: {self._FORBIDDEN & local_modules}")
        allowed_local = {"bigcherry", "core", "paths", "apply"}
        self.assertTrue(local_modules <= allowed_local,
                        f"registry local imports exceed the allowed set: {local_modules - allowed_local}")


if __name__ == "__main__":
    unittest.main()

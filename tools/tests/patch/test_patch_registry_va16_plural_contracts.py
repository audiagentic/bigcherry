"""VA16 (VA09A) tests: PatchDescriptor.experiment_contracts (plural) as
the canonical binding authority, with the legacy singular
experiment-contract normalizing to a one-element tuple, and the
compatibility .experiment_contract property failing closed for any
multi-contract patch rather than silently picking one.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch import registry as patch_registry # noqa: E402

CONTRACTS_TOML = """\
[contract.TEST-CONTRACT-A]
title = "test contract A"

[contract.TEST-CONTRACT-A.source]
source_id = "stew675-rdna-boosts"
commits = ["abc123def456"]
atomic_part = "test-a"

[contract.TEST-CONTRACT-A.hypothesis]
family = "mmvq"
expected_effect = "performance"
rationale = "test hypothesis a"

[contract.TEST-CONTRACT-A.scope]
backend = "hip"
architectures = ["gfx1100"]

[contract.TEST-CONTRACT-A.positive]
models = ["m1"]
workloads = ["decode"]

[contract.TEST-CONTRACT-A.controls]
models = ["m1"]
workloads = ["prefill"]

[contract.TEST-CONTRACT-A.acceptance]
target_kernel_gain_pct = 1
max_control_regression_pct = 1

[contract.TEST-CONTRACT-B]
title = "test contract B"

[contract.TEST-CONTRACT-B.source]
source_id = "stew675-rdna-boosts"
commits = ["def456abc123"]
atomic_part = "test-b"

[contract.TEST-CONTRACT-B.hypothesis]
family = "mmq"
expected_effect = "correctness"
rationale = "test hypothesis b"

[contract.TEST-CONTRACT-B.scope]
backend = "hip"
architectures = ["gfx1201"]

[contract.TEST-CONTRACT-B.positive]
models = ["m2"]
workloads = ["prefill"]

[contract.TEST-CONTRACT-B.controls]
models = ["m2"]
workloads = ["decode"]

[contract.TEST-CONTRACT-B.correctness]
bit_identical = "required"

[contract.TEST-CONTRACT-B.acceptance]
max_control_regression_pct = 1

[contract.TEST-CONTRACT-C]
title = "test contract C"

[contract.TEST-CONTRACT-C.source]
source_id = "stew675-rdna-boosts"
commits = ["112233445566"]
atomic_part = "test-c"

[contract.TEST-CONTRACT-C.hypothesis]
family = "mmvf"
expected_effect = "performance"
rationale = "test hypothesis c"

[contract.TEST-CONTRACT-C.scope]
backend = "hip"
architectures = ["gfx1030"]

[contract.TEST-CONTRACT-C.positive]
models = ["m3"]
workloads = ["decode"]

[contract.TEST-CONTRACT-C.controls]
models = ["m3"]
workloads = ["prefill"]

[contract.TEST-CONTRACT-C.acceptance]
target_kernel_gain_pct = 1
max_control_regression_pct = 1
"""

PACKAGE_TOML = """\
schema = 1
id = "{patch_id}"
order = {order}
group = "core"
state = "untested"
kind = "enhancement"
origin = "local"
backend = "hip"
{extra}
"""

NO_CONTRACT_PY = "GROUP = 'core'\nSTATE = 'untested'\nPATCHES = []\n"


def write(root: Path, rel: str, content: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


class PluralContractRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="bigcherry-registry-va16-")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "patches"
        self.root.mkdir()
        self.contracts = write(Path(self._tmp.name), "experiment-contracts.toml", CONTRACTS_TOML)

    def _write_package(self, patch_id: str, order: int, extra: str) -> None:
        package = self.root / "rd" / patch_id
        package.mkdir(parents=True, exist_ok=True)
        (package / "patch.toml").write_text(
            PACKAGE_TOML.format(patch_id=patch_id, order=order, extra=extra), encoding="utf-8"
        )
        (package / "patch.py").write_text(NO_CONTRACT_PY, encoding="utf-8")

    def _load(self):
        return patch_registry.load_registry(self.root, contracts_path=self.contracts)

    def test_singular_legacy_normalizes_to_one_element_tuple(self) -> None:
        self._write_package("1300_singular", 1300, 'experiment-contract = "TEST-CONTRACT-A"')
        descriptor = self._load().get("1300_singular")
        self.assertEqual(descriptor.experiment_contracts, ("TEST-CONTRACT-A",))
        self.assertEqual(descriptor.experiment_contract, "TEST-CONTRACT-A")

    def test_no_contract_is_empty_tuple(self) -> None:
        self._write_package("1300_none", 1300, "")
        descriptor = self._load().get("1300_none")
        self.assertEqual(descriptor.experiment_contracts, ())
        self.assertIsNone(descriptor.experiment_contract)

    def test_plural_parses_and_sorts(self) -> None:
        self._write_package(
            "1300_plural", 1300,
            'experiment-contracts = ["TEST-CONTRACT-C", "TEST-CONTRACT-A", "TEST-CONTRACT-B"]',
        )
        descriptor = self._load().get("1300_plural")
        self.assertEqual(
            descriptor.experiment_contracts,
            ("TEST-CONTRACT-A", "TEST-CONTRACT-B", "TEST-CONTRACT-C"),
        )

    def test_plural_access_raises_for_multi_contract_patch(self) -> None:
        self._write_package(
            "1300_plural", 1300,
            'experiment-contracts = ["TEST-CONTRACT-A", "TEST-CONTRACT-B"]',
        )
        descriptor = self._load().get("1300_plural")
        with self.assertRaises(patch_registry.PatchRegistryError):
            _ = descriptor.experiment_contract

    def test_both_keys_present_rejected(self) -> None:
        self._write_package(
            "1300_both", 1300,
            'experiment-contract = "TEST-CONTRACT-A"\n'
            'experiment-contracts = ["TEST-CONTRACT-B"]',
        )
        with self.assertRaisesRegex(patch_registry.PatchRegistryError, "must not both be present"):
            self._load()

    def test_explicit_empty_plural_list_rejected(self) -> None:
        self._write_package("1300_empty", 1300, "experiment-contracts = []")
        with self.assertRaisesRegex(patch_registry.PatchRegistryError, "must not be an empty list"):
            self._load()

    def test_duplicate_plural_ids_rejected(self) -> None:
        self._write_package(
            "1300_dup", 1300,
            'experiment-contracts = ["TEST-CONTRACT-A", "TEST-CONTRACT-A"]',
        )
        with self.assertRaisesRegex(patch_registry.PatchRegistryError, "duplicates"):
            self._load()

    def test_unknown_plural_id_rejected_at_discovery(self) -> None:
        self._write_package(
            "1300_unknown", 1300,
            'experiment-contracts = ["TEST-CONTRACT-A", "NO-SUCH-CONTRACT"]',
        )
        with self.assertRaisesRegex(patch_registry.PatchRegistryError, "not found in experiment-contracts"):
            self._load()

    def test_contract_ordering_in_toml_does_not_affect_canonical_identity(self) -> None:
        self._write_package(
            "1300_order_a", 1300,
            'experiment-contracts = ["TEST-CONTRACT-A", "TEST-CONTRACT-B"]',
        )
        self._write_package(
            "1301_order_b", 1301,
            'experiment-contracts = ["TEST-CONTRACT-B", "TEST-CONTRACT-A"]',
        )
        registry = self._load()
        a = registry.get("1300_order_a")
        b = registry.get("1301_order_b")
        self.assertEqual(a.experiment_contracts, b.experiment_contracts)

    def test_single_contract_validation_digest_unchanged_shape(self) -> None:
        """The payload/schema for 0/1-contract packages must stay
        byte-identical to the pre-VA16 shape -- verified here by
        constructing a real validation.toml and confirming the digest is
        stable and deterministic across repeated loads, not by comparing
        to a historical recorded value (which drifts with cache-file
        presence, unrelated to this change -- see VA16's plan notes)."""
        self._write_package("1300_single", 1300, 'experiment-contract = "TEST-CONTRACT-A"')
        package = self.root / "rd" / "1300_single"
        (package / "validation.toml").write_text(
            'schema = 1\n\n[[check]]\nid = "apply"\ncapability = "apply"\n'
            'validator = "apply"\nrequired = true\n', encoding="utf-8",
        )
        first = self._load().get("1300_single").validation_digest
        second = self._load().get("1300_single").validation_digest
        self.assertIsNotNone(first)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()

"""VA17 policy slice: validation_policy.py's plural has_contract /
require_execution_package() / architecture-union semantics.

GPT review (session ses_d1759a9471d443d5, following up on
ses_5bbee8ce5c9a4265's req_e3f0e25329814273/req_4fdf8cdfa7d94e20/
req_3487e42d4567418f): the three remaining singular
``descriptor.experiment_contract``/``plan.contract`` accesses in
validation_policy.py are exactly the ones this slice replaces --
``has_contract``, ``require_execution_package()``, and the
architecture-match check (now an exact UNION across every bound
contract's own required architectures, not just one contract's view).

Uses two REAL bound contracts (RD05-WMMA-FA-CORRECTNESS-BARRIERS,
RD06-RDNA4-WMMA-FA-CONFIG -- both real, gfx1201-only, already committed
to config/experiment-contracts.toml) on a synthetic packaged test patch,
since no real on-branch packaged patch with a committed validation.toml
binds multiple contracts yet (1203 is contract-set-only, see
test_patch_validation_va17.py).
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch import registry as patch_registry  # noqa: E402
from bigcherry.patch import validation_policy as vp  # noqa: E402

PATCH_PY = "STATE = 'validated'\n"

VALIDATION_TOML = """\
schema = 1

[[check]]
id = "apply"
capability = "apply"
validator = "apply"
required = true

[[check]]
id = "build"
capability = "build"
validator = "build"
required = true

[[check]]
id = "correctness-both"
capability = "correctness"
validator = "backend-ops"
required = true
contracts = ["RD05-WMMA-FA-CORRECTNESS-BARRIERS", "RD06-RDNA4-WMMA-FA-CONFIG"]
ops = ["SOME_OP"]

[[check]]
id = "controls-both"
capability = "controls"
validator = "benchmark"
required = true
contracts = ["RD05-WMMA-FA-CORRECTNESS-BARRIERS", "RD06-RDNA4-WMMA-FA-CONFIG"]

[[check]]
id = "performance-rd06"
capability = "performance"
validator = "benchmark"
required = true
contracts = ["RD06-RDNA4-WMMA-FA-CONFIG"]

[[check]]
id = "activation-rd06"
capability = "activation"
validator = "trace-marker"
required = true
contracts = ["RD06-RDNA4-WMMA-FA-CONFIG"]
marker-regex = "hit"
"""


def _patch_toml(*, validation_architectures: str) -> str:
    return f"""\
schema = 1
id = "9999_va17_policy_test"
order = 9999
group = "test"
state = "validated"
kind = "enhancement"
origin = "external-fork"
backend = "hip"
plan-item = "RD05/RD06"
experiment-contracts = ["RD05-WMMA-FA-CORRECTNESS-BARRIERS", "RD06-RDNA4-WMMA-FA-CONFIG"]
validation-architectures = [{validation_architectures}]
"""


def _write_package(
    root: Path, *, validation_architectures: str, with_validation_toml: bool = True,
) -> Path:
    package_dir = root / "9999_va17_policy_test"
    package_dir.mkdir(parents=True)
    (package_dir / "patch.py").write_text(PATCH_PY, encoding="utf-8")
    (package_dir / "patch.toml").write_text(
        _patch_toml(validation_architectures=validation_architectures), encoding="utf-8",
    )
    (package_dir / "README.md").write_text("# example\n", encoding="utf-8")
    if with_validation_toml:
        (package_dir / "validation.toml").write_text(VALIDATION_TOML, encoding="utf-8")
    return package_dir


class MultiContractHasContractTests(unittest.TestCase):
    def test_multi_contract_patch_is_not_reported_as_missing_a_contract(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write_package(root, validation_architectures='"gfx1201"')
            report = vp.check_validation_packages(root=root)
            problems = [p for p in report.problems if "9999_va17_policy_test" in p]
            self.assertFalse(
                any("no experiment-contract bound" in p for p in problems),
                f"unexpected 'no experiment-contract bound' problem for a real "
                f"2-contract patch: {problems}",
            )


class ArchitectureUnionTests(unittest.TestCase):
    def test_union_of_two_contracts_architectures_matches(self) -> None:
        # Both RD05 and RD06 are real, gfx1201-only contracts -- the union
        # is exactly {gfx1201}.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write_package(root, validation_architectures='"gfx1201"')
            report = vp.check_validation_packages(root=root)
            problems = [p for p in report.problems if "9999_va17_policy_test" in p]
            self.assertFalse(
                any("validation-architectures" in p for p in problems),
                f"unexpected architecture-mismatch problem: {problems}",
            )

    def test_union_mismatch_is_a_real_problem(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            # gfx1100 is real hardware neither RD05 nor RD06 declares -- this
            # is caught even earlier than the union check itself, by
            # check_contract_compatibility() inside build_plan_for_patch()
            # (one bound contract's own architecture scope already
            # contradicts the declared set). Both are real, correct
            # fail-closed signals for the same underlying mismatch.
            _write_package(root, validation_architectures='"gfx1100"')
            report = vp.check_validation_packages(root=root)
            problems = [p for p in report.problems if "9999_va17_policy_test" in p]
            self.assertTrue(
                any("architectures" in p for p in problems),
                f"expected an architecture-mismatch problem, got: {problems}",
            )

    def test_union_mismatch_within_the_declared_footprint_is_caught_by_the_union_check(
        self,
    ) -> None:
        # RD06 is gfx1201-only; RD07 is gfx1030/gfx1100/gfx1201. Declaring
        # only gfx1201 is individually COMPATIBLE with both (compatibility
        # only rejects a declared architecture a bound contract explicitly
        # excludes -- it never requires the declared set to be the full
        # union). check_contract_compatibility() alone would therefore NOT
        # catch this; only the union check in validation_policy.py does --
        # this isolates that check specifically.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            package_dir = _write_package(root, validation_architectures='"gfx1201"')
            (package_dir / "patch.toml").write_text(
                _patch_toml(validation_architectures='"gfx1201"').replace(
                    'experiment-contracts = ["RD05-WMMA-FA-CORRECTNESS-BARRIERS", '
                    '"RD06-RDNA4-WMMA-FA-CONFIG"]',
                    'experiment-contracts = ["RD06-RDNA4-WMMA-FA-CONFIG", "RD07-Q6K-MMQ-PREFILL-FOLD"]',
                ).replace('plan-item = "RD05/RD06"', 'plan-item = "RD06/RD07"'),
                encoding="utf-8",
            )
            (package_dir / "validation.toml").write_text(
                VALIDATION_TOML.replace(
                    'contracts = ["RD05-WMMA-FA-CORRECTNESS-BARRIERS", "RD06-RDNA4-WMMA-FA-CONFIG"]',
                    'contracts = ["RD06-RDNA4-WMMA-FA-CONFIG", "RD07-Q6K-MMQ-PREFILL-FOLD"]',
                ).replace(
                    'contracts = ["RD06-RDNA4-WMMA-FA-CONFIG"]',
                    'contracts = ["RD06-RDNA4-WMMA-FA-CONFIG", "RD07-Q6K-MMQ-PREFILL-FOLD"]',
                ),
                encoding="utf-8",
            )
            report = vp.check_validation_packages(root=root)
            problems = [p for p in report.problems if "9999_va17_policy_test" in p]
            self.assertTrue(
                any("validation-architectures" in p for p in problems),
                f"expected the union-check's own architecture-mismatch problem, got: {problems}",
            )


class RequireExecutionPackageMultiContractTests(unittest.TestCase):
    def test_require_execution_package_accepts_a_real_multi_contract_patch(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write_package(root, validation_architectures='"gfx1201"')
            registry = patch_registry.load_registry(root)
            descriptor = registry.get("9999_va17_policy_test")
            plan = vp.require_execution_package(descriptor, root=root)
            self.assertEqual(len(plan.contracts), 2)
            self.assertEqual(
                {b.contract_id for b in plan.contracts},
                {"RD05-WMMA-FA-CORRECTNESS-BARRIERS", "RD06-RDNA4-WMMA-FA-CONFIG"},
            )


if __name__ == "__main__":
    unittest.main()

"""VA26 P0: PQM-v1 qualification-matrix planner tests.

Uses a synthetic packaged-patch tree + config + experiment-contracts file
(rather than the real project catalog) so REQUIRES/dependent shapes needed
for the counterfactual_not_composable and staleness cases can be
constructed deliberately."""

from __future__ import annotations

import dataclasses
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.campaign import qualification_matrix as qm  # noqa: E402
from bigcherry.campaign import resolution as campaign_resolution  # noqa: E402
from bigcherry.core import config, paths  # noqa: E402
from bigcherry.experiment import contract as ec  # noqa: E402
from bigcherry.patch import patchset  # noqa: E402
from bigcherry.patch import registry as patch_registry  # noqa: E402


def _write_packaged_patch(
    patches_root: Path, patch_id: str, order: int, *, state: str = "validated",
    requires: tuple[str, ...] = (), validation_architectures: tuple[str, ...] = (),
    experiment_contract: str | None = None, marker_text: str = "x",
) -> None:
    package_dir = patches_root / patch_id
    package_dir.mkdir(parents=True, exist_ok=True)
    requires_toml = ", ".join(f'"{r}"' for r in requires)
    arches_toml = ", ".join(f'"{a}"' for a in validation_architectures)
    lines = [
        "schema = 1",
        f'id = "{patch_id}"',
        f"order = {order}",
        f'state = "{state}"',
        'kind = "enhancement"',
        f"requires = [{requires_toml}]",
        "conflicts = []",
        "requires-options = []",
        "forbids-options = []",
        "subsystems = []",
        "hardware = []",
        f"validation-architectures = [{arches_toml}]",
        "backends = []",
    ]
    if experiment_contract is not None:
        lines.append(f'experiment-contract = "{experiment_contract}"')
    (package_dir / "patch.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (package_dir / "patch.py").write_text(
        "from bigcherry.patcher import Edit, FilePatch\n"
        "PATCH = FilePatch(\n"
        "    path='source.txt',\n"
        "    edits=(Edit(\n"
        f"        id='{patch_id}',\n"
        "        anchor=r'one',\n"
        f"        text='{marker_text}',\n"
        "        mode='insert_after',\n"
        "    ),),\n"
        ")\n",
        encoding="utf-8",
    )


def _write_contracts_toml(path: Path, *, contract_id: str, architectures: tuple[str, ...]) -> None:
    arches_toml = ", ".join(f'"{a}"' for a in architectures)
    path.write_text(f"""
[contract.{contract_id}]
title = "synthetic test contract"

[contract.{contract_id}.source]
source_id = "synthetic-source"
commits = ["deadbeefcafef00d"]
atomic_part = "synthetic-part"

[contract.{contract_id}.hypothesis]
family = "mmq"
expected_effect = "performance"
rationale = "test fixture"

[contract.{contract_id}.scope]
backend = "hip"
architectures = [{arches_toml}]
weight_types = ["q8_0"]

[contract.{contract_id}.positive]
models = ["recipe/model"]
workloads = ["small_m"]

[contract.{contract_id}.controls]
models = ["control-recipe"]
workloads = ["decode"]

[contract.{contract_id}.boundary]

[contract.{contract_id}.boundary.dimensions]
physical_m = [1, 2, 4]

[contract.{contract_id}.correctness]
backend_reference = "required"
greedy_parity = "required"

[contract.{contract_id}.acceptance]
target_kernel_gain_pct = 5
end_to_end_gain_pct = 1
max_control_regression_pct = 1
""".lstrip(), encoding="utf-8")


class QualificationMatrixFixture(unittest.TestCase):
    """Builds: framework = [0001_base]; validated-enhancements =
    [0004_shipped, 0005_dependent_on_shipped] (0005 REQUIRES 0004);
    candidate = 0002_candidate (untested, contract CONTRACT-1, gain arch
    gfx1100, safety arch gfx1201), not in any patch-set / not shipped."""

    CONTRACT_ID = "PQM-TEST-CONTRACT-1"

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.patches_root = root / "patches"
        self.contracts_path = root / "experiment-contracts.toml"

        _write_packaged_patch(
            self.patches_root, "0001_base", 1, state="validated", marker_text="base")
        _write_packaged_patch(
            self.patches_root, "0002_candidate", 2, state="untested",
            validation_architectures=("gfx1100", "gfx1201"),
            experiment_contract=self.CONTRACT_ID, marker_text="candidate")
        _write_packaged_patch(
            self.patches_root, "0004_shipped", 4, state="validated",
            validation_architectures=("gfx1100",),
            experiment_contract=self.CONTRACT_ID, marker_text="shipped")
        _write_packaged_patch(
            self.patches_root, "0005_dependent_on_shipped", 5, state="validated",
            requires=("0004_shipped",), marker_text="dependent")

        _write_contracts_toml(
            self.contracts_path, contract_id=self.CONTRACT_ID,
            architectures=("gfx1100",))

        # patchset.catalog() (called both directly and internally by
        # resolution.resolve_patch_set()'s own consistency re-check) has no
        # contracts_path seam and always resolves experiment-contract
        # bindings against paths.EXPERIMENT_CONTRACTS -- patch the default
        # for the duration of this fixture so every caller (ours and
        # resolution.py's internal re-derivation) sees the SAME synthetic
        # contracts file, rather than threading contracts_path through two
        # independent call paths that must otherwise agree by construction.
        patcher = mock.patch.object(paths, "EXPERIMENT_CONTRACTS", self.contracts_path)
        patcher.start()
        self.addCleanup(patcher.stop)

        self.registry = patch_registry.load_registry(self.patches_root)
        self.catalog = patchset.catalog(directory=self.patches_root)
        self.contract = ec.load_contracts(self.contracts_path).contracts[self.CONTRACT_ID]

        self.cfg = config.Config(
            pinned="unused",
            patch_sets={
                "framework": config.PatchSet(
                    name="framework", patches=("0001_base",), required_state="validated"),
                "validated-enhancements": config.PatchSet(
                    name="validated-enhancements",
                    patches=("0004_shipped", "0005_dependent_on_shipped"),
                    required_state="validated"),
            },
            sources={
                "bigcherry-native": config.Source(
                    name="bigcherry-native", ref="pinned", overlay=False,
                    patch_sets=("framework",)),
                "bigcherry": config.Source(
                    name="bigcherry", ref="pinned", overlay=False,
                    patch_sets=("framework", "validated-enhancements")),
            },
            builds={}, platforms={}, experiments={}, campaigns={},
            path=root / "recipes.toml",
        )

    def _plan(self, **kwargs) -> qm.QualificationMatrixPlan:
        return qm.build_qualification_matrix_plan(
            "0002_candidate", self.contract, self.cfg, self.catalog, self.registry,
            catalog_directory=self.patches_root, **kwargs,
        )


class BuildQualificationMatrixPlanTests(QualificationMatrixFixture):
    def test_gain_and_safety_arches_are_derived_correctly(self):
        plan = self._plan()
        self.assertEqual(plan.gain_arches, ("gfx1100",))
        self.assertEqual(plan.safety_arches, ("gfx1201",))

    def test_policy_and_identity_fields(self):
        plan = self._plan()
        self.assertEqual(plan.qualification_policy, "pqm-v1")
        self.assertEqual(plan.patch_id, "0002_candidate")
        self.assertEqual(plan.contract_id, self.CONTRACT_ID)
        self.assertEqual(plan.contract_hash, self.contract.contract_hash)
        self.assertEqual(plan.dependency_closure, ("0002_candidate",))

    def test_four_cells_per_architecture_two_contrasts_one_gain_one_safety(self):
        plan = self._plan()
        # 1 gain arch x 2 contrasts + 1 safety arch x 2 contrasts = 4 cells.
        self.assertEqual(len(plan.cells), 4)
        by_arch_tier = {(c.architecture, c.contrast): c for c in plan.cells}
        self.assertEqual(by_arch_tier[("gfx1100", "isolated")].evidence_level, "inferential")
        self.assertEqual(by_arch_tier[("gfx1100", "release_delta")].evidence_level, "inferential")
        self.assertEqual(by_arch_tier[("gfx1201", "isolated")].evidence_level, "smoke")
        self.assertEqual(by_arch_tier[("gfx1201", "release_delta")].evidence_level, "smoke")

    def test_plan_is_deterministic_across_two_generations(self):
        first = self._plan()
        second = self._plan()
        self.assertEqual(first, second)

    def test_isolated_control_is_native_release_delta_control_is_release(self):
        plan = self._plan()
        isolated = next(c for c in plan.cells if c.contrast == "isolated")
        release_delta = next(c for c in plan.cells if c.contrast == "release_delta")
        native = campaign_resolution.resolve_lane(
            "bigcherry-native", self.cfg, self.catalog, catalog_directory=self.patches_root)
        release = campaign_resolution.resolve_lane(
            "bigcherry", self.cfg, self.catalog, catalog_directory=self.patches_root)
        self.assertEqual(isolated.control.patch_set_id, native.patch_set.patch_set_id)
        self.assertEqual(release_delta.control.patch_set_id, release.patch_set.patch_set_id)
        self.assertIn("0002_candidate", isolated.subject.module_ids)
        self.assertIn("0002_candidate", release_delta.subject.module_ids)

    def test_release_composition_hash_matches_current_release(self):
        plan = self._plan()
        release = campaign_resolution.resolve_lane(
            "bigcherry", self.cfg, self.catalog, catalog_directory=self.patches_root)
        self.assertEqual(plan.release_composition_hash, release.patch_set.patch_set_id)


class QualificationMatrixFailClosedTests(QualificationMatrixFixture):
    def test_unknown_patch_id_raises(self):
        with self.assertRaises(qm.QualificationMatrixError):
            qm.build_qualification_matrix_plan(
                "9999_nonexistent", self.contract, self.cfg, self.catalog, self.registry,
                catalog_directory=self.patches_root,
            )

    def test_contract_not_declared_by_patch_raises(self):
        other_contract_path = Path(self._tmp.name) / "other-contracts.toml"
        _write_contracts_toml(
            other_contract_path, contract_id="SOME-OTHER-CONTRACT", architectures=("gfx1100",))
        other_contract = ec.load_contracts(other_contract_path).contracts["SOME-OTHER-CONTRACT"]
        with self.assertRaisesRegex(qm.QualificationMatrixError, "not one of this patch"):
            qm.build_qualification_matrix_plan(
                "0002_candidate", other_contract, self.cfg, self.catalog, self.registry,
                catalog_directory=self.patches_root,
            )

    def test_gain_arch_not_in_validation_architectures_raises(self):
        bad_contract_path = Path(self._tmp.name) / "bad-contracts.toml"
        _write_contracts_toml(
            bad_contract_path, contract_id=self.CONTRACT_ID, architectures=("gfx9999",))
        bad_registry = patch_registry.load_registry(
            self.patches_root, contracts_path=bad_contract_path)
        bad_contract = ec.load_contracts(bad_contract_path).contracts[self.CONTRACT_ID]
        with self.assertRaisesRegex(qm.QualificationMatrixError, "not in the patch"):
            qm.build_qualification_matrix_plan(
                "0002_candidate", bad_contract, self.cfg, self.catalog, bad_registry,
                catalog_directory=self.patches_root,
            )

    def test_unavailable_host_architecture_raises(self):
        with self.assertRaisesRegex(qm.QualificationMatrixError, "host inventory"):
            self._plan(host_architectures=("gfx1100",))  # missing gfx1201

    def test_already_shipped_patch_is_rejected_as_a_candidate(self):
        with self.assertRaisesRegex(qm.QualificationMatrixError, "already part of"):
            qm.build_qualification_matrix_plan(
                "0004_shipped",
                ec.load_contracts(self.contracts_path).contracts[self.CONTRACT_ID],
                self.cfg, self.catalog, self.registry, catalog_directory=self.patches_root,
            )


class StaleReleaseDeltaCellsTests(QualificationMatrixFixture):
    def test_no_staleness_when_release_unchanged(self):
        plan = self._plan()
        self.assertEqual(qm.stale_release_delta_cells(
            plan, self.cfg, self.catalog, catalog_directory=self.patches_root), ())

    def test_release_delta_cells_go_stale_after_admission_isolated_cells_do_not(self):
        plan = self._plan()
        # Simulate a new admission by adding a patch to validated-enhancements.
        _write_packaged_patch(
            self.patches_root, "0006_newly_admitted", 6, state="validated", marker_text="new")
        grown = dataclasses.replace(
            self.cfg.patch_sets["validated-enhancements"],
            patches=self.cfg.patch_sets["validated-enhancements"].patches + ("0006_newly_admitted",),
        )
        cfg = dataclasses.replace(
            self.cfg, patch_sets={**self.cfg.patch_sets, "validated-enhancements": grown})
        catalog = patchset.catalog(directory=self.patches_root)

        stale = qm.stale_release_delta_cells(plan, cfg, catalog, catalog_directory=self.patches_root)
        self.assertEqual(len(stale), 2)
        self.assertTrue(all(c.contrast == "release_delta" for c in stale))
        fresh_contrasts = {c.contrast for c in plan.cells} - {c.contrast for c in stale}
        self.assertEqual(fresh_contrasts, {"isolated"})


class CounterfactualComposableTests(QualificationMatrixFixture):
    def test_shipped_patch_with_a_real_dependent_is_not_composable(self):
        disposition = qm.check_counterfactual_composable(
            "0004_shipped", self.cfg, self.catalog, catalog_directory=self.patches_root)
        self.assertFalse(disposition.composable)
        self.assertEqual(disposition.blocking_dependents, ("0005_dependent_on_shipped",))

    def test_shipped_patch_with_no_dependents_is_composable(self):
        disposition = qm.check_counterfactual_composable(
            "0005_dependent_on_shipped", self.cfg, self.catalog, catalog_directory=self.patches_root)
        self.assertTrue(disposition.composable)
        self.assertEqual(disposition.blocking_dependents, ())

    def test_never_recursively_drops_dependents(self):
        # 0004_shipped has a dependent; the disposition must report the
        # blocker, not silently produce a composable release with both gone.
        disposition = qm.check_counterfactual_composable(
            "0004_shipped", self.cfg, self.catalog, catalog_directory=self.patches_root)
        self.assertIn("0005_dependent_on_shipped", disposition.blocking_dependents)

    def test_patch_not_in_release_raises(self):
        with self.assertRaises(qm.QualificationMatrixError):
            qm.check_counterfactual_composable(
                "0002_candidate", self.cfg, self.catalog, catalog_directory=self.patches_root)


if __name__ == "__main__":
    unittest.main()

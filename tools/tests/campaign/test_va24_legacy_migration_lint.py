"""VA24 registry lint + qualification rule (design: dev-gpt-agent req_358ce8a96f49469b).

The two rules are deliberately keyed differently:
  * LINT (edit time)          -> contract ID
  * QUALIFICATION (evidence)  -> exact baseline_contract_hash

Keying the lint by hash was the first proposal and is wrong: contract_hash
covers rationale prose and source provenance, so a typo fix would evaporate a
waiver and demand a fresh hardware qualification -- impossible for contracts
targeting hardware this project does not own, making them uneditable.
"""
import dataclasses
import unittest
from pathlib import Path

from bigcherry.experiment import contract as ec

REGISTRY = Path("config/experiment-contracts.toml")
MANIFEST = Path(ec.LEGACY_MANIFEST_PATH)


class LegacyMigrationLintTests(unittest.TestCase):
    def setUp(self):
        self.registry = ec.load_contracts(REGISTRY)
        self.waivers = ec.load_legacy_waivers(MANIFEST)

    def test_real_registry_is_clean(self):
        self.assertEqual(ec.lint_effect_evidence_policy(self.registry, self.waivers), [])

    def test_every_waived_contract_actually_exists_and_declares_a_gain(self):
        for contract_id in self.waivers:
            self.assertIn(contract_id, self.registry.contracts, contract_id)
            self.assertTrue(
                ec.declares_gain_threshold(self.registry[contract_id]),
                f"{contract_id} is waived but declares no gain threshold",
            )

    def test_new_gain_contract_without_a_waiver_fails_lint(self):
        base = self.registry["RD08-Q6K-MMVQ-VDR2"]
        fresh = dataclasses.replace(base, id="RDXX-BRAND-NEW")
        registry = ec.ContractRegistry(contracts={"RDXX-BRAND-NEW": fresh})
        problems = ec.lint_effect_evidence_policy(registry, self.waivers)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("RDXX-BRAND-NEW", problems[0])
        self.assertIn("ci95_threshold_bound_v1", problems[0])

    def test_correctness_only_contract_needs_no_waiver(self):
        """A contract with no gain threshold has nothing to evidence with an
        interval, so it must not be dragged into the migration."""
        base = self.registry["RD08-Q6K-MMVQ-VDR2"]
        acceptance = dataclasses.replace(
            base.acceptance, target_kernel_gain_pct=None, end_to_end_gain_pct=None)
        fresh = dataclasses.replace(base, id="RDXX-CORRECTNESS-ONLY", acceptance=acceptance)
        registry = ec.ContractRegistry(contracts={"RDXX-CORRECTNESS-ONLY": fresh})
        self.assertEqual(ec.lint_effect_evidence_policy(registry, self.waivers), [])

    def test_a_lost_manifest_fails_closed(self):
        """No manifest must mean 'everything needs the strong policy', never
        'everything is waived'."""
        empty = ec.load_legacy_waivers(Path("config/does-not-exist.toml"))
        self.assertEqual(empty, {})
        problems = ec.lint_effect_evidence_policy(self.registry, empty)
        self.assertTrue(problems)

    # ---------------- qualification rule: exact hash ----------------

    def test_unedited_legacy_contract_keeps_its_evidence_waiver(self):
        for contract_id in self.waivers:
            self.assertTrue(
                ec.legacy_evidence_is_honoured(self.registry[contract_id], self.waivers),
                contract_id,
            )

    def test_edited_contract_stays_lint_clean_but_loses_its_evidence_waiver(self):
        """The whole point of splitting the two rules.

        A prose edit changes contract_hash. The contract remains EDITABLE
        (lint clean, keyed by ID) but can no longer acquire fresh legacy
        qualification (keyed by exact hash), so it must migrate before it is
        next qualified.
        """
        base = self.registry["RD08-Q6K-MMVQ-VDR2"]
        edited_hypothesis = dataclasses.replace(
            base.hypothesis, rationale=base.hypothesis.rationale + " (typo fixed)")
        edited = dataclasses.replace(base, hypothesis=edited_hypothesis)
        self.assertNotEqual(edited.contract_hash, base.contract_hash)

        registry = ec.ContractRegistry(contracts={edited.id: edited})
        self.assertEqual(
            ec.lint_effect_evidence_policy(registry, self.waivers), [],
            "a prose edit must not make a legacy contract un-committable")
        self.assertFalse(
            ec.legacy_evidence_is_honoured(edited, self.waivers),
            "an edited contract must not keep legacy evidence validity")


if __name__ == "__main__":
    unittest.main()

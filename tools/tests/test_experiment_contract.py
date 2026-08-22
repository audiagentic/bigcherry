"""Experiment Contract schema, validator and registry tests (EC01/EC02/EC12)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import experiment_contract as ec  # noqa: E402


def _write(text: str) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False, encoding="utf-8")
    with handle:
        handle.write(text)
    return Path(handle.name)


def _base_doc(**overrides: object) -> dict:
    doc = {
        "title": "tiny-M Q8 MMQ specialization",
        "source": {
            "source_id": "stew675-rdna-boosts",
            "commits": ["deadbeefcafef00d"],
            "atomic_part": "tiny-m-q8-gate",
        },
        "hypothesis": {
            "family": "mmq",
            "expected_effect": "performance",
            "rationale": "why this should win",
        },
        "prerequisites": [],
        "scope": {
            "backend": "hip",
            "architectures": ["gfx1100", "gfx1201"],
            "weight_types": ["q8_0"],
        },
        "positive": {"models": ["recipe/model"], "workloads": ["small_m", "mtp_verify"]},
        "controls": {"models": ["control-recipe"], "workloads": ["decode", "prefill"]},
        "boundary": {"dimensions": {"physical_m": [1, 2, 3, 4, 8, 16, 32, 64, 128]}},
        "correctness": {"backend_reference": "required", "greedy_parity": "required"},
        "acceptance": {
            "target_kernel_gain_pct": 5,
            "end_to_end_gain_pct": 1,
            "max_control_regression_pct": 1,
        },
    }
    doc.update(overrides)
    return doc


class ParseContractTests(unittest.TestCase):
    def test_parses_the_guide_worked_example(self):
        contract = ec.parse_contract(_base_doc(), contract_id="RDNA-EXT-001")
        self.assertEqual(contract.id, "RDNA-EXT-001")
        self.assertEqual(contract.hypothesis.family, "mmq")
        self.assertEqual(contract.correctness.required_checks, ("backend_reference", "greedy_parity"))
        self.assertEqual(
            contract.boundary.dimensions,
            (("physical_m", (1, 2, 3, 4, 8, 16, 32, 64, 128)),),
        )

    def test_unknown_kernel_family_rejected(self):
        doc = _base_doc()
        doc["hypothesis"] = dict(doc["hypothesis"], family="not-a-family")
        with self.assertRaisesRegex(ec.ExperimentContractError, "not-a-family"):
            ec.parse_contract(doc, contract_id="X")

    def test_unknown_workload_tag_rejected(self):
        doc = _base_doc()
        doc["positive"] = dict(doc["positive"], workloads=["not-a-workload"])
        with self.assertRaisesRegex(ec.ExperimentContractError, "not-a-workload"):
            ec.parse_contract(doc, contract_id="X")

    def test_unknown_correctness_check_rejected(self):
        doc = _base_doc()
        doc["correctness"] = {"not-a-check": "required"}
        with self.assertRaisesRegex(ec.ExperimentContractError, "not-a-check"):
            ec.parse_contract(doc, contract_id="X")

    def test_unknown_acceptance_field_rejected(self):
        doc = _base_doc()
        doc["acceptance"] = dict(doc["acceptance"], bogus_pct=1.0)
        with self.assertRaisesRegex(ec.ExperimentContractError, "bogus_pct"):
            ec.parse_contract(doc, contract_id="X")

    def test_unknown_top_level_field_rejected(self):
        doc = _base_doc(extra_field="oops")
        with self.assertRaisesRegex(ec.ExperimentContractError, "extra_field"):
            ec.parse_contract(doc, contract_id="X")

    def test_missing_regression_budget_rejected(self):
        doc = _base_doc()
        doc["acceptance"] = {"target_kernel_gain_pct": 5}
        with self.assertRaisesRegex(ec.ExperimentContractError, "max_control_regression_pct"):
            ec.parse_contract(doc, contract_id="X")

    def test_correctness_only_contract_is_valid(self):
        # Guide Appendix A: RD20/RD22/RD26-class contracts have no expected
        # throughput gain; performance is a regression budget only.
        doc = _base_doc()
        doc["acceptance"] = {"max_control_regression_pct": 1}
        contract = ec.parse_contract(doc, contract_id="X")
        self.assertIsNone(contract.acceptance.target_kernel_gain_pct)
        self.assertEqual(contract.acceptance.max_control_regression_pct, 1)

    def test_duplicate_boundary_values_rejected(self):
        doc = _base_doc()
        doc["boundary"] = {"dimensions": {"physical_m": [1, 1, 2]}}
        with self.assertRaisesRegex(ec.ExperimentContractError, "duplicate"):
            ec.parse_contract(doc, contract_id="X")

    def test_duplicate_model_or_workload_rejected(self):
        doc = _base_doc()
        doc["positive"] = dict(doc["positive"], models=["a", "a"])
        with self.assertRaisesRegex(ec.ExperimentContractError, "duplicates"):
            ec.parse_contract(doc, contract_id="X")

    def test_negative_gain_threshold_rejected(self):
        doc = _base_doc()
        doc["acceptance"] = dict(doc["acceptance"], target_kernel_gain_pct=-5)
        with self.assertRaisesRegex(ec.ExperimentContractError, "target_kernel_gain_pct"):
            ec.parse_contract(doc, contract_id="X")

    def test_negative_regression_budget_rejected(self):
        doc = _base_doc()
        doc["acceptance"] = {"max_control_regression_pct": -1}
        with self.assertRaisesRegex(ec.ExperimentContractError, "max_control_regression_pct"):
            ec.parse_contract(doc, contract_id="X")

    def test_nan_threshold_rejected(self):
        doc = _base_doc()
        doc["acceptance"] = dict(doc["acceptance"], target_kernel_gain_pct=float("nan"))
        with self.assertRaisesRegex(ec.ExperimentContractError, "target_kernel_gain_pct"):
            ec.parse_contract(doc, contract_id="X")

    def test_infinite_threshold_rejected(self):
        doc = _base_doc()
        doc["acceptance"] = {"max_control_regression_pct": float("inf")}
        with self.assertRaisesRegex(ec.ExperimentContractError, "max_control_regression_pct"):
            ec.parse_contract(doc, contract_id="X")

    def test_negative_infinite_threshold_rejected(self):
        doc = _base_doc()
        doc["acceptance"] = {"max_control_regression_pct": float("-inf")}
        with self.assertRaisesRegex(ec.ExperimentContractError, "max_control_regression_pct"):
            ec.parse_contract(doc, contract_id="X")


class TargetClassificationTests(unittest.TestCase):
    """EC16: the orthogonal target.kind/target.family classification."""

    def test_legacy_shape_no_target_section_derives_kernel_family(self):
        contract = ec.parse_contract(_base_doc(), contract_id="X")
        self.assertEqual(contract.target.kind, "kernel_family")
        self.assertEqual(contract.target.family, "mmq")
        self.assertEqual(contract.hypothesis.family, "mmq")

    def test_legacy_shape_missing_hypothesis_family_rejected(self):
        doc = _base_doc()
        doc["hypothesis"] = {k: v for k, v in doc["hypothesis"].items() if k != "family"}
        with self.assertRaisesRegex(ec.ExperimentContractError, "hypothesis.family"):
            ec.parse_contract(doc, contract_id="X")

    def test_explicit_kernel_family_target_matches_hypothesis(self):
        doc = _base_doc()
        doc["target"] = {"kind": "kernel_family", "family": "mmq"}
        contract = ec.parse_contract(doc, contract_id="X")
        self.assertEqual(contract.target.kind, "kernel_family")
        self.assertEqual(contract.target.family, "mmq")
        self.assertEqual(contract.hypothesis.family, "mmq")

    def test_explicit_kernel_family_target_disagreeing_with_hypothesis_rejected(self):
        doc = _base_doc()
        doc["target"] = {"kind": "kernel_family", "family": "mmvq"}
        with self.assertRaisesRegex(ec.ExperimentContractError, "disagrees"):
            ec.parse_contract(doc, contract_id="X")

    def test_kernel_family_target_without_family_rejected(self):
        doc = _base_doc()
        doc["target"] = {"kind": "kernel_family"}
        with self.assertRaisesRegex(ec.ExperimentContractError, "target.family"):
            ec.parse_contract(doc, contract_id="X")

    def test_non_kernel_family_target_drops_hypothesis_family_requirement(self):
        doc = _base_doc()
        doc["hypothesis"] = {k: v for k, v in doc["hypothesis"].items() if k != "family"}
        doc["target"] = {"kind": "attention"}
        contract = ec.parse_contract(doc, contract_id="X")
        self.assertEqual(contract.target.kind, "attention")
        self.assertIsNone(contract.target.family)
        self.assertIsNone(contract.hypothesis.family)

    def test_non_kernel_family_target_rejects_family_field(self):
        doc = _base_doc()
        doc["target"] = {"kind": "attention", "family": "mmq"}
        with self.assertRaisesRegex(ec.ExperimentContractError, "target.family"):
            ec.parse_contract(doc, contract_id="X")

    def test_non_kernel_family_target_rejects_leftover_hypothesis_family(self):
        doc = _base_doc()
        doc["target"] = {"kind": "graph_fusion"}
        # hypothesis.family="mmq" is still present from _base_doc() -- must
        # be explicitly removed for a non-kernel_family target, not silently
        # ignored (that would let a contract carry conflicting classification).
        with self.assertRaisesRegex(ec.ExperimentContractError, "hypothesis.family must be absent"):
            ec.parse_contract(doc, contract_id="X")

    def test_unknown_target_kind_rejected(self):
        doc = _base_doc()
        doc["hypothesis"] = {k: v for k, v in doc["hypothesis"].items() if k != "family"}
        doc["target"] = {"kind": "not_a_real_kind"}
        with self.assertRaisesRegex(ec.ExperimentContractError, "target.kind"):
            ec.parse_contract(doc, contract_id="X")

    def test_all_target_kinds_parse(self):
        for kind in ec.TARGET_KINDS:
            with self.subTest(kind=kind):
                doc = _base_doc()
                if kind == "kernel_family":
                    doc["target"] = {"kind": kind, "family": "mmq"}
                else:
                    doc["hypothesis"] = {k: v for k, v in doc["hypothesis"].items() if k != "family"}
                    doc["target"] = {"kind": kind}
                contract = ec.parse_contract(doc, contract_id="X")
                self.assertEqual(contract.target.kind, kind)


class ExistingBackfilledContractsRegressionTests(unittest.TestCase):
    """EC16 must not break the 5 contracts EC02 already backfilled into
    config/experiment-contracts.toml -- none of those TOML entries were
    rewritten; they must still parse via the legacy (no [target] section)
    path exactly as before."""

    def test_all_five_existing_contracts_still_parse(self):
        from bigcherry import paths as _paths
        registry = ec.load_contracts(_paths.EXPERIMENT_CONTRACTS)
        expected = {
            "RD07-Q6K-MMQ-PREFILL-FOLD": "mmq",
            "RD08-Q6K-MMVQ-VDR2": "mmvq",
            "RD12-PAIRED-MMVQ-DUAL": "mmvq",
            "RD17-MOE-TOPK-DOWN-FOLD": "mmvq",
            "RD21-GFX1151-MMVQ-NWARPS": "mmvq",
        }
        for contract_id, family in expected.items():
            with self.subTest(contract_id=contract_id):
                contract = registry[contract_id]
                self.assertEqual(contract.target.kind, "kernel_family")
                self.assertEqual(contract.target.family, family)
                self.assertEqual(contract.hypothesis.family, family)

    def test_all_seventeen_contracts_in_the_real_registry_parse(self):
        # EC17 regression proof: adding [source-evidence] as an optional
        # section must not break any of the 5 original (EC02) or 12
        # EC16-backfilled contracts already committed to
        # config/experiment-contracts.toml.
        from bigcherry import paths as _paths
        registry = ec.load_contracts(_paths.EXPERIMENT_CONTRACTS)
        self.assertEqual(len(registry.contracts), 17)


class SourceEvidenceTests(unittest.TestCase):
    """EC17: [source-evidence] is optional, structurally separate from
    [acceptance], and never blocks parsing on a mismatch."""

    def test_source_evidence_absent_by_default(self):
        contract = ec.parse_contract(_base_doc(), contract_id="X")
        self.assertIsNone(contract.source_evidence)
        self.assertIsNone(ec.source_evidence_mismatch_warning(contract))

    def test_source_evidence_parses_when_present(self):
        doc = _base_doc()
        doc["source-evidence"] = {
            "metric": "tg128", "value_pct": 7.4,
            "hardware": "gfx1151", "workload": "Qwen3.6-35B-A3B Q4_K_M",
        }
        contract = ec.parse_contract(doc, contract_id="X")
        self.assertIsNotNone(contract.source_evidence)
        self.assertEqual(contract.source_evidence.metric, "tg128")
        self.assertEqual(contract.source_evidence.value_pct, 7.4)
        self.assertEqual(contract.source_evidence.hardware, "gfx1151")
        self.assertEqual(contract.source_evidence.workload, "Qwen3.6-35B-A3B Q4_K_M")

    def test_source_evidence_unknown_field_rejected(self):
        doc = _base_doc()
        doc["source-evidence"] = {
            "metric": "tg128", "value_pct": 7.4, "hardware": "gfx1151",
            "workload": "w", "bogus": "x",
        }
        with self.assertRaisesRegex(ec.ExperimentContractError, "bogus"):
            ec.parse_contract(doc, contract_id="X")

    def test_source_evidence_non_finite_value_rejected(self):
        doc = _base_doc()
        doc["source-evidence"] = {
            "metric": "tg128", "value_pct": float("nan"),
            "hardware": "gfx1151", "workload": "w",
        }
        with self.assertRaises(ec.ExperimentContractError):
            ec.parse_contract(doc, contract_id="X")

    def test_mismatch_warning_when_acceptance_exceeds_source(self):
        # Acceptance requiring MORE than the source itself ever reported --
        # an impossible-to-meet bar.
        doc = _base_doc()
        doc["acceptance"] = dict(doc["acceptance"], target_kernel_gain_pct=10)
        doc["source-evidence"] = {
            "metric": "tg128", "value_pct": 5.0,
            "hardware": "gfx1100", "workload": "w",
        }
        contract = ec.parse_contract(doc, contract_id="X")
        warning = ec.source_evidence_mismatch_warning(contract)
        self.assertIsNotNone(warning)
        self.assertIn("exceeds", warning)

    def test_mismatch_warning_when_acceptance_far_below_source(self):
        doc = _base_doc()
        doc["acceptance"] = dict(doc["acceptance"], target_kernel_gain_pct=1)
        doc["source-evidence"] = {
            "metric": "tg128", "value_pct": 10.0,
            "hardware": "gfx1100", "workload": "w",
        }
        contract = ec.parse_contract(doc, contract_id="X")
        warning = ec.source_evidence_mismatch_warning(contract)
        self.assertIsNotNone(warning)
        self.assertIn("less than half", warning)

    def test_no_warning_when_acceptance_reasonably_aligned_with_source(self):
        doc = _base_doc()
        doc["acceptance"] = dict(doc["acceptance"], target_kernel_gain_pct=4)
        doc["source-evidence"] = {
            "metric": "tg128", "value_pct": 5.0,
            "hardware": "gfx1100", "workload": "w",
        }
        contract = ec.parse_contract(doc, contract_id="X")
        self.assertIsNone(ec.source_evidence_mismatch_warning(contract))

    def test_no_warning_when_acceptance_has_no_target_kernel_gain(self):
        # Correctness-only contract: nothing numeric to compare.
        doc = _base_doc()
        doc["acceptance"] = {"max_control_regression_pct": 1}
        doc["source-evidence"] = {
            "metric": "tg128", "value_pct": 5.0,
            "hardware": "gfx1100", "workload": "w",
        }
        contract = ec.parse_contract(doc, contract_id="X")
        self.assertIsNone(ec.source_evidence_mismatch_warning(contract))

    def test_real_rd21_contract_source_evidence_matches_its_own_acceptance(self):
        # RD21's real backfilled contract: source-evidence and acceptance
        # were set to the SAME real documented number (~+0.6% decode) --
        # must not spuriously warn.
        from bigcherry import paths as _paths
        registry = ec.load_contracts(_paths.EXPERIMENT_CONTRACTS)
        contract = registry["RD21-GFX1151-MMVQ-NWARPS"]
        self.assertIsNotNone(contract.source_evidence)
        self.assertEqual(contract.source_evidence.value_pct, 0.6)
        self.assertIsNone(ec.source_evidence_mismatch_warning(contract))


class ContractHashTests(unittest.TestCase):
    def test_hash_is_stable_across_reparse(self):
        first = ec.parse_contract(_base_doc(), contract_id="X")
        second = ec.parse_contract(_base_doc(), contract_id="X")
        self.assertEqual(first.contract_hash, second.contract_hash)

    def test_hash_changes_on_semantic_edit(self):
        first = ec.parse_contract(_base_doc(), contract_id="X")
        doc = _base_doc()
        doc["acceptance"] = dict(doc["acceptance"], target_kernel_gain_pct=6)
        second = ec.parse_contract(doc, contract_id="X")
        self.assertNotEqual(first.contract_hash, second.contract_hash)

    def test_hash_changes_on_id(self):
        first = ec.parse_contract(_base_doc(), contract_id="X")
        second = ec.parse_contract(_base_doc(), contract_id="Y")
        self.assertNotEqual(first.contract_hash, second.contract_hash)

    def test_hash_independent_of_dict_key_order(self):
        doc_a = _base_doc()
        doc_b = {key: doc_a[key] for key in reversed(list(doc_a))}
        first = ec.parse_contract(doc_a, contract_id="X")
        second = ec.parse_contract(doc_b, contract_id="X")
        self.assertEqual(first.contract_hash, second.contract_hash)


class LoadContractsTests(unittest.TestCase):
    TOML = """
[contract.RDNA-EXT-001]
title = "tiny-M Q8 MMQ specialization"

[contract.RDNA-EXT-001.source]
source_id = "stew675-rdna-boosts"
commits = ["deadbeef"]
atomic_part = "tiny-m-q8-gate"

[contract.RDNA-EXT-001.hypothesis]
family = "mmq"
expected_effect = "performance"
rationale = "why this should win"

[contract.RDNA-EXT-001.scope]
backend = "hip"
architectures = ["gfx1100", "gfx1201"]
weight_types = ["q8_0"]

[contract.RDNA-EXT-001.positive]
models = ["recipe/model"]
workloads = ["small_m", "mtp_verify"]

[contract.RDNA-EXT-001.controls]
models = ["control-recipe"]
workloads = ["decode", "prefill"]

[contract.RDNA-EXT-001.boundary.dimensions]
physical_m = [1, 2, 3, 4, 8, 16, 32, 64, 128]

[contract.RDNA-EXT-001.correctness]
backend_reference = "required"
greedy_parity = "required"

[contract.RDNA-EXT-001.acceptance]
target_kernel_gain_pct = 5
end_to_end_gain_pct = 1
max_control_regression_pct = 1
"""

    def test_loads_a_real_toml_file(self):
        path = _write(self.TOML)
        registry = ec.load_contracts(path)
        self.assertEqual(len(registry), 1)
        self.assertEqual(registry["RDNA-EXT-001"].hypothesis.family, "mmq")

    def test_unknown_source_id_rejected_when_checked(self):
        path = _write(self.TOML)
        with self.assertRaisesRegex(ec.ExperimentContractError, "stew675-rdna-boosts"):
            ec.load_contracts(path, known_source_ids=frozenset({"some-other-source"}))

    def test_known_source_id_accepted(self):
        path = _write(self.TOML)
        registry = ec.load_contracts(path, known_source_ids=frozenset({"stew675-rdna-boosts"}))
        self.assertEqual(len(registry), 1)

    def test_unknown_top_level_field_rejected(self):
        path = _write('bogus = true\n' + self.TOML)
        with self.assertRaisesRegex(ec.ExperimentContractError, "bogus"):
            ec.load_contracts(path)

    def test_unknown_prerequisite_rejected(self):
        toml = self.TOML.replace(
            "[contract.RDNA-EXT-001]",
            '[contract.RDNA-EXT-001]\nprerequisites = ["does-not-exist"]',
        )
        path = _write(toml)
        with self.assertRaisesRegex(ec.ExperimentContractError, "does-not-exist"):
            ec.load_contracts(path)

    def test_missing_file_rejected(self):
        with self.assertRaises(ec.ExperimentContractError):
            ec.load_contracts("/no/such/experiment-contracts.toml")

    def test_prerequisite_cycle_rejected(self):
        toml = f"""
[contract.A]
title = "A"
prerequisites = ["B"]
{self.TOML.split("[contract.RDNA-EXT-001.source]", 1)[1].replace("RDNA-EXT-001", "A")[:0]}
"""
        # Build two minimal, mutually-prerequisite contracts directly rather
        # than string-surgery on the fixture above.
        toml = """
[contract.A]
title = "A"
prerequisites = ["B"]

[contract.A.source]
source_id = "s"
commits = ["c"]
atomic_part = "p"

[contract.A.hypothesis]
family = "mmq"
expected_effect = "performance"
rationale = "r"

[contract.A.scope]
backend = "hip"
architectures = ["gfx1100"]

[contract.A.positive]
models = ["m"]
workloads = ["decode"]

[contract.A.controls]
models = ["m"]
workloads = ["decode"]

[contract.A.acceptance]
max_control_regression_pct = 1

[contract.B]
title = "B"
prerequisites = ["A"]

[contract.B.source]
source_id = "s"
commits = ["c"]
atomic_part = "p"

[contract.B.hypothesis]
family = "mmq"
expected_effect = "performance"
rationale = "r"

[contract.B.scope]
backend = "hip"
architectures = ["gfx1100"]

[contract.B.positive]
models = ["m"]
workloads = ["decode"]

[contract.B.controls]
models = ["m"]
workloads = ["decode"]

[contract.B.acceptance]
max_control_regression_pct = 1
"""
        path = _write(toml)
        with self.assertRaisesRegex(ec.ExperimentContractError, "cycle"):
            ec.load_contracts(path)

    def test_shipped_registry_loads_cleanly(self):
        # The real repo-root experiment-contracts.toml parses and its own
        # source_ids all cross-check clean against the real external-sources
        # registry (EC02's backfill populates it with real [contract.*]
        # entries; this stays true before and after that population).
        from bigcherry import paths
        registry = ec.load_contracts(
            paths.EXPERIMENT_CONTRACTS,
            known_source_ids=ec.known_source_ids_from_external_sources(),
        )
        self.assertGreaterEqual(len(registry), 0)


class KnownSourceIdsFromExternalSourcesTests(unittest.TestCase):
    def test_real_registry_yields_known_ids(self):
        ids = ec.known_source_ids_from_external_sources()
        self.assertIn("stew675-rdna-boosts", ids)
        self.assertIn("amd-ecosystem-llama-cpp", ids)

    def test_explicit_path_used_over_default(self):
        path = _write("""
[[sources]]
id = "only-one"
repo = "https://example.invalid/x"
locator = "l"

[[sources.snapshots]]
label = "v1"
head = "0000000000000000000000000000000000000000"
base = "0000000000000000000000000000000000000000"
active = true
""")
        ids = ec.known_source_ids_from_external_sources(path)
        self.assertEqual(ids, frozenset({"only-one"}))

    def test_missing_file_rejected(self):
        with self.assertRaises(ec.ExperimentContractError):
            ec.known_source_ids_from_external_sources("/no/such/external-sources.toml")

    def test_load_contracts_rejects_shipped_registry_against_real_sources_if_broken(self):
        # Cross-checking the shipped registry (used above) proves the two
        # files stay mutually consistent -- this test documents *why*:
        # a contract citing a source_id that isn't registered would be
        # caught here, not silently accepted.
        from bigcherry import paths
        registry = ec.load_contracts(
            paths.EXPERIMENT_CONTRACTS,
            known_source_ids=ec.known_source_ids_from_external_sources(),
        )
        for contract in registry:
            self.assertIn(
                contract.source.source_id,
                ec.known_source_ids_from_external_sources(),
            )


if __name__ == "__main__":
    unittest.main()


def _minimal_contract(**overrides) -> ec.ExperimentContract:
    doc = {
        "title": "t",
        "source": {"source_id": "s", "commits": ["c"], "atomic_part": "p"},
        "hypothesis": {"family": "mmq", "expected_effect": "performance", "rationale": "r"},
        "scope": {"backend": "hip", "architectures": ["gfx1100"]},
        "positive": {"models": ["m"], "workloads": ["decode"]},
        "controls": {"models": ["m"], "workloads": ["decode"]},
        "acceptance": {"max_control_regression_pct": 1},
    }
    doc.update(overrides)
    return ec.parse_contract(doc, contract_id="X")


class EvidenceBindingTests(unittest.TestCase):
    FAKE_PROVENANCE = {
        "schema_version": 2, "project": {}, "source": {}, "build": {},
        "workload": {}, "campaign": {},
    }

    def test_evidence_ref_carries_contract_identity_not_runtime_identity(self):
        contract = _minimal_contract()
        evidence = ec.evidence_ref_for_lane(
            contract, role="positive", workload_tag="decode", model_ref="m")
        self.assertEqual(evidence.contract_id, "X")
        self.assertEqual(evidence.contract_hash, contract.contract_hash)
        self.assertEqual(evidence.optimization_id, "p")

    def test_invalid_role_rejected(self):
        contract = _minimal_contract()
        with self.assertRaises(ec.ExperimentContractError):
            ec.evidence_ref_for_lane(contract, role="not-a-role")

    def test_attach_does_not_mutate_or_overwrite_caller_document(self):
        contract = _minimal_contract()
        evidence = ec.evidence_ref_for_lane(contract, role="control", workload_tag="decode")
        original = dict(self.FAKE_PROVENANCE)
        attached = ec.attach_to_document(original, evidence)
        self.assertEqual(original, self.FAKE_PROVENANCE)  # untouched
        self.assertNotIn("contract_evidence", original)
        self.assertIn("contract_evidence", attached)
        # every original key survives unchanged in the new document
        for key, value in self.FAKE_PROVENANCE.items():
            self.assertEqual(attached[key], value)

    def test_double_attach_rejected(self):
        contract = _minimal_contract()
        evidence = ec.evidence_ref_for_lane(contract, role="boundary", boundary_dimension="physical_m",
                                            boundary_value="4")
        attached = ec.attach_to_document(dict(self.FAKE_PROVENANCE), evidence)
        with self.assertRaisesRegex(ec.ExperimentContractError, "contract_evidence"):
            ec.attach_to_document(attached, evidence)

    def test_read_from_document_round_trips(self):
        contract = _minimal_contract()
        evidence = ec.evidence_ref_for_lane(
            contract, role="boundary", boundary_dimension="physical_m", boundary_value="8")
        attached = ec.attach_to_document(dict(self.FAKE_PROVENANCE), evidence)
        readback = ec.read_from_document(attached)
        self.assertEqual(readback, evidence)

    def test_read_from_document_returns_none_when_unbound(self):
        self.assertIsNone(ec.read_from_document(dict(self.FAKE_PROVENANCE)))

    def test_read_from_document_rejects_malformed_sidecar(self):
        doc = dict(self.FAKE_PROVENANCE)
        doc["contract_evidence"] = {"role": "not-a-role"}
        with self.assertRaises(ec.ExperimentContractError):
            ec.read_from_document(doc)


class AggregateContractEffectsTests(unittest.TestCase):
    def test_target_gain_is_mean_of_positive_role_only(self):
        contract = _minimal_contract()
        effects = [
            ec.LaneEffect(role="positive", metric="tg", geometric_effect_pct=6.0),
            ec.LaneEffect(role="positive", metric="tg", geometric_effect_pct=4.0),
            ec.LaneEffect(role="control", metric="tg", geometric_effect_pct=100.0),
            ec.LaneEffect(role="boundary", metric="tg", geometric_effect_pct=-100.0),
        ]
        result = ec.aggregate_contract_effects(contract, effects, target_metric="tg")
        self.assertEqual(result["target_kernel_gain_pct"], 5.0)

    def test_max_control_regression_is_worst_not_average(self):
        contract = _minimal_contract()
        effects = [
            ec.LaneEffect(role="positive", metric="tg", geometric_effect_pct=5.0),
            ec.LaneEffect(role="control", metric="tg", geometric_effect_pct=-3.0),
            ec.LaneEffect(role="control", metric="tg", geometric_effect_pct=0.5),
            ec.LaneEffect(role="control", metric="tg", geometric_effect_pct=1.0),
        ]
        result = ec.aggregate_contract_effects(contract, effects, target_metric="tg")
        self.assertEqual(result["max_control_regression_pct"], 3.0)

    def test_no_regression_when_all_controls_improve(self):
        contract = _minimal_contract()
        effects = [
            ec.LaneEffect(role="positive", metric="tg", geometric_effect_pct=5.0),
            ec.LaneEffect(role="control", metric="tg", geometric_effect_pct=1.0),
            ec.LaneEffect(role="control", metric="tg", geometric_effect_pct=2.0),
        ]
        result = ec.aggregate_contract_effects(contract, effects, target_metric="tg")
        self.assertEqual(result["max_control_regression_pct"], 0.0)

    def test_missing_positive_effects_rejected(self):
        contract = _minimal_contract()
        effects = [ec.LaneEffect(role="control", metric="tg", geometric_effect_pct=1.0)]
        with self.assertRaisesRegex(ec.ExperimentContractError, "positive"):
            ec.aggregate_contract_effects(contract, effects, target_metric="tg")

    def test_missing_control_effects_rejected_not_reported_as_zero(self):
        # The whole point: an empty control set must never silently read
        # as "no regression" (0.0) -- it must fail loudly instead.
        contract = _minimal_contract()
        effects = [ec.LaneEffect(role="positive", metric="tg", geometric_effect_pct=5.0)]
        with self.assertRaisesRegex(ec.ExperimentContractError, "control"):
            ec.aggregate_contract_effects(contract, effects, target_metric="tg")

    def test_end_to_end_metric_falls_back_to_target_metric(self):
        contract = _minimal_contract()
        effects = [
            ec.LaneEffect(role="positive", metric="tg", geometric_effect_pct=5.0),
            ec.LaneEffect(role="control", metric="tg", geometric_effect_pct=0.0),
        ]
        result = ec.aggregate_contract_effects(contract, effects, target_metric="tg")
        self.assertEqual(result["end_to_end_gain_pct"], 5.0)

    def test_end_to_end_metric_when_named_separately(self):
        contract = _minimal_contract()
        effects = [
            ec.LaneEffect(role="positive", metric="kernel_us", geometric_effect_pct=20.0),
            ec.LaneEffect(role="positive", metric="pp512", geometric_effect_pct=1.5),
            ec.LaneEffect(role="control", metric="kernel_us", geometric_effect_pct=0.0),
        ]
        result = ec.aggregate_contract_effects(
            contract, effects, target_metric="kernel_us", end_to_end_metric="pp512")
        self.assertEqual(result["target_kernel_gain_pct"], 20.0)
        self.assertEqual(result["end_to_end_gain_pct"], 1.5)

    def test_end_to_end_gain_is_none_when_metric_never_measured(self):
        contract = _minimal_contract()
        effects = [
            ec.LaneEffect(role="positive", metric="kernel_us", geometric_effect_pct=20.0),
            ec.LaneEffect(role="control", metric="kernel_us", geometric_effect_pct=0.0),
        ]
        result = ec.aggregate_contract_effects(
            contract, effects, target_metric="kernel_us", end_to_end_metric="pp512")
        self.assertIsNone(result["end_to_end_gain_pct"])


class CorrectnessGateTests(unittest.TestCase):
    def test_passes_when_every_required_check_passes(self):
        contract = _minimal_contract(correctness={"greedy_parity": "required"})
        results = {"greedy_parity": ec.CorrectnessResult(check="greedy_parity", passed=True)}
        gate = ec.evaluate_correctness_gate(contract, results)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["missing_checks"], [])
        self.assertEqual(gate["failed_checks"], [])

    def test_fails_when_a_required_check_is_missing(self):
        contract = _minimal_contract(
            correctness={"greedy_parity": "required", "bit_identical": "required"})
        results = {"greedy_parity": ec.CorrectnessResult(check="greedy_parity", passed=True)}
        gate = ec.evaluate_correctness_gate(contract, results)
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["missing_checks"], ["bit_identical"])

    def test_fails_when_a_required_check_failed(self):
        contract = _minimal_contract(correctness={"greedy_parity": "required"})
        results = {"greedy_parity": ec.CorrectnessResult(check="greedy_parity", passed=False, detail="diverged")}
        gate = ec.evaluate_correctness_gate(contract, results)
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["failed_checks"], ["greedy_parity"])

    def test_correctness_only_contract_with_no_requirements_passes_trivially(self):
        # No correctness section at all -- a pure-performance contract.
        contract = _minimal_contract()
        gate = ec.evaluate_correctness_gate(contract, {})
        self.assertTrue(gate["passed"])

    def test_extra_unrequested_results_do_not_affect_pass_fail(self):
        contract = _minimal_contract(correctness={"greedy_parity": "required"})
        results = {
            "greedy_parity": ec.CorrectnessResult(check="greedy_parity", passed=True),
            "bit_identical": ec.CorrectnessResult(check="bit_identical", passed=False),
        }
        gate = ec.evaluate_correctness_gate(contract, results)
        self.assertTrue(gate["passed"])


class GeneralisationHandoffTests(unittest.TestCase):
    def test_floor_matches_generalise_py_required_thresholds(self):
        from bigcherry import generalise
        self.assertEqual(ec.generalisation_floor(), dict(generalise.REQUIRED_THRESHOLDS))

    def test_none_returns_the_floor_unchanged(self):
        self.assertEqual(ec.require_generalisation_policy(None), ec.generalisation_floor())

    def test_stricter_min_threshold_accepted(self):
        floor = ec.generalisation_floor()
        stricter = {"min_holdout_calls": floor["min_holdout_calls"] + 50}
        result = ec.require_generalisation_policy(stricter)
        self.assertEqual(result["min_holdout_calls"], floor["min_holdout_calls"] + 50)

    def test_stricter_max_threshold_accepted(self):
        floor = ec.generalisation_floor()
        stricter = {"max_median_regret_pct": floor["max_median_regret_pct"] / 2}
        result = ec.require_generalisation_policy(stricter)
        self.assertEqual(result["max_median_regret_pct"], floor["max_median_regret_pct"] / 2)

    def test_looser_min_threshold_rejected(self):
        floor = ec.generalisation_floor()
        looser = {"min_holdout_calls": floor["min_holdout_calls"] - 1}
        with self.assertRaisesRegex(ec.ExperimentContractError, "min_holdout_calls"):
            ec.require_generalisation_policy(looser)

    def test_looser_max_threshold_rejected(self):
        floor = ec.generalisation_floor()
        looser = {"max_median_regret_pct": floor["max_median_regret_pct"] + 1}
        with self.assertRaisesRegex(ec.ExperimentContractError, "max_median_regret_pct"):
            ec.require_generalisation_policy(looser)

    def test_partial_override_keeps_other_thresholds_at_floor(self):
        floor = ec.generalisation_floor()
        result = ec.require_generalisation_policy({"min_holdout_calls": floor["min_holdout_calls"] + 1})
        for name, value in floor.items():
            if name == "min_holdout_calls":
                continue
            self.assertEqual(result[name], value)


class PromotionGateTests(unittest.TestCase):
    PASSING_CORRECTNESS = {"passed": True, "missing_checks": [], "failed_checks": []}
    FAILING_CORRECTNESS = {"passed": False, "missing_checks": ["greedy_parity"], "failed_checks": []}

    def test_performance_contract_promotes_when_all_thresholds_met(self):
        contract = _minimal_contract(acceptance={
            "target_kernel_gain_pct": 5, "end_to_end_gain_pct": 1, "max_control_regression_pct": 1,
        })
        effects = {"target_kernel_gain_pct": 6.0, "end_to_end_gain_pct": 1.5,
                   "max_control_regression_pct": 0.5}
        gate = ec.evaluate_promotion_gate(
            contract, correctness_gate=self.PASSING_CORRECTNESS, aggregated_effects=effects)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["reasons"], [])

    def test_fails_when_target_gain_below_threshold(self):
        contract = _minimal_contract(acceptance={"target_kernel_gain_pct": 5, "max_control_regression_pct": 1})
        effects = {"target_kernel_gain_pct": 3.0, "max_control_regression_pct": 0.5}
        gate = ec.evaluate_promotion_gate(
            contract, correctness_gate=self.PASSING_CORRECTNESS, aggregated_effects=effects)
        self.assertFalse(gate["passed"])
        self.assertTrue(any("target_kernel_gain_pct" in r for r in gate["reasons"]))

    def test_fails_when_regression_budget_exceeded(self):
        contract = _minimal_contract(acceptance={"max_control_regression_pct": 1})
        effects = {"max_control_regression_pct": 2.5}
        gate = ec.evaluate_promotion_gate(
            contract, correctness_gate=self.PASSING_CORRECTNESS, aggregated_effects=effects)
        self.assertFalse(gate["passed"])
        self.assertTrue(any("max_control_regression_pct" in r for r in gate["reasons"]))

    def test_fails_when_correctness_gate_failed_even_with_great_performance(self):
        contract = _minimal_contract(acceptance={"target_kernel_gain_pct": 5, "max_control_regression_pct": 1})
        effects = {"target_kernel_gain_pct": 50.0, "max_control_regression_pct": 0.0}
        gate = ec.evaluate_promotion_gate(
            contract, correctness_gate=self.FAILING_CORRECTNESS, aggregated_effects=effects)
        self.assertFalse(gate["passed"])
        self.assertTrue(any("correctness gate failed" in r for r in gate["reasons"]))

    def test_correctness_only_contract_promotes_on_correctness_alone(self):
        # Guide Appendix A: RD20/RD22/RD26-class contracts -- no performance
        # claim, correctness + regression budget only.
        contract = _minimal_contract(acceptance={"max_control_regression_pct": 1})
        effects = {"max_control_regression_pct": 0.2}
        gate = ec.evaluate_promotion_gate(
            contract, correctness_gate=self.PASSING_CORRECTNESS, aggregated_effects=effects)
        self.assertTrue(gate["passed"])

    def test_missing_generalisation_proof_blocks_when_supplied_and_failed(self):
        contract = _minimal_contract(acceptance={"max_control_regression_pct": 1})
        effects = {"max_control_regression_pct": 0.0}
        gate = ec.evaluate_promotion_gate(
            contract, correctness_gate=self.PASSING_CORRECTNESS, aggregated_effects=effects,
            generalisation_result={"passed": False})
        self.assertFalse(gate["passed"])

    def test_generalisation_proof_absent_does_not_block(self):
        contract = _minimal_contract(acceptance={"max_control_regression_pct": 1})
        effects = {"max_control_regression_pct": 0.0}
        gate = ec.evaluate_promotion_gate(
            contract, correctness_gate=self.PASSING_CORRECTNESS, aggregated_effects=effects,
            generalisation_result=None)
        self.assertTrue(gate["passed"])


class RenderReportTests(unittest.TestCase):
    PASSING_CORRECTNESS = {"passed": True, "missing_checks": [], "failed_checks": []}

    def test_report_has_all_nine_sections_for_a_promoted_contract(self):
        contract = _minimal_contract(acceptance={
            "target_kernel_gain_pct": 5, "end_to_end_gain_pct": 1, "max_control_regression_pct": 1,
        })
        effects = {"target_kernel_gain_pct": 6.0, "end_to_end_gain_pct": 1.5,
                   "max_control_regression_pct": 0.2}
        correctness = self.PASSING_CORRECTNESS
        promotion = ec.evaluate_promotion_gate(
            contract, correctness_gate=correctness, aggregated_effects=effects)
        report = ec.render_report(
            contract, correctness_gate=correctness, aggregated_effects=effects,
            promotion_gate=promotion)
        for heading in ("Hypothesis", "Source", "Scope", "Winners", "Non-trigger",
                        "Controls", "Correctness", "Generalised rule", "Promotion decision"):
            self.assertIn(heading, report)
        self.assertIn("passed: True", report)

    def test_report_renders_fully_even_for_a_rejected_contract(self):
        # guide section 12 step 12: rejected optimizations are useful
        # evidence -- the report must not short-circuit.
        contract = _minimal_contract(acceptance={"target_kernel_gain_pct": 5, "max_control_regression_pct": 1})
        effects = {"target_kernel_gain_pct": 1.0, "max_control_regression_pct": 3.0}
        correctness = self.PASSING_CORRECTNESS
        promotion = ec.evaluate_promotion_gate(
            contract, correctness_gate=correctness, aggregated_effects=effects)
        self.assertFalse(promotion["passed"])
        report = ec.render_report(
            contract, correctness_gate=correctness, aggregated_effects=effects,
            promotion_gate=promotion)
        for heading in ("Hypothesis", "Source", "Scope", "Winners", "Non-trigger",
                        "Controls", "Correctness", "Generalised rule", "Promotion decision"):
            self.assertIn(heading, report)
        self.assertIn("blocked by:", report)
        self.assertIn(contract.id, report)

    def test_report_shows_boundary_dimensions_when_declared(self):
        contract = _minimal_contract(
            acceptance={"max_control_regression_pct": 1},
            boundary={"dimensions": {"physical_m": [1, 2, 4]}},
        )
        effects = {"max_control_regression_pct": 0.0}
        promotion = ec.evaluate_promotion_gate(
            contract, correctness_gate=self.PASSING_CORRECTNESS, aggregated_effects=effects)
        report = ec.render_report(
            contract, correctness_gate=self.PASSING_CORRECTNESS, aggregated_effects=effects,
            promotion_gate=promotion)
        self.assertIn("physical_m", report)
        self.assertIn("1, 2, 4", report)

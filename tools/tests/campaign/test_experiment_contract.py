"""Experiment Contract schema, validator and registry tests (EC01/EC02/EC12)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.experiment import contract as ec # noqa: E402


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

    def test_lane_in_both_positive_and_controls_rejected(self):
        """A lane is (model, workload). Naming the same lane in both roles asks
        one measurement to satisfy two contradictory requirements, and makes the
        regression budget self-referential: the 'control' IS the treatment, so
        it can never detect the collateral damage a control exists to catch."""
        doc = _base_doc()
        doc["positive"] = {"models": ["shared-model"], "workloads": ["decode"]}
        doc["controls"] = {"models": ["shared-model"], "workloads": ["decode"]}
        with self.assertRaisesRegex(ec.ExperimentContractError,
                                    r"shared-model/decode.*BOTH positive and controls"):
            ec.parse_contract(doc, contract_id="X")

    def test_partial_lane_overlap_rejected_naming_only_the_shared_lanes(self):
        """Overlap is per-LANE, not per-set: sharing a model is fine so long as
        no (model, workload) pair is claimed by both roles. Here decode overlaps
        and prefill does not, so only decode may be reported."""
        doc = _base_doc()
        doc["positive"] = {"models": ["m"], "workloads": ["decode", "small_m"]}
        doc["controls"] = {"models": ["m"], "workloads": ["decode", "prefill"]}
        with self.assertRaises(ec.ExperimentContractError) as caught:
            ec.parse_contract(doc, contract_id="X")
        message = str(caught.exception)
        self.assertIn("m/decode", message)
        self.assertNotIn("m/prefill", message)
        self.assertNotIn("m/small_m", message)

    def test_same_workload_on_a_different_model_is_a_valid_control(self):
        """The rule must not over-reach into the common, correct pattern of
        controlling a workload on a model the hypothesis does not claim."""
        doc = _base_doc()
        doc["positive"] = {"models": ["claimed"], "workloads": ["decode"]}
        doc["controls"] = {"models": ["unclaimed"], "workloads": ["decode"]}
        contract = ec.parse_contract(doc, contract_id="X")
        self.assertEqual(contract.controls.models, ("unclaimed",))

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
        from bigcherry.core import paths as _paths
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

    def test_all_nineteen_contracts_in_the_real_registry_parse(self):
        # EC17 regression proof: adding [source-evidence] as an optional
        # section must not break any of the 5 original (EC02) or 12
        # EC16-backfilled contracts already committed to
        # config/experiment-contracts.toml, plus VA05's
        # RD58-PIN-STATE-BUFFER-MULTIGPU-RESTORE (18th) and VA06's
        # RD73-STABLE-GRAPH-CACHE-KEY (19th). GPT review
        # (req_3616cc1d90dc4512): keep this an exact count, not a lower
        # bound -- a >= assertion silently stops catching a contract that
        # fails to load/register at all, which is exactly the regression
        # this test exists to guard against.
        from bigcherry.core import paths as _paths
        registry = ec.load_contracts(_paths.EXPERIMENT_CONTRACTS)
        self.assertEqual(len(registry.contracts), 19)


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
        from bigcherry.core import paths as _paths
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
workloads = ["prefill"]

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
workloads = ["prefill"]

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
        from bigcherry.core import paths
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
        from bigcherry.core import paths
        registry = ec.load_contracts(
            paths.EXPERIMENT_CONTRACTS,
            known_source_ids=ec.known_source_ids_from_external_sources(),
        )
        for contract in registry:
            self.assertIn(
                contract.source.source_id,
                ec.known_source_ids_from_external_sources(),
            )


_MODEL_CHECK_TOML = """
[contract.A]
title = "t"

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
models = ["{positive}"]
workloads = ["decode"]

[contract.A.controls]
models = ["{controls}"]
workloads = ["prefill"]

[contract.A.acceptance]
max_control_regression_pct = 1
"""


class KnownModelIdsFromModelsRegistryTests(unittest.TestCase):
    """Model refs in an evaluation set were unvalidated free text: _evaluation_set()
    constrained `workloads` to WORKLOAD_TAGS and `models` to nothing. A typo or a
    model that never existed validated clean and failed only at hardware time --
    or worse, did not fail at all, since the ref is a LABEL and the real gguf
    arrives separately via --model, so evidence could claim a lane it never
    measured."""

    def test_real_registry_yields_known_ids(self):
        ids = ec.known_model_ids_from_models_registry()
        self.assertIn("tierA-qwen4b-q6k", ids)
        self.assertIn("tierL-qwen27b-q8", ids)
        self.assertIn("tierM-gptoss20b-q6k", ids)

    def test_explicit_path_used_over_default(self):
        path = _write("""
version = 1

[[models]]
id = "only-one"
family = "f"
path = "f/x.gguf"
quantisation = "Q8_0"
parameters = "1B"
size-bytes = 1
mtp = false
""")
        self.assertEqual(
            ec.known_model_ids_from_models_registry(path), frozenset({"only-one"})
        )

    def test_missing_file_rejected(self):
        with self.assertRaises(ec.ExperimentContractError):
            ec.known_model_ids_from_models_registry("/no/such/models.toml")

    def test_unregistered_positive_model_rejected(self):
        path = _write(_MODEL_CHECK_TOML.format(
            positive="no-such-model", controls="tierA-qwen4b-q6k"))
        with self.assertRaises(ec.ExperimentContractError) as caught:
            ec.load_contracts(path, known_model_ids=frozenset({"tierA-qwen4b-q6k"}))
        self.assertIn("no-such-model", str(caught.exception))
        self.assertIn("positive", str(caught.exception))

    def test_unregistered_control_model_rejected(self):
        # Controls matter as much as positives: an unresolvable control lane
        # means the regression budget is measured against nothing.
        path = _write(_MODEL_CHECK_TOML.format(
            positive="tierA-qwen4b-q6k", controls="no-such-model"))
        with self.assertRaises(ec.ExperimentContractError) as caught:
            ec.load_contracts(path, known_model_ids=frozenset({"tierA-qwen4b-q6k"}))
        self.assertIn("controls", str(caught.exception))

    def test_check_is_opt_in_and_skipped_when_not_requested(self):
        # Callers building a contract in isolation must not be forced to
        # maintain a models.toml fixture -- same contract as the source-id check.
        path = _write(_MODEL_CHECK_TOML.format(
            positive="anything-at-all", controls="tierA-qwen4b-q6k"))
        self.assertEqual(len(ec.load_contracts(path)), 1)

    def test_shipped_registry_cross_checks_clean(self):
        from bigcherry.core import paths
        known = ec.known_model_ids_from_models_registry()
        registry = ec.load_contracts(paths.EXPERIMENT_CONTRACTS, known_model_ids=known)
        for contract in registry:
            for model in (*contract.positive.models, *contract.controls.models):
                self.assertIn(model, known)


def _ratios(*percents: float) -> tuple[float, ...]:
    return tuple(1.0 + pct / 100.0 for pct in percents)


class BootstrapSessionEffectTests(unittest.TestCase):
    """The estimator exists because a within-run interval structurally cannot
    see drift BETWEEN runs. RD73 measured one build three times -- +1.855%,
    +1.717%, +1.249% -- and the third run's point estimate fell below the
    second run's ci95_low, so its single-run interval overstated precision."""

    # Two sessions that disagree, in the way RD73's runs 2 and 3 disagreed.
    DRIFTED = [
        _ratios(2.55, 1.87, 1.95, 2.37, 1.33, 0.73, 1.14, 1.53, 1.79, 1.93),
        _ratios(3.22, -0.20, 1.63, 0.37, 1.72, 0.17, 1.69, 1.30, 2.62, 0.04),
    ]

    def test_too_few_sessions_returns_none_rather_than_a_narrow_guess(self):
        for count in range(1, ec.MIN_BOOTSTRAP_SESSIONS):
            with self.subTest(sessions=count):
                self.assertIsNone(
                    ec.bootstrap_session_effect([self.DRIFTED[0]] * count)
                )

    def test_at_the_threshold_it_estimates(self):
        result = ec.bootstrap_session_effect(
            [self.DRIFTED[0]] * ec.MIN_BOOTSTRAP_SESSIONS
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["sessions"], ec.MIN_BOOTSTRAP_SESSIONS)

    def test_empty_or_nonfinite_session_returns_none(self):
        base = [self.DRIFTED[0]] * ec.MIN_BOOTSTRAP_SESSIONS
        self.assertIsNone(ec.bootstrap_session_effect(base[:-1] + [()]))
        self.assertIsNone(ec.bootstrap_session_effect(base[:-1] + [(1.0, -0.5)]))
        self.assertIsNone(
            ec.bootstrap_session_effect(base[:-1] + [(1.0, float("nan"))])
        )

    def test_drift_widens_the_interval_beyond_a_single_run(self):
        # THE reason this estimator exists. Sessions that disagree must yield
        # a wider interval than the same number of identical sessions, whose
        # only variation is within-run.
        drifted = ec.bootstrap_session_effect(self.DRIFTED * 2)
        steady = ec.bootstrap_session_effect([self.DRIFTED[0]] * 4)
        drifted_width = drifted["ci95_high_pct"] - drifted["ci95_low_pct"]
        steady_width = steady["ci95_high_pct"] - steady["ci95_low_pct"]
        self.assertGreater(drifted_width, steady_width)
        self.assertGreater(drifted["between_session_sd_pct"], 0.0)
        self.assertAlmostEqual(steady["between_session_sd_pct"], 0.0, places=9)

    def test_sessions_are_weighted_equally_not_by_pair_count(self):
        # A session that happened to collect more pairs must not speak louder
        # about where the true effect lies -- the session is the unit.
        few = _ratios(4.0, 4.0)
        many = _ratios(*([0.0] * 40))
        result = ec.bootstrap_session_effect([few, many, few, many])
        # Equal weighting puts the point estimate midway between the two
        # session effects (~4% and ~0%); pair weighting would drag it to ~0.4%.
        self.assertAlmostEqual(result["geometric_effect_pct"], 2.0, delta=0.05)

    def test_point_estimate_is_the_mean_of_per_session_effects(self):
        result = ec.bootstrap_session_effect(self.DRIFTED * 2)
        per_session = result["per_session_effect_pct"]
        self.assertEqual(len(per_session), 4)
        self.assertAlmostEqual(
            result["geometric_effect_pct"], sum(per_session) / 4, places=9
        )

    def test_is_deterministic_for_a_given_seed(self):
        first = ec.bootstrap_session_effect(self.DRIFTED * 2, seed=7)
        second = ec.bootstrap_session_effect(self.DRIFTED * 2, seed=7)
        self.assertEqual(first, second)
        other = ec.bootstrap_session_effect(self.DRIFTED * 2, seed=8)
        self.assertNotEqual(first["ci95_low_pct"], other["ci95_low_pct"])

    def test_interval_brackets_the_point_estimate(self):
        result = ec.bootstrap_session_effect(self.DRIFTED * 2)
        self.assertLessEqual(result["ci95_low_pct"], result["geometric_effect_pct"])
        self.assertLessEqual(result["geometric_effect_pct"], result["ci95_high_pct"])

    def test_reports_total_pairs_across_sessions(self):
        result = ec.bootstrap_session_effect(self.DRIFTED * 2)
        self.assertEqual(result["paired_rounds_total"], 40)


def _session_contract(**acceptance_overrides):
    acceptance = {
        "end_to_end_gain_pct": 1.0,
        "max_control_regression_pct": 1.0,
        "effect_evidence_policy": "session_ci95_threshold_bound_v1",
        "min_paired_rounds": 10,
        "min_sessions": 4,
        "max_sessions": 8,
        "max_ci95_width_pct": 1.0,
    }
    acceptance.update(acceptance_overrides)
    return _minimal_contract(acceptance=acceptance)


class SessionEvidencePolicyParsingTests(unittest.TestCase):
    def test_policy_requires_the_whole_stopping_rule(self):
        for missing in ("min_sessions", "max_sessions", "max_ci95_width_pct"):
            with self.subTest(missing=missing):
                acceptance = {
                    "end_to_end_gain_pct": 1.0, "max_control_regression_pct": 1.0,
                    "effect_evidence_policy": "session_ci95_threshold_bound_v1",
                    "min_paired_rounds": 10, "min_sessions": 4, "max_sessions": 8,
                    "max_ci95_width_pct": 1.0,
                }
                del acceptance[missing]
                with self.assertRaises(ec.ExperimentContractError) as caught:
                    _minimal_contract(acceptance=acceptance)
                self.assertIn(missing, str(caught.exception))

    def test_min_sessions_below_the_bootstrap_floor_rejected(self):
        with self.assertRaises(ec.ExperimentContractError) as caught:
            _session_contract(min_sessions=ec.MIN_BOOTSTRAP_SESSIONS - 1)
        self.assertIn("min_sessions", str(caught.exception))

    def test_max_sessions_below_min_rejected(self):
        with self.assertRaises(ec.ExperimentContractError):
            _session_contract(min_sessions=6, max_sessions=5)

    def test_stopping_rule_without_the_policy_rejected(self):
        # A declared-but-unconsulted stopping rule is worse than none: it
        # reads as though the run was governed when nothing enforced it.
        with self.assertRaises(ec.ExperimentContractError) as caught:
            _minimal_contract(acceptance={
                "end_to_end_gain_pct": 1.0, "max_control_regression_pct": 1.0,
                "min_sessions": 4,
            })
        self.assertIn("min_sessions", str(caught.exception))

    def test_valid_session_contract_parses(self):
        contract = _session_contract()
        self.assertEqual(contract.acceptance.min_sessions, 4)
        self.assertEqual(contract.acceptance.max_sessions, 8)
        self.assertEqual(contract.acceptance.max_ci95_width_pct, 1.0)


class SessionStoppingRuleGateTests(unittest.TestCase):
    """The stopping rule must consult session count and interval WIDTH only.
    A rule that can see where ci95_low sits relative to the threshold is a
    rule that stops when it likes the answer."""

    CORRECTNESS = {"passed": True, "missing_checks": [], "failed_checks": []}

    def _gate(self, *, sessions, low, high, contract=None):
        return ec.evaluate_promotion_gate(
            contract or _session_contract(),
            correctness_gate=self.CORRECTNESS,
            aggregated_effects={
                "end_to_end_gain_pct": (low + high) / 2,
                "end_to_end_gain_pct_ci95_low": low,
                "end_to_end_gain_pct_ci95_high": high,
                "end_to_end_gain_pct_sessions": sessions,
                "end_to_end_gain_pct_paired_rounds": 10 * sessions,
                "max_control_regression_pct": 0.0,
                "max_control_regression_pct_ci95_high": 0.0,
                "max_control_regression_pct_paired_rounds": 10 * sessions,
            },
        )

    def test_too_few_sessions_is_inconclusive_not_a_fail(self):
        result = self._gate(sessions=3, low=1.5, high=1.9)
        self.assertEqual(result["status"], "invalid")

    def test_wide_interval_below_max_sessions_is_inconclusive(self):
        # Width 1.4 > target 1.0, and 5 < max 8 -> collect another session.
        result = self._gate(sessions=5, low=1.1, high=2.5)
        self.assertEqual(result["status"], "invalid")
        self.assertTrue(
            any("INCONCLUSIVE" in str(r) for r in result.get("reasons", []))
        )

    def test_wide_interval_at_max_sessions_is_decided_not_deferred(self):
        # At max_sessions the rule must stop deferring and decide on what it
        # has, otherwise a noisy patch defers for ever.
        result = self._gate(sessions=8, low=1.1, high=2.5)
        self.assertEqual(result["status"], "pass")

    def test_precise_enough_and_above_threshold_passes(self):
        self.assertEqual(self._gate(sessions=5, low=1.2, high=1.9)["status"], "pass")

    def test_precise_enough_and_below_threshold_is_a_real_fail(self):
        # Precision satisfied, bound not met -> an ordinary FAIL, never
        # "collect more until it passes".
        result = self._gate(sessions=5, low=0.4, high=1.1)
        self.assertEqual(result["status"], "fail")

    def test_stopping_decision_ignores_which_side_of_the_bar_it_lands(self):
        # THE direction-blindness property. Two runs with the SAME session
        # count and the SAME interval width, one comfortably above the bar and
        # one below it, must reach the same STOPPING decision -- both decided,
        # differing only in pass/fail.
        width = 0.8
        above = self._gate(sessions=5, low=1.6, high=1.6 + width)
        below = self._gate(sessions=5, low=0.2, high=0.2 + width)
        self.assertEqual(above["status"], "pass")
        self.assertEqual(below["status"], "fail")
        for result in (above, below):
            self.assertNotIn(
                "INCONCLUSIVE", " ".join(str(r) for r in result.get("reasons", []))
            )

    def test_missing_session_count_is_invalid(self):
        result = ec.evaluate_promotion_gate(
            _session_contract(), correctness_gate=self.CORRECTNESS,
            aggregated_effects={
                "end_to_end_gain_pct": 1.5,
                "end_to_end_gain_pct_ci95_low": 1.2,
                "end_to_end_gain_pct_ci95_high": 1.8,
                "end_to_end_gain_pct_paired_rounds": 40,
                "max_control_regression_pct": 0.0,
                "max_control_regression_pct_ci95_high": 0.0,
                "max_control_regression_pct_paired_rounds": 40,
            },
        )
        self.assertEqual(result["status"], "invalid")


if __name__ == "__main__":
    unittest.main()


def _minimal_contract(**overrides) -> ec.ExperimentContract:
    doc = {
        "title": "t",
        "source": {"source_id": "s", "commits": ["c"], "atomic_part": "p"},
        "hypothesis": {"family": "mmq", "expected_effect": "performance", "rationale": "r"},
        "scope": {"backend": "hip", "architectures": ["gfx1100"]},
        "positive": {"models": ["m"], "workloads": ["decode"]},
        # Deliberately a different workload from the positive lane: a lane
        # cannot be both the thing that must improve and the thing that must
        # hold constant (parse_contract rejects the overlap).
        "controls": {"models": ["m"], "workloads": ["prefill"]},
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

    def test_control_regression_computed_across_a_different_metric_than_target(self):
        """GPT round 2 (req_240634997c1a4ee9): a control lane very often
        measures a structurally different workload than the positive lane
        (e.g. RD08: positive=decode/tg128, control=prefill/pp512) and
        reports its own natural metric -- max_control_regression_pct must
        be computed from ALL control-role effects regardless of metric,
        never filtered to target_metric."""
        contract = _minimal_contract()
        effects = [
            ec.LaneEffect(role="positive", metric="tg128", geometric_effect_pct=5.0),
            ec.LaneEffect(role="control", metric="pp512", geometric_effect_pct=-2.0),
        ]
        result = ec.aggregate_contract_effects(contract, effects, target_metric="tg128")
        self.assertEqual(result["target_kernel_gain_pct"], 5.0)
        self.assertEqual(result["max_control_regression_pct"], 2.0)

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


class TriggerEvidenceTests(unittest.TestCase):
    def test_requires_at_least_one_count_field(self):
        with self.assertRaisesRegex(ec.ExperimentContractError, "candidate_launches.*expected_route_selected"):
            ec.TriggerEvidence(role="positive", lane_id="lane-a")

    def test_accepts_launches_only(self):
        evidence = ec.TriggerEvidence(role="positive", lane_id="lane-a", candidate_launches=5)
        self.assertEqual(evidence.candidate_launches, 5)
        self.assertIsNone(evidence.expected_route_selected)

    def test_accepts_route_selected_only(self):
        evidence = ec.TriggerEvidence(role="positive", lane_id="lane-a", expected_route_selected=3)
        self.assertEqual(evidence.expected_route_selected, 3)

    def test_rejects_negative_launches(self):
        with self.assertRaisesRegex(ec.ExperimentContractError, "non-negative"):
            ec.TriggerEvidence(role="positive", lane_id="lane-a", candidate_launches=-1)

    def test_rejects_bool_as_launches(self):
        with self.assertRaisesRegex(ec.ExperimentContractError, "non-negative"):
            ec.TriggerEvidence(role="positive", lane_id="lane-a", candidate_launches=True)


class TriggerProofTests(unittest.TestCase):
    def test_passes_when_every_positive_lane_has_launches(self):
        evidence = [
            ec.TriggerEvidence(role="positive", lane_id="a", candidate_launches=10),
            ec.TriggerEvidence(role="positive", lane_id="b", candidate_launches=1),
        ]
        result = ec.evaluate_trigger_proof(evidence)
        self.assertTrue(result["passed"])
        self.assertEqual(result["reasons"], [])
        self.assertEqual(result["checked_lanes"], 2)
        self.assertEqual(result["untriggered_lanes"], [])

    def test_passes_via_route_selected_alone(self):
        evidence = [ec.TriggerEvidence(role="positive", lane_id="a", expected_route_selected=2)]
        result = ec.evaluate_trigger_proof(evidence)
        self.assertTrue(result["passed"])

    def test_fails_when_a_positive_lane_has_zero_launches_and_zero_routes(self):
        evidence = [
            ec.TriggerEvidence(role="positive", lane_id="a", candidate_launches=10),
            ec.TriggerEvidence(role="positive", lane_id="b", candidate_launches=0,
                                expected_route_selected=0),
        ]
        result = ec.evaluate_trigger_proof(evidence)
        self.assertFalse(result["passed"])
        self.assertEqual(result["untriggered_lanes"], ["b"])
        self.assertTrue(any("b" in r for r in result["reasons"]))

    def test_fails_closed_on_empty_evidence(self):
        result = ec.evaluate_trigger_proof([])
        self.assertFalse(result["passed"])
        self.assertEqual(result["checked_lanes"], 0)

    def test_control_role_lanes_are_not_checked(self):
        # A control lane deliberately should not trigger the candidate in
        # most contracts -- only positive-role lanes are checked.
        evidence = [
            ec.TriggerEvidence(role="positive", lane_id="a", candidate_launches=5),
            ec.TriggerEvidence(role="control", lane_id="c", candidate_launches=0),
        ]
        result = ec.evaluate_trigger_proof(evidence)
        self.assertTrue(result["passed"])
        self.assertEqual(result["checked_lanes"], 1)

    def test_boundary_role_lanes_are_not_checked(self):
        evidence = [
            ec.TriggerEvidence(role="positive", lane_id="a", candidate_launches=5),
            ec.TriggerEvidence(role="boundary", lane_id="bd", candidate_launches=0),
        ]
        result = ec.evaluate_trigger_proof(evidence)
        self.assertTrue(result["passed"])
        self.assertEqual(result["checked_lanes"], 1)


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

    # -------- VA24 multi-lane (dev-gpt-agent req_a667633429fa4c9e) --------

    def test_regression_interval_endpoints_reverse_under_negation(self):
        """HIGH-risk invariant, pinned deliberately.

        regression = max(0, -effect), so R_high derives from E_ci95_LOW, not
        from E_ci95_high. Using the effect's upper bound would report the most
        OPTIMISTIC case as the worst case.
        """
        contract = _minimal_contract(acceptance={
            "target_kernel_gain_pct": 1.0, "max_control_regression_pct": 1.0})
        effects = [
            ec.LaneEffect(role="positive", metric="m", geometric_effect_pct=2.0,
                          ci95_low_pct=1.5, ci95_high_pct=2.5, paired_rounds=10),
            # effect CI [-0.9, +0.4] -> worst plausible regression is 0.9,
            # which comes from the LOW end. Taking the high end would give 0.0.
            ec.LaneEffect(role="control", metric="c", geometric_effect_pct=-0.2,
                          ci95_low_pct=-0.9, ci95_high_pct=0.4, paired_rounds=10),
        ]
        agg = ec.aggregate_contract_effects(contract, effects, target_metric="m")
        self.assertAlmostEqual(agg["max_control_regression_pct_ci95_high"], 0.9)

    def test_multi_control_stays_fail_closed(self):
        """Independent per-lane 95% bounds are not a 95% FAMILY guarantee, and
        no contract in the registry exercises K>1 yet, so the correction is
        deliberately unimplemented rather than untested-and-shipped."""
        contract = _minimal_contract(acceptance={
            "target_kernel_gain_pct": 1.0, "max_control_regression_pct": 1.0})
        effects = [
            ec.LaneEffect(role="positive", metric="m", geometric_effect_pct=2.0,
                          ci95_low_pct=1.5, ci95_high_pct=2.5, paired_rounds=10),
            ec.LaneEffect(role="control", metric="c1", geometric_effect_pct=0.0,
                          ci95_low_pct=-0.3, ci95_high_pct=0.3, paired_rounds=10),
            ec.LaneEffect(role="control", metric="c2", geometric_effect_pct=0.0,
                          ci95_low_pct=-0.3, ci95_high_pct=0.3, paired_rounds=10),
        ]
        agg = ec.aggregate_contract_effects(contract, effects, target_metric="m")
        self.assertNotIn("max_control_regression_pct_ci95_high", agg)

    def test_multi_positive_bootstraps_the_fixed_composite_mean(self):
        """Two fixed positive lanes now yield an aggregate interval, computed
        by resampling WITHIN each lane -- never resampling lane identity."""
        contract = _minimal_contract(acceptance={
            "target_kernel_gain_pct": 1.0, "max_control_regression_pct": 1.0})
        lane_a = tuple([1.03] * 10)   # ~+3%
        lane_b = tuple([1.01] * 10)   # ~+1%
        effects = [
            ec.LaneEffect(role="positive", metric="m", geometric_effect_pct=3.0,
                          ci95_low_pct=3.0, ci95_high_pct=3.0, paired_rounds=10,
                          pair_ratios=lane_a),
            ec.LaneEffect(role="positive", metric="m", geometric_effect_pct=1.0,
                          ci95_low_pct=1.0, ci95_high_pct=1.0, paired_rounds=10,
                          pair_ratios=lane_b),
            ec.LaneEffect(role="control", metric="c", geometric_effect_pct=0.0,
                          ci95_low_pct=-0.2, ci95_high_pct=0.2, paired_rounds=10),
        ]
        agg = ec.aggregate_contract_effects(contract, effects, target_metric="m")
        self.assertIn("target_kernel_gain_pct_ci95_low", agg)
        # zero within-lane variance -> the aggregate collapses on the mean of
        # the two fixed lane effects, ~2%.
        self.assertAlmostEqual(agg["target_kernel_gain_pct_ci95_low"], 2.0, places=6)
        self.assertEqual(agg["target_kernel_gain_pct_paired_rounds"], 10)

    def test_multi_positive_rounds_is_the_weakest_contributor(self):
        contract = _minimal_contract(acceptance={
            "target_kernel_gain_pct": 1.0, "max_control_regression_pct": 1.0})
        effects = [
            ec.LaneEffect(role="positive", metric="m", geometric_effect_pct=3.0,
                          ci95_low_pct=3.0, ci95_high_pct=3.0, paired_rounds=10,
                          pair_ratios=tuple([1.03] * 10)),
            ec.LaneEffect(role="positive", metric="m", geometric_effect_pct=1.0,
                          ci95_low_pct=1.0, ci95_high_pct=1.0, paired_rounds=4,
                          pair_ratios=tuple([1.01] * 4)),
            ec.LaneEffect(role="control", metric="c", geometric_effect_pct=0.0,
                          ci95_low_pct=-0.2, ci95_high_pct=0.2, paired_rounds=10),
        ]
        agg = ec.aggregate_contract_effects(contract, effects, target_metric="m")
        self.assertEqual(agg["target_kernel_gain_pct_paired_rounds"], 4)

    def test_multi_positive_without_ratios_yields_no_interval(self):
        contract = _minimal_contract(acceptance={
            "target_kernel_gain_pct": 1.0, "max_control_regression_pct": 1.0})
        effects = [
            ec.LaneEffect(role="positive", metric="m", geometric_effect_pct=3.0,
                          ci95_low_pct=2.5, ci95_high_pct=3.5, paired_rounds=10),
            ec.LaneEffect(role="positive", metric="m", geometric_effect_pct=1.0,
                          ci95_low_pct=0.5, ci95_high_pct=1.5, paired_rounds=10),
            ec.LaneEffect(role="control", metric="c", geometric_effect_pct=0.0,
                          ci95_low_pct=-0.2, ci95_high_pct=0.2, paired_rounds=10),
        ]
        agg = ec.aggregate_contract_effects(contract, effects, target_metric="m")
        self.assertNotIn("target_kernel_gain_pct_ci95_low", agg)

    # --------- VA24 P0 hardening (dev-gpt-agent req_d563bd481bcf4324) ---------

    def test_ci_policy_requires_an_explicit_rounds_floor(self):
        """An interval policy with no evidence-depth floor is weaker than it
        looks: run_paired_lane() accepts pairs=1, whose bootstrap yields a
        degenerate interval that can look arbitrarily significant."""
        with self.assertRaises(ec.ExperimentContractError) as caught:
            _minimal_contract(acceptance={
                "end_to_end_gain_pct": 1.0, "max_control_regression_pct": 1.0,
                "effect_evidence_policy": "ci95_threshold_bound_v1"})
        self.assertIn("min_paired_rounds", str(caught.exception))

    def test_rounds_floor_applies_to_the_control_lane_too(self):
        """A regression budget from one usable pair is as untrustworthy as a
        gain from one; the floor must not be gain-only."""
        gate = self._ci_gate(
            end_to_end_gain_pct=1.855, end_to_end_gain_pct_ci95_low=1.482,
            max_control_regression_pct=0.0, max_control_regression_pct_ci95_high=0.2,
            max_control_regression_pct_paired_rounds=1)
        self.assertEqual(gate["status"], "invalid", gate)
        self.assertTrue(any("control interval" in r for r in gate["reasons"]), gate)

    def test_inverted_source_interval_is_rejected_before_regression_derivation(self):
        """max(0, -effect) can hide an inverted source interval, so the
        LaneEffect must be validated atomically before the transform."""
        contract = _minimal_contract(acceptance={
            "target_kernel_gain_pct": 1.0, "max_control_regression_pct": 1.0})
        effects = [
            ec.LaneEffect(role="positive", metric="m", geometric_effect_pct=2.0,
                          ci95_low_pct=1.5, ci95_high_pct=2.5, paired_rounds=10),
            # low > point: incoherent, and the regression transform would
            # otherwise still yield a plausible non-negative bound.
            ec.LaneEffect(role="control", metric="c", geometric_effect_pct=-0.2,
                          ci95_low_pct=0.9, ci95_high_pct=1.5, paired_rounds=10),
        ]
        agg = ec.aggregate_contract_effects(contract, effects, target_metric="m")
        self.assertNotIn("max_control_regression_pct_ci95_high", agg)

    def test_point_estimate_outside_interval_is_not_usable(self):
        contract = _minimal_contract(acceptance={
            "target_kernel_gain_pct": 1.0, "max_control_regression_pct": 1.0})
        effects = [
            ec.LaneEffect(role="positive", metric="m", geometric_effect_pct=9.0,
                          ci95_low_pct=1.0, ci95_high_pct=2.0, paired_rounds=10),
            ec.LaneEffect(role="control", metric="c", geometric_effect_pct=0.0,
                          ci95_low_pct=-0.3, ci95_high_pct=0.3, paired_rounds=10),
        ]
        agg = ec.aggregate_contract_effects(contract, effects, target_metric="m")
        self.assertNotIn("target_kernel_gain_pct_ci95_low", agg)

    # ------------- VA24: interval plumbing through aggregation -------------

    def test_single_lane_interval_is_carried_through_exactly(self):
        contract = _minimal_contract(acceptance={
            "target_kernel_gain_pct": 1.0, "max_control_regression_pct": 1.0})
        effects = [
            ec.LaneEffect(role="positive", metric="m", geometric_effect_pct=1.855,
                          ci95_low_pct=1.482, ci95_high_pct=2.169, paired_rounds=10),
            ec.LaneEffect(role="control", metric="c", geometric_effect_pct=-0.2,
                          ci95_low_pct=-0.6, ci95_high_pct=0.3, paired_rounds=10),
        ]
        agg = ec.aggregate_contract_effects(contract, effects, target_metric="m")
        self.assertEqual(agg["target_kernel_gain_pct_ci95_low"], 1.482)
        self.assertEqual(agg["target_kernel_gain_pct_paired_rounds"], 10)
        # regression = max(0, -effect), so its UPPER bound comes from the
        # effect's LOWER bound: max(0, -(-0.6)) == 0.6
        self.assertAlmostEqual(agg["max_control_regression_pct_ci95_high"], 0.6)

    def test_multi_lane_refuses_to_invent_an_aggregate_interval(self):
        """mean(lane ci95_lows) is NOT the ci95_low of the mean effect.

        Rather than emit a plausible-looking but statistically invalid
        number, aggregation omits the interval entirely for multi-lane
        contracts; the gate then reports "invalid" under an interval policy.
        """
        contract = _minimal_contract(acceptance={
            "target_kernel_gain_pct": 1.0, "max_control_regression_pct": 1.0})
        effects = [
            ec.LaneEffect(role="positive", metric="m", geometric_effect_pct=2.0,
                          ci95_low_pct=1.5, ci95_high_pct=2.5, paired_rounds=10),
            ec.LaneEffect(role="positive", metric="m", geometric_effect_pct=1.0,
                          ci95_low_pct=0.5, ci95_high_pct=1.5, paired_rounds=10),
            ec.LaneEffect(role="control", metric="c", geometric_effect_pct=-0.2,
                          ci95_low_pct=-0.6, ci95_high_pct=0.3, paired_rounds=10),
        ]
        agg = ec.aggregate_contract_effects(contract, effects, target_metric="m")
        self.assertEqual(agg["target_kernel_gain_pct"], 1.5)          # point estimate still averaged
        self.assertNotIn("target_kernel_gain_pct_ci95_low", agg)      # interval withheld

    def test_multi_lane_under_interval_policy_is_invalid_not_pass(self):
        contract = _minimal_contract(acceptance={
            "target_kernel_gain_pct": 1.0, "max_control_regression_pct": 1.0,
            "effect_evidence_policy": "ci95_threshold_bound_v1",
            "min_paired_rounds": 10})
        effects = [
            ec.LaneEffect(role="positive", metric="m", geometric_effect_pct=2.0,
                          ci95_low_pct=1.5, ci95_high_pct=2.5, paired_rounds=10),
            ec.LaneEffect(role="positive", metric="m", geometric_effect_pct=1.8,
                          ci95_low_pct=1.4, ci95_high_pct=2.2, paired_rounds=10),
            ec.LaneEffect(role="control", metric="c", geometric_effect_pct=0.1,
                          ci95_low_pct=-0.2, ci95_high_pct=0.4, paired_rounds=10),
        ]
        agg = ec.aggregate_contract_effects(contract, effects, target_metric="m")
        gate = ec.evaluate_promotion_gate(
            contract, correctness_gate=self.PASSING_CORRECTNESS, aggregated_effects=agg)
        self.assertEqual(gate["status"], "invalid", gate)

    def test_lane_effects_without_intervals_omit_them(self):
        contract = _minimal_contract(acceptance={
            "target_kernel_gain_pct": 1.0, "max_control_regression_pct": 1.0})
        effects = [
            ec.LaneEffect(role="positive", metric="m", geometric_effect_pct=2.0),
            ec.LaneEffect(role="control", metric="c", geometric_effect_pct=0.0),
        ]
        agg = ec.aggregate_contract_effects(contract, effects, target_metric="m")
        self.assertNotIn("target_kernel_gain_pct_ci95_low", agg)
        self.assertNotIn("max_control_regression_pct_ci95_high", agg)

    # ---------------- VA24: ci95_threshold_bound_v1 ----------------

    CI_ACCEPT = {
        "end_to_end_gain_pct": 1.0, "max_control_regression_pct": 1.0,
        "effect_evidence_policy": "ci95_threshold_bound_v1", "min_paired_rounds": 10,
    }

    def _ci_gate(self, **effects):
        base = {"end_to_end_gain_pct_paired_rounds": 10,
                "max_control_regression_pct_paired_rounds": 10}
        base.update(effects)
        return ec.evaluate_promotion_gate(
            _minimal_contract(acceptance=dict(self.CI_ACCEPT)),
            correctness_gate=self.PASSING_CORRECTNESS, aggregated_effects=base)

    def test_ci_policy_requires_lower_bound_to_reach_the_threshold(self):
        """The BOUND must be established, not merely positivity.

        +1.1% with CI [0.1, 2.1] "excludes zero" but never establishes the
        declared 1.0% gain, so it must FAIL. This is the specific weakness
        of a "CI excludes zero" rule (dev-gpt-agent req_cd86e5fd4a3b4328).
        """
        gate = self._ci_gate(
            end_to_end_gain_pct=1.1, end_to_end_gain_pct_ci95_low=0.1,
            max_control_regression_pct=0.0, max_control_regression_pct_ci95_high=0.2)
        self.assertEqual(gate["status"], "fail", gate)
        self.assertTrue(any("ci95_low" in r for r in gate["reasons"]), gate)

    def test_ci_policy_passes_when_lower_bound_clears_the_threshold(self):
        # RD73's real shape: point 1.855, ci95_low 1.482, threshold 1.0.
        gate = self._ci_gate(
            end_to_end_gain_pct=1.855, end_to_end_gain_pct_ci95_low=1.482,
            max_control_regression_pct=0.0, max_control_regression_pct_ci95_high=0.3)
        self.assertEqual(gate["status"], "pass", gate)

    def test_ci_policy_missing_interval_is_invalid_not_fail(self):
        gate = self._ci_gate(
            end_to_end_gain_pct=1.855,
            max_control_regression_pct=0.0, max_control_regression_pct_ci95_high=0.3)
        self.assertEqual(gate["status"], "invalid", gate)
        self.assertFalse(gate["passed"])

    def test_ci_policy_insufficient_paired_rounds_is_invalid(self):
        """pairs=1 yields a degenerate bootstrap interval that can look
        arbitrarily significant; a rounds floor is what makes the interval
        policy actually stronger than the point estimate."""
        gate = self._ci_gate(
            end_to_end_gain_pct=1.855, end_to_end_gain_pct_ci95_low=1.482,
            end_to_end_gain_pct_paired_rounds=1,
            max_control_regression_pct=0.0, max_control_regression_pct_ci95_high=0.3)
        self.assertEqual(gate["status"], "invalid", gate)
        self.assertTrue(any("paired rounds" in r for r in gate["reasons"]), gate)

    def test_ci_policy_incoherent_interval_is_invalid(self):
        gate = self._ci_gate(
            end_to_end_gain_pct=1.0, end_to_end_gain_pct_ci95_low=2.0,
            max_control_regression_pct=0.0, max_control_regression_pct_ci95_high=0.3)
        self.assertEqual(gate["status"], "invalid", gate)
        self.assertTrue(any("incoherent" in r for r in gate["reasons"]), gate)

    def test_ci_policy_regression_upper_bound_must_sit_inside_budget(self):
        """Noise absorbed, uncertain over-budget regression rejected."""
        ok = self._ci_gate(
            end_to_end_gain_pct=1.855, end_to_end_gain_pct_ci95_low=1.482,
            max_control_regression_pct=-0.2, max_control_regression_pct_ci95_high=0.4)
        self.assertEqual(ok["status"], "pass", ok)
        bad = self._ci_gate(
            end_to_end_gain_pct=1.855, end_to_end_gain_pct_ci95_low=1.482,
            max_control_regression_pct=-0.2, max_control_regression_pct_ci95_high=1.2)
        self.assertEqual(bad["status"], "fail", bad)
        self.assertTrue(any("ci95_high" in r for r in bad["reasons"]), bad)

    def test_legacy_contracts_keep_point_estimate_behaviour(self):
        contract = _minimal_contract(acceptance={
            "end_to_end_gain_pct": 1.0, "max_control_regression_pct": 1.0})
        self.assertEqual(contract.acceptance.effect_evidence_policy, "point_estimate_v1")
        gate = ec.evaluate_promotion_gate(
            contract, correctness_gate=self.PASSING_CORRECTNESS,
            aggregated_effects={"end_to_end_gain_pct": 1.5,
                                "max_control_regression_pct": 0.5})
        self.assertEqual(gate["status"], "pass", gate)

    def test_nan_effect_does_not_satisfy_any_threshold(self):
        """NaN must never PASS a bound.

        Regression test for a real gate hole (dev-gpt-agent review,
        req_cd86e5fd4a3b4328): the gate used isinstance(x, (int, float)),
        and every ordered comparison against NaN is False --
        `nan < required_gain` is False and `nan > regression_budget` is
        False -- so a NaN effect silently satisfied BOTH the gain and the
        regression check and produced a PASS from malformed evidence.
        """
        nan = float("nan")
        contract = _minimal_contract(acceptance={
            "target_kernel_gain_pct": 5, "end_to_end_gain_pct": 1,
            "max_control_regression_pct": 1,
        })
        effects = {"target_kernel_gain_pct": nan, "end_to_end_gain_pct": nan,
                   "max_control_regression_pct": nan}
        gate = ec.evaluate_promotion_gate(
            contract, correctness_gate=self.PASSING_CORRECTNESS, aggregated_effects=effects)
        self.assertFalse(gate["passed"], gate)
        joined = " ".join(gate["reasons"])
        self.assertIn("target_kernel_gain_pct", joined)
        self.assertIn("end_to_end_gain_pct", joined)
        self.assertIn("max_control_regression_pct", joined)

    def test_infinite_regression_does_not_satisfy_budget(self):
        contract = _minimal_contract(acceptance={"max_control_regression_pct": 1})
        effects = {"max_control_regression_pct": float("-inf")}
        gate = ec.evaluate_promotion_gate(
            contract, correctness_gate=self.PASSING_CORRECTNESS, aggregated_effects=effects)
        self.assertFalse(gate["passed"], gate)

    def test_bool_is_not_accepted_as_a_measured_effect(self):
        """bool is a subclass of int; True must not be read as 1.0."""
        contract = _minimal_contract(acceptance={"target_kernel_gain_pct": 0.5,
                                                 "max_control_regression_pct": 1})
        effects = {"target_kernel_gain_pct": True, "max_control_regression_pct": 0.0}
        gate = ec.evaluate_promotion_gate(
            contract, correctness_gate=self.PASSING_CORRECTNESS, aggregated_effects=effects)
        self.assertFalse(gate["passed"], gate)

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

    def test_passing_gate_reports_status_pass(self):
        contract = _minimal_contract(acceptance={"max_control_regression_pct": 1})
        effects = {"max_control_regression_pct": 0.0}
        gate = ec.evaluate_promotion_gate(
            contract, correctness_gate=self.PASSING_CORRECTNESS, aggregated_effects=effects)
        self.assertEqual(gate["status"], "pass")

    def test_failing_gate_reports_status_fail(self):
        contract = _minimal_contract(acceptance={"max_control_regression_pct": 1})
        effects = {"max_control_regression_pct": 5.0}
        gate = ec.evaluate_promotion_gate(
            contract, correctness_gate=self.PASSING_CORRECTNESS, aggregated_effects=effects)
        self.assertEqual(gate["status"], "fail")

    def test_missing_trigger_proof_short_circuits_to_invalid_even_with_perfect_effects(self):
        # EC18: a benchmark whose target code path never ran cannot be
        # evidence of pass OR fail, regardless of how good the numbers
        # otherwise look -- trigger proof is checked before anything else.
        contract = _minimal_contract(acceptance={
            "target_kernel_gain_pct": 5, "max_control_regression_pct": 1,
        })
        effects = {"target_kernel_gain_pct": 50.0, "max_control_regression_pct": 0.0}
        trigger_proof = ec.evaluate_trigger_proof(
            [ec.TriggerEvidence(role="positive", lane_id="a", candidate_launches=0)])
        gate = ec.evaluate_promotion_gate(
            contract, correctness_gate=self.PASSING_CORRECTNESS, aggregated_effects=effects,
            trigger_proof=trigger_proof)
        self.assertEqual(gate["status"], "invalid")
        self.assertFalse(gate["passed"])
        self.assertTrue(any("never exercised" in r or "positive-role" in r for r in gate["reasons"]))

    def test_passing_trigger_proof_does_not_block_a_real_pass(self):
        contract = _minimal_contract(acceptance={"max_control_regression_pct": 1})
        effects = {"max_control_regression_pct": 0.0}
        trigger_proof = ec.evaluate_trigger_proof(
            [ec.TriggerEvidence(role="positive", lane_id="a", candidate_launches=10)])
        gate = ec.evaluate_promotion_gate(
            contract, correctness_gate=self.PASSING_CORRECTNESS, aggregated_effects=effects,
            trigger_proof=trigger_proof)
        self.assertEqual(gate["status"], "pass")
        self.assertTrue(gate["passed"])

    def test_trigger_proof_absent_does_not_block_backward_compatibility(self):
        contract = _minimal_contract(acceptance={"max_control_regression_pct": 1})
        effects = {"max_control_regression_pct": 0.0}
        gate = ec.evaluate_promotion_gate(
            contract, correctness_gate=self.PASSING_CORRECTNESS, aggregated_effects=effects,
            trigger_proof=None)
        self.assertEqual(gate["status"], "pass")


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

    def test_report_flags_invalid_status_distinctly_from_pass_or_fail(self):
        contract = _minimal_contract(acceptance={"max_control_regression_pct": 1})
        effects = {"max_control_regression_pct": 0.0}
        trigger_proof = ec.evaluate_trigger_proof(
            [ec.TriggerEvidence(role="positive", lane_id="a", candidate_launches=0)])
        promotion = ec.evaluate_promotion_gate(
            contract, correctness_gate=self.PASSING_CORRECTNESS, aggregated_effects=effects,
            trigger_proof=trigger_proof)
        report = ec.render_report(
            contract, correctness_gate=self.PASSING_CORRECTNESS, aggregated_effects=effects,
            promotion_gate=promotion)
        self.assertIn("status: invalid", report)
        self.assertIn("INVALID", report)

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

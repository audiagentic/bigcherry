"""VA18 slice 1: build_contract_evidence_for_persistence() -- the pure
function factored out of run()'s persistence plumbing so it can be
unit-tested without a real hardware campaign. Proves the exact contract
GPT specified: contracts=[{id,hash},...] from validation_plan.contracts,
verdicts from contract_promotions with an explicit BLOCKED entry for any
bound contract with no produced promotion (never an inferred PASS), and
that a genuine RD08 PASS survives unchanged through this plumbing.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch import validation_campaign as vc  # noqa: E402


class _FakeBinding:
    def __init__(self, contract_id: str, contract_hash: str) -> None:
        self.contract_id = contract_id
        self.contract_hash = contract_hash


class BuildContractEvidenceForPersistenceTests(unittest.TestCase):
    def test_rd08_genuine_pass_survives_persistence_unchanged(self) -> None:
        binding = _FakeBinding("RD08-Q6K-MMVQ-VDR2", "hash08")
        promotions = {"RD08-Q6K-MMVQ-VDR2": {"passed": True, "status": "pass", "reasons": []}}
        contracts, verdicts = vc.build_contract_evidence_for_persistence((binding,), promotions)
        self.assertEqual(contracts, [{"id": "RD08-Q6K-MMVQ-VDR2", "hash": "hash08"}])
        self.assertEqual(verdicts["RD08-Q6K-MMVQ-VDR2"]["passed"], True)
        self.assertEqual(verdicts["RD08-Q6K-MMVQ-VDR2"]["status"], "pass")
        self.assertEqual(verdicts["RD08-Q6K-MMVQ-VDR2"]["detail"], promotions["RD08-Q6K-MMVQ-VDR2"])

    def test_bound_contract_with_no_promotion_gets_explicit_blocked_never_inferred_pass(self) -> None:
        binding = _FakeBinding("RD73-STABLE-GRAPH-CACHE-KEY", "hash73")
        contracts, verdicts = vc.build_contract_evidence_for_persistence((binding,), {})
        self.assertEqual(contracts, [{"id": "RD73-STABLE-GRAPH-CACHE-KEY", "hash": "hash73"}])
        self.assertEqual(verdicts["RD73-STABLE-GRAPH-CACHE-KEY"]["passed"], False)
        self.assertEqual(verdicts["RD73-STABLE-GRAPH-CACHE-KEY"]["status"], "blocked")

    def test_multi_contract_mixed_pass_and_blocked(self) -> None:
        rd05 = _FakeBinding("RD05", "h5")
        rd06 = _FakeBinding("RD06", "h6")
        rd07 = _FakeBinding("RD07", "h7")
        promotions = {"RD05": {"passed": True, "status": "pass"}, "RD06": {"passed": False, "status": "fail"}}
        contracts, verdicts = vc.build_contract_evidence_for_persistence((rd05, rd06, rd07), promotions)
        self.assertEqual(
            contracts,
            [{"id": "RD05", "hash": "h5"}, {"id": "RD06", "hash": "h6"}, {"id": "RD07", "hash": "h7"}],
        )
        self.assertEqual(verdicts["RD05"]["passed"], True)
        self.assertEqual(verdicts["RD06"]["passed"], False)
        # RD07 never appeared in contract_promotions at all -- explicit
        # BLOCKED, not silently absent from contract_verdicts.
        self.assertEqual(verdicts["RD07"]["passed"], False)
        self.assertEqual(verdicts["RD07"]["status"], "blocked")

    def test_no_bound_contracts_produces_empty_evidence(self) -> None:
        contracts, verdicts = vc.build_contract_evidence_for_persistence((), {})
        self.assertEqual(contracts, [])
        self.assertEqual(verdicts, {})

    def test_none_contract_promotions_treated_as_empty(self) -> None:
        binding = _FakeBinding("RD04-BF16-FLASH-ATTN-TILE", "h04")
        contracts, verdicts = vc.build_contract_evidence_for_persistence((binding,), None)
        self.assertEqual(verdicts["RD04-BF16-FLASH-ATTN-TILE"]["passed"], False)
        self.assertEqual(verdicts["RD04-BF16-FLASH-ATTN-TILE"]["status"], "blocked")


if __name__ == "__main__":
    unittest.main()

"""VA15 real-hardware finding (GPT session ses_5bbee8ce5c9a4265,
req_bc329f6ae30c4e4c): the generic S1-S7 record/tune/promote/replay/bench/
report campaign is unrelated to a contract's own evidence and must never
be a hard prerequisite of it -- a real, honest tune-campaign run can
legitimately promote zero candidates, and that must not block the
contract's own lane/correctness/trigger/promotion evidence from ever
being collected. After the 1204/RD08 producer migration, --run-rd73-contract
is the last specialized execution mode guarded here (the RD08 guards were
retired with the dedicated flags).

run() is a large integration entry point (real source materialization,
7 real cmake builds, the real e2e_smoke_campaign.Campaign class) that
cannot be reasonably unit-tested end to end without real hardware --
consistent with VA14's established scope boundary, this proves the exact
structural invariant GPT flagged via direct source inspection of the
committed function body, rather than mocking the entire pipeline.
"""

from __future__ import annotations

import inspect
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patch import validation_campaign as vc  # noqa: E402
from bigcherry.patch.campaign import producer as campaign_producer  # noqa: E402


class LegacyRunPathCorrectnessGateTests(unittest.TestCase):
    """VA15 wiring: the legacy run() path resolves a REAL ExperimentContract
    (not the lightweight ContractBinding projection) before calling
    compute_contract_correctness_gate().

    (The historical VA15 invariants that a contract run skips the generic
    S1-S7 campaign / trace probe / report.md were retired with the RD73
    legacy compatibility retirement; the generic pipeline now always runs on
    the legacy run() path.)"""

    def setUp(self) -> None:
        self.source = inspect.getsource(vc.run)

    def test_contract_correctness_gate_uses_a_real_experiment_contract_not_a_binding(self) -> None:
        # VA15 real-hardware finding (req_bc329f6ae30c4e4c follow-up):
        # validation_plan.contract is a patch_validation.ContractBinding --
        # a lightweight projection that deliberately does NOT carry
        # .correctness/.acceptance -- passing it directly to
        # compute_contract_correctness_gate() crashed on real hardware
        # with AttributeError: 'ContractBinding' object has no attribute
        # 'correctness'. The real committed code must resolve a real
        # ExperimentContract before calling compute_contract_correctness_gate():
        # the refactor does this via bound_contracts[0] (guarded by
        # len(bound_contracts) == 1) -- never validation_plan.contract.
        self.assertNotIn(
            "compute_contract_correctness_gate(\n            validation_plan.contract,",
            self.source,
        )
        # The refactor moved the gate resolution out of run() into the
        # producer-dispatch path, so check the full module source (not just
        # run()'s body) -- now campaign/producer.py -- for the bound_contracts[0] resolution.
        full_source = inspect.getsource(campaign_producer)
        match = re.search(
            r"compute_contract_correctness_gate\(\s*\n\s*bound_contracts\[0\],",
            full_source,
        )
        self.assertIsNotNone(
            match,
            "compute_contract_correctness_gate() must be called with a real "
            "ExperimentContract (bound_contracts[0], guarded by "
            "len(bound_contracts) == 1) -- never validation_plan.contract "
            "(a ContractBinding projection with no .correctness field)",
        )


if __name__ == "__main__":
    unittest.main()

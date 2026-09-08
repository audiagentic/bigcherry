"""VA25 golden-thread test: RD73's three server-driven lanes
(run_rd73_mtp_server_lane, run_rd73_decode_control_lane,
run_rd73_resource_burst_session) were the named P0 coverage gap -- the
underlying attestation.py machinery (ExecutionIdentity/
compare_execution_identity/parse_llama_server_attestation) already
existed, but nothing joined it to these three functions. That is exactly
the recurring defect shape this plan item's own notes name ("a field or
check exists at one end and nothing joins it to the path that needs it").

This test does not re-verify attestation CONTENT correctness (that is
test_attested_server_session.py and test_execution_attestation.py's job).
It verifies the STRUCTURAL property: these three functions cannot get a
measurable server without going through AttestedServerSession, so a
future fourth RD73-style lane added by copy-pasting one of these three
inherits attestation automatically rather than needing to remember it --
which is the actual failure mode that created this gap in the first
place.
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = Path(__file__).resolve().parents[3]
VALIDATION_CAMPAIGN_PATH = ROOT / "tools" / "bigcherry" / "patch" / "validation_campaign.py"

_RD73_SERVER_LANE_FUNCTIONS = (
    "run_rd73_mtp_server_lane",
    "run_rd73_decode_control_lane",
    "run_rd73_resource_burst_session",
)


class Rd73LaneAttestationGoldenThreadTests(unittest.TestCase):
    """AST-based, not a text grep: proves the property from the actual
    parsed function bodies, so a reformatting/rename that a grep would
    miss cannot silently defeat this test the way a purely textual check
    could."""

    @classmethod
    def setUpClass(cls) -> None:
        source = VALIDATION_CAMPAIGN_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(source, filename=str(VALIDATION_CAMPAIGN_PATH))
        cls.functions_by_name = {
            node.name: node
            for node in ast.walk(cls.tree)
            if isinstance(node, ast.FunctionDef)
        }

    def _calls_in(self, func: ast.FunctionDef) -> list[str]:
        names = []
        for node in ast.walk(func):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    names.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    names.append(node.func.attr)
        return names

    def test_all_three_named_functions_still_exist(self) -> None:
        # If one of these gets renamed, the golden-thread guarantee below
        # would silently stop covering it -- fail loudly instead.
        for name in _RD73_SERVER_LANE_FUNCTIONS:
            with self.subTest(function=name):
                self.assertIn(name, self.functions_by_name, f"{name} not found in validation_campaign.py")

    def test_no_rd73_server_lane_constructs_a_raw_server_runner(self) -> None:
        for name in _RD73_SERVER_LANE_FUNCTIONS:
            with self.subTest(function=name):
                func = self.functions_by_name[name]
                calls = self._calls_in(func)
                self.assertNotIn(
                    "ServerRunner", calls,
                    f"{name} constructs a raw ServerRunner directly -- it must go through "
                    f"AttestedServerSession, or its measurements are unattested (VA25)",
                )

    def test_every_rd73_server_lane_uses_attested_server_session(self) -> None:
        for name in _RD73_SERVER_LANE_FUNCTIONS:
            with self.subTest(function=name):
                func = self.functions_by_name[name]
                calls = self._calls_in(func)
                self.assertIn(
                    "AttestedServerSession", calls,
                    f"{name} never constructs an AttestedServerSession -- it launches a "
                    f"server without attestation (VA25)",
                )

    def test_every_rd73_server_lane_requires_expected_execution(self) -> None:
        # The parameter that carries the caller's declared identity into
        # AttestedServerSession -- a lane that dropped this argument could
        # still construct AttestedServerSession with a stale/default
        # identity and silently defeat the check.
        for name in _RD73_SERVER_LANE_FUNCTIONS:
            with self.subTest(function=name):
                func = self.functions_by_name[name]
                arg_names = [a.arg for a in func.args.kwonlyargs]
                self.assertIn(
                    "expected_execution", arg_names,
                    f"{name} does not require an expected_execution parameter (VA25)",
                )


if __name__ == "__main__":
    unittest.main()

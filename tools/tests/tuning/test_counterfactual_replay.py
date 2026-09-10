from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.tuning.counterfactual_replay import (  # noqa: E402
    CounterfactualProvenanceError,
    UNREPLAYABLE_PROVENANCE,
    build_pair,
    sha256,
)


def _selection():
    return {
        "schema_version": 1,
        "selection_id": "sel-1",
        "dispatch": "d1",
        "winner": {"name": "winner", "config_sha256": sha256({"name": "winner"})},
        "runner_up": {"name": "runner", "config_sha256": sha256({"name": "runner"})},
        "source_decision_sha256": sha256({"decision": 1}),
    }


def test_pair_changes_only_target_dispatch_and_is_analysis_only():
    pair = build_pair(_selection(), {"dispatches": {"d1": {"name": "old"}, "d2": {"name": "same"}}})
    assert pair["analysis_only"] is True
    assert pair["control_manifest"]["dispatches"]["d1"]["name"] == "winner"
    assert pair["counterfactual_manifest"]["dispatches"]["d1"]["name"] == "runner"
    assert pair["control_manifest"]["dispatches"]["d2"] == pair["counterfactual_manifest"]["dispatches"]["d2"]


@pytest.mark.parametrize("field", ["runner_up", "source_decision_sha256"])
def test_missing_provenance_fails_closed(field):
    selection = _selection()
    selection.pop(field)
    with pytest.raises(CounterfactualProvenanceError, match=UNREPLAYABLE_PROVENANCE):
        build_pair(selection, {"dispatches": {"d1": {}}})


def test_mutating_current_manifest_does_not_change_source_selection():
    selection = _selection()
    before = copy.deepcopy(selection)
    build_pair(selection, {"dispatches": {"d1": {}}})
    assert selection == before


def test_missing_target_dispatch_fails_closed():
    with pytest.raises(CounterfactualProvenanceError, match=UNREPLAYABLE_PROVENANCE):
        build_pair(_selection(), {"dispatches": {"other": {}}})

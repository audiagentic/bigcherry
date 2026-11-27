"""Contracts for deterministic native screening, not a second scheduler."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools/lab/radiance" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


screen = load("scheduler_screen")
sim = load("scheduler_sim")


def test_policy_grid_and_pareto():
    policies = list(screen.policies())
    assert len(policies) == 34
    assert len({tuple(sorted(p.items())) for p in policies}) == len(policies)
    assert screen.dominates([1, 2], [1, 3])
    assert not screen.dominates([1, 2], [1, 2])
    assert not screen.dominates([1, 3], [2, 2])


def test_native_seam_fails_closed():
    with pytest.raises(ValueError, match="seam changed"):
        sim.extend_native_sim("int main() {}")


TRACE = """BC_STEP 0 512 0 100000000 1:1:512:512:0
BC_EMIT 1 1 100000000
BC_STEP 1 0 1 20000000 1:0:1:513:1
BC_EMIT 1 2 120000000
BC_RESULT 0.12 0.10 0.02 0.02 2 1 1 0
"""


@pytest.mark.parametrize("output,rc,valid", [
    (TRACE, 0, True), (TRACE, 1, False),
    (TRACE.replace("BC_EMIT 1 2", "BC_EMIT 1 1"), 0, False),
    (TRACE.replace("BC_STEP 1", "BC_STEP 5"), 0, False),
    (TRACE + "BC_RESULT 0.12 0.10 0.02 0.02 2 1 1 0\n", 0, False),
    (TRACE.replace("BC_RESULT", "NO_RESULT"), 0, False),
    (TRACE.replace("0.10", "nan"), 0, False),
])
def test_native_result_and_trace_contract(monkeypatch, output, rc, valid):
    monkeypatch.setattr(screen.subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(returncode=rc, stdout=output, stderr=""))
    if valid:
        a = screen.native_run(Path("sched_test"), {"SIM_N": 1, "SIM_OUT": 2})
        assert a["steps"] == 2
        assert a == screen.native_run(Path("sched_test"), {"SIM_N": 1, "SIM_OUT": 2})
    else:
        with pytest.raises(ValueError):
            screen.native_run(Path("sched_test"), {"SIM_N": 1, "SIM_OUT": 2})

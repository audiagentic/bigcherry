"""VA26: live gfx1100 parity check between the existing RD08 hand-rolled
qualification path and the new campaign.qualification_execution
orchestrator + campaign.qualification_rd08 adapter.

Diagnostic-only, not a full PQM-v1 qualification (per dev-gpt-agent design
review req_a5e5f0987117413b): gfx1100 only, 2 cells (isolated +
release_delta), classified as an orchestrator parity proof. RD08's real
prior hardware evidence already shows a genuine bit_identical correctness
failure (independently confirmed, not an artifact) -- this run is
EXPECTED to reproduce promotion.status == "fail" via that same failure,
not to produce a passing qualification.

Run on the build server, in an isolated clone (never the shared checkout):
    cd ~/bigcherry-va26-rd08
    source tools/env/bigcherry-env.sh
    export HIP_VISIBLE_DEVICES=0,1
    export ROCR_VISIBLE_DEVICES=0,1
    PYTHONPATH=tools python tools/lab/va26-rd08-parity/run.py
"""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.campaign import qualification_execution as qe  # noqa: E402
from bigcherry.campaign import qualification_matrix as qm  # noqa: E402
from bigcherry.campaign import qualification_rd08 as rd08  # noqa: E402
from bigcherry.core import config, paths  # noqa: E402
from bigcherry.experiment import contract as ec  # noqa: E402
from bigcherry.patch import patchset, registry as patch_registry  # noqa: E402

PATCH_ID = "1204_rd08_q6k_mmvq_vdr2"
CONTRACT_ID = "RD08-Q6K-MMVQ-VDR2"
GFX1100 = "gfx1100"


def main() -> int:
    for required in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        if not os.environ.get(required):
            print(f"refusing to run: {required} is not set (see this file's own docstring)")
            return 2

    repo_root = Path(__file__).resolve().parents[3]
    bc_env = {
        k: os.environ[k] for k in (
            "BC_HOST", "BC_MODEL_ROOT", "BC_ROCM_SHIM", "BC_CACHE",
        ) if k in os.environ
    }
    print(f"env: {bc_env}")

    cfg = config.load(paths.RECIPES)
    catalog = patchset.catalog()
    registry = patch_registry.load_registry(paths.PATCHES)
    contracts = ec.load_contracts(repo_root / "config" / "experiment-contracts.toml")
    contract = contracts.contracts[CONTRACT_ID]

    full_plan = qm.build_qualification_matrix_plan(PATCH_ID, contract, cfg, catalog, registry)
    gfx1100_cells = tuple(cell for cell in full_plan.cells if cell.architecture == GFX1100)
    diagnostic_plan = dataclasses.replace(full_plan, cells=gfx1100_cells)
    print(f"diagnostic plan: {len(diagnostic_plan.cells)} cell(s) on {GFX1100}")
    for cell in diagnostic_plan.cells:
        print(f"  {cell.architecture}/{cell.contrast}: {cell.evidence_level}")

    model_root = Path(os.environ["BC_MODEL_ROOT"])
    hip_path = Path(os.environ["BC_ROCM_SHIM"])
    run_root = repo_root / ".." / "bigcherry-va26-rd08-run"
    run_root = run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    run_cell = rd08.make_rd08_run_cell(
        contract=contract, base_revision=cfg.pinned, hip_path=hip_path,
        model=model_root / "gpt-oss-20B" / "gguf" / "gpt-oss-20b-UD-Q6_K_XL.gguf",
        model_ref="tierM-gptoss20b-q6k",
        marker_regex=r"BIGCHERRY_PATCH_HIT patch=1204_rd08 path=q6k_mmvq_vdr2",
        worktree_root=run_root / "worktrees", build_root=run_root / "builds",
        run_root=run_root / "evidence", build_env=dict(os.environ), pairs=3,
    )

    result = qe.execute_qualification_plan(
        diagnostic_plan, contract, run_cell=run_cell, target_metric="tg128",
    )

    print()
    print(f"fully_evidenced: {result.fully_evidenced}")
    print(f"correctness_gate: {result.correctness_gate}")
    print(f"trigger_proof: {result.trigger_proof}")
    print(f"promotion: {result.promotion}")

    expected_fail_via_bit_identical = (
        result.fully_evidenced
        and result.correctness_gate is not None
        and not result.correctness_gate.get("passed")
        and "bit_identical" in (result.correctness_gate.get("failed_checks") or [])
        and result.promotion.get("status") == "fail"
    )
    print()
    print(f"PARITY CHECK (expected: fail via bit_identical): {expected_fail_via_bit_identical}")
    return 0 if expected_fail_via_bit_identical else 1


if __name__ == "__main__":
    raise SystemExit(main())

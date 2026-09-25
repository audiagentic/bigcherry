"""PVPS05: production dual-GPU no-regression lane.

Single-GPU lanes cannot see how a patch behaves inside the production serving
shape: RD73 (1233) passed its single-GPU contract and later measured -2% on
the dual RX 7900 XTX, -sm tensor, MTP speculative-decode server. This lane
reproduces that shape for every patch that opts in (``--production-lane``):
validated BC (control) vs validated BC + patch (subject), fresh llama-server
per request, paired and interleaved, metric = client wall-clock tokens/s.

Verdict: pass when the 95% interval of the subject/control effect does not
reach below -1% (the regression budget). Mean draft acceptance per arm is
recorded as work-equivalence evidence: differing acceptance means the arms
did different work, which is reported, not silently averaged away.

Host specifics are never committed: the model and device list resolve from
``${BIGCHERRY_PRODUCTION_MODEL}`` / ``${BIGCHERRY_PRODUCTION_DEVICES}`` (the
environment or the ``[env]`` table of config/environment.local.toml).
"""

from __future__ import annotations

import dataclasses
import json
import statistics
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from bigcherry.core import config as bc_config
from bigcherry.core import paths as bc_paths

ARTIFACT = "production-lane.json"
SCHEMA = 1
REGRESSION_BUDGET_PCT = 1.0
ARCHITECTURE = "gfx1100"
# Production serving flags (the dual-xtx-27b profile), minus host paths.
PRODUCTION_SERVER_ARGS = (
    "--parallel", "1", "--metrics", "-sm", "tensor", "--fit", "off",
    "--spec-type", "draft-mtp", "--spec-draft-n-max", "4",
    "--flash-attn", "on", "-ub", "512", "-b", "2048", "-ngl", "99", "--threads", "8",
    "-ctkd", "q8_0", "-ctvd", "q8_0", "-c", "16384",
)
CORPUS = bc_paths.REPO_ROOT / "tools" / "bigcherry" / "bench" / "corpora" / "mtp-27b-v1.jsonl"


class ProductionLaneError(RuntimeError):
    pass


def production_target() -> tuple[Path, tuple[int, ...]]:
    model = bc_config.require_resolved(
        bc_config.expand_host_value("${BIGCHERRY_PRODUCTION_MODEL}"), "production-lane model"
    )
    devices = bc_config.require_resolved(
        bc_config.expand_host_value("${BIGCHERRY_PRODUCTION_DEVICES}"), "production-lane devices"
    )
    ids = tuple(int(d) for d in devices.split(",") if d.strip())
    if len(ids) != 2:
        raise ProductionLaneError(f"production lane needs exactly two devices, got {devices!r}")
    path = Path(model)
    if not path.is_file():
        raise ProductionLaneError(f"production model not found: {path}")
    return path, ids


def verdict(effect: dict[str, Any]) -> dict[str, Any]:
    low = effect.get("ci95_low_pct")
    passed = isinstance(low, (int, float)) and low >= -REGRESSION_BUDGET_PCT
    return {
        "passed": passed,
        "regression_budget_pct": REGRESSION_BUDGET_PCT,
        "reason": None if passed else (
            f"ci95_low {low} reaches below -{REGRESSION_BUDGET_PCT}% on the production dual-GPU MTP server"
        ),
    }


def mean_acceptance(records: list[dict[str, Any]]) -> float | None:
    values = [r["draft_acceptance"] for r in records if isinstance(r.get("draft_acceptance"), (int, float))]
    return statistics.fmean(values) if values else None


def run_production_lane(
    *, control_server: Path, subject_server: Path, workdir: Path, measured_pairs: int = 8
) -> dict[str, Any]:
    from bigcherry.experiment.attestation import ExecutionIdentity
    from bigcherry.patch import producer_support as support

    model, devices = production_target()
    ctx = SimpleNamespace(model=model, corpus=CORPUS, workdir=workdir)
    effect, records, logs = support.mtp_server_lane(
        ctx,
        control_binary=control_server,
        subject_binary=subject_server,
        expected=ExecutionIdentity(backend="rocm", architectures=(ARCHITECTURE, ARCHITECTURE)),
        env={"HIP_VISIBLE_DEVICES": ",".join(str(d) for d in devices)},
        label="production-lane",
        role="control",
        server_args=PRODUCTION_SERVER_ARGS,
        measured_pairs=measured_pairs,
    )
    effect_doc = dataclasses.asdict(effect)
    return {
        "schema": SCHEMA,
        "model": str(model),
        "devices": list(devices),
        "server_args": list(PRODUCTION_SERVER_ARGS),
        "metric": "mtp_wall_tps",
        "effect": effect_doc,
        "draft_acceptance": {arm: mean_acceptance(rows) for arm, rows in records.items()},
        "verdict": verdict(effect_doc),
        "logs": {arm: str(path) for arm, path in logs.items()},
    }


def write(run_dir: Path, payload: dict[str, Any]) -> Path:
    path = run_dir / ARTIFACT
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path

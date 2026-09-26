"""Summarise plan-qualification campaign runs (one line per run).

Usage: python3 tools/lab/plan-qualification/summarize.py [run-name-glob]
Reads work/runs/<run>/campaign/producer-execution.json, the bound
performance artifact and the PVPS03 reference ladder; prints eligibility,
blocking reasons, each lane's
effect and CI95, correctness and activation status.
"""

from __future__ import annotations

import fnmatch
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _work_root() -> Path:
    """$BIGCHERRY_WORK_ROOT, else environment.local.toml [env], else work/."""
    if os.environ.get("BIGCHERRY_WORK_ROOT"):
        return Path(os.environ["BIGCHERRY_WORK_ROOT"])
    import tomllib

    local = ROOT / "config" / "environment.local.toml"
    env = tomllib.loads(local.read_text(encoding="utf-8")).get("env", {}) if local.is_file() else {}
    return Path(env.get("BIGCHERRY_WORK_ROOT") or ROOT / "work")


WORK = _work_root()


def _lanes(run_dir: Path) -> list[str]:
    out = []
    for artifact in sorted((run_dir / "campaign" / "artifacts").glob("*.json")):
        try:
            payload = json.loads(artifact.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        metrics = payload.get("metrics") if isinstance(payload, dict) else None
        if not isinstance(metrics, dict):
            continue
        for name, effect in metrics.items():
            if isinstance(effect, dict) and "geometric_effect_pct" in effect:
                out.append(
                    f"{name} {effect['geometric_effect_pct']:+.3f}% "
                    f"[{effect.get('ci95_low_pct', float('nan')):+.3f},{effect.get('ci95_high_pct', float('nan')):+.3f}]"
                )
    return out


def _ladder(run_dir: Path) -> list[str]:
    """PVPS03 reference ladder: each arm's mean and % vs stock llama.cpp."""
    path = run_dir / "campaign" / "reference-ladder.json"
    if not path.is_file():
        return []
    doc = json.loads(path.read_text(encoding="utf-8"))
    if "error" in doc:
        return [f"ladder error: {doc['error']}"]
    out = []
    for metric, data in (doc.get("metrics") or {}).items():
        cells = [
            f"{arm} {mean:.2f} ({data['pct_vs_stock'].get(arm, 0.0):+.2f}%)"
            for arm, mean in data.get("mean", {}).items()
        ]
        out.append(f"ladder {metric}: " + " | ".join(cells))
    return out


def _production_lane(run_dir: Path) -> list[str]:
    """PVPS05 production dual-GPU MTP no-regression lane."""
    path = run_dir / "campaign" / "production-lane.json"
    if not path.is_file():
        return []
    doc = json.loads(path.read_text(encoding="utf-8"))
    if "error" in doc:
        return [f"production lane ERROR: {doc['error']}"]
    e = doc["effect"]
    v = doc["verdict"]
    acc = doc.get("draft_acceptance", {})
    return [
        f"production lane {'PASS' if v['passed'] else 'FAIL'}: mtp_wall_tps {e['geometric_effect_pct']:+.3f}% "
        f"[{e['ci95_low_pct']:+.3f},{e['ci95_high_pct']:+.3f}] acceptance "
        + " ".join(f"{arm}={val:.3f}" for arm, val in acc.items() if val is not None)
    ]


def main() -> int:
    pattern = sys.argv[1] if len(sys.argv) > 1 else "*"
    for run_dir in sorted((WORK / "runs").iterdir()):
        if not run_dir.is_dir() or not fnmatch.fnmatch(run_dir.name, pattern):
            continue
        execution = run_dir / "campaign" / "producer-execution.json"
        if not execution.is_file():
            print(f"{run_dir.name}: no producer-execution.json")
            continue
        doc = json.loads(execution.read_text(encoding="utf-8"))
        checks = {k: v.get("status") for k, v in (doc.get("check_results") or {}).items() if isinstance(v, dict)}
        verdicts = {
            cid: (v.get("status"), (v.get("detail") or {}).get("reasons"))
            for cid, v in (doc.get("contract_verdicts") or {}).items()
        }
        print(f"{run_dir.name}: eligible={doc.get('eligible')} checks={checks}")
        for lane in _lanes(run_dir):
            print(f"    {lane}")
        for line in _ladder(run_dir):
            print(f"    {line}")
        for line in _production_lane(run_dir):
            print(f"    {line}")
        for cid, (status, reasons) in verdicts.items():
            print(f"    {cid}: {status} {reasons or ''}")
        if doc.get("reasons"):
            print(f"    reasons: {doc['reasons']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

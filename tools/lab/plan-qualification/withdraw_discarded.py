"""Withdraw evidence records produced by runs that were set aside.

A run moved to <work>/runs-discarded/<run>.<reason> (e.g. ``.parallel`` --
measured while another GPU job ran concurrently; ``.overlap-check``;
``.rerun-dup``) still has its evidence record in the patch's evidence file.
This matches each record to its run by the positive lane's pair ratios and
appends it to the patch's ``withdrawn.json`` (evidence itself stays
append-only; only session pooling skips withdrawn records).

Usage: python3 tools/lab/plan-qualification/withdraw_discarded.py [--dry-run]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))

from bigcherry.patch import evidence  # noqa: E402


def _work_root() -> Path:
    import subprocess

    out = subprocess.run([str(ROOT / "tools/lab/plan-qualification/work-root.sh"), str(ROOT)],
                         capture_output=True, text=True, check=True)
    return Path(out.stdout.strip())


def _ratio_keys(payload: dict) -> set[str]:
    keys: set[str] = set()
    for metric in (payload.get("metrics") or {}).values():
        ratios = metric.get("pair_ratios") or []
        if ratios:
            keys.add(repr(round(float(ratios[0]), 12)))
    return keys


def main() -> int:
    dry = "--dry-run" in sys.argv
    discarded = _work_root() / "runs-discarded"
    by_patch: dict[str, list[tuple[Path, str, set[str]]]] = {}
    for run_dir in sorted(p for p in discarded.iterdir() if p.is_dir()):
        reason = run_dir.name.split(".", 1)[1] if "." in run_dir.name else "discarded"
        execution = run_dir / "campaign" / "producer-execution.json"
        if not execution.is_file():
            continue
        patch_id = json.loads(execution.read_text(encoding="utf-8")).get("patch_id")
        keys: set[str] = set()
        for artifact in (run_dir / "campaign" / "artifacts").glob("*-performance.json"):
            keys |= _ratio_keys(json.loads(artifact.read_text(encoding="utf-8")))
        if patch_id and keys:
            by_patch.setdefault(patch_id, []).append((run_dir, reason, keys))
    for patch_id, runs in sorted(by_patch.items()):
        records = evidence.load_records(patch_id)
        already = evidence.withdrawn_digests(patch_id)
        path = evidence.evidence_path(patch_id).parent / evidence.WITHDRAWALS_FILE
        document = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"withdrawn": []}
        added = 0
        for record in records:
            digest = record.get("record_digest")
            if not isinstance(digest, str) or digest in already:
                continue
            record_keys: set[str] = set()
            for effect in record.get("lane_effects") or []:
                ratios = effect.get("pair_ratios") if isinstance(effect, dict) else None
                if ratios:
                    record_keys.add(repr(round(float(ratios[0]), 12)))
            for run_dir, reason, keys in runs:
                if keys & record_keys:
                    document["withdrawn"].append({
                        "record_digest": digest, "run": run_dir.name,
                        "reason": {"parallel": "measured while another GPU job ran concurrently",
                                   "overlap-check": "a pre-flight check overlapped the measurement",
                                   "rerun-dup": "accidental duplicate re-run of a finished session"
                                   }.get(reason, f"run set aside ({reason})"),
                    })
                    added += 1
                    print(f"{patch_id}: withdraw {digest[:16]} <- {run_dir.name}")
                    break
        if added and not dry:
            path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

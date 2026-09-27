"""Withdraw evidence records produced by runs that were set aside.

A run moved to <work>/runs-discarded/<run>.<reason> (e.g. ``.parallel`` --
measured while another GPU job ran concurrently; ``.overlap-check``;
``.rerun-dup``) still has its evidence record in the patch's evidence file.
This links each record to its run by EXACT equality of the complete set of
pair-ratio vectors (every lane, every round) and withdraws it only when that
set matches exactly one set-aside run and no kept run; ambiguous or partial
links are reported and skipped. It appends to the patch's ``withdrawn.json`` (evidence itself stays
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


Vectors = frozenset[tuple[float, ...]]


def _run_vectors(payload: dict) -> Vectors:
    return frozenset(
        tuple(round(float(x), 12) for x in metric["pair_ratios"])
        for metric in (payload.get("metrics") or {}).values() if metric.get("pair_ratios")
    )


def _record_vectors(record: dict) -> Vectors:
    return frozenset(
        tuple(round(float(x), 12) for x in effect["pair_ratios"])
        for effect in record.get("lane_effects") or []
        if isinstance(effect, dict) and effect.get("pair_ratios")
    )


def _vectors_of(run_dir: Path) -> Vectors:
    vectors: set[tuple[float, ...]] = set()
    for artifact in (run_dir / "campaign" / "artifacts").glob("*-performance.json"):
        vectors |= _run_vectors(json.loads(artifact.read_text(encoding="utf-8")))
    return frozenset(vectors)


def main() -> int:
    dry = "--dry-run" in sys.argv
    work = _work_root()
    discarded = work / "runs-discarded"
    kept = {d.name: _vectors_of(d) for d in (work / "runs").iterdir() if d.is_dir()}
    by_patch: dict[str, list[tuple[Path, str, Vectors]]] = {}
    for run_dir in sorted(p for p in discarded.iterdir() if p.is_dir()):
        reason = run_dir.name.split(".", 1)[1] if "." in run_dir.name else "discarded"
        execution = run_dir / "campaign" / "producer-execution.json"
        if not execution.is_file():
            continue
        patch_id = json.loads(execution.read_text(encoding="utf-8")).get("patch_id")
        vectors = _vectors_of(run_dir)
        if patch_id and vectors:
            by_patch.setdefault(patch_id, []).append((run_dir, reason, vectors))
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
            record_vectors = _record_vectors(record)
            if not record_vectors:
                continue
            matches = [(run_dir, reason) for run_dir, reason, vectors in runs if vectors == record_vectors]
            kept_matches = [name for name, vectors in kept.items() if vectors == record_vectors]
            if len(matches) != 1 or kept_matches:
                if matches or kept_matches:
                    print(f"{patch_id}: SKIP {digest[:16]} ambiguous link: set-aside={[m[0].name for m in matches]} kept={kept_matches}")
                continue
            for run_dir, reason in matches:
                if True:
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

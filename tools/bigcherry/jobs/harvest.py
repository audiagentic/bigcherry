"""Verified, explicit-path evidence harvest from attempt worktrees.

A validation attempt writes patch evidence inside its detached worktree. This
module identifies only records added by that attempt relative to its frozen
commit, verifies them against durable run identity, then merges them into the
canonical append-only evidence file under a maintenance fence.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from bigcherry.core.tree_activity import MaintenanceLock
from bigcherry.patch import evidence as patch_evidence
from bigcherry.tuning.journal import atomic_write

from .gitops import (
    commit_staged,
    require_clean_index,
    require_paths_clean,
    show_text,
    stage_exact,
)
from .store import RunStore


class HarvestError(RuntimeError):
    pass


def _container_records(text: str | None, patch_id: str) -> tuple[dict[str, Any], ...]:
    if text is None:
        return ()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HarvestError(f"baseline evidence JSON is invalid for {patch_id}: {exc}") from exc
    if not isinstance(value, dict) or value.get("patch_id") != patch_id:
        raise HarvestError(f"baseline evidence container identity mismatch for {patch_id}")
    rows = value.get("records")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise HarvestError(f"baseline evidence records are invalid for {patch_id}")
    return tuple(dict(row) for row in rows)


def _record_digest(record: dict[str, Any]) -> str:
    value = record.get("record_digest")
    if not isinstance(value, str) or not value:
        raise HarvestError("validation evidence record has no record_digest")
    expected = patch_evidence._record_digest(record)  # type: ignore[attr-defined]
    if value != expected:
        raise HarvestError("validation evidence record_digest does not match payload")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    atomic_write(
        path,
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True).encode("ascii")
        + b"\n",
    )


def _attempt_added_records(
    store: RunStore,
    run: dict[str, Any],
) -> tuple[int, tuple[dict[str, Any], ...]]:
    run_id = str(run["run_id"])
    attempt_no = store.latest_attempt(run_id)
    if attempt_no is None:
        raise HarvestError(f"run {run_id} has no attempt")
    attempt_root = store.attempt_root(run_id, attempt_no)
    result = store.read_optional(attempt_root / "executor-result.json")
    if not result or int(result.get("returncode", 1)) != 0:
        raise HarvestError(f"run {run_id} latest attempt is not successfully complete")
    attempt = store.attempt(run_id, attempt_no)
    project_root = Path(str(attempt["project_root"])).resolve()
    patch_id = str(run["job"]["patch"])
    source_root = project_root / "patches"
    source_path = patch_evidence.evidence_path(patch_id, root=source_root)
    if not source_path.is_file():
        raise HarvestError(
            f"run {run_id} produced no tracked validation evidence at {source_path}"
        )
    source_records = patch_evidence.load_records(patch_id, root=source_root)
    relpath = source_path.relative_to(project_root).as_posix()
    baseline = _container_records(
        show_text(project_root, str(attempt["commit"]), relpath), patch_id
    )
    baseline_digests = {
        str(row.get("record_digest")) for row in baseline if row.get("record_digest")
    }
    added = tuple(
        dict(record)
        for record in source_records
        if _record_digest(dict(record)) not in baseline_digests
    )
    if not added:
        raise HarvestError(
            f"run {run_id} has no evidence record added relative to frozen commit {attempt['commit']}"
        )
    return attempt_no, added


def _verify_record_for_run(record: dict[str, Any], run: dict[str, Any]) -> str:
    record_digest = _record_digest(record)
    patch_id = str(run["job"]["patch"])
    if record.get("patch_id") != patch_id:
        raise HarvestError(
            f"evidence patch_id {record.get('patch_id')!r} does not match run {patch_id!r}"
        )
    science = run.get("scientific_identity")
    focal = science.get("focal") if isinstance(science, dict) else None
    implementation = focal.get("implementation_digest") if isinstance(focal, dict) else None
    if not isinstance(implementation, str) or not implementation:
        raise HarvestError("run scientific identity has no focal implementation digest")
    if record.get("patch_implementation_digest") != implementation:
        raise HarvestError("evidence patch bytes do not match frozen run scientific identity")

    if isinstance(science, dict) and science.get("schema") == "bigcherry.scientific-identity.v2":
        validation_digest = focal.get("validation_digest") if isinstance(focal, dict) else None
        if record.get("validation_implementation_digest") != validation_digest:
            raise HarvestError(
                "evidence validation adapter bytes do not match frozen run identity"
            )
        expected_contracts = focal.get("contract_bindings") if isinstance(focal, dict) else None
        actual_contracts = record.get("contracts")
        if not isinstance(expected_contracts, list) or not isinstance(actual_contracts, list):
            raise HarvestError("evidence/frozen contract binding is malformed")
        canonical_expected = sorted(
            ({"id": str(row["id"]), "hash": str(row["hash"])} for row in expected_contracts),
            key=lambda row: row["id"],
        )
        canonical_actual = sorted(
            ({"id": str(row["id"]), "hash": str(row["hash"])} for row in actual_contracts),
            key=lambda row: row["id"],
        )
        if canonical_actual != canonical_expected:
            raise HarvestError(
                "evidence experiment-contract hashes do not match frozen series identity"
            )

    architecture = str(run["job"]["architecture"])
    archs = record.get("gpu_architectures")
    if not isinstance(archs, list) or architecture not in {str(item) for item in archs}:
        raise HarvestError(
            f"evidence does not bind the run architecture {architecture!r}"
        )
    return record_digest


def harvest_series(
    *,
    store: RunStore,
    series_id: str,
    project_root: Path,
    work_root: Path,
    commit: bool,
) -> dict[str, Any]:
    """Harvest one complete series into the canonical repository.

    ``commit=False`` stages only the exact evidence destination and records a
    non-verified harvest result. Review readiness requires ``commit=True``.
    """
    project_root = project_root.resolve()
    work_root = work_root.resolve()
    series = store.series(series_id)
    runs = tuple(
        sorted(
            (dict(run) for run in store.list_runs() if run.get("series_id") == series_id),
            key=lambda run: int(run["session"]),
        )
    )
    planned = int(series["series_material"]["planned_sessions"])
    if len(runs) != planned:
        raise HarvestError(
            f"series {series_id} has {len(runs)} durable runs, expected {planned}"
        )
    patch_ids = {str(run["job"]["patch"]) for run in runs}
    if len(patch_ids) != 1:
        raise HarvestError("one scientific series must bind exactly one focal patch")
    patch_id = next(iter(patch_ids))

    by_run: list[dict[str, Any]] = []
    records_to_merge: list[dict[str, Any]] = []
    seen_digests: set[str] = set()
    for run in runs:
        attempt_no, added = _attempt_added_records(store, run)
        digests: list[str] = []
        for record in added:
            record_digest = _verify_record_for_run(record, run)
            digests.append(record_digest)
            if record_digest not in seen_digests:
                seen_digests.add(record_digest)
                records_to_merge.append(record)
        by_run.append(
            {
                "run_id": str(run["run_id"]),
                "session": int(run["session"]),
                "attempt": attempt_no,
                "record_digests": sorted(digests),
            }
        )

    canonical_root = project_root / "patches"
    destination = patch_evidence.evidence_path(patch_id, root=canonical_root)
    try:
        destination_rel = destination.relative_to(project_root).as_posix()
    except ValueError as exc:
        raise HarvestError("canonical evidence destination escapes repository") from exc

    manifest = {
        "schema": "bigcherry.jobs.verified-evidence.v1",
        "series_id": series_id,
        "patch_id": patch_id,
        "scientific_identity_hash": series["series_material"]["scientific_identity_hash"],
        "hardware_cohort_hash": series["series_material"]["hardware_cohort_hash"],
        "destination": destination_rel,
        "runs": by_run,
        "record_digests": sorted(seen_digests),
    }
    result_path = store.root / "series" / series_id / "harvest.json"
    verified_path = store.root / "series" / series_id / "verified-evidence.json"

    existing_verified = store.read_optional(verified_path)
    canonical_digests = {
        str(record.get("record_digest"))
        for record in patch_evidence.load_records(patch_id, root=canonical_root)
    }
    if (
        existing_verified
        and existing_verified.get("manifest") == manifest
        and set(manifest["record_digests"]).issubset(canonical_digests)
    ):
        return dict(existing_verified, already_harvested=True)

    with MaintenanceLock(work_root, project_root):
        require_clean_index(project_root)
        require_paths_clean(project_root, (destination_rel,))
        for record in records_to_merge:
            patch_evidence.write_record(record, root=canonical_root)
        stage_exact(project_root, (destination_rel,))
        commit_sha: str | None = None
        if commit:
            commit_sha = commit_staged(
                project_root,
                message=f"evidence: harvest {series_id}",
                expected_paths=(destination_rel,),
            )
        result = {
            "schema": "bigcherry.jobs.harvest-result.v1",
            "series_id": series_id,
            "manifest": manifest,
            "committed": commit,
            "commit": commit_sha,
            "verified": bool(commit and commit_sha),
            "finished_ns": time.time_ns(),
        }
        _atomic_json(result_path, result)
        if result["verified"]:
            _atomic_json(verified_path, result)
            store.append_event(
                kind="series.evidence-harvested",
                severity="notify",
                data={"series_id": series_id, "commit": commit_sha},
            )
        else:
            store.append_event(
                kind="series.evidence-staged",
                data={"series_id": series_id, "destination": destination_rel},
            )
        return result

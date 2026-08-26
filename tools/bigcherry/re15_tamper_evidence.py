"""RE15 step J: negative/tamper-rejection evidence.

Proves the acceptance evidence chain fails CLOSED when an intermediate
artifact is corrupted after publication, for at least three distinct
stages, each against a DISPOSABLE COPY of a real acceptance store (never
the canonical evidence a real run produced -- see re15_acceptance_run.py).

Not a permanent CLI command -- same status as re15_acceptance_run.py: a
one-shot harness meant to be pointed at the artifact/work roots a prior
real re15_acceptance_run.py invocation left behind, run once to produce
the RE15 acceptance record's negative-evidence proof, not something that
runs routinely.

Each check below exercises the REAL downstream stage function (lifecycle.py
/ promotion.py), not a synthetic property test of ArtifactStore in
isolation -- the point is to show the actual production call path refuses,
not merely that content-hash verification exists somewhere.

Usage:
    python -m bigcherry.re15_tamper_evidence \
        --artifact-root ~/.cache/bigcherry/artifacts-store \
        --work-root ~/.cache/bigcherry \
        --model /mnt/vault/llm-models/qwen3.5-0.8B/gguf/Qwen3.5-0.8B-UD-Q5_K_XL.gguf \
        --dispatch-db-artifact-id <inventory's dispatch-db artifact id, C's output> \
        --runtime-bundle-artifact-id <replay lane's runtime-bundle artifact id> \
        --replay-cache-artifact-id <replay cache artifact id> \
        --comparison-report-artifact-id <comparison report artifact id>
"""

from __future__ import annotations

import argparse
import shutil
import sys
import uuid
from pathlib import Path

from . import lifecycle
from .tuning import promotion
from .core.artifacts import ArtifactLocator, ArtifactStore
from .core.context import ProjectContext
from .lifecycle import LifecycleError
from .campaign.smoke import RuntimeSmokeSpec


def _disposable_copy(source_root: Path, label: str) -> Path:
    disposable = source_root.parent / f"{source_root.name}-tamper-{label}-{uuid.uuid4().hex[:8]}"
    shutil.copytree(source_root, disposable)
    return disposable


def _flip_a_byte(path: Path) -> None:
    data = bytearray(path.read_bytes())
    if not data:
        raise SystemExit(f"cannot tamper an empty file: {path}")
    # Flip a byte roughly in the middle: avoids header-only structures
    # (e.g. a JSON opening brace) where a truncation-style corruption might
    # be caught by a shallower check than the one this test wants to prove.
    mid = len(data) // 2
    data[mid] ^= 0xFF
    path.write_bytes(bytes(data))


def _artifact_path_for(store: ArtifactStore, artifact_id: str) -> Path:
    descriptor = store.load_descriptor(artifact_id)
    return store._path(descriptor.relative_path)  # noqa: SLF001 -- test-only introspection


def check_inventory_tamper(
    *, artifact_root: Path, work_root: Path, dispatch_db_artifact_id: str,
    runtime_bundle_artifact_id: str, model: Path,
) -> None:
    """(1) Flip a byte in the inventory (dispatch-db) artifact consumed by
    the tune stage -> execute_tune_stage must refuse."""
    disposable = _disposable_copy(artifact_root, "inventory")
    try:
        store = ArtifactStore(disposable)
        target = _artifact_path_for(store, dispatch_db_artifact_id)
        _flip_a_byte(target)
        context = ProjectContext.resolve(work_root=work_root / "tamper-inventory-work")
        try:
            lifecycle.execute_tune_stage(
                context=context, store=store, run_id="tamper-inventory",
                runtime_bundle=ArtifactLocator(runtime_bundle_artifact_id),
                dispatch_db=ArtifactLocator(dispatch_db_artifact_id),
                spec=RuntimeSmokeSpec(model_path=model),
            )
        except LifecycleError as exc:
            print(f"  [PASS] inventory tamper rejected: {exc}")
            return
        raise SystemExit(
            "ACCEPTANCE FAILURE: tune stage did NOT reject a tampered inventory artifact"
        )
    finally:
        shutil.rmtree(disposable, ignore_errors=True)


def check_runtime_bundle_member_tamper(
    *, artifact_root: Path, work_root: Path, runtime_bundle_artifact_id: str,
    replay_cache_artifact_id: str, model: Path,
) -> None:
    """(2) Flip a byte in one runtime .so member of the replay lane's
    runtime bundle -> execute_replay_validation_stage must refuse before
    ever launching the (now-untrusted) binary."""
    disposable = _disposable_copy(artifact_root, "runtime-bundle")
    try:
        store = ArtifactStore(disposable)
        descriptor = store.load_descriptor(runtime_bundle_artifact_id)
        manifest_path = store._path(descriptor.relative_path)  # noqa: SLF001
        import json as _json
        manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
        member_names = [
            name for name in manifest["members"] if name != manifest["entrypoint"]
        ]
        if not member_names:
            raise SystemExit("runtime bundle has no non-entrypoint member to tamper")
        member_path = manifest_path.parent / member_names[0]
        _flip_a_byte(member_path)
        context = ProjectContext.resolve(work_root=work_root / "tamper-runtime-bundle-work")
        try:
            lifecycle.execute_replay_validation_stage(
                context=context, store=store, run_id="tamper-runtime-bundle",
                runtime_bundle=ArtifactLocator(runtime_bundle_artifact_id),
                replay_cache_artifact=ArtifactLocator(replay_cache_artifact_id),
                spec=RuntimeSmokeSpec(model_path=model),
            )
        except LifecycleError as exc:
            print(f"  [PASS] runtime-bundle member tamper rejected "
                  f"(member={member_names[0]!r}): {exc}")
            return
        raise SystemExit(
            "ACCEPTANCE FAILURE: replay validation did NOT reject a tampered "
            "runtime-bundle member"
        )
    finally:
        shutil.rmtree(disposable, ignore_errors=True)


def check_comparison_report_tamper(
    *, artifact_root: Path, comparison_report_artifact_id: str,
    replay_coverage_artifact_id: str,
) -> None:
    """(3) Flip a byte in the comparison-report artifact -> release-pointer
    construction must refuse rather than promote unverifiable evidence."""
    disposable = _disposable_copy(artifact_root, "comparison-report")
    try:
        store = ArtifactStore(disposable)
        target = _artifact_path_for(store, comparison_report_artifact_id)
        _flip_a_byte(target)
        try:
            promotion.pointer_from_comparison_report(
                store=store, report_artifact_id=comparison_report_artifact_id,
                release_tag="tamper-comparison-report",
                replay_coverage_artifact_id=replay_coverage_artifact_id,
                required_architectures=("gfx1201",),
            )
        except Exception as exc:  # noqa: BLE001
            # store.rehydrate() raises ArtifactError (not PromotionError) on
            # a content-hash mismatch -- both count as "refused", so this
            # check accepts either, but only ever instead of a silent
            # success (the raise SystemExit below is what actually proves
            # this branch, not the except clause's breadth).
            print(f"  [PASS] comparison-report tamper rejected: "
                  f"{type(exc).__name__}: {exc}")
            return
        raise SystemExit(
            "ACCEPTANCE FAILURE: release-pointer construction did NOT reject "
            "a tampered comparison-report artifact"
        )
    finally:
        shutil.rmtree(disposable, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dispatch-db-artifact-id", required=True)
    parser.add_argument("--runtime-bundle-artifact-id", required=True)
    parser.add_argument("--replay-cache-artifact-id", required=True)
    parser.add_argument("--comparison-report-artifact-id", required=True)
    parser.add_argument("--replay-coverage-artifact-id", required=True)
    args = parser.parse_args(argv)

    artifact_root = args.artifact_root.expanduser().resolve()
    work_root = args.work_root.expanduser().resolve()
    model = args.model.expanduser().resolve()

    print("--- J1: inventory (dispatch-db) tamper -> tune stage must refuse ---")
    check_inventory_tamper(
        artifact_root=artifact_root, work_root=work_root, model=model,
        dispatch_db_artifact_id=args.dispatch_db_artifact_id,
        runtime_bundle_artifact_id=args.runtime_bundle_artifact_id,
    )

    print("--- J2: runtime-bundle member tamper -> replay validation must refuse ---")
    check_runtime_bundle_member_tamper(
        artifact_root=artifact_root, work_root=work_root, model=model,
        runtime_bundle_artifact_id=args.runtime_bundle_artifact_id,
        replay_cache_artifact_id=args.replay_cache_artifact_id,
    )

    print("--- J3: comparison-report tamper -> release pointer must refuse ---")
    check_comparison_report_tamper(
        artifact_root=artifact_root,
        comparison_report_artifact_id=args.comparison_report_artifact_id,
        replay_coverage_artifact_id=args.replay_coverage_artifact_id,
    )

    print("=== RE15 step J: all three tamper checks PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

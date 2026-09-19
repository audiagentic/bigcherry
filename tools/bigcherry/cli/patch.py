"""CLI presentation handlers for patch lifecycle commands."""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING

from .. import patch_admission
from ..core import config as campaign_config
from ..core import paths
from ..core.context import ProjectContext
from ..patch import apply as patcher
from ..patch import catalog as patch_catalog
from ..patch import disposition as patch_disposition
from ..patch import docs as patch_docs
from ..patch import gates as patch_gates
from ..patch import lifecycle as patch_lifecycle
from ..patch import overlay as patch_overlay
from ..patch import patchset
from ..patch import rebase as patch_rebase
from ..patch import registry as patch_registry
from ..patch import selection as patch_selection
from ..release import records as releases
from ..source.workspace import UpstreamRepository, WorkspaceError

if TYPE_CHECKING:
    from ..patch.selection import CliPatchSelection

DISPOSITIONS_DIR = paths.DISPOSITIONS


_copy_overlay = patch_overlay.copy_overlay
_restore_overlay = patch_overlay.restore_overlay
_record_for = releases.record_for_checkout


def _apply_exact_selection(
    root: Path,
    selection: CliPatchSelection,
    *,
    force: bool = False,
    dry_run: bool = False,
    allow_stale_validation_evidence: bool = False,
) -> bool:
    """Install the overlay and apply one exact ``--source`` selection."""
    live_revision = patch_rebase._git(root, "rev-parse", "HEAD")
    # Resolve symbolic refs in the same checkout before comparing them with
    # live HEAD. A correctly pinned checkout may use a tag rather than a SHA.
    try:
        expected_revision = patch_rebase._git(
            root, "rev-parse", f"{selection.source_ref}^{{commit}}"
        )
    except Exception as exc:  # noqa: BLE001 -- clear apply-time diagnostic
        print(
            f"apply: --source {selection.source_name!r}'s ref "
            f"{selection.source_ref!r} does not resolve in this checkout: {exc}",
            file=sys.stderr,
        )
        return False
    if expected_revision != live_revision:
        print(
            f"apply: --source {selection.source_name!r} expects upstream "
            f"revision {selection.source_ref!r} ({expected_revision}), but "
            f"the live checkout is at {live_revision!r} -- pull first, or "
            "re-run patch-rebase-check",
            file=sys.stderr,
        )
        return False

    if selection.source_name is None:
        print(
            "apply: selection has no source name -- refusing to guess",
            file=sys.stderr,
        )
        return False
    fresh = patch_selection._resolve_exact_selection(selection.source_name)
    if (
        fresh.patch_set_id != selection.patch_set_id
        or fresh.patch_ids != selection.patch_ids
    ):
        print(
            f"apply: --source {selection.source_name!r}'s composition changed "
            "since it was resolved (config/recipes.toml or the patch "
            "registry moved concurrently) -- re-run to pick up the current "
            "composition",
            file=sys.stderr,
        )
        return False
    selection = fresh

    if selection.overlay is None:
        print(
            f"apply: --source {selection.source_name!r}'s overlay flag was "
            "never resolved -- refusing to guess whether to install it",
            file=sys.stderr,
        )
        return False

    admission = patch_admission.admit(
        selection.patch_ids,
        mode="apply",
        pinned_ref=selection.source_ref,
        resolved_base_revision=live_revision,
        allow_stale_validation_evidence=allow_stale_validation_evidence,
    )
    for warning in admission.warnings:
        print(f"apply: admission warning: {warning}", file=sys.stderr)
    if not admission.admissible:
        detail = "; ".join(admission.failures) or admission.status
        print(f"apply: patch admission failed: {detail}", file=sys.stderr)
        return False

    record = _record_for(root)
    if not force and not record.audit.get("passed"):
        print(
            "refusing to patch a tree that has not passed a strict audit.\n"
            "  run `python -m bigcherry audit` first, or pass --force.",
            file=sys.stderr,
        )
        return False

    overlay_backup: dict[str, str | None] = {}
    overlay_sim: dict[str, str] = {}
    written: list[str] = []
    if selection.overlay:
        written = _copy_overlay(
            root,
            dry_run=dry_run,
            backup=overlay_backup,
            sim_texts=overlay_sim,
        )

    resolved = patchset.resolve_exact(selection.patch_ids, allow_rejected=False)
    patches = patchset.load_resolved(resolved)
    results = patcher.apply_all(
        patches, root, dry_run=dry_run, initial_texts=overlay_sim
    )
    ok = all(result.ok for result in results)
    if not ok and not dry_run and overlay_backup:
        _restore_overlay(root, overlay_backup)

    intended_tree_state = selection.tree_state_key(live_revision)
    selection_changed = record.tree_state != intended_tree_state
    tree_mutated = bool(written) or any(result.changed for result in results)

    if ok:
        verb = "would write" if dry_run else "wrote"
        print(
            f"overlay: {verb} {len(written)} file(s)"
            if selection.overlay
            else "overlay: none (source overlay=false)"
        )
    else:
        verb = "not written (dry run)" if dry_run else "rolled back"
        print(f"overlay: {verb} -- patches failed")
    print(f"patches ({len(patches)} file(s)):")
    print(patcher.format_results(results))

    if not dry_run:
        record = _record_for(root)
        record.patches = releases.summarise_patches(results)
        releases.record_apply_result(
            record, ok, mutated=selection_changed or tree_mutated
        )
        if not ok:
            record.notes = "patches failed: " + ", ".join(
                record.patches["failed_edits"]
            )
        elif record.notes.startswith("patches failed:"):
            record.notes = ""
        record.tree_state = intended_tree_state if ok else ""
        record.save()
    return ok


def cmd_apply(args: Namespace) -> int:
    root = paths.llama_root(args.llama_root)
    report_path = getattr(args, "rebase_report", None)
    known_good = bool(getattr(args, "known_good", False))
    allow_stale_validation_evidence = bool(
        getattr(args, "allow_stale_validation_evidence", False)
    )

    if bool(report_path) != known_good:
        print(
            "apply: --rebase-report PATH and --known-good must be supplied together",
            file=sys.stderr,
        )
        return 2

    if report_path:
        if allow_stale_validation_evidence:
            print(
                "apply: --allow-stale-validation-evidence applies only to "
                "direct --source admission; --rebase-report remains fail-closed",
                file=sys.stderr,
            )
            return 2
        if getattr(args, "source", None):
            print(
                "apply: --rebase-report owns the exact logical selection; "
                "do not combine it with --source",
                file=sys.stderr,
            )
            return 2
        try:
            result = patch_rebase.apply_known_good(
                root,
                Path(report_path),
                force=args.force,
                dry_run=args.dry_run,
            )
        except (patch_rebase.RebaseCheckError, OSError, ValueError) as exc:
            print(f"apply: {exc}", file=sys.stderr)
            return 2

        selected = len(result.selected_patch_ids)
        applied = len(result.known_good_patch_ids)
        print(f"selection: rebase-report={report_path} known-good={applied}/{selected}")
        if result.partial:
            print("  NOTE: partial reconciliation apply; release stage not advanced")
        print("  RESULT: " + ("PASS" if result.ok else "FAIL"))
        return 0 if result.ok else 1

    try:
        selection = patch_selection.resolve_cli_selection(args)
    except patch_selection.SelectionError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if selection.select_all:
        print("apply: --source is required", file=sys.stderr)
        return 2

    ok = _apply_exact_selection(
        root,
        selection,
        force=args.force,
        dry_run=args.dry_run,
        allow_stale_validation_evidence=allow_stale_validation_evidence,
    )
    print(f"selection: {selection.label}")
    print("  RESULT: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def cmd_patch_rebase_check(args: Namespace) -> int:
    """PA16: revision-specific patch compatibility report. Observational --
    never advances release stage, never mutates the vendor checkout."""
    root = paths.llama_root(args.llama_root)
    try:
        report = patch_rebase.run_rebase_check(
            root,
            source_name=getattr(args, "source", None),
            all_patches=bool(getattr(args, "all_patches", False)),
            experiment=getattr(args, "experiment", None),
            focal_overlay_patch_id=getattr(args, "focal_overlay", None),
            context_lines=args.context_lines,
        )
    except patch_rebase.RebaseCheckError as exc:
        print(f"patch-rebase-check: {exc}", file=sys.stderr)
        return 2

    if args.json:
        patch_rebase.write_report(Path(args.json), report)

    print(patch_rebase.render_report(report))
    if args.json:
        print(f"report: {args.json}")

    return 1 if report["summary"]["reconciliation_required"] else 0


def cmd_patch_status(args: Namespace) -> int:
    """Render computed patch lifecycle status."""
    statuses = patch_lifecycle.compute_all()
    if args.item:
        statuses = {key: value for key, value in statuses.items() if key == args.item}
        if not statuses:
            print(
                f"no lifecycle signal found for plan-item {args.item!r}",
                file=sys.stderr,
            )
            return 1
    if not statuses:
        print(
            "no plan-items with any tracked/materialized/contracted signal found",
            file=sys.stderr,
        )
        return 1
    print(patch_lifecycle.render_table(statuses))
    return 0


def cmd_patch_explain(args: Namespace) -> int:
    try:
        snapshot = patch_catalog.build_snapshot()
    except ValueError as exc:
        print(
            f"patch explain: could not load patches/catalog.toml: {exc}",
            file=sys.stderr,
        )
        return 2
    from ..core import config as campaign_config

    try:
        cfg = campaign_config.load(paths.RECIPES)
    except (campaign_config.ConfigError, OSError):
        cfg = None
    try:
        info = patch_catalog.explain(args.patch_id, snapshot, cfg)
    except KeyError as exc:
        print(f"patch explain: {exc}", file=sys.stderr)
        return 1
    print(patch_catalog.render_explanation(info))
    return 0


def cmd_patch_graph(args: Namespace) -> int:
    """Render the patch dependency topology."""
    try:
        snapshot = patch_catalog.build_snapshot()
    except ValueError as exc:
        print(
            f"patch graph: could not load patches/catalog.toml: {exc}", file=sys.stderr
        )
        return 2
    try:
        print(patch_catalog.dependency_graph(snapshot, roots=tuple(args.roots or ())))
    except ValueError as exc:
        print(f"patch graph: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_patch_lint(args: Namespace) -> int:
    """Run the non-mutating catalog/package lint gate. Purely static (VA02,
    docs/reference/testing/PATCH_VALIDATION.md): never verifies validation
    *evidence* content/freshness (that's patch-verify-evidence's job, VA08)
    -- cross_check() is called with verify_validation_evidence=False and its
    SUMMARY branch disabled because the shared lint gate owns that check. The
    HI83 dynamic branch (which silently skips packaged RD patches, since it
    only checks catalog.toml entries) never runs here."""
    problems = list(
        patch_catalog.cross_check(
            verify_validation_evidence=False,
            allow_legacy_grandfather=True,
            check_summaries=False,
        )
    )
    lint_report = patch_gates.evaluate_repository_lint_gates()
    problems.extend(lint_report.problems)
    if args.json:
        print(
            json.dumps(
                {
                    "passed": not problems,
                    "problems": problems,
                    "grandfathered": list(lint_report.grandfathered),
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        for problem in problems:
            print(problem, file=sys.stderr)
        for patch_id in lint_report.grandfathered:
            print(
                f"{patch_id}: structurally grandfathered (non-current, not failing)",
                file=sys.stderr,
            )
    return 0 if not problems else 1


def cmd_patch_disposition(args: Namespace) -> int:
    """HI152: record/list/clear a revision-bound known_broken disposition
    for a non-recipe patch -- see patch/disposition.py's module docstring.
    Never a standing waiver: it stops applying the instant target_revision
    or patch_digest changes."""
    action = args.disposition_action
    if action == "set":
        record = patch_disposition.Disposition(
            patch_id=args.patch_id,
            target_revision=args.revision,
            patch_digest=args.digest,
            disposition="known_broken",
            failure_status=args.failure_status,
            reason=args.reason,
            owner=args.owner,
            tracking_item=args.tracking_item,
        )
        try:
            path = patch_disposition.save_disposition(DISPOSITIONS_DIR, record)
        except patch_disposition.DispositionError as exc:
            print(f"patch-disposition: {exc}", file=sys.stderr)
            return 2
        print(f"disposition recorded: {path}")
        return 0
    if action == "clear":
        removed = patch_disposition.clear_disposition(DISPOSITIONS_DIR, args.patch_id)
        print(
            f"disposition cleared: {args.patch_id}"
            if removed
            else f"no disposition on file for {args.patch_id}"
        )
        return 0
    # action == "list"
    records = patch_disposition.list_dispositions(DISPOSITIONS_DIR)
    if args.json:
        print(
            json.dumps(
                {pid: r.__dict__ for pid, r in sorted(records.items())},
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not records:
        print("no dispositions on file")
        return 0
    for pid, record in sorted(records.items()):
        print(
            f"{pid}: revision={record.target_revision[:12]} digest={record.patch_digest[:12]} "
            f"owner={record.owner} tracking_item={record.tracking_item} reason={record.reason!r}"
        )
    return 0


def cmd_patch_verify_evidence(args: Namespace) -> int:
    """Report current validation evidence for selected patches."""
    from ..core import config as campaign_config
    from ..source.workspace import UpstreamRepository, WorkspaceError

    cfg = campaign_config.load(paths.RECIPES)
    modules = patchset.catalog()
    if args.patch_id is not None:
        modules = [module for module in modules if module.patch_id == args.patch_id]
        if not modules:
            print(f"unknown patch {args.patch_id!r}", file=sys.stderr)
            return 1
    # VA08 resolved-pin-SHA slice (GPT session ses_330ae3c055084f38,
    # req_1c131ba025834afe): a record's base_revision must match the pin's
    # actual RESOLVED commit, not just a string with the same ref name --
    # otherwise stale evidence from before the pin moved could still
    # "match" cfg.pinned by name alone. resolve_ref() is local-only (no
    # fetch); if the configured pin cannot resolve locally, fail closed
    # with a real CLI error rather than silently skipping the check.
    try:
        import os

        mirror = ProjectContext.resolve(
            work_root=os.environ.get("BC_CACHE")
        ).upstream_repo
        upstream = mirror if mirror.exists() else paths.llama_root()
        resolved_base_revision = UpstreamRepository(upstream).resolve_ref(cfg.pinned)
    except WorkspaceError as exc:
        print(
            f"patch-verify-evidence: cannot resolve pin {cfg.pinned!r} locally: {exc}",
            file=sys.stderr,
        )
        return 2
    statuses = patch_catalog.validation_evidence_statuses(
        [module.patch_id for module in modules],
        pinned_ref=cfg.pinned,
        resolved_base_revision=resolved_base_revision,
        allow_legacy_grandfather=not args.no_legacy_grandfather,
    )
    if args.json:
        print(
            json.dumps(
                {
                    patch_id: {
                        "status": status.status,
                        "problems": list(status.problems),
                        "campaign_digests": list(status.campaign_digests),
                    }
                    for patch_id, status in sorted(statuses.items())
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        for patch_id, status in sorted(statuses.items()):
            print(f"{patch_id}: {status.status}")
            for problem in status.problems:
                print(f"  - {problem}")
    return 0 if all(status.ok for status in statuses.values()) else 1


def cmd_patch_validate(args: Namespace) -> int:
    """PA33: execute declared validation through PA34+PA36 authorities.

    When --validation-producer is provided, resolves the selector, catalog
    snapshot, validation plan, bound contracts, platform/device identity,
    model/fixture inputs, and producer selection, then executes through
    PA36's shared primitives and binds the evidence into the existing
    schema. Otherwise, falls back to the read-only verify-evidence behavior.
    """
    validation_producer = getattr(args, "validation_producer", None)
    if validation_producer is None:
        # No producer selected: fall back to read-only verify-evidence
        return cmd_patch_verify_evidence(args)

    # PA33: fail-fast validation before doing expensive work
    # 1. Selector validation
    if args.patch_id is None:
        print("patch-validate: patch_id is required", file=sys.stderr)
        return 2
    # 2. PA34 selector qualifier validation
    source = getattr(args, "source", None)
    experiment = getattr(args, "experiment", None)
    focal_overlay = getattr(args, "focal_overlay", False)
    if (experiment is not None or focal_overlay) and not source:
        print("patch-validate: --experiment/--focal-overlay require --source", file=sys.stderr)
        return 2
    # 3. Producer input validation
    producer_input = getattr(args, "producer_input", [])
    for entry in producer_input:
        if "=" not in entry:
            print(f"patch-validate: --producer-input {entry!r} must be NAME=VALUE", file=sys.stderr)
            return 2
    # 4. Device map validation
    device_map = getattr(args, "device_map", None)
    if device_map is not None:
        for entry in device_map.split(","):
            if "=" not in entry:
                print(f"patch-validate: --device-map entry {entry!r} must be ARCH=ID[,ID...]", file=sys.stderr)
                return 2
    # 5. Producer corpus validation
    producer_corpus = getattr(args, "producer_corpus", None)
    if producer_corpus is not None:
        corpus_path = Path(producer_corpus)
        if not corpus_path.is_file():
            print(f"patch-validate: --producer-corpus {producer_corpus!r} is not a file", file=sys.stderr)
            return 2

    # PA33: execute through PA36's shared primitives
    from ..patch import validation_campaign as validation_campaign_module

    # Build a minimal args namespace for the validation campaign
    campaign_args = Namespace(
        patch=args.patch_id,
        validation_producer=validation_producer,
        producer_input=getattr(args, "producer_input", []),
        producer_corpus=getattr(args, "producer_corpus", None),
        amdgpu_targets=getattr(args, "amdgpu_targets", None),
        device_map=getattr(args, "device_map", None),
        hip_path=getattr(args, "hip_path", None),
        model=getattr(args, "model", None),
        workdir=getattr(args, "workdir", None),
        worktree_root=getattr(args, "worktree_root", None),
        baseline_source=getattr(args, "baseline_source", None),
        # PA34 selector arguments
        source=getattr(args, "source", None),
        experiment=getattr(args, "experiment", None),
        focal_overlay=getattr(args, "focal_overlay", False),
        # Defaults for other required fields
        run=None,
        tune=None,
        replay=None,
        stock=None,
        control=None,
        validation_subject=None,
        run_rd04_contract=False,
        run_rd04_benchmark=False,
        rd04_corpus=None,
        run_rd08_lanes=False,
        run_rd08_contract=False,
        run_rd12_contract=False,
        run_rd13_contract=False,
        run_rd26_contract=False,
        run_rd58_state_restore=False,
        run_rd73_contract=False,
        run_rd30_correctness=False,
        run_rd17_ppl_check=False,
        run_rd19_ppl_check=False,
        run_rd43_ppl_check=False,
        run_rd05_contract_correctness=False,
        run_rd06_contract_correctness=False,
        run_rd07_contract_correctness=False,
        run_1203_backend_reference_contract_correctness=False,
        run_1000_exact_shape_backend_ops=False,
        run_1000_q2k_model_lane=False,
        run_0300_framework_configuration=False,
    )

    # Execute through the validation campaign's main()
    return validation_campaign_module.main(campaign_args)


def cmd_patch_gates(args: Namespace) -> int:
    """Evaluate the shared PA21 gate registry for one exact composition.

    This handler resolves inputs and presents results only.  Gate policy
    remains in ``patch.gates`` and its existing domain authorities; this
    command must not grow a second set of patch-selection or evidence rules.
    """
    from ..campaign import resolution as campaign_resolution

    intent = patch_gates.GateIntent(args.intent)
    try:
        cfg = campaign_config.load(paths.RECIPES)
        registry = patch_registry.load_registry(paths.PATCHES)
        descriptor = registry.get(args.patch_id)
        modules = patchset.catalog(directory=paths.PATCHES)

        # PA34: the exact selection is named by its canonical identity.
        # --experiment / --focal-overlay qualify a --source selection; the
        # no-source path evaluates the focal's own dependency closure.
        experiment = getattr(args, "experiment", None)
        focal_overlay = bool(getattr(args, "focal_overlay", False))
        if (experiment is not None or focal_overlay) and not args.source:
            raise ValueError("--experiment/--focal-overlay require --source")
        selection = None
        if args.source:
            if args.source not in cfg.sources:
                raise ValueError(f"unknown source {args.source!r}")
            selection = campaign_resolution.resolve_canonical_selection(
                args.source,
                cfg,
                modules,
                catalog_directory=paths.PATCHES,
                experiment=experiment,
                focal_overlay_patch_id=args.patch_id if focal_overlay else None,
            )
            selected_ids = selection.patch_ids
            if args.patch_id not in selected_ids:
                # NOT_EVALUATED (PA34): the focal patch is not in the
                # resolved selector -- no gate runs; this is an explicit
                # selection-status result, not a resolution error.
                payload = {
                    "schema_version": 1,
                    "patch_id": args.patch_id,
                    "intent": intent.value,
                    "source": args.source,
                    "selection_status": "NOT_EVALUATED",
                    "selector": selection.identity.to_payload(),
                    "composition": [],
                    "gates": [],
                }
                if args.json:
                    print(json.dumps(payload, indent=2, sort_keys=True))
                else:
                    print(
                        f"patch-gates: {args.patch_id} ({intent.value}) "
                        f"NOT_EVALUATED -- not selected by "
                        f"{selection.identity.selector_name!r}"
                    )
                return 1
        else:
            if intent in (patch_gates.GateIntent.BUILD, patch_gates.GateIntent.REBASE):
                raise ValueError(f"--source is required for {intent.value}")
            selected_ids = patchset.expand_composition(
                (args.patch_id,),
                directory=paths.PATCHES,
            ).expanded
        composition = patchset.resolve_exact(
            tuple(selected_ids),
            directory=paths.PATCHES,
            allow_rejected=False,
        )
    except (
        campaign_config.ConfigError,
        campaign_resolution.ResolutionError,
        patch_registry.PatchRegistryError,
        ValueError,
        OSError,
    ) as exc:
        print(f"patch-gates: cannot resolve composition: {exc}", file=sys.stderr)
        return 2

    source_root = paths.llama_root(getattr(args, "llama_root", None))
    try:
        repository = UpstreamRepository(source_root)
        resolved_base_revision = repository.resolve_ref(cfg.pinned)
        target_revision = repository.resolve_ref("HEAD")
    except (WorkspaceError, OSError, ValueError) as exc:
        print(
            f"patch-gates: cannot resolve pin {cfg.pinned!r} locally: {exc}",
            file=sys.stderr,
        )
        return 2

    def load_report(path_value: str | None, label: str) -> dict[str, object] | None:
        if path_value is None:
            return None
        try:
            report = patch_rebase.load_report(Path(path_value))
        except (patch_rebase.RebaseCheckError, OSError, ValueError) as exc:
            raise ValueError(f"{label} is not a readable report: {exc}") from exc
        if not isinstance(report, dict):
            raise ValueError(f"{label} must contain a JSON object")
        return report

    try:
        rebase_report = load_report(args.rebase_report, "--rebase-report")
        coverage_report = load_report(args.all_report, "--all-report")
    except ValueError as exc:
        print(f"patch-gates: {exc}", file=sys.stderr)
        return 2

    source_cfg = cfg.sources.get(args.source) if args.source else None
    backend = (
        source_cfg.backend
        if source_cfg is not None
        else descriptor.backend or "agnostic"
    )
    context = patch_gates.GateContext(
        descriptor=descriptor,
        composition=composition,
        pinned_ref=cfg.pinned,
        intent=intent,
        patch_context=patch_catalog.PatchContext(
            backend=backend,
            source=args.source,
        ),
        patches_dir=paths.PATCHES,
        resolved_base_revision=resolved_base_revision,
        catalog_path=paths.PATCH_CATALOG,
        evidence_root=None,
        external_sources_path=paths.EXTERNAL_SOURCES,
        validation_baseline_path=paths.VALIDATION_PACKAGE_GRANDFATHER,
        dispositions_dir=paths.DISPOSITIONS,
        source_root=source_root,
        expected_selector=selection.identity if selection is not None else None,
        rebase_report=rebase_report,
        allow_legacy_grandfather=not args.no_legacy_grandfather,
        catalog_states={module.patch_id: module.state for module in modules},
        coverage_report=coverage_report,
        recipe_patch_ids=(
            frozenset(module.patch_id for module in composition.modules)
            if intent in (patch_gates.GateIntent.BUILD, patch_gates.GateIntent.REBASE)
            else None
        ),
        target_revision=target_revision,
    )
    results = patch_gates.evaluate_patch_gates(context)

    # PA34 (dev-gpt-agent req_41a3133e657340c7 Q1/P2#5): the successful path
    # reports its selection status and the selector it was evaluated under.
    # A named selection carries its canonical identity; the no-source path
    # carries the rebase report's selector (the only selector in evidence).
    selector_payload = (
        selection.identity.to_payload()
        if selection is not None
        else (rebase_report.get("selector") if rebase_report is not None else None)
    )
    payload = {
        "schema_version": 1,
        "patch_id": args.patch_id,
        "intent": intent.value,
        "source": args.source,
        "selection_status": "EVALUATED",
        "selector": selector_payload,
        "composition": [module.patch_id for module in composition.modules],
        "gates": [
            {
                "id": result.id.value,
                "status": result.status.value,
                "phase": result.phase,
                "authority": result.authority,
                "detail": list(result.detail),
            }
            for result in results
        ],
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        source_label = args.source or "focal patch dependency closure"
        print(f"patch-gates: {args.patch_id} ({intent.value})")
        print(f"source: {source_label}")
        print("composition: " + ", ".join(payload["composition"]))
        for result in payload["gates"]:
            suffix = " -- " + "; ".join(result["detail"]) if result["detail"] else ""
            print(
                f"  {result['id']}: {result['status']} ({result['authority']}){suffix}"
            )
        overall = (
            "PASS"
            if all(
                result.status
                in (patch_gates.GateStatus.PASS, patch_gates.GateStatus.NA)
                for result in results
            )
            else "FAIL"
        )
        print("RESULT: " + overall)

    return (
        0
        if all(
            result.status in (patch_gates.GateStatus.PASS, patch_gates.GateStatus.NA)
            for result in results
        )
        else 1
    )


def cmd_patch_doc(args: Namespace) -> int:
    """Merge the selected patches' SUMMARY.md into one release doc.

    Reuses patch-rebase-check's exact selection logic (--source NAME, or
    --all) so "what's in this doc" always matches "what's in this build" --
    no separate selection language to drift out of sync.
    """
    root = paths.llama_root(args.llama_root)
    source_name = getattr(args, "source", None)
    all_patches = bool(getattr(args, "all_patches", False))
    try:
        patch_ids = patch_rebase._selection_patch_ids(
            source_name=source_name,
            all_patches=all_patches,
        )
    except patch_rebase.RebaseCheckError as exc:
        print(f"patch-doc: {exc}", file=sys.stderr)
        return 2

    upstream_revision = patch_rebase._git(root, "rev-parse", "HEAD")
    bigcherry_revision = patch_rebase._git(paths.REPO_ROOT, "rev-parse", "HEAD")
    pin_info = {
        "llama.cpp revision": upstream_revision,
        "bigcherry revision": bigcherry_revision,
    }
    selection_label = f"--source {source_name}" if source_name else "--all"

    try:
        doc = patch_docs.render_patch_selection_doc(
            patch_ids=patch_ids,
            pin_info=pin_info,
            selection_label=selection_label,
        )
    except patch_docs.PatchDocError as exc:
        print(f"patch-doc: {exc}", file=sys.stderr)
        return 2

    if args.out:
        Path(args.out).write_text(doc, encoding="utf-8")
        print(f"release doc: {args.out} ({len(patch_ids)} patch(es))")
    else:
        print(doc)
    return 0


def cmd_patches(args: Namespace) -> int:
    """Show every patch, its metadata, and whether a selection takes it.

    --kind/--backend/--origin filter against patch metadata: catalog.toml
    for legacy flat patches (RE30 phase 1's declarative metadata), and
    patch.toml for packaged patches (patch-system PA02: patches/ may now
    hold <id>/ package directories -- the metadata, not a directory move,
    answers "which patches form the framework / are HIP vs Vulkan / came
    from an external fork").

    RE39: reads patches/ and patches/catalog.toml exactly once via a single
    CatalogSnapshot, instead of the two independent scans (patchset.describe()
    + patch_catalog.load_catalog()) this command used to make.
    """
    try:
        snapshot = patch_catalog.build_snapshot()
    except ValueError as exc:
        print(f"patches: could not load patches/catalog.toml: {exc}", file=sys.stderr)
        return 2
    if not snapshot.modules:
        print("no patches found", file=sys.stderr)
        return 1

    try:
        selection = patch_selection.resolve_cli_selection(args)
    except patch_selection.SelectionError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    catalog_filter_active = bool(args.kind or args.backend or args.origin)

    root = paths.llama_root(args.llama_root)
    print(f"selection: {selection.label}")
    print(f"checkout:  {root}")
    if catalog_filter_active:
        print(
            f"catalog:   kind={args.kind or 'any'} backend={args.backend or 'any'} "
            f"origin={args.origin or 'any'}"
        )
    print()

    rows, problems, selected = [], [], 0
    for module in snapshot.modules:
        entry = snapshot.entry_for(module.patch_id)
        if catalog_filter_active:
            if entry is None:
                continue
            if args.kind and entry.kind != args.kind:
                continue
            if args.backend and entry.backend != args.backend:
                continue
            if args.origin and entry.origin != args.origin:
                continue

        taken = selection.matches(module)
        selected += taken

        note = ""
        if module.upstream:
            landed = patchset.upstream_landed(module.upstream, root)
            if landed:
                note = f"upstream {module.upstream[:8]} landed -- redundant here"
            elif landed is None:
                note = f"upstream {module.upstream[:8]} unknown"
            else:
                note = f"upstream {module.upstream[:8]} not in this checkout"

        if module.state not in patchset.STATES:
            problems.append(
                f"{module.patch_id}: STATE={module.state!r} is not one of "
                f"{', '.join(patchset.STATES)}"
            )

        catalog_label = f"{entry.kind}/{entry.backend}" if entry is not None else ""
        rows.append(
            (
                "[x]" if taken else "[ ]",
                module.patch_id,
                ", ".join(module.tags) or "-",
                module.state,
                catalog_label,
                note,
            )
        )

    if not rows:
        print("no patches match the given --kind/--backend/--origin filter")
        return 0

    widths = [max(len(r[i]) for r in rows) for i in range(5)]
    for mark, name, tags, state, catalog_label, note in rows:
        line = (
            f"{mark} {name:<{widths[1]}}  {tags:<{widths[2]}}  "
            f"{state:<{widths[3]}}  {catalog_label:<{widths[4]}}"
        )
        print(f"{line}  {note}".rstrip())

    print(
        f"\n{selected} of {len(rows)} shown selected"
        + (
            ""
            if not catalog_filter_active
            else f" ({len(snapshot.modules)} total in catalog)"
        )
    )
    for problem in problems:
        print(f"warning: {problem}", file=sys.stderr)
    return 1 if problems else 0

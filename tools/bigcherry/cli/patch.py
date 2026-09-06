"""CLI presentation handlers for patch lifecycle commands."""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

from ..core import paths
from ..patch import catalog as patch_catalog
from ..patch import disposition as patch_disposition
from ..patch import lifecycle as patch_lifecycle
from ..patch import patchset
from ..patch import selection as patch_selection
from ..patch import rebase as patch_rebase
from ..patch import docs as patch_docs
from ..patch import validation_policy as patch_validation_policy

DISPOSITIONS_DIR = paths.DISPOSITIONS


def cmd_apply(args: Namespace) -> int:
    from .. import __main__ as legacy

    root = paths.llama_root(args.llama_root)
    report_path = getattr(args, "rebase_report", None)
    known_good = bool(getattr(args, "known_good", False))

    if bool(report_path) != known_good:
        print(
            "apply: --rebase-report PATH and --known-good must be supplied together",
            file=sys.stderr,
        )
        return 2

    if report_path:
        if getattr(args, "source", None):
            print(
                "apply: --rebase-report owns the exact logical selection; "
                "do not combine it with --source",
                file=sys.stderr,
            )
            return 2
        try:
            result = patch_rebase.apply_known_good(
                root, Path(report_path),
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

    ok = legacy._apply_exact_selection(
        root, selection, force=args.force, dry_run=args.dry_run
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
    -- cross_check() is called with verify_validation_evidence=False so the
    HI83 dynamic branch (which silently skips packaged RD patches, since it
    only checks catalog.toml entries) never runs here."""
    problems = list(
        patch_catalog.cross_check(
            verify_validation_evidence=False, allow_legacy_grandfather=True
        )
    )
    package_report = patch_validation_policy.check_validation_packages()
    problems.extend(package_report.problems)
    if args.json:
        print(
            json.dumps(
                {
                    "passed": not problems,
                    "problems": problems,
                    "grandfathered": list(package_report.grandfathered),
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        for problem in problems:
            print(problem, file=sys.stderr)
        for patch_id in package_report.grandfathered:
            print(f"{patch_id}: structurally grandfathered (non-current, not failing)", file=sys.stderr)
    return 0 if not problems else 1


def cmd_patch_disposition(args: Namespace) -> int:
    """HI152: record/list/clear a revision-bound known_broken disposition
    for a non-recipe patch -- see patch/disposition.py's module docstring.
    Never a standing waiver: it stops applying the instant target_revision
    or patch_digest changes."""
    action = args.disposition_action
    if action == "set":
        record = patch_disposition.Disposition(
            patch_id=args.patch_id, target_revision=args.revision,
            patch_digest=args.digest, disposition="known_broken",
            failure_status=args.failure_status, reason=args.reason,
            owner=args.owner, tracking_item=args.tracking_item,
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
        print(f"disposition cleared: {args.patch_id}" if removed
              else f"no disposition on file for {args.patch_id}")
        return 0
    # action == "list"
    records = patch_disposition.list_dispositions(DISPOSITIONS_DIR)
    if args.json:
        print(json.dumps(
            {pid: r.__dict__ for pid, r in sorted(records.items())}, indent=2, sort_keys=True,
        ))
        return 0
    if not records:
        print("no dispositions on file")
        return 0
    for pid, record in sorted(records.items()):
        print(f"{pid}: revision={record.target_revision[:12]} digest={record.patch_digest[:12]} "
              f"owner={record.owner} tracking_item={record.tracking_item} reason={record.reason!r}")
    return 0


def cmd_patch_verify_evidence(args: Namespace) -> int:
    """Report current validation evidence for selected patches."""
    from ..core import config as campaign_config
    from ..core.context import ProjectContext
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
        mirror = ProjectContext.resolve(work_root=os.environ.get("BC_CACHE")).upstream_repo
        upstream = mirror if mirror.exists() else paths.llama_root()
        resolved_base_revision = UpstreamRepository(upstream).resolve_ref(cfg.pinned)
    except WorkspaceError as exc:
        print(f"patch-verify-evidence: cannot resolve pin {cfg.pinned!r} locally: {exc}", file=sys.stderr)
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
    """Verify existing evidence; hardware campaigns remain explicit."""
    return cmd_patch_verify_evidence(args)


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
            source_name=source_name, all_patches=all_patches,
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
            patch_ids=patch_ids, pin_info=pin_info, selection_label=selection_label,
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
                module.group,
                module.state,
                catalog_label,
                note,
            )
        )

    if not rows:
        print("no patches match the given --kind/--backend/--origin filter")
        return 0

    widths = [max(len(r[i]) for r in rows) for i in range(5)]
    for mark, name, group, state, catalog_label, note in rows:
        line = (
            f"{mark} {name:<{widths[1]}}  {group:<{widths[2]}}  "
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

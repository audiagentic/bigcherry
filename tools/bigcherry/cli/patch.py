"""CLI presentation handlers for patch lifecycle commands."""

from __future__ import annotations

import json
import sys
from argparse import Namespace

from .. import paths, recipes
from ..patch import catalog as patch_catalog
from ..patch import lifecycle as patch_lifecycle
from ..patch import patchset


def cmd_apply(args: Namespace) -> int:
    from .. import __main__ as legacy

    root = paths.llama_root(args.llama_root)
    try:
        groups, states, label = legacy._resolve_selection(args)
    except recipes.RecipeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    ok = legacy._apply_selection(
        root, groups, states, force=args.force, dry_run=args.dry_run
    )
    print(f"selection: {label}")
    print("  RESULT: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def cmd_patches(args: Namespace) -> int:
    from .. import __main__ as legacy

    try:
        snapshot = patch_catalog.build_snapshot()
    except ValueError as exc:
        print(f"patches: could not load patches/catalog.toml: {exc}", file=sys.stderr)
        return 2
    if not snapshot.modules:
        print("no patches found", file=sys.stderr)
        return 1
    try:
        groups, states, label = legacy._resolve_selection(args)
    except recipes.RecipeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    filtered = bool(args.kind or args.backend or args.origin)
    root = paths.llama_root(args.llama_root)
    print(f"selection: {label}\ncheckout:  {root}")
    if filtered:
        print(
            f"catalog:   kind={args.kind or 'any'} backend={args.backend or 'any'} origin={args.origin or 'any'}"
        )
    print()
    rows, problems, selected = [], [], 0
    for module in snapshot.modules:
        entry = snapshot.entry_for(module.patch_id)
        if filtered and (
            entry is None
            or (args.kind and entry.kind != args.kind)
            or (args.backend and entry.backend != args.backend)
            or (args.origin and entry.origin != args.origin)
        ):
            continue
        taken = (groups is None or module.group in groups) and (
            states is None or module.state in states
        )
        selected += taken
        note = ""
        if module.upstream:
            landed = patchset.upstream_landed(module.upstream, root)
            note = (
                f"upstream {module.upstream[:8]} landed -- redundant here"
                if landed
                else f"upstream {module.upstream[:8]} unknown"
                if landed is None
                else f"upstream {module.upstream[:8]} not in this checkout"
            )
        if module.state not in patchset.STATES:
            problems.append(
                f"{module.patch_id}: STATE={module.state!r} is not one of {', '.join(patchset.STATES)} -- no recipe will select it"
            )
        label_value = f"{entry.kind}/{entry.backend}" if entry is not None else ""
        rows.append(
            (
                "[x]" if taken else "[ ]",
                module.patch_id,
                module.group,
                module.state,
                label_value,
                note,
            )
        )
    if not rows:
        print("no patches match the given --kind/--backend/--origin filter")
        return 0
    widths = [max(len(row[i]) for row in rows) for i in range(5)]
    for mark, name, group, state, catalog_label, note in rows:
        print(
            f"{mark} {name:<{widths[1]}}  {group:<{widths[2]}}  {state:<{widths[3]}}  {catalog_label:<{widths[4]}}  {note}".rstrip()
        )
    print(
        f"\n{selected} of {len(rows)} shown selected"
        + ("" if not filtered else f" ({len(snapshot.modules)} total in catalog)")
    )
    for problem in problems:
        print(f"warning: {problem}", file=sys.stderr)
    return 1 if problems else 0


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
    from .. import config as campaign_config

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
    """Run the non-mutating catalog/package lint gate."""
    problems = patch_catalog.cross_check(allow_legacy_grandfather=True)
    if args.json:
        print(
            json.dumps(
                {"passed": not problems, "problems": problems}, indent=2, sort_keys=True
            )
        )
    else:
        for problem in problems:
            print(problem, file=sys.stderr)
    return 0 if not problems else 1


def cmd_patch_verify_evidence(args: Namespace) -> int:
    """Report current validation evidence for selected patches."""
    from .. import config as campaign_config

    cfg = campaign_config.load(paths.RECIPES)
    modules = patchset.catalog()
    if args.patch_id is not None:
        modules = [module for module in modules if module.patch_id == args.patch_id]
        if not modules:
            print(f"unknown patch {args.patch_id!r}", file=sys.stderr)
            return 1
    statuses = patch_catalog.validation_evidence_statuses(
        [module.patch_id for module in modules],
        pinned_ref=cfg.pinned,
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
    from .. import __main__ as legacy

    try:
        snapshot = patch_catalog.build_snapshot()
    except ValueError as exc:
        print(f"patches: could not load patches/catalog.toml: {exc}", file=sys.stderr)
        return 2
    if not snapshot.modules:
        print("no patches found", file=sys.stderr)
        return 1

    try:
        groups, states, label = legacy._resolve_selection(args)
    except recipes.RecipeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    catalog_filter_active = bool(args.kind or args.backend or args.origin)

    root = paths.llama_root(args.llama_root)
    print(f"selection: {label}")
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

        taken = (groups is None or module.group in groups) and (
            states is None or module.state in states
        )
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
                f"{', '.join(patchset.STATES)} -- no recipe will select it"
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

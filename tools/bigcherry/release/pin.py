"""Canonical release/pin command handlers.

TR03 promotes the pin CLI surface without changing parser contracts or the
underlying pin-status and transition state machines. The compatibility names
in ``bigcherry.__main__`` continue to delegate here during migration.
"""

from __future__ import annotations

import argparse
import json
import sys

from ..core import config as campaign_config
from ..core import paths
from ..release import pin_status
from .. import pin_transition
from .. import recipes
from ..source import sources, upstream


def resolve_pin_sha(ref: str) -> str:
    """Resolve a pin ref to a full commit SHA in the vendor clone."""
    root = paths.llama_root()
    if not (root / ".git").exists():
        raise upstream.UpstreamError(
            f"vendor checkout missing at {root}; run 'python -m bigcherry pull' "
            "before repinning"
        )
    checkout_ref = upstream.ensure_ref(root, ref)
    resolved = ref if checkout_ref == ref else checkout_ref
    return upstream._git(root, "rev-parse", f"{resolved}^{{commit}}").strip()


def cmd_repin(args: argparse.Namespace) -> int:
    """Move the pin and declare the transition without changing semantics."""
    try:
        target = args.ref or upstream.latest_release()
    except upstream.UpstreamError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    try:
        old = recipes.repin(target)
    except recipes.RecipeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if old == target:
        print(f"already pinned to {target}")
        return 0

    try:
        from_sha = resolve_pin_sha(old)
        to_sha = resolve_pin_sha(target)
        declaring = upstream._git(paths.REPO_ROOT, "rev-parse", "HEAD")
    except upstream.UpstreamError as exc:
        print(
            f"pin moved ({old} -> {target}) but the transition marker "
            f"could not be resolved: {exc}",
            file=sys.stderr,
        )
        print(
            "restore the previous pin or re-run repin; a pin move without "
            "a marker reads as drift in pin-status.",
            file=sys.stderr,
        )
        return 1

    pin_transition.write(from_sha, to_sha, target, declaring)
    print(f"pinned: {old} -> {target}")
    print(
        "recipes following the pin now build from it; recipes naming their "
        "own ref are unchanged."
    )
    print(f"transition marker: {pin_transition.MARKER_PATH} ({old} -> {target})")

    try:
        report = sources.baseline_candidates_at_pin(target)
        sources.print_baseline_candidates(report)
    except Exception as exc:  # noqa: BLE001 -- advisory report is non-fatal
        print(
            f"ancestry gate: report unavailable: {exc}; "
            "pin move remains valid and no commits are asserted redundant",
            file=sys.stderr,
        )

    print(
        "next: COMMIT recipes.toml + the marker together, then "
        "python -m bigcherry pull --recipe <name>"
    )
    return 0


def pin_status_paths() -> tuple[pin_status.RepoPaths, object, list]:
    """Build local RepoPaths and configured remote trees from real config."""
    cfg = campaign_config.load(paths.RECIPES)
    repo_paths = pin_status.RepoPaths(
        repo_root=paths.REPO_ROOT,
        llama_root=paths.llama_root(),
        releases_dir=paths.REPO_ROOT / "releases",
        artifacts_dir=paths.REPO_ROOT / "artifacts",
    )
    return repo_paths, cfg, list(cfg.trees)


def render_pin_status(report, trees) -> str:
    """Render the established human-readable pin-status report."""
    lines: list[str] = []
    local = report.local
    lines.append("pin-status: local")
    pinned = (
        f"{local.pinned_ref} -> {local.pinned_sha[:12]}"
        if local.pinned_sha
        else (local.pinned_ref or "(none)")
    )
    lines.append(f"  pin           {pinned}")
    vendor = f"{local.vendor_head[:12]}" if local.vendor_head else "(absent)"
    if local.vendor_tags:
        vendor += f" (tag {', '.join(local.vendor_tags)})"
    if local.vendor_head:
        vendor += (
            f"  [{local.vendor_modified} modified, {local.vendor_untracked} untracked]"
        )
    lines.append(f"  vendor HEAD   {vendor}")
    if local.records:
        recs = ", ".join(
            f"{revision[:12]} {stage}"
            for revision, stage in sorted(
                local.records.items(), key=lambda item: item[0]
            )
        )
        lines.append(f"  releases      {recs}")
    if local.descriptors:
        desc = ", ".join(f"{descriptor[:12]}" for descriptor in local.descriptors)
        lines.append(f"  descriptors   {desc}")
    lines.append(f"  bigcherry     {(local.bigcherry_head or '?')[:12]}")
    if local.marker:
        lines.append(
            f"  marker        {local.marker.from_sha[:12]} -> "
            f"{local.marker.to_sha[:12]} ({local.marker.tag}) "
            f"[{local.marker_state}]"
        )
    lines.append(
        f"  VERDICT       {local.verdict}"
        + (f"  ({'; '.join(local.reasons)})" if local.reasons else "")
    )

    for status in report.remotes:
        lines.append(f"\npin-status: {status.name} (remote)")
        if not status.reachable:
            lines.append(f"  VERDICT       unreachable  ({'; '.join(status.reasons)})")
            continue
        lines.append(f"  vendor HEAD   {(status.vendor_head or 'none')[:12]}")
        lines.append(
            f"  pin           {status.pinned_ref or 'none'} "
            f"-> {(status.pinned_sha or '?')[:12]}"
        )
        lines.append(f"  bigcherry     {(status.bigcherry_head or 'none')[:12]}")
        lines.append(
            f"  VERDICT       {status.verdict}"
            + (f"  ({'; '.join(status.reasons)})" if status.reasons else "")
        )

    if report.remotes:
        if report.converged == True:  # noqa: E712 -- tri-state: None = no remotes
            lines.append("\nAGGREGATE  converged")
        elif report.converged == False:  # noqa: E712 -- tri-state: None = no remotes
            lines.append(
                f"\nAGGREGATE  DIVERGED  ({'; '.join(report.aggregate_reasons)})"
            )
    return "\n".join(lines)


def cmd_pin_status(args: argparse.Namespace) -> int:
    """Name the pin state; this command only reads."""
    try:
        repo_paths, _cfg, trees = pin_status_paths()
    except Exception as exc:  # ConfigError, RecipeError, OSError
        print(f"pin-status: config error: {exc}", file=sys.stderr)
        return 2

    if args.remote:
        trees = [tree for tree in trees if tree.name == args.remote]
        if not trees:
            print(
                f"pin-status: no configured tree named {args.remote!r}",
                file=sys.stderr,
            )
            return 2
    elif not args.all_remotes:
        trees = []

    try:
        report = pin_status.build_report(repo_paths, tuple(trees))
    except Exception as exc:
        print(f"pin-status: {exc}", file=sys.stderr)
        return 1

    if args.json:
        document = {
            "local": {
                "pinned_ref": report.local.pinned_ref,
                "pinned_sha": report.local.pinned_sha,
                "vendor_head": report.local.vendor_head,
                "vendor_tags": list(report.local.vendor_tags),
                "vendor_modified": report.local.vendor_modified,
                "vendor_untracked": report.local.vendor_untracked,
                "marker": report.local.marker.to_json()
                if report.local.marker
                else None,
                "marker_state": report.local.marker_state,
                "records": report.local.records,
                "descriptors": list(report.local.descriptors),
                "bigcherry_head": report.local.bigcherry_head,
                "verdict": report.local.verdict,
                "reasons": list(report.local.reasons),
            },
            "remotes": [
                {
                    "name": status.name,
                    "reachable": status.reachable,
                    "vendor_head": status.vendor_head,
                    "pinned_ref": status.pinned_ref,
                    "pinned_sha": status.pinned_sha,
                    "bigcherry_head": status.bigcherry_head,
                    "verdict": status.verdict,
                    "reasons": list(status.reasons),
                }
                for status in report.remotes
            ],
            "converged": report.converged,
            "aggregate_reasons": list(report.aggregate_reasons),
        }
        print(json.dumps(document, indent=2, sort_keys=True))
    else:
        print(render_pin_status(report, trees))

    if args.complete:
        failures = pin_status.complete_failures(report, tuple(trees))
        if not args.json:
            print(f"\nCOMPLETION  {'PASS' if not failures else 'FAIL'}")
            for reason in failures:
                print(f"  - {reason}")
        return 1 if failures else 0

    if args.strict:
        failures = pin_status.strict_failure(report)
        if failures:
            for reason in failures:
                print(f"pin-status --strict FAIL: {reason}", file=sys.stderr)
            return 1
        if report.local.verdict == pin_status.VERDICT_MID_REBASE:
            print(
                "WARNING: tree is mid-rebase (a declared bump is in flight); "
                "proceeding only because the pipeline's source identity is "
                "revision-bound.",
                file=sys.stderr,
            )
        return 0
    return 0

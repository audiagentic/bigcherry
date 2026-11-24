"""`bigcherry run new|status` and `bigcherry clean`: the managed work root (WSL01, WSL03 skeleton).

Both are platform commands (no --engine): the work root is shared by every engine. run cleanup stays dry-run only; `clean --builds --apply` removes eligible build caches.
"""

from __future__ import annotations

import json
import math
import os
import sys
from argparse import Namespace
from pathlib import Path

from ..core import paths, retention, workspace
from ..core import run_manifest as rm


def work_root(override: str | None) -> Path:
    return Path(override or os.environ.get("BC_WORK_ROOT") or paths.primary_root() / "work").expanduser().resolve()


def _blocked(root: Path) -> str | None:
    """A condition that must stop any mutation: a maintenance lock or a live runner lease under this work root."""
    from ..core import tree_activity

    try:
        across_worktrees = tree_activity.cleanup_activity_blocker(root)
        if across_worktrees:
            return across_worktrees
        if tree_activity.maintenance_is_held(root, paths.REPO_ROOT):
            return "a maintenance lock is held"
        live = tree_activity.list_active_leases(root, paths.REPO_ROOT)
    except OSError as exc:
        return f"the runner leases could not be read ({exc})"
    return f"{len(live)} runner lease(s) are active" if live else None


def cmd_run_new(args: Namespace) -> int:
    root = work_root(args.work_root)
    try:
        h = workspace.create_run(root, args.kind, args.topic, plan_item=args.plan, engine=args.run_engine or "none",
                                 argv=sys.argv[:], purpose=args.purpose or "", repo=paths.REPO_ROOT)
    except (workspace.WorkspaceError, rm.ManifestError) as exc:
        print(f"run new: ERROR: {exc}", file=sys.stderr)
        return 2
    print(h.path)
    return 0


def _find(root: Path, run_id: str):
    # Diagnostic lookup must still resolve corrupt nested manifests so status
    # can report their actual error. Include adopted flat runs, not unadopted.
    hits = [e.path for e in retention.inspect_workspace(root, measure=False).runs
            if e.run_id.startswith(run_id) and
            (e.kind != "legacy" or e.manifest is not None)]
    return hits


def cmd_run_status(args: Namespace) -> int:
    root = work_root(args.work_root)
    inv = retention.inspect_workspace(root, measure=False)
    if args.run_id:
        hits = _find(root, args.run_id)
        if len(hits) != 1:
            print(f"run status: {len(hits)} runs match {args.run_id!r}", file=sys.stderr)
            return 2
        try:
            print(json.dumps(rm.read_manifest(hits[0]), indent=1, sort_keys=True))
        except rm.ManifestError as exc:
            print(f"run status: {exc}", file=sys.stderr)
            return 1
        return 0
    for e in inv.runs:
        m = e.manifest
        state = f"{m['state']}/{m['harvest']['state']}" if m else "unmanaged"
        print(f"{e.kind:8} {state:22} {e.run_id}")
    print(f"{len(inv.runs)} run(s); {len(inv.unknown)} path(s) outside the layout", file=sys.stderr)
    return 0


def cmd_run_harvest(args: Namespace) -> int:
    from ..core import harvest

    root = work_root(args.work_root)
    hits = _find(root, args.run_id)
    if len(hits) != 1:
        print(f"run harvest: {len(hits)} runs match {args.run_id!r}", file=sys.stderr)
        return 2
    why = _blocked(root)
    if why:
        print(f"run harvest: refused: {why}", file=sys.stderr)
        return 3
    try:
        doc = harvest.harvest(root, hits[0])
    except (harvest.HarvestError, rm.ManifestError) as exc:
        print(f"run harvest: ERROR: {exc}", file=sys.stderr)
        return 1
    flag = " (diagnostic only)" if doc["diagnostic_only"] else ""
    print(f"harvested {doc['run_id']}{flag}: {len(doc['files'])} file(s) kept, {doc['bulk_bytes']} byte(s) of bulk listed "
          f"-> {harvest.bundle_dir(root, doc['run_id'])}")
    return 0


def _one_run(args: Namespace):
    root = work_root(args.work_root)
    hits = _find(root, args.run_id)
    if len(hits) != 1:
        print(f"{args.run_action}: {len(hits)} runs match {args.run_id!r}", file=sys.stderr)
        return root, None
    return root, hits[0]


def _cmd_quarantine_family(args: Namespace) -> int:
    from ..core import quarantine

    root, run = _one_run(args)
    if run is None:
        return 2
    blocked = _blocked(root)
    try:
        if args.run_action == "quarantine":
            policy = retention.Policy(min_age_days=args.min_age_days, newest_k=args.keep_newest)
            out = quarantine.quarantine(root, run, policy=policy, blocked=blocked, apply=args.apply)
        elif args.run_action == "restore":
            out = quarantine.restore(root, run, blocked=blocked)
        else:
            out = quarantine.purge(root, run, grace_days=args.grace_days, blocked=blocked)
    except (quarantine.QuarantineError, rm.ManifestError) as exc:
        print(f"run {args.run_action}: refused: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(out, sort_keys=True))
    return 0


def cmd_clean(args: Namespace) -> int:
    root = work_root(args.work_root)
    if not root.is_dir():
        print(f"clean: {root} does not exist", file=sys.stderr)
        return 2
    if getattr(args, "legacy", False) and getattr(args, "builds", False):
        print("clean: choose --legacy or --builds", file=sys.stderr)
        return 2
    if getattr(args, "builds", False):
        from ..core import build_gc

        if not math.isfinite(args.min_age_days) or args.min_age_days < 0 or args.keep_newest < 0:
            print("clean: retention policy must be finite and non-negative", file=sys.stderr)
            return 2
        if args.apply and args.no_size:
            print("clean: --apply requires measured sizes (omit --no-size)", file=sys.stderr)
            return 2
        policy = retention.Policy(min_age_days=args.min_age_days, newest_k=args.keep_newest)
        blocker = _blocked(root)
        entries = retention.inspect_cache_roots(root, policy=policy, measure=not args.no_size, blocked=blocker)
        outcome = None
        if args.apply:
            if blocker:
                print(f"clean --builds: refused: {blocker}", file=sys.stderr)
                return 3
            try:
                outcome = build_gc.apply(root, policy=policy, blocked_check=lambda: _blocked(root))
            except (build_gc.BuildGCError, OSError) as exc:
                print(f"clean --builds: refused: {exc}", file=sys.stderr)
                return 3
        if args.json:
            report = {"root": str(root), "read_only": not args.apply, "entries": entries}
            if outcome is not None:
                report["apply"] = outcome
            json.dump(report, sys.stdout, indent=1, sort_keys=True)
            print()
        else:
            for entry in entries:
                size = entry["size_bytes"]
                age = entry["age_days"]
                size_display = "unknown" if size is None else str(size)
                age_display = "unknown" if age is None else f"{age:.1f}d"
                print(f'{entry["kind"]:6} {entry["action"]:9} {size_display:>12} {age_display:>9} '
                      f'{entry["metadata"]:20} {entry["path"]}: {"; ".join(entry["reasons"])}')
            if outcome is None:
                print("dry run: no cache entries changed")
            else:
                print(f'build GC applied: {len(outcome["removed"])} removed, '
                      f'{len(outcome["skipped"])} skipped; record {outcome["record"]}')
                for removed in outcome["removed"]:
                    if removed.get("warning"):
                        print(f'  warning {removed["path"]}: {removed["warning"]}')
                for skipped in outcome["skipped"]:
                    print(f'  skipped {skipped["path"]}: {skipped["reason"]}')
                    if skipped.get("quarantine_path"):
                        print(f'  quarantine leftover: {skipped["quarantine_path"]}')
        return 0 if outcome is None or not outcome["skipped"] else 3
    if args.apply:
        print("clean: --apply requires --builds (run cleanup remains dry-run only)", file=sys.stderr)
        return 2
    if getattr(args, "legacy", False):
        from ..core.legacy_inventory import inventory_legacy
        report = inventory_legacy(root, measure=not args.no_size)
        if args.json:
            json.dump(report, sys.stdout, indent=1, sort_keys=True)
            print()
        else:
            for item in report["entries"]:
                print(f'{item["category"]:24} protected {item["path"]}')
            print(f'read-only: {report["totals"]["entries"]} protected items; '
                  f'{report["totals"]["unmeasured_entries"]} unmeasured or incomplete')
        return 0
    policy = retention.Policy(min_age_days=args.min_age_days, newest_k=args.keep_newest)
    plan = retention.plan_cleanup(retention.inspect_workspace(root, measure=not args.no_size), policy=policy, blocked=_blocked(root))
    if args.json:
        json.dump({"root": str(root), "blocked": plan.blocked, "totals": plan.totals(),
                   "unknown": [str(p) for p in plan.unknown],
                   "decisions": [{"path": str(d.path), "kind": d.kind, "run_id": d.run_id, "action": d.action,
                                 "size_bytes": d.size_bytes, "reasons": d.reasons} for d in plan.decisions]},
                  sys.stdout, indent=1, sort_keys=True)
        print()
    else:
        print(retention.render(plan, root, verbose=args.verbose))
    return 0


def cmd_legacy_runs(args: Namespace) -> int:
    from ..core import legacy_runs

    root = work_root(args.work_root)
    report = legacy_runs.migrate(root, apply=args.apply)
    if args.json:
        json.dump(report, sys.stdout, indent=1, sort_keys=True)
        print()
    else:
        for row in report["runs"]:
            pins = ", ".join(ref["id"] for ref in row["references"]) or "-"
            print(f'{row["status"]:12} {row["path"]}  build refs: {pins}'
                  + (f'  ERROR: {row["error"]}' if row.get("error") else ""))
        for row in report["loose_files"]:
            pins = ", ".join(ref["id"] for ref in row.get("references", [])) or "-"
            print(f'loose {row["status"]:7} {row["path"]}  build refs: {pins}'
                  + (f'  ERROR: {row["error"]}' if row.get("error") else ""))
        print(f'legacy runs: {json.dumps(report["totals"], sort_keys=True)}'
              + (" (dry run: no files changed)" if not args.apply else ""))
        if report.get("error"):
            print(report["error"], file=sys.stderr)
    return 3 if report["totals"]["errors"] else 0


def register(sub) -> None:
    run = sub.add_parser("run", help="create and inspect managed runs under the work root (run new | status | harvest | quarantine | restore | purge)")
    rsub = run.add_subparsers(dest="run_action", required=True)
    new = rsub.add_parser("new", help="create runs/<kind>/<run-id>/ with its manifest and print the path")
    new.add_argument("kind", choices=sorted(rm.NESTED_RUN_KINDS))
    new.add_argument("topic", help="short description; becomes part of the run id")
    new.add_argument("--plan", help="plan item this run belongs to (e.g. RAD12)")
    new.add_argument("--run-engine", dest="run_engine", help="engine the run is about (informational)")
    new.add_argument("--purpose", help="one line: what question the run answers")
    new.add_argument("--work-root")
    new.set_defaults(func=cmd_run_new)
    st = rsub.add_parser("status", help="list runs, or print one run's manifest (id or unique prefix)")
    st.add_argument("run_id", nargs="?")
    st.add_argument("--work-root")
    st.set_defaults(func=cmd_run_status)
    hv = rsub.add_parser("harvest", help="keep what a finished run proved: write its hash-verified evidence bundle (deletes nothing)")
    hv.add_argument("run_id", help="run id or unique prefix")
    hv.add_argument("--work-root")
    hv.set_defaults(func=cmd_run_harvest)
    qa = rsub.add_parser("quarantine", help="move a harvested candidate run's bulk into trash/ (dry run unless --apply)")
    qa.add_argument("run_id")
    qa.add_argument("--apply", action="store_true")
    qa.add_argument("--min-age-days", type=float, default=1.0)
    qa.add_argument("--keep-newest", type=int, default=0)
    qa.add_argument("--work-root")
    qa.set_defaults(func=_cmd_quarantine_family)
    rs = rsub.add_parser("restore", help="move a quarantined run's bulk back")
    rs.add_argument("run_id")
    rs.add_argument("--work-root")
    rs.set_defaults(func=_cmd_quarantine_family)
    pg = rsub.add_parser("purge", help="delete a quarantined run's journalled bulk after the grace period")
    pg.add_argument("run_id")
    pg.add_argument("--grace-days", type=float, default=7.0)
    pg.add_argument("--work-root")
    pg.set_defaults(func=_cmd_quarantine_family)

    legacy = sub.add_parser("legacy-runs", help="inspect/adopt flat historical runs (dry-run by default)")
    legacy_sub = legacy.add_subparsers(dest="legacy_action", required=True)
    migrate = legacy_sub.add_parser("migrate", help="scan every file; write missing v1 manifests only with --apply")
    migrate.add_argument("--work-root", help="the work root (default: BC_WORK_ROOT)")
    migrate.add_argument("--apply", action="store_true", help="write manifests using the run manifest lock")
    migrate.add_argument("--json", action="store_true", help="detailed per-run report")
    migrate.set_defaults(func=cmd_legacy_runs)

    cl = sub.add_parser("clean", help="dry-run report of what cleanup would reclaim and why everything else is protected")
    cl.add_argument("--work-root")
    cl.add_argument("--apply", action="store_true", help="delete eligible builds when --builds is selected")
    cl.add_argument("--min-age-days", type=float, default=1.0)
    cl.add_argument("--keep-newest", type=int, default=0, help="newest K runs of each kind or builds per recipe are kept (default 0: only age, a live process or an installed release keeps a build)")
    cl.add_argument("--legacy", action="store_true", help="read-only protected legacy work-root inventory")
    cl.add_argument("--builds", action="store_true", help="build GC candidates and protected reasons; sources remain read-only")
    cl.add_argument("--no-size", action="store_true", help="skip measuring directory sizes (fast)")
    cl.add_argument("--verbose", action="store_true")
    cl.add_argument("--json", action="store_true")
    cl.set_defaults(func=cmd_clean)

"""`bigcherry run new|status` and `bigcherry clean`: the managed work root (WSL01, WSL03 skeleton).

Both are platform commands (no --engine): the work root is shared by every engine. `clean` is a dry run only in this slice; it
reads the manifests and says what it would do and why, and refuses to delete.
"""

from __future__ import annotations

import json
import os
import sys
from argparse import Namespace
from pathlib import Path

from ..core import paths, retention, run_manifest as rm, workspace


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
    hits = [p for k in rm.RUN_KINDS for p in (root / "runs" / k).glob(run_id + "*") if p.is_dir()]
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
    if args.apply:
        print("clean: --apply is not available yet: deletion needs harvest (WSL02) and quarantine first; this is a dry run only",
              file=sys.stderr)
        return 2
    root = work_root(args.work_root)
    if not root.is_dir():
        print(f"clean: {root} does not exist", file=sys.stderr)
        return 2
    if getattr(args, "legacy", False):
        if getattr(args, "builds", False):
            print("clean: choose --legacy or --builds", file=sys.stderr)
            return 2
        from ..core import legacy_inventory
        try:
            report = legacy_inventory.inventory_legacy(
                root, measure=not args.no_size, repo_root=paths.REPO_ROOT,
                extra_roots=tuple(Path(p).expanduser().resolve() for p in getattr(args, "extra_root", None) or ()))
        except ValueError as exc:
            print(f"clean: {exc}", file=sys.stderr)
            return 2
        if args.json:
            json.dump(report, sys.stdout, indent=1, sort_keys=True)
            print()
        else:
            print(legacy_inventory.render(report))
        return 0
    if getattr(args, "builds", False):
        entries = retention.inspect_cache_roots(root, measure=not args.no_size)
        if args.json:
            json.dump({"root": str(root), "read_only": True, "entries": entries},
                      sys.stdout, indent=1, sort_keys=True)
            print()
        else:
            for entry in entries:
                size = entry["size_bytes"]
                display_size = "unknown" if size is None else str(size)
                print(f'{entry["kind"]:6} protected {display_size:>12} {entry["metadata"]:20} {entry["path"]}')
            print("read-only: every cache entry is protected; no GC candidates")
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


def register(sub) -> None:
    run = sub.add_parser("run", help="create and inspect managed runs under the work root (run new | status | harvest | quarantine | restore | purge)")
    rsub = run.add_subparsers(dest="run_action", required=True)
    new = rsub.add_parser("new", help="create runs/<kind>/<run-id>/ with its manifest and print the path")
    new.add_argument("kind", choices=sorted(rm.RUN_KINDS))
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
    qa.add_argument("--min-age-days", type=float, default=7.0)
    qa.add_argument("--keep-newest", type=int, default=3)
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

    cl = sub.add_parser("clean", help="dry-run report of what cleanup would reclaim and why everything else is protected")
    cl.add_argument("--work-root")
    cl.add_argument("--apply", action="store_true", help="not available yet")
    cl.add_argument("--min-age-days", type=float, default=7.0)
    cl.add_argument("--keep-newest", type=int, default=3, help="newest K runs of each kind are always kept")
    cl.add_argument("--legacy", action="store_true", help="read-only protected legacy work-root inventory")
    cl.add_argument("--extra-root", action="append", metavar="PATH",
                    help="with --legacy: also inventory this legacy root (repeatable; read-only)")
    cl.add_argument("--builds", action="store_true", help="read-only build/source cache inventory (all entries protected)")
    cl.add_argument("--no-size", action="store_true", help="skip measuring directory sizes (fast)")
    cl.add_argument("--verbose", action="store_true")
    cl.add_argument("--json", action="store_true")
    cl.set_defaults(func=cmd_clean)

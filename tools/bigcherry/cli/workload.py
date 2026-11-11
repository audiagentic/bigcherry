"""Read-only benchmark index commands; 'run' sends traffic to an already-running server."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..bench import workloads
from ..core.context import resolve_work_root


def cmd_workload(args: argparse.Namespace) -> int:
    try:
        specs = workloads.catalog()
        if args.workload_action == "list":
            for key, cfg in specs.items():
                print(f"{key}@{cfg['version']:<3} {cfg['description']}")
            return 0
        if args.workload_action == "compare":
            cfg = specs.get(args.id)
            if cfg is None:
                raise workloads.WorkloadError(f"unknown workload {args.id!r}")
            print(workloads.compare(workloads.read_index(args.index), args.id, cfg["version"],
                                    layout=args.layout, since=args.since))
            return 0
        if args.workload_action == "seed":
            from ..bench.seed_workloads import seed
            count = seed(args.index, args.work_root, specs)
            print(f"seeded {count} new rows into {args.index}")
            return 0
        cfg = specs.get(args.id)
        if cfg is None:
            raise workloads.WorkloadError(f"unknown workload {args.id!r}")
        if args.manifest and not args.manifest.is_file():
            raise workloads.WorkloadError(f"manifest does not exist: {args.manifest}")
        run = args.runs_root / f"{workloads.timestamp()[:10]}-{args.id}-{__import__('uuid').uuid4().hex[:8]}"
        # Read-only metadata snapshots: no power writes, GPU queries, or leases.
        info = workloads.live_info(args.url, args.work_root, args.manifest)
        before = workloads.power_state()
        metrics, details = workloads.execute(args.id, args.url, args.repo_root, cfg)
        after = workloads.power_state()
        details["power_state_changed_during_run"] = before != after
        details["runner_commit"] = workloads.git_head(args.repo_root)
        details["workload_config"] = cfg
        details["url"] = args.url
        if not info["manifest"]:
            details["metadata_warning"] = "No verified active manifest; provide --manifest for precise build/layout/options."
            print("WARNING: server manifest unverified; build/layout/policy not inferred", file=sys.stderr)
        run.mkdir(parents=True, exist_ok=False)
        row = workloads.make_row(args.id, cfg, metrics, details, info, run_folder=str(run), power=before)
        (run / "result.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
        workloads.append_index(args.index, row)
        print(f"{args.id}@{cfg['version']} {row['row_id']} -> {run}")
        for name, metric in metrics.items():
            if name != "board_watts":
                print(f"  {name}: {metric['median']:.3f} {metric['unit']}")
        print(f"index: {args.index}")
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(f"workload: {exc}", file=sys.stderr)
        return 2


def register(sub) -> None:
    parser = sub.add_parser("workload", help="versioned inference workloads and comparable indexed results")
    actions = parser.add_subparsers(dest="workload_action", required=True)
    actions.add_parser("list", help="list canonical workload ids and versions").set_defaults(func=cmd_workload)
    run = actions.add_parser("run", help="benchmark an already-running server (never starts/stops/leases a GPU)")
    run.add_argument("id")
    run.add_argument("--url", required=True, help="OpenAI-compatible server base URL")
    run.add_argument("--manifest", type=Path, help="exact manifest for the running server; auto-read from live engine cmdline otherwise")
    run.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3],
                     help="source checkout for document slices and runner commit")
    run.add_argument("--work-root", type=Path, default=resolve_work_root())
    run.add_argument("--runs-root", type=Path, default=workloads.DEFAULT_RUNS)
    run.add_argument("--index", type=Path, default=workloads.DEFAULT_INDEX)
    run.set_defaults(func=cmd_workload)
    compare = actions.add_parser("compare", help="compare indexed rows for exactly this id and version")
    compare.add_argument("id")
    compare.add_argument("--layout", help="filter to one layout")
    compare.add_argument("--since", help="inclusive YYYY-MM-DD date")
    compare.add_argument("--index", type=Path, default=workloads.DEFAULT_INDEX)
    compare.set_defaults(func=cmd_workload)
    seed = actions.add_parser("seed", help="idempotently import known historical logs (no GPU)")
    seed.add_argument("--work-root", type=Path, default=resolve_work_root())
    seed.add_argument("--index", type=Path, default=workloads.DEFAULT_INDEX)
    seed.set_defaults(func=cmd_workload)

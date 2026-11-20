"""Workload commands; 'run' sends traffic to an already-running server and publishes one run record."""
from __future__ import annotations

import argparse
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
        cfg = specs.get(args.id)
        if cfg is None:
            raise workloads.WorkloadError(f"unknown workload {args.id!r}")
        if args.manifest and not args.manifest.is_file():
            raise workloads.WorkloadError(f"manifest does not exist: {args.manifest}")
        run = args.runs_root / f"{workloads.timestamp()[:10]}-{args.id}-{__import__('uuid').uuid4().hex[:8]}"
        # Pin native manifest content while it exists. Historical paths alone
        # cannot reconstitute deleted manifests from the 2026-11-17 purge.
        info = workloads.live_info(args.url, args.work_root, args.manifest)
        if not info.get("manifest"):
            raise workloads.WorkloadError("verified live manifest required for v1 measurement")
        manifest_bytes = Path(info["manifest"]).read_bytes()
        # Verify the native owner's SHA-sealed start receipt before the first request.
        host_start = workloads.host_facts.read_owner_start(Path(info["manifest"]).parent)
        before = workloads.host_facts.power_state()
        metrics, details = workloads.execute(args.id, args.url, args.repo_root, cfg)
        after = workloads.host_facts.power_state()
        details["power_state_changed_during_run"] = before != after
        details["runner_commit"] = workloads.git_head(args.repo_root)
        details["workload_config"] = cfg
        details["url"] = args.url
        if not info["manifest"]:
            details["metadata_warning"] = "No verified active manifest; provide --manifest for precise build/layout/options."
            print("WARNING: server manifest unverified; build/layout/policy not inferred", file=sys.stderr)
        run.mkdir(parents=True, exist_ok=False)
        row = workloads.make_row(args.id, cfg, metrics, details, info, run_folder=str(run), power=before)
        import os

        env_keys = ("ROCM_PATH", "HIP_PATH", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
        digest = workloads.publish_record(
            row, manifest_bytes=manifest_bytes, work_root=args.work_root,
            argv=list(sys.argv),
            environment={name: os.environ[name] for name in env_keys if name in os.environ},
            host_snapshot=host_start,
        )
        (run / "record.sha256").write_text(digest + "\n")
        print(f"{args.id}@{cfg['version']} {row['row_id']} -> {run}")
        for name, metric in metrics.items():
            if name != "board_watts":
                print(f"  {name}: {metric['median']:.3f} {metric['unit']}")
        print(f"run-record v1: {digest}")
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(f"workload: {exc}", file=sys.stderr)
        return 2


def register(sub) -> None:
    parser = sub.add_parser("workload", help="versioned inference workloads, measured into run records")
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
    run.set_defaults(func=cmd_workload)

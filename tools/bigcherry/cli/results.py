"""Platform measurement results CLI (MRC05); no engine or lab imports."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from ..core import engines, results, run_records
from ..core.context import resolve_work_root


def _read_baseline(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        doc = json.load(stream, object_pairs_hook=run_records._pairs_no_duplicates,
                        parse_constant=lambda x: (_ for _ in ()).throw(
                            results.ResultsError(f"invalid numeric constant {x}")
                        ))
    return results.validate_baseline(doc)


def _write_baseline(path: Path, baseline: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".results-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(baseline, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _select(args, store: run_records.RunRecordStore, *, reference=None) -> list[dict]:
    identity = reference["identity"] if reference else {}
    engine = args.engine or identity.get("engine")
    layout = args.layout or identity.get("layout")
    workload = args.workload or (identity.get("workload") or {}).get("name")
    if not (engine and layout and workload):
        raise results.ResultsError("engine, layout and workload must be selected")
    if args.record:
        rows = [{"sha256": sha, "record": store.get_record(sha)} for sha in args.record]
        for row in rows:
            rec = row["record"]
            if (rec["identity"]["engine"], rec["identity"]["layout"],
                    rec["workload"]["name"]) != (engine, layout, workload):
                raise results.ResultsError("selected record is outside requested engine/layout/workload")
            if args.since and rec["recorded_at"] < args.since:
                raise results.ResultsError("selected record precedes --since")
            if args.until and rec["recorded_at"] > args.until:
                raise results.ResultsError("selected record exceeds --until")
        return sorted(rows, key=lambda r: (r["record"]["recorded_at"], r["sha256"]))
    return store.query(engine=engine, layout=layout, workload=workload,
                       since=args.since, until=args.until)


def _show_comparison(report: dict) -> None:
    print(f"baseline {report['baseline']} -> {report['launches']} launches")
    for note in report["warnings"]:
        print("WARNING:", note)
    print("provenance differences (host_facts, feature_states, kernel, config, heat, policy, power):")
    differences = report["provenance_differences"]
    for category in ("host_facts", "feature_states", "kernel", "config", "heat",
                     "policy", "power", "inputs", "environment", "argv", "build"):
        changes = [change for change in differences if change["category"] == category]
        print(f"  {category}:")
        if not changes:
            print("    (none)")
        for change in changes:
            before = json.dumps(change["baseline"], sort_keys=True)
            after = json.dumps(change["now"], sort_keys=True)
            print(f"    {change['path']}: {before} -> {after}")
    print(f"{'metric':30s} {'baseline':>12s} {'now':>12s} {'change%':>10s} {'allow%':>10s}  verdict")
    for name, m in report["metrics"].items():
        def fmt(n, pattern=".4g"):
            return "-" if n is None else format(n, pattern)
        tolerance = m["tolerance"] * 100 if m["tolerance"] is not None else None
        print(f"{name:30s} {fmt(m['baseline']):>12s} {fmt(m['now']):>12s} "
              f"{fmt(m['change_percent'], '+.2f'):>10s} {fmt(tolerance, '.2f'):>10s}  {m['verdict']}")
    print("FAIL" if report["failed"] else "PASS")


def _show_history(report: dict) -> None:
    print(f"{report['engine']} / {report['layout']} / {report['workload']}: {report['metric']}")
    print(f"{'UTC':20s} {'launch':18s} {'value':>12s} {'baseline drift':>15s} {'trailing drift':>15s}")
    for point in report["points"]:
        def flag(value):
            return "-" if value is None else ("DRIFT" if value else "ok")
        print(f"{point['recorded_at']:20s} {point['run_id']:18s} "
              f"{point['value']:12.4g} {flag(point['baseline_drift']):>15s} "
              f"{flag(point['trailing_drift']):>15s}")


def cmd_results(args: argparse.Namespace) -> int:
    try:
        store = run_records.RunRecordStore(args.work_root)
        if args.results_action == "list":
            from ..bench import workloads
            rows = []
            for item in store.query(engine=args.engine, layout=args.layout,
                                    workload=args.workload, since=args.since, until=args.until):
                rec = item["record"]
                rows.append({
                    "status": rec.get("validity", {}).get("state", "current"),
                    "sha256": item["sha256"], "run_id": rec["run_id"],
                    "engine": rec["identity"]["engine"],
                    "layout": rec["identity"]["layout"],
                    "workload": rec["workload"]["name"],
                    "recorded_at": rec["recorded_at"],
                })
            # Explicit read-only bridge for the one retired Radiance index.
            # Do not recreate a manifest, promote its metrics or import it into
            # the verified record store.
            if args.engine in (None, engines.RADIANCE):
                index = resolve_work_root(args.work_root) / "results/index.jsonl"
                for legacy in workloads.read_index(index):
                    if args.layout and legacy.get("layout") != args.layout:
                        continue
                    if args.workload and legacy.get("workload_id") != args.workload:
                        continue
                    stamp = str(legacy.get("timestamp", "")).replace("+00:00", "Z")
                    if args.since and stamp < args.since:
                        continue
                    if args.until and stamp > args.until:
                        continue
                    rows.append({
                        "status": "legacy",
                        "reason": "missing_content_provenance",
                        "row_id": legacy["row_id"], "engine": engines.RADIANCE,
                        "layout": legacy.get("layout"),
                        "workload": legacy["workload_id"],
                        "recorded_at": stamp,
                    })
            rows.sort(key=lambda row: (row["recorded_at"], row.get("sha256", row.get("row_id", ""))))
            if args.json:
                print(json.dumps(rows, indent=2, sort_keys=True))
            else:
                for row in rows:
                    print(f"{row['recorded_at']} {row['status']:<12} "
                          f"{row['engine']} {row['layout']} {row['workload']} "
                          f"{row.get('sha256', row.get('row_id'))}")
            return 0
        if args.results_action == "baseline":
            baseline = results.make_baseline(
                store, _select(args, store), args.name
            )
            _write_baseline(args.to, baseline)
            if args.json:
                print(json.dumps(baseline, indent=2, sort_keys=True))
            else:
                print(f"baseline {args.name}: {args.to} ({len(baseline['records'])} launches, "
                      f"{len(baseline['blobs'])} embedded provenance blobs)")
                for name, metric in baseline["metrics"].items():
                    if metric["gated"]:
                        print(f"  {name} {metric['value']:.4g} "
                              f"{metric['direction']} tolerance {metric['tolerance'] * 100:.2f}%")
            return 0
        baseline = _read_baseline(args.baseline) if args.baseline else None
        if args.results_action == "compare":
            report = results.compare_records(store, baseline,
                                             _select(args, store, reference=baseline))
            if args.json:
                print(json.dumps(report, indent=2, sort_keys=True))
            else:
                _show_comparison(report)
            return 1 if report["failed"] else 0
        if args.results_action == "history":
            report = results.history(
                store, engine=args.engine, layout=args.layout,
                workload=args.workload, metric=args.metric, baseline=baseline,
                trailing_window=args.window, since=args.since, until=args.until,
            )
            if args.json:
                print(json.dumps(report, indent=2, sort_keys=True))
            else:
                _show_history(report)
            return 0
        raise results.ResultsError(f"unknown results command {args.results_action}")
    except (results.ResultsError, run_records.RecordError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"results: {exc}", file=sys.stderr)
        return 2


def register(sub) -> None:
    parser = sub.add_parser("results", help="baseline, compare and history for immutable run records")
    actions = parser.add_subparsers(dest="results_action", required=True)

    def common(command, *, require_scope=False):
        command.add_argument("--work-root", type=Path, default=None,
                             help="BC_WORK_ROOT holding the MRC01 run-record store")
        command.add_argument("--engine", required=require_scope, help="engine identity")
        command.add_argument("--layout", required=require_scope, help="layout identity")
        command.add_argument("--workload", required=require_scope, help="workload name")
        command.add_argument("--since", help="inclusive UTC YYYY-MM-DDTHH:MM:SSZ")
        command.add_argument("--until", help="inclusive UTC YYYY-MM-DDTHH:MM:SSZ")
        command.add_argument("--json", action="store_true", help="emit structured JSON")
        command.set_defaults(func=cmd_results)

    listing = actions.add_parser("list", help="list verified run records and read-only legacy workload rows")
    common(listing)

    baseline = actions.add_parser("baseline", help="create portable spread-derived baseline (two or more launches)")
    common(baseline, require_scope=True)
    baseline.add_argument("--name", required=True)
    baseline.add_argument("--to", type=Path, required=True, help="output JSON baseline")
    baseline.add_argument("--record", action="append", default=[], metavar="SHA256",
                          help="exact record sha256; repeat (otherwise query by scope and time)")

    compare = actions.add_parser("compare", help="compare two or more stored launches to a baseline")
    common(compare)
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--record", action="append", default=[], metavar="SHA256",
                         help="exact candidate record sha256; repeat")

    history = actions.add_parser("history", help="one phase-qualified metric over time with drift flags")
    common(history, require_scope=True)
    history.add_argument("--metric", required=True, help="e.g. decode.tokens_per_second")
    history.add_argument("--baseline", type=Path, help="optional portable baseline for drift thresholds")
    history.add_argument("--window", type=int, default=3, help="earlier-launch trailing reference (minimum 2)")

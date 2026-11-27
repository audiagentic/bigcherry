"""Platform measurement results CLI (MRC05); no engine or lab imports."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from ..core import engines, results, run_manifest, run_records
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
        if args.results_action == "put":
            # Input files are operator-supplied transport only; the store
            # remains the exclusive reader and validates all SHA references.
            with args.record.open("r", encoding="utf-8") as stream:
                doc = json.load(stream, object_pairs_hook=run_records._pairs_no_duplicates)
            run_records.validate_record(doc)
            for entry in args.blob:
                sha, separator, raw_path = entry.partition("=")
                if not separator or not raw_path:
                    raise run_records.RecordError("--blob must be SHA256=PATH")
                run_records._check_digest(sha)
                path = Path(raw_path)
                if path.is_symlink() or not path.is_file():
                    raise run_records.RecordError(f"unsafe blob input: {path}")
                ref = store.put_blob_file(path)
                if ref["sha256"] != sha:
                    raise run_records.RecordError(f"sha256 mismatch in supplied blob: {sha}")
            digest = store.put_record(doc)
            if args.run_dir is not None:
                run_manifest.link_measurement_record(args.run_dir, store, digest)
            if args.json:
                print(json.dumps({"sha256": digest}, sort_keys=True))
            else:
                print(f"run-record v1: {digest}")
            return 0
        if args.results_action == "query":
            rows = store.query(
                run_id=args.run_id, engine=args.engine, layout=args.layout,
                workload=args.workload, metric=args.metric, since=args.since, until=args.until
            )
            if args.json:
                print(json.dumps(rows, indent=2, sort_keys=True))
            else:
                for item in rows:
                    rec = item["record"]
                    print(f"{rec['recorded_at']} {rec['run_id']} "
                          f"{rec['identity']['engine']} {rec['identity']['layout']} "
                          f"{rec['workload']['name']} {item['sha256']}")
            return 0
        if args.results_action == "show":
            rec = store.get_record(args.sha256)
            result = {"sha256": args.sha256, "record": rec}
            if args.json:
                print(json.dumps(result, indent=2, sort_keys=True))
            else:
                print(f"sha256: {args.sha256}")
                print(f"run: {rec['run_id']} ({rec['recorded_at']})")
                print(f"identity: {rec['identity']['engine']} / {rec['identity']['layout']} / "
                      f"{rec['identity']['model']}")
                print(f"workload: {rec['workload']['name']}")
                for phase, readings in rec["metrics"].items():
                    for name, reading in sorted(readings.items()):
                        print(f"  {phase}.{name}: {reading}")
                print("provenance inputs:")
                for item in rec["provenance"]["inputs"]:
                    print(f"  {item['role']}: {item['blob']['sha256']}")
                print(f"validity: {rec.get('validity', {'state':'unknown'})}")
            return 0
        if args.results_action == "ingest-native-lab":
            from ..bench.radiance_lab_records import publish_native_lab

            sha = publish_native_lab(args.run_dir, args.measurement_file,
                                     work_root=args.work_root)
            if args.json:
                print(json.dumps({"sha256": sha}, sort_keys=True))
            else:
                print(f"native lab run-record v1: {sha}")
            return 0
        if args.results_action == "list":
            if args.today:
                if args.since or args.until:
                    raise results.ResultsError("--today cannot combine with --since/--until")
                day = datetime.now(UTC).strftime("%Y-%m-%d")
                args.since, args.until = day + "T00:00:00Z", day + "T23:59:59Z"
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
            rows.sort(key=lambda row: (row["recorded_at"], row["sha256"]))
            if args.json:
                print(json.dumps(rows, indent=2, sort_keys=True))
            else:
                for row in rows:
                    print(f"{row['recorded_at']} {row['status']:<12} "
                          f"{row['engine']} {row['layout']} {row['workload']} "
                          f"{row['sha256']}")
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
    except (results.ResultsError, run_records.RecordError, run_manifest.ManifestError,
            OSError, ValueError, KeyError, TypeError) as exc:
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

    putting = actions.add_parser("put", help="publish a complete record with SHA-verified input blobs")
    putting.add_argument("--record", type=Path, required=True, help="record JSON input")
    putting.add_argument("--blob", action="append", default=[], metavar="SHA256=PATH",
                         help="exact source bytes (repeat for all blobs not already in the store)")
    putting.add_argument("--run-dir", type=Path, help="atomically add a typed edge to its WSL01 manifest")
    putting.add_argument("--work-root", type=Path, default=None)
    putting.add_argument("--json", action="store_true")
    putting.set_defaults(func=cmd_results)

    querying = actions.add_parser("query", help="filter only SHA-verified v1 records")
    common(querying)
    querying.add_argument("--run-id", help="WSL01 run identity")
    querying.add_argument("--metric", help="group-qualified metric, e.g. decode.tokens_per_second or quality.kld_mean")

    listing = actions.add_parser("list", help="list verified run records")
    common(listing)
    listing.add_argument("--today", action="store_true",
                         help="restrict to today's UTC date (use --engine radiance --workload parity)")

    show = actions.add_parser("show", help="read a verified immutable run record by SHA-256")
    show.add_argument("sha256")
    show.add_argument("--work-root", type=Path, default=None)
    show.add_argument("--json", action="store_true")
    show.set_defaults(func=cmd_results)

    native_lab = actions.add_parser("ingest-native-lab",
                                    help="ingest typed measurements from a stopped native Radiance owner")
    native_lab.add_argument("--run-dir", type=Path, required=True)
    native_lab.add_argument("--measurement-file", type=Path, required=True)
    native_lab.add_argument("--work-root", type=Path, default=None)
    native_lab.add_argument("--json", action="store_true")
    native_lab.set_defaults(func=cmd_results)

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

    history = actions.add_parser("history", help="one group-qualified metric over time with drift flags")
    common(history, require_scope=True)
    history.add_argument("--metric", required=True, help="e.g. decode.tokens_per_second or quality.kld_mean")
    history.add_argument("--baseline", type=Path, help="optional portable baseline for drift thresholds")
    history.add_argument("--window", type=int, default=3, help="earlier-launch trailing reference (minimum 2)")

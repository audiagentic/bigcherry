"""Tuning measurement analysis and reporting (HI21).

Reads tuning results from either a .measurements.jsonl file or a SQLite
database and produces formatted reports for analysis.

Usage:

    python -m bigcherry report signatures --measurements tune.measurements.jsonl
    python -m bigcherry report summary --measurements tune.measurements.jsonl
    python -m bigcherry report families --measurements tune.measurements.jsonl \\
        --dispatch abc123...
    python -m bigcherry report hot --database tune.sqlite
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_measurements_jsonl(path: Path) -> list[dict[str, Any]]:
    """Parse a measurements JSONL, returning only result records."""
    results: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if row.get("kind") == "result":
                results.append(row)
    return results


def read_measurements_sqlite(
    path: Path, dispatch_filter: str | None = None
) -> list[dict[str, Any]]:
    """Read measurement and winner data from SQLite, returning one dict per
    dispatch digest with the same shape as the JSONL format.

    When ``dispatch_filter`` is set, only that dispatch digest is returned.
    """
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row  # type: ignore[attr-defined]
    try:
        if dispatch_filter:
            where = "WHERE m.dispatch_digest = ?"
            params: tuple[bytes, ...] = (bytes.fromhex(dispatch_filter),)
        else:
            where = ""
            params = ()

        cursor = connection.execute(
            f"""SELECT m.dispatch_digest, c.stable_name AS candidate_name,
                       c.family, m.median_us, m.gpu_mad_us, m.p95_us,
                       m.host_median_us, m.nmse, m.max_abs_err,
                       m.workspace_bytes, m.samples, m.accepted,
                       m.reject_reason, w.stable_name AS winner_name,
                       w.improvement_pct, w.is_native, w.reason AS winner_reason
                FROM measurement m
                JOIN candidate c ON m.candidate_id = c.candidate_id
                LEFT JOIN winner w
                  ON m.dispatch_digest = w.dispatch_digest
                  AND m.build_id = w.build_id
                  AND m.objective = w.objective
                {where}
                ORDER BY m.dispatch_digest, m.median_us""",
            params,
        )

        by_dispatch: dict[str, dict[str, Any]] = {}
        for row in cursor:
            dispatch_hex = (
                row["dispatch_digest"].hex()
                if isinstance(row["dispatch_digest"], bytes)
                else str(row["dispatch_digest"])
            )
            entry = by_dispatch.setdefault(
                dispatch_hex,
                {
                    "dispatch": dispatch_hex,
                    "winner": "",
                    "improvement_pct": 0.0,
                    "reason": "",
                    "generated": 0,
                    "eligible": 0,
                    "measured": 0,
                    "candidates": [],
                },
            )
            entry["candidates"].append(
                {
                    "name": row["candidate_name"],
                    "family": row["family"],
                    "median_us": row["median_us"],
                    "mad_us": row["gpu_mad_us"],
                    "p95_us": row["p95_us"],
                    "host_median_us": row["host_median_us"],
                    "nmse": row["nmse"],
                    "max_abs": row["max_abs_err"],
                    "workspace": row["workspace_bytes"],
                    "samples": row["samples"],
                    "status": "ok"
                    if row["accepted"]
                    else (
                        row["reject_reason"].removeprefix("GGML_HIP_REJECT_").lower()
                        if row["reject_reason"]
                        else "unknown"
                    ),
                }
            )
            if not entry["winner"]:
                winner_name_val = row["winner_name"]
                entry["winner"] = winner_name_val if winner_name_val else ""
                imp_val = row["improvement_pct"]
                entry["improvement_pct"] = imp_val if imp_val is not None else 0.0
                reason_val = row["winner_reason"]
                entry["reason"] = reason_val if reason_val else ""

        # Fill in generated/eligible/measured counts
        for entry in by_dispatch.values():
            candidates = entry["candidates"]
            entry["measured"] = sum(
                1 for c in candidates if c["status"] == "ok" and c["samples"]
            )
            # Eligible = not filtered by hard-identity (architecture, ineligible).
            # Tolerance-rejected candidates were launched but failed correctness.
            entry["eligible"] = sum(
                1
                for c in candidates
                if c["status"].lower() not in ("architecture", "ineligible")
            )
            entry["generated"] = len(candidates)

        return list(by_dispatch.values())
    finally:
        connection.close()


def read_hot_signatures(path: Path, n: int = 20) -> list[dict[str, Any]]:
    """Read top-N signatures by call count from observation table."""
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row  # type: ignore[attr-defined]
    try:
        cursor = connection.execute(
            """SELECT o.calls, s.canonical_json, o.native_stable_name,
                      w.stable_name AS winner_name, w.improvement_pct
               FROM observation o
               LEFT JOIN signature s ON o.signature_id = s.signature_id
               LEFT JOIN winner w
                 ON o.build_id = w.build_id
                 AND o.hardware_id = w.hardware_id
                 AND o.signature_id = w.signature_id
               ORDER BY o.calls DESC
               LIMIT ?""",
            (n,),
        )
        rows = []
        for row in cursor:
            canonical = {}
            with contextlib.suppress(json.JSONDecodeError):
                canonical = json.loads(row["canonical_json"] or "{}")
            m_val = canonical.get("m", 0)
            n2_val = canonical.get("n", 0)
            k_val = canonical.get("k", 0)
            winner_name_val = row["winner_name"]
            imp_val = row["improvement_pct"]
            rows.append(
                {
                    "calls": row["calls"],
                    "op": canonical.get("op", "") or "",
                    "src0_type": canonical.get("src0_type", "") or "",
                    "src1_type": canonical.get("src1_type", "") or "",
                    "m": m_val,
                    "n": n2_val,
                    "k": k_val,
                    "native": row["native_stable_name"] or "",
                    "winner": winner_name_val if winner_name_val else "",
                    "improvement_pct": imp_val if imp_val is not None else 0.0,
                }
            )
        return rows
    finally:
        connection.close()


def read_tuning_overview(path: Path) -> dict[str, Any]:
    """Read a quick overview of the tuning run from SQLite.

    Returns a compact dict suitable for human-readable output or JSON.
    """
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row  # type: ignore[attr-defined]
    try:
        # Total stats
        cursor = connection.execute("SELECT COUNT(*) as sigs FROM winner")
        sigs = cursor.fetchone()["sigs"]

        # Improvement distribution
        cursor = connection.execute(
            "SELECT "
            "  COUNT(CASE WHEN improvement_pct > 10 THEN 1 END) as over_10,"
            "  COUNT(CASE WHEN improvement_pct > 5 THEN 1 END) as over_5,"
            "  COUNT(CASE WHEN improvement_pct > 1 THEN 1 END) as over_1,"
            "  COUNT(CASE WHEN improvement_pct > 0.5 THEN 1 END) as over_05,"
            "  COUNT(CASE WHEN stable_name LIKE '%:native:%' THEN 1 END) as native_winners,"
            "  AVG(improvement_pct) as avg_imp,"
            "  MAX(improvement_pct) as max_imp"
            " FROM winner"
        )
        imp = cursor.fetchone()

        # Pipeline stats
        cursor = connection.execute("SELECT COUNT(*) as total FROM measurement")
        total_cands = cursor.fetchone()["total"]

        cursor = connection.execute(
            "SELECT COUNT(*) as count, reject_reason FROM measurement "
            "WHERE accepted = 0 AND reject_reason IS NOT NULL "
            "GROUP BY reject_reason ORDER BY count DESC"
        )
        rejections: dict[str, int] = {r["reject_reason"]: r["count"] for r in cursor}

        cursor = connection.execute(
            "SELECT COUNT(*) as count FROM measurement WHERE accepted = 1"
        )
        measured = cursor.fetchone()["count"]

        eligible = total_cands - rejections.get("GGML_HIP_REJECT_INELIGIBLE", 0)
        eligible -= rejections.get("GGML_HIP_REJECT_ARCHITECTURE", 0)

        # Winner by family
        cursor = connection.execute(
            "SELECT SUBSTR(stable_name, 1, INSTR(stable_name, ':')-1) as family,"
            "  COUNT(*) as count "
            "FROM winner GROUP BY family ORDER BY count DESC"
        )
        families: dict[str, int] = {r["family"]: r["count"] for r in cursor}

        # Top tuned winners (non-native with improvement > 0)
        cursor = connection.execute(
            "SELECT stable_name, improvement_pct, median_us "
            "FROM winner "
            "WHERE stable_name NOT LIKE '%:native:%' AND improvement_pct > 0 "
            "ORDER BY improvement_pct DESC LIMIT 10"
        )
        top_tuned = [
            {
                "name": r["stable_name"],
                "improvement_pct": r["improvement_pct"],
                "median_us": r["median_us"],
            }
            for r in cursor
        ]

        return {
            "signatures": sigs,
            "pipeline": {
                "generated": total_cands,
                "eligible": eligible,
                "measured": measured,
            },
            "improvement": {
                ">10%": imp["over_10"] or 0,
                ">5%": imp["over_5"] or 0,
                ">1%": imp["over_1"] or 0,
                ">0.5%": imp["over_05"] or 0,
                "native_retained": imp["native_winners"] or 0,
            },
            "families": families,
            "rejections": rejections,
            "top_tuned": top_tuned,
        }
    finally:
        connection.close()


# ------------------------------------------------------------------ formatting


def _fmt_us(v: float | None) -> str:
    if v is None:
        return "-"
    if v < 1:
        return f"{v:.3f}"
    return f"{v:.2f}"


def _family(name: str) -> str:
    return name.split(":", 1)[0] if ":" in name else ""


# ------------------------------------------------------------------ commands


def cmd_signatures(args: argparse.Namespace) -> int:
    """Per-signature detail tables."""
    if args.measurements:
        results = read_measurements_jsonl(Path(args.measurements))
    elif args.database:
        results = read_measurements_sqlite(
            Path(args.database), dispatch_filter=args.dispatch or None
        )
    else:
        print("error: must specify --measurements or --database", file=sys.stderr)
        return 2

    for result in results[: args.limit if args.limit else len(results)]:
        dispatch = (
            result["dispatch"][:16] + "..."
            if len(result["dispatch"]) > 16
            else result["dispatch"]
        )
        winner = result.get("winner", "N/A")
        improvement = result.get("improvement_pct", 0.0)
        gen_elig_meas = f"{result.get('generated', 0)}/{result.get('eligible', 0)}/{result.get('measured', 0)}"

        if args.json:
            print(json.dumps(result, indent=2))
            continue

        print(
            f"\nDispatch: {dispatch}  Winner: {winner} ({improvement:+.1f}% vs native)"
        )
        print(f"Generated/Eligible/Measured: {gen_elig_meas}")
        print()
        print(
            f"{'Candidate':<52} {'Status':<14} {'Median':>8} {'MAD':>7} "
            f"{'P95':>8} {'WS':>8}"
        )
        print("-" * 115)

        candidates = sorted(
            result.get("candidates", []),
            key=lambda c: c.get("median_us") or float("inf"),
        )
        for c in candidates:
            name = c.get("name", "?")
            status = c.get("status", "?")
            median = _fmt_us(c.get("median_us"))
            mad = _fmt_us(c.get("mad_us"))
            p95 = _fmt_us(c.get("p95_us"))
            ws_bytes = c.get("workspace", 0) or 0
            ws = f"{ws_bytes // 1024}KB" if ws_bytes else "-"
            is_winner = " <<" if name == winner else ""
            print(
                f"  {name:<50} {status:<14} {median:>8} {mad:>7} "
                f"{p95:>8} {ws:>8}{is_winner}"
            )

        # Correctness data for measured candidates
        measured = [
            c
            for c in result.get("candidates", [])
            if c.get("samples") and c.get("status") == "ok"
        ]
        if measured:
            print()
            print(f"{'Candidate':<52} {'NMSE':>10} {'MaxAbs':>10} {'Samples':>8}")
            print("-" * 86)
            for c in measured:
                nmse = f"{c.get('nmse', 0):.0e}" if c.get("nmse") else "-"
                maxabs = f"{c.get('max_abs', 0):.0e}" if c.get("max_abs") else "-"
                samples = str(c.get("samples", "?"))
                print(
                    f"  {c.get('name', '?'):<50} {nmse:>10} {maxabs:>10} {samples:>8}"
                )

    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    """Aggregate statistics across all tuning results."""
    if args.measurements:
        results = read_measurements_jsonl(Path(args.measurements))
    elif args.database:
        results = read_measurements_sqlite(Path(args.database))
    else:
        print("error: must specify --measurements or --database", file=sys.stderr)
        return 2

    total_sigs = len(results)
    if not total_sigs:
        print("no tuning results found")
        return 0

    # Improvement distribution
    improvements: list[float] = [r.get("improvement_pct", 0.0) for r in results]
    native_count = sum(1 for r in results if "native" in r.get("winner", ""))

    # Family breakdown
    family_winners: Counter[str] = Counter()
    family_native: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    total_generated = 0
    total_eligible = 0
    total_measured = 0

    for r in results:
        winner_name = r.get("winner", "")
        family_winners[_family(winner_name)] += 1

        for c in r.get("candidates", []):
            if _family(c.get("name", "")) == "native" or ":native:" in c.get(
                "name", ""
            ):
                family_native[c.get("family", "")] += 1
            status = c.get("status", "ok")
            if status != "ok":
                rejection_counts[status] += 1

        total_generated += r.get("generated", 0)
        total_eligible += r.get("eligible", 0)
        total_measured += r.get("measured", 0)

    if args.json:
        data = {
            "total_signatures": total_sigs,
            "native_retained": native_count,
            "improvement_distribution": {
                "over_10pct": sum(1 for i in improvements if i > 10),
                "over_5pct": sum(1 for i in improvements if i > 5),
                "over_1pct": sum(1 for i in improvements if i > 1),
                "over_0.5pct": sum(1 for i in improvements if i > 0.5),
            },
            "family_winners": dict(family_winners),
            "rejection_breakdown": dict(rejection_counts),
            "total_generated": total_generated,
            "total_eligible": total_eligible,
            "total_measured": total_measured,
        }
        print(json.dumps(data, indent=2))
        return 0

    print(f"Tuning Summary ({total_sigs} signatures)")
    print()
    print("Improvement distribution:")
    for threshold, label in [(10, ">10%"), (5, ">5%"), (1, ">1%"), (0.5, ">0.5%")]:
        count = sum(1 for i in improvements if i > threshold)
        pct = 100 * count / total_sigs
        bar = "█" * int(pct / 2)
        print(f"  {label:>6}: {count:4d} ({pct:5.1f}%) {bar}")
    print(
        f"  Native retained: {native_count:4d} ({100 * native_count / total_sigs:.1f}%)"
    )

    print()
    print("Candidate pipeline:")
    print(f"  Generated:  {total_generated:>6d}")
    print(
        f"  Eligible:   {total_eligible:>6d} "
        f"({100 * total_eligible / max(total_generated, 1):.1f}% of generated)"
    )
    print(
        f"  Measured:   {total_measured:>6d} "
        f"({100 * total_measured / max(total_eligible, 1):.1f}% of eligible)"
    )

    print()
    print("Winner by family:")
    for family, count in family_winners.most_common():
        pct = 100 * count / total_sigs
        print(f"  {family:<8} {count:4d} ({pct:.1f}%)")

    if rejection_counts:
        print()
        print("Rejection breakdown:")
        for reason, count in rejection_counts.most_common():
            print(f"  {reason:<20} {count}")

    return 0


def cmd_families(args: argparse.Namespace) -> int:
    """Cross-family comparison for a specific dispatch digest."""
    if not args.dispatch:
        print("error: --dispatch is required for families view", file=sys.stderr)
        return 2

    if args.measurements:
        results = read_measurements_jsonl(Path(args.measurements))
        match = [r for r in results if r.get("dispatch") == args.dispatch]
        if not match:
            print(f"dispatch digest {args.dispatch} not found", file=sys.stderr)
            return 1
    elif args.database:
        results = read_measurements_sqlite(Path(args.database), args.dispatch)
    else:
        print("error: must specify --measurements or --database", file=sys.stderr)
        return 2

    result = results[0]
    dispatch = (
        result["dispatch"][:16] + "..."
        if len(result["dispatch"]) > 16
        else result["dispatch"]
    )

    # Group by family
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in result.get("candidates", []):
        family = c.get("family", _family(c.get("name", "")))
        if not family:
            continue
        by_family[family].append(c)

    if args.json:
        data = {
            "dispatch": args.dispatch,
            "winner": result.get("winner", ""),
            "by_family": dict(by_family),
        }
        print(json.dumps(data, indent=2))
        return 0

    print(f"Dispatch: {dispatch}")
    print(
        f"Winner:   {result.get('winner', 'N/A')} ({result.get('improvement_pct', 0):+0.1f}% vs native)"
    )
    print()

    for family in sorted(by_family):
        candidates = by_family[family]
        measured = [
            c for c in candidates if c.get("samples") and c.get("status") == "ok"
        ]
        best = (
            min(measured, key=lambda c: c.get("median_us") or float("inf"))
            if measured
            else None
        )

        print(f"--- {family.upper()} ---")
        print(f"{'Candidate':<52} {'Median':>8} {'MAD':>7} {'P95':>8} {'NMSE':>10}")
        print("-" * 92)

        for c in sorted(measured, key=lambda c: c.get("median_us") or float("inf")):
            name = c.get("name", "?")
            median = _fmt_us(c.get("median_us"))
            mad = _fmt_us(c.get("mad_us"))
            p95 = _fmt_us(c.get("p95_us"))
            nmse = f"{c.get('nmse', 0):.0e}" if c.get("nmse") else "-"
            marker = " << best" if best and c.get("name") == best.get("name") else ""
            print(f"  {name:<50} {median:>8} {mad:>7} {p95:>8} {nmse:>10}{marker}")

        # Rejected candidates for this family
        rejected = [c for c in candidates if c.get("status") != "ok"]
        if rejected:
            reject_statuses = {c["status"] for c in rejected}
            print(f"  (rejected: {len(rejected)} — {', '.join(reject_statuses)})")
        print()

    return 0


def cmd_overview(args: argparse.Namespace) -> int:
    """Quick one-liner overview of the tuning run."""
    if not args.database:
        print(
            "error: --database is required for overview",
            file=sys.stderr,
        )
        return 2

    data = read_tuning_overview(Path(args.database))

    if args.json:
        print(json.dumps(data, indent=2))
        return 0

    sigs = data["signatures"]
    pipe = data["pipeline"]
    imp = data["improvement"]
    rejections = data["rejections"]
    families = data["families"]
    top_tuned = data["top_tuned"]

    print(f"Tuning: {sigs} signatures")
    print(
        f"Pipeline: {pipe['generated']} generated -> {pipe['eligible']} eligible -> {pipe['measured']} measured"
    )
    print(
        f"Winners: {imp['native_retained']} native, >1%={imp[('>1%')]}, >5%={imp[('>5%')]}, >10%={imp[('>10%')]}"
    )

    if rejections:
        rej_summary = ", ".join(
            f"{k.split('_')[-1]}:{v}" for k, v in rejections.items()
        )
        print(f"Rejections: {rej_summary}")

    if top_tuned:
        print("\nTop tuned winners:")
        for t in top_tuned:
            print(
                f"  +{t['improvement_pct']:.1f}%  {t['name']:60s} median={t['median_us']:.1f}us"
            )

    return 0


def cmd_hot(args: argparse.Namespace) -> int:
    """Top-N signatures by call count."""
    if not args.database:
        print(
            "error: --database is required for hot view (needs observation table)",
            file=sys.stderr,
        )
        return 2

    rows = read_hot_signatures(Path(args.database), n=args.limit)

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    print(f"Top {len(rows)} signatures by call count")
    print()
    print(
        f"{'Calls':>8} {'Op':<12} {'Shape':<20} {'Native':<35} {'Winner':<35} {'Δ%':>6}"
    )
    print("-" * 126)

    for row in rows:
        m = row.get("m", 0) or 0
        n = row.get("n", 0) or 0
        k = row.get("k", 0) or 0
        shape = f"{m}×{n}×{k}"
        native = (
            (row.get("native", "?")[:32] + "..")
            if len(row.get("native", "")) > 32
            else row.get("native", "?")
        )
        winner = (
            (row.get("winner", "?")[:32] + "..")
            if len(row.get("winner", "")) > 32
            else row.get("winner", "?")
        )
        delta = f"{row.get('improvement_pct', 0):+0.1f}" if row.get("winner") else "-"
        print(
            f"  {row['calls']:>8} {row.get('op', '?'):<12} {shape:<20} "
            f"{native:<35} {winner:<35} {delta:>6}"
        )

    return 0


# ------------------------------------------------------------------ parser


def build_parser(subparsers) -> None:
    """Register report subcommands on the given subparser container."""
    report = subparsers.add_parser("report", help="analyze tuning measurements")
    sub = report.add_subparsers(dest="report_command", required=True)

    sig = sub.add_parser("signatures", help="per-signature detail tables")
    sig.add_argument("--measurements", default=None, help="JSONL file from tuning run")
    sig.add_argument("--database", default=None, help="SQLite database path")
    sig.add_argument(
        "--dispatch", default=None, help="filter to one dispatch digest (hex)"
    )
    sig.add_argument(
        "--limit", type=int, default=0, help="max results to show (0 = all)"
    )
    sig.add_argument("--json", action="store_true")
    sig.set_defaults(func=cmd_signatures)

    summ = sub.add_parser("summary", help="aggregate statistics")
    summ.add_argument("--measurements", default=None, help="JSONL file from tuning run")
    summ.add_argument("--database", default=None, help="SQLite database path")
    summ.add_argument("--json", action="store_true")
    summ.set_defaults(func=cmd_summary)

    over = sub.add_parser("overview", help="quick one-line summary of the tuning run")
    over.add_argument("--database", default=None, help="SQLite database path")
    over.add_argument("--json", action="store_true")
    over.set_defaults(func=cmd_overview)

    fam = sub.add_parser("families", help="cross-family comparison for one digest")
    fam.add_argument("--dispatch", required=True, help="dispatch digest (hex)")
    fam.add_argument("--measurements", default=None, help="JSONL file from tuning run")
    fam.add_argument("--database", default=None, help="SQLite database path")
    fam.add_argument("--json", action="store_true")
    fam.set_defaults(func=cmd_families)

    hot = sub.add_parser("hot", help="top-N signatures by call count")
    hot.add_argument(
        "--database",
        default=None,
        help="SQLite database path (needs observation table)",
    )
    hot.add_argument("--limit", type=int, default=20)
    hot.add_argument("--json", action="store_true")
    hot.set_defaults(func=cmd_hot)

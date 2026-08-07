"""The ``bigcherry`` command line.

One command per stage of taking a new llama.cpp release into production:

    pull -> audit -> apply -> generate -> build

Stages are idempotent, and each refuses to run on a tree that has not passed
the stage before it. That ordering is the whole point: patches are only
meaningful against a tree whose shape has been verified, and a build is only
meaningful against a manifest generated from that same tree.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import __version__
from . import paths
from . import patcher
from . import patchset
from . import releases
from . import source_audit

UPSTREAM_URL = "https://github.com/ggml-org/llama.cpp"


def _run(
    args: list[str], cwd: Path | None = None, *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, check=check)


def _git_out(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ("git", "-C", str(root)) + args, capture_output=True, text=True, check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""
    return result.stdout.strip()


def _bigcherry_revision() -> str:
    return _git_out(paths.REPO_ROOT, "rev-parse", "HEAD")


def _record_for(root: Path) -> releases.ReleaseRecord:
    revision, _ = source_audit.git_revision(root)
    # `git describe --tags` gives the upstream release tag (b1234) when HEAD is
    # at or after one; a shallow clone often has no tags, hence the fallback.
    tag = _git_out(root, "describe", "--tags", "--exact-match")
    record = releases.load(revision, tag)
    record.revision = revision
    record.release_tag = tag
    record.bigcherry_revision = _bigcherry_revision()
    return record


# --------------------------------------------------------------------- pull


def cmd_pull(args: argparse.Namespace) -> int:
    root = paths.llama_root(args.llama_root)
    if not (root / ".git").exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        print(f"cloning {UPSTREAM_URL} -> {root}")
        depth = [] if args.full else ["--depth", "1"]
        _run(["git", "clone", *depth, UPSTREAM_URL, str(root)])
        # Upstream is LF throughout. Letting git rewrite line endings in the
        # working tree would make every generated diff unreviewable.
        _run(["git", "-C", str(root), "config", "core.autocrlf", "false"])
    else:
        print(f"fetching into {root}")
        fetch = ["git", "-C", str(root), "fetch", "--tags"]
        if not args.full:
            fetch += ["--depth", "1"]
        fetch += ["origin", args.ref or "HEAD"]
        _run(fetch)

    if args.ref:
        print(f"checking out {args.ref}")
        _run(["git", "-C", str(root), "checkout", "--force", args.ref])

    record = _record_for(root)
    record.advance_to("pulled")
    record.save()
    print(
        f"at {record.revision[:12]}"
        f"{' (' + record.release_tag + ')' if record.release_tag else ''}"
    )
    return 0


# -------------------------------------------------------------------- audit


def cmd_audit(args: argparse.Namespace) -> int:
    root = paths.llama_root(args.llama_root)
    report = source_audit.audit(root)
    report["strict"] = args.strict

    out = paths.artifact_dir(report["source_revision"]) / "source-audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    good = source_audit.passed(report, strict=args.strict)
    print(f"source audit of {root}")
    print(
        f"  revision {report['source_revision'][:12]}"
        f"{' (dirty)' if report['source_dirty'] else ''}"
    )
    print(source_audit.format_report(report, verbose=args.verbose))
    print(f"  report: {out}")
    print("  RESULT: " + ("PASS" if good else "FAIL"))

    record = _record_for(root)
    record.audit = releases.summarise_audit(report, strict=args.strict)
    record.advance_to("audited" if good else "broken")
    if not good:
        record.notes = "source audit failed: " + ", ".join(
            record.audit["failed_checks"]
        )
    record.save()
    return 0 if good else 1


# -------------------------------------------------------------------- apply


def _copy_overlay(root: Path, *, dry_run: bool) -> list[str]:
    """Mirror ``src/`` onto the checkout. Returns the paths written."""
    written: list[str] = []
    for source in sorted(paths.SRC_OVERLAY.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(paths.SRC_OVERLAY)
        target = root / relative
        text = source.read_text(encoding="utf-8")
        if target.is_file() and target.read_text(encoding="utf-8") == text:
            continue
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8", newline="")
        written.append(str(relative).replace("\\", "/"))
    return written


def cmd_apply(args: argparse.Namespace) -> int:
    root = paths.llama_root(args.llama_root)
    record = _record_for(root)

    if not args.force and not record.audit.get("passed"):
        print(
            "refusing to patch a tree that has not passed a strict audit.\n"
            "  run `python -m bigcherry audit` first, or pass --force.",
            file=sys.stderr,
        )
        return 2

    patches = patchset.load_patches()
    results = patcher.apply_all(patches, root, dry_run=args.dry_run)
    ok = all(r.ok for r in results)

    if ok:
        written = _copy_overlay(root, dry_run=args.dry_run)
        verb = "would write" if args.dry_run else "wrote"
        print(f"overlay: {verb} {len(written)} file(s)")
        for path in written:
            print(f"    {path}")
    else:
        print("overlay: skipped -- patches failed")

    print(f"patches ({len(patches)} file(s)):")
    print(patcher.format_results(results))

    if not args.dry_run:
        record.patches = releases.summarise_patches(results)
        record.advance_to("patched" if ok else "broken")
        if not ok:
            record.notes = "patches failed: " + ", ".join(
                record.patches["failed_edits"]
            )
        record.save()

    print("  RESULT: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


# ----------------------------------------------------------------- generate


def cmd_generate(args: argparse.Namespace) -> int:
    from . import autotune_catalog

    root = paths.llama_root(args.llama_root)
    record = _record_for(root)

    # Generating against an unpatched tree would emit a registry referencing
    # launcher symbols that do not exist yet -- a link error much later, with
    # nothing pointing back to the real cause.
    if not args.force and record.stage not in (
        "patched",
        "generated",
        "built",
        "tested",
        "tuned",
        "validated",
    ):
        print(
            "refusing to generate against an unpatched tree.\n"
            "  run `python -m bigcherry apply` first, or pass --force.",
            file=sys.stderr,
        )
        return 2

    forwarded = ["--variant-set", args.variant_set, "--arch", args.arch]
    if args.llama_root:
        forwarded += ["--llama-root", args.llama_root]
    if args.inventory:
        forwarded += ["--inventory", args.inventory]
    if args.winners:
        forwarded += ["--winners", args.winners]
    if args.dry_run:
        forwarded += ["--dry-run"]

    status = autotune_catalog.main(forwarded)
    if status == 0 and not args.dry_run:
        manifest_path = (
            paths.artifact_dir(record.revision) / "hip-autotune-manifest.json"
        )
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            record.manifest_hash = manifest["manifest_hash"]
        record.advance_to("generated")
        record.save()
    return status


# ------------------------------------------------------------------- status


def cmd_status(args: argparse.Namespace) -> int:
    root = paths.llama_root(args.llama_root)
    revision, dirty = source_audit.git_revision(root)
    print(f"bigcherry {__version__}")
    print(f"  repo:     {paths.REPO_ROOT}")
    print(f"  checkout: {root}")
    print(f"  revision: {revision[:12]}{' (dirty)' if dirty else ''}")
    print()
    records = releases.all_records()
    if not records:
        print("  no releases recorded yet")
        return 0
    print(f"  {'release':<16} {'stage':<12} {'audit':<7} manifest")
    for record in records:
        audit = (
            "pass" if record.audit.get("passed") else ("fail" if record.audit else "-")
        )
        print(
            f"  {record.slug():<16} {record.stage:<12} {audit:<7} "
            f"{record.manifest_hash[:12] or '-'}"
        )
    return 0


# --------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bigcherry", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--llama-root",
        default=None,
        help="llama.cpp checkout (default: vendor/llama.cpp)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pull = sub.add_parser("pull", help="clone or update the llama.cpp checkout")
    pull.add_argument(
        "--ref", default=None, help="tag, branch or sha to check out (e.g. b1234)"
    )
    pull.add_argument(
        "--full",
        action="store_true",
        help="full clone instead of depth-1 (needed to check out "
        "arbitrary older revisions)",
    )
    pull.set_defaults(func=cmd_pull)

    audit = sub.add_parser("audit", help="verify upstream invariants")
    audit.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="treat warnings as failures (default)",
    )
    audit.add_argument("--no-strict", dest="strict", action="store_false")
    audit.add_argument("-v", "--verbose", action="store_true")
    audit.set_defaults(func=cmd_audit)

    apply_cmd = sub.add_parser("apply", help="apply the overlay and patches")
    apply_cmd.add_argument(
        "--dry-run", action="store_true", help="report what would change, write nothing"
    )
    apply_cmd.add_argument(
        "--force", action="store_true", help="patch even without a passing audit"
    )
    apply_cmd.set_defaults(func=cmd_apply)

    from . import autotune_schema as _schema

    generate = sub.add_parser(
        "generate", help="generate the candidate catalog and its artifacts"
    )
    generate.add_argument(
        "--variant-set", default="inventory", choices=_schema.VARIANT_SETS
    )
    generate.add_argument(
        "--arch",
        default="all",
        help="comma-separated architectures or group names "
        f"({', '.join(sorted(_schema.ARCHITECTURE_GROUPS))})",
    )
    generate.add_argument(
        "--inventory",
        default=None,
        help="inventory JSON from a record-mode run (required for workload-max)",
    )
    generate.add_argument(
        "--winners",
        default=None,
        help="measurements JSONL from a tuning run (required for replay-slim)",
    )
    generate.add_argument("--dry-run", action="store_true")
    generate.add_argument(
        "--force", action="store_true", help="generate even against an unpatched tree"
    )
    generate.set_defaults(func=cmd_generate)

    status = sub.add_parser("status", help="show checkout and release status")
    status.set_defaults(func=cmd_status)

    from . import report as _report

    _report.build_parser(sub)

    # Inventory: convert record JSONL → SQLite + inventory JSON, or load tuning measurements.
    inventory = sub.add_parser(
        "inventory",
        help="Convert record JSONL to inventory/DB, or load tuning measurements",
    )
    inv_sub = inventory.add_subparsers(dest="inv_subcommand")

    # Record mode: JSONL → SQLite + inventory JSON (existing behavior)
    inv_record = inv_sub.add_parser(
        "record", help="Convert record-mode JSONL to inventory + DB"
    )
    inv_record.add_argument("record", help="JSONL written by GGML_HIP_DISPATCH_DB")
    inv_record.add_argument(
        "--inventory",
        default=None,
        help="inventory JSON to write (default: alongside)",
    )
    inv_record.add_argument(
        "--database",
        default=None,
        help="SQLite database to write (default: alongside)",
    )
    inv_record.set_defaults(func=lambda args: cmd_inventory(args, subcmd="record"))

    # Tuning mode: measurements JSONL → SQLite with winners/measurements/candidates
    inv_tuning = inv_sub.add_parser(
        "tuning", help="Load tuning measurements into SQLite"
    )
    inv_tuning.add_argument(
        "measurements",
        help="JSONL written by GGML_HIP_DISPATCH_DB (the .measurements.jsonl file)",
    )
    inv_tuning.add_argument(
        "--database",
        default=None,
        help="SQLite database path (default: alongside measurements, .sqlite extension)",
    )
    inv_tuning.add_argument(
        "--manifest",
        default=None,
        help="Manifest JSON for full candidate data (artifacts/<rev>/hip-autotune-manifest.json)",
    )
    inv_tuning.add_argument(
        "--signature-source",
        action="append",
        default=[],
        help="JSONL record/replay diagnostics file containing canonical shapes; may be repeated",
    )
    inv_tuning.set_defaults(func=lambda args: cmd_inventory(args, subcmd="tuning"))

    return parser


def cmd_inventory(args: argparse.Namespace, *, subcmd: str) -> int:
    """Dispatch to inventory record/tuning subcommand."""
    from . import inventory as inv_mod
    from pathlib import Path

    if subcmd == "record":
        record_path = Path(args.record)
        if not record_path.is_file():
            print(f"no such record file: {record_path}", file=sys.stderr)
            return 2
        try:
            record = inv_mod.read_jsonl(record_path)
        except inv_mod.RecordError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        inventory = inv_mod.build_inventory(record)
        inventory_path = (
            Path(args.inventory)
            if args.inventory
            else record_path.with_suffix(".inventory.json")
        )
        inventory_path.write_text(
            json.dumps(inventory, indent=2) + "\n", encoding="utf-8", newline=""
        )

        database_path = (
            Path(args.database) if args.database else record_path.with_suffix(".sqlite")
        )
        counts = inv_mod.build_database(
            record, database_path, paths.SQL / "dispatch-db.sql"
        )

        print(f"read {len(record.observations)} observation(s) from {record_path}")
        print(f"  types: mmq={inventory['mmq_types']} mmvq={inventory['mmvq_types']}")
        print(f"         mmvf={inventory['mmvf_types']} mmf={inventory['mmf_types']}")
        print(f"  widths: {inventory['widths']}")
        print(f"  blas observed: {inventory['uses_blas']}")
        print(f"  inventory: {inventory_path}")
        print(
            f"  database:  {database_path} "
            f"({counts['signatures']} signatures, {counts['hardware']} hardware)"
        )
        return 0

    elif subcmd == "tuning":
        meas_path = Path(args.measurements)
        if not meas_path.is_file():
            print(f"no such measurements file: {meas_path}", file=sys.stderr)
            return 2

        db_path = (
            Path(args.database) if args.database else meas_path.with_suffix(".sqlite")
        )
        manifest_path = Path(args.manifest) if args.manifest else None

        counts = inv_mod.load_measurements(
            meas_path,
            db_path,
            paths.SQL / "dispatch-db.sql",
            manifest_path=manifest_path,
            signature_source_paths=[Path(p) for p in args.signature_source],
        )

        print(
            f"loaded {counts['results']} result(s) with "
            f"{counts['measurements']} measurement(s) and "
            f"{counts['candidates']} candidate(s) into {db_path}"
        )
        return 0

    else:
        # Backward compat: positional arg means record mode (no subcommand)
        return cmd_inventory(args, subcmd="record")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

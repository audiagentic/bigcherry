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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

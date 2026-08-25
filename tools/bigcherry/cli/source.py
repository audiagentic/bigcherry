"""CLI presentation handlers for source lifecycle workflows."""

from __future__ import annotations

import json
from argparse import Namespace

from .. import paths
from ..release import records as releases
from ..source import audit as source_audit


def cmd_audit(args: Namespace) -> int:
    from .. import __main__ as legacy

    root = paths.llama_root(args.llama_root)
    report = source_audit.audit(root)
    report["strict"] = args.strict
    out = paths.artifact_dir(report["source_revision"]) / "source-audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    good = source_audit.passed(report, strict=args.strict)
    print(f"source audit of {root}")
    print(
        f"  revision {report['source_revision'][:12]}{' (dirty)' if report['source_dirty'] else ''}"
    )
    print(source_audit.format_report(report, verbose=args.verbose))
    print(f"  report: {out}")
    print("  RESULT: " + ("PASS" if good else "FAIL"))
    record = legacy._record_for(root)
    record.audit = releases.summarise_audit(report, strict=args.strict)
    if not good:
        record.advance_to("broken")
    elif record.stage == "pulled":
        record.advance_to("audited")
    if not good:
        record.notes = "source audit failed: " + ", ".join(
            record.audit["failed_checks"]
        )
    elif record.notes.startswith("source audit failed:"):
        record.notes = ""
    record.save()
    return 0 if good else 1


def cmd_pull(args: Namespace) -> int:
    from .. import __main__ as legacy

    root = legacy.paths.llama_root(args.llama_root)

    # RE48: never move the checkout over an uncommitted pin change.
    uncommitted_pin = legacy._uncommitted_pin_change()
    if uncommitted_pin is not None and args.ref:
        print(
            f"refusing: the pin in config/recipes.toml is uncommitted "
            f"(working tree pins {uncommitted_pin!r}).",
            file=legacy.sys.stderr,
        )
        print(
            "commit the pin move (and the repin transition marker, if one "
            "was written) first, then pull -- or revert the pin change.",
            file=legacy.sys.stderr,
        )
        return 2

    # RE48: never move the checkout while a pin-transition marker exists
    # but is uncommitted (the bump's declaration commit is still missing).
    marker_path = legacy.paths.REPO_ROOT / "releases" / "pin-transition.json"
    if (
        marker_path.is_file()
        and legacy.pin_transition.committed_state(marker_path) != "committed-clean"
    ):
        print(
            "refusing: releases/pin-transition.json (the pin-transition "
            "marker) is uncommitted.",
            file=legacy.sys.stderr,
        )
        print(
            "commit it together with config/recipes.toml first (see "
            "docs/reference/PIN_BUMP.md), then pull -- or delete it if the "
            "bump was abandoned.",
            file=legacy.sys.stderr,
        )
        return 2

    # Ref resolution order: explicit --ref, else the recipe's, else stay put.
    # `latest` resolves against the remote here so what gets recorded is the
    # tag that was actually built, not a moving alias.
    ref = args.ref
    if ref is None and getattr(args, "recipe", None):
        try:
            ref = legacy.recipes.get(args.recipe).ref
        except legacy.recipes.RecipeError as exc:
            print(str(exc), file=legacy.sys.stderr)
            return 2
    try:
        if ref:
            resolved = legacy.upstream.resolve_ref(ref)
            if resolved != ref:
                print(f"{ref} -> {resolved}")
            ref = resolved
    except legacy.upstream.UpstreamError as exc:
        print(str(exc), file=legacy.sys.stderr)
        return 1

    if not (root / ".git").exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        print(f"cloning {legacy.UPSTREAM_URL} -> {root}")
        depth = [] if args.full else ["--depth", "1"]
        legacy._run(["git", "clone", *depth, legacy.UPSTREAM_URL, str(root)])
        # Upstream is LF throughout. Letting git rewrite line endings in the
        # working tree would make every generated diff unreviewable.
        legacy._run(["git", "-C", str(root), "config", "core.autocrlf", "false"])
    else:
        # A lock left by a killed git process (a prior timeout, an
        # interrupted step) blocks every ref write that touches it, silently
        # or with a hang depending on the codepath -- see upstream.py's
        # `clear_stale_locks` docstring for the incident this fixes. Not
        # calling this unconditionally elsewhere: it is only safe when we are
        # about to run the one git operation ourselves, i.e. right here.
        stale = legacy.upstream.clear_stale_locks(root)
        if stale:
            print(
                f"cleared {len(stale)} stale git lock(s) from an earlier "
                "interrupted run"
            )

        print(f"fetching into {root}")
        fetch = ["git", "-C", str(root), "fetch", "--no-tags"]
        if not args.full:
            fetch += ["--depth", "1"]
        fetch += ["origin", ref or "HEAD"]
        legacy._run(fetch)

    checkout_target = ref
    if ref:
        # A shallow checkout cannot reach a tag it never fetched, and plain
        # `fetch --tags` does not bring one down under a master-only refspec.
        # Fetch exactly this ref rather than making everyone unshallow.
        try:
            checkout_target = legacy.upstream.ensure_ref(
                root, ref, deepen=not args.full
            )
        except legacy.upstream.UpstreamError as exc:
            print(f"could not make {ref} available: {exc}", file=legacy.sys.stderr)
            return 1
        label = ref if checkout_target == ref else f"{ref} ({checkout_target})"
        print(f"checking out {label}")
        legacy._run(["git", "-C", str(root), "checkout", "--force", checkout_target])

    record = legacy._record_for(root)
    record.advance_to("pulled")
    record.save()
    print(
        f"at {record.revision[:12]}"
        f"{' (' + record.release_tag + ')' if record.release_tag else ''}"
    )
    return 0

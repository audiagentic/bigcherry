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
import re
import subprocess
import sys
from pathlib import Path

from . import __version__
from . import doctor
from . import paths
from . import patch_catalog
from . import patch_lifecycle
from . import patcher
from . import patchset
from . import pin_status
from . import pin_transition
from . import recipes
from . import releases
from . import source_audit
from . import sources
from . import upstream

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


_PIN_LINE = re.compile(r'^pinned\s*=\s*"([^"]+)"', re.MULTILINE)


def _uncommitted_pin_change() -> str | None:
    """RE48 enforcement hook A: the working-tree pin, when it differs from
    the committed config.

    A pin change that has not been committed is not yet a declared bump --
    the transition (pin commit + marker) is the declaration. Moving the
    checkout over an uncommitted pin change is exactly the manual path that
    produces drift, so pull refuses until it is committed (or reverted).
    """
    try:
        text = paths.RECIPES.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    working = _PIN_LINE.search(text)
    if working is None:
        return None
    committed = _git_out(
        paths.REPO_ROOT,
        "show",
        "HEAD:" + str(paths.RECIPES.relative_to(paths.REPO_ROOT)).replace("\\", "/"),
    )
    match = _PIN_LINE.search(committed) if committed else None
    committed_pin = match.group(1) if match else None
    if committed_pin is not None and committed_pin != working.group(1):
        return working.group(1)
    return None


def cmd_pull(args: argparse.Namespace) -> int:
    root = paths.llama_root(args.llama_root)

    # RE48: never move the checkout over an uncommitted pin change.
    uncommitted_pin = _uncommitted_pin_change()
    if uncommitted_pin is not None and args.ref:
        print(
            f"refusing: the pin in config/recipes.toml is uncommitted "
            f"(working tree pins {uncommitted_pin!r}).",
            file=sys.stderr,
        )
        print(
            "commit the pin move (and the repin transition marker, if one "
            "was written) first, then pull -- or revert the pin change.",
            file=sys.stderr,
        )
        return 2

    # RE48: never move the checkout while a pin-transition marker exists
    # but is uncommitted (the bump's declaration commit is still missing).
    marker_path = paths.REPO_ROOT / "releases" / "pin-transition.json"
    if (
        marker_path.is_file()
        and pin_transition.committed_state(marker_path) != "committed-clean"
    ):
        print(
            "refusing: releases/pin-transition.json (the pin-transition "
            "marker) is uncommitted.",
            file=sys.stderr,
        )
        print(
            "commit it together with config/recipes.toml first (see "
            "docs/reference/PIN_BUMP.md), then pull -- or delete it if the "
            "bump was abandoned.",
            file=sys.stderr,
        )
        return 2

    # Ref resolution order: explicit --ref, else the recipe's, else stay put.
    # `latest` resolves against the remote here so what gets recorded is the
    # tag that was actually built, not a moving alias.
    ref = args.ref
    if ref is None and getattr(args, "recipe", None):
        try:
            ref = recipes.get(args.recipe).ref
        except recipes.RecipeError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    try:
        if ref:
            resolved = upstream.resolve_ref(ref)
            if resolved != ref:
                print(f"{ref} -> {resolved}")
            ref = resolved
    except upstream.UpstreamError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not (root / ".git").exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        print(f"cloning {UPSTREAM_URL} -> {root}")
        depth = [] if args.full else ["--depth", "1"]
        _run(["git", "clone", *depth, UPSTREAM_URL, str(root)])
        # Upstream is LF throughout. Letting git rewrite line endings in the
        # working tree would make every generated diff unreviewable.
        _run(["git", "-C", str(root), "config", "core.autocrlf", "false"])
    else:
        # A lock left by a killed git process (a prior timeout, an
        # interrupted step) blocks every ref write that touches it, silently
        # or with a hang depending on the codepath -- see upstream.py's
        # `clear_stale_locks` docstring for the incident this fixes. Not
        # calling this unconditionally elsewhere: it is only safe when we are
        # about to run the one git operation ourselves, i.e. right here.
        stale = upstream.clear_stale_locks(root)
        if stale:
            print(
                f"cleared {len(stale)} stale git lock(s) from an earlier "
                f"interrupted run"
            )

        print(f"fetching into {root}")
        fetch = ["git", "-C", str(root), "fetch", "--no-tags"]
        if not args.full:
            fetch += ["--depth", "1"]
        fetch += ["origin", ref or "HEAD"]
        _run(fetch)

    checkout_target = ref
    if ref:
        # A shallow checkout cannot reach a tag it never fetched, and plain
        # `fetch --tags` does not bring one down under a master-only refspec.
        # Fetch exactly this ref rather than making everyone unshallow.
        try:
            checkout_target = upstream.ensure_ref(root, ref, deepen=not args.full)
        except upstream.UpstreamError as exc:
            print(f"could not make {ref} available: {exc}", file=sys.stderr)
            return 1
        label = ref if checkout_target == ref else f"{ref} ({checkout_target})"
        print(f"checking out {label}")
        _run(["git", "-C", str(root), "checkout", "--force", checkout_target])

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
    if not good:
        record.advance_to("broken")
    elif record.stage == "pulled":
        record.advance_to("audited")
    # A repeated audit is observational.  If later evidence already exists,
    # do not attempt to move its monotonic release stage backwards to
    # ``audited``; preserve the later stage while refreshing the audit report.
    if not good:
        record.notes = "source audit failed: " + ", ".join(
            record.audit["failed_checks"]
        )
    elif record.notes.startswith("source audit failed:"):
        record.notes = ""
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


def _apply_selection(
    root: Path,
    groups: frozenset[str] | None,
    states: frozenset[str] | None,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> bool:
    """Install the overlay and apply one patch selection. True if all placed.

    Used by ``apply``, the mutable-checkout diagnostic/development path
    (RE14 non-goal: raw-Path/mutable-checkout compatibility stays an
    explicit imported-legacy/development boundary for ad-hoc tuning; only
    the ``build`` command's own tree-flipping mechanics were retired, RE23).
    """
    record = _record_for(root)
    if not force and not record.audit.get("passed"):
        print(
            "refusing to patch a tree that has not passed a strict audit.\n"
            "  run `python -m bigcherry audit` first, or pass --force.",
            file=sys.stderr,
        )
        return False

    # An anchored edit extends an overlay source; install overlays before
    # resolving anchors so a fresh upstream clone follows the same path.
    written = _copy_overlay(root, dry_run=dry_run)
    patches = patchset.load_patches(groups=groups, states=states)
    results = patcher.apply_all(patches, root, dry_run=dry_run)
    ok = all(r.ok for r in results)
    intended_tree_state = recipes.tree_state_key(
        record.release_tag or record.revision, groups, states
    )
    selection_changed = record.tree_state != intended_tree_state
    tree_mutated = bool(written) or any(result.changed for result in results)

    if ok:
        verb = "would write" if dry_run else "wrote"
        print(f"overlay: {verb} {len(written)} file(s)")
    else:
        print("overlay: skipped -- patches failed")
    print(f"patches ({len(patches)} file(s)):")
    print(patcher.format_results(results))

    if not dry_run:
        record = _record_for(root)
        record.patches = releases.summarise_patches(results)
        releases.record_apply_result(
            record, ok, mutated=selection_changed or tree_mutated
        )
        if not ok:
            record.notes = "patches failed: " + ", ".join(
                record.patches["failed_edits"]
            )
        elif record.notes.startswith("patches failed:"):
            record.notes = ""
        # Key what the tree now carries, so `build` can tell whether it needs
        # a reset or can compile against this selection as-is.
        record.tree_state = intended_tree_state if ok else ""
        record.save()
    return ok


def cmd_apply(args: argparse.Namespace) -> int:
    root = paths.llama_root(args.llama_root)

    try:
        selection = _resolve_selection(args)
    except recipes.RecipeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    groups, states, label = selection

    ok = _apply_selection(root, groups, states, force=args.force, dry_run=args.dry_run)
    print(f"selection: {label}")
    print("  RESULT: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def _resolve_pin_sha(ref: str) -> str:
    """Resolve a pin ref to a full commit SHA in the vendor clone.

    RE48: the transition marker is SHA-keyed; an unresolvable ref fails the
    repin rather than writing a marker the verdict engine cannot match."""
    root = paths.llama_root()
    if not (root / ".git").exists():
        raise upstream.UpstreamError(
            f"vendor checkout missing at {root}; run 'python -m bigcherry pull' "
            "before repinning"
        )
    checkout_ref = upstream.ensure_ref(root, ref)
    resolved = ref if checkout_ref == ref else checkout_ref
    return upstream._git(root, "rev-parse", f"{resolved}^{{commit}}").strip()


def cmd_repin(args: argparse.Namespace) -> int:
    """Move the pin to the newest upstream release and declare the transition.

    RE48/RV78: repin now writes the pin-transition marker (releases/
    pin-transition.json) atomically with the recipes.toml rewrite. Commit
    both together -- the committed marker is what makes the in-flight state
    'mid-rebase' instead of 'drift' (see docs/reference/PIN_BUMP.md)."""
    try:
        target = args.ref or upstream.latest_release()
    except upstream.UpstreamError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    try:
        old = recipes.repin(target)
    except recipes.RecipeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if old == target:
        print(f"already pinned to {target}")
        return 0

    try:
        from_sha = _resolve_pin_sha(old)
        to_sha = _resolve_pin_sha(target)
        declaring = upstream._git(paths.REPO_ROOT, "rev-parse", "HEAD")
    except upstream.UpstreamError as exc:
        print(
            f"pin moved ({old} -> {target}) but the transition marker "
            f"could not be resolved: {exc}",
            file=sys.stderr,
        )
        print(
            "restore the previous pin or re-run repin; a pin move without "
            "a marker reads as drift in pin-status.",
            file=sys.stderr,
        )
        return 1

    pin_transition.write(from_sha, to_sha, target, declaring)
    print(f"pinned: {old} -> {target}")
    print(
        "recipes following the pin now build from it; recipes naming their "
        "own ref are unchanged."
    )
    print(f"transition marker: {pin_transition.MARKER_PATH} ({old} -> {target})")
    print(
        "next: COMMIT recipes.toml + the marker together, then "
        "python -m bigcherry pull --recipe <name>"
    )
    return 0


def _pin_status_paths() -> tuple[pin_status.RepoPaths, object, list]:
    """Local RepoPaths + config trees + the llama root, from the real config."""
    from . import config as campaign_config

    cfg = campaign_config.load(paths.RECIPES)
    repo_paths = pin_status.RepoPaths(
        repo_root=paths.REPO_ROOT,
        llama_root=paths.llama_root(),
        releases_dir=paths.REPO_ROOT / "releases",
        artifacts_dir=paths.REPO_ROOT / "artifacts",
    )
    return repo_paths, cfg, list(cfg.trees)


def _render_pin_status(report, trees) -> str:
    lines: list[str] = []
    local = report.local
    lines.append("pin-status: local")
    pinned = (
        f"{local.pinned_ref} -> {local.pinned_sha[:12]}"
        if local.pinned_sha
        else (local.pinned_ref or "(none)")
    )
    lines.append(f"  pin           {pinned}")
    vendor = f"{local.vendor_head[:12]}" if local.vendor_head else "(absent)"
    if local.vendor_tags:
        vendor += f" (tag {', '.join(local.vendor_tags)})"
    if local.vendor_head:
        vendor += (
            f"  [{local.vendor_modified} modified, {local.vendor_untracked} untracked]"
        )
    lines.append(f"  vendor HEAD   {vendor}")
    if local.records:
        recs = ", ".join(
            f"{r[:12]} {s}"
            for r, s in sorted(local.records.items(), key=lambda kv: kv[0])
        )
        lines.append(f"  releases      {recs}")
    if local.descriptors:
        desc = ", ".join(f"{d[:12]}" for d in local.descriptors)
        lines.append(f"  descriptors   {desc}")
    lines.append(f"  bigcherry     {(local.bigcherry_head or '?')[:12]}")
    if local.marker:
        lines.append(
            f"  marker        {local.marker.from_sha[:12]} -> "
            f"{local.marker.to_sha[:12]} ({local.marker.tag}) "
            f"[{local.marker_state}]"
        )
    lines.append(
        f"  VERDICT       {local.verdict}"
        + (f"  ({'; '.join(local.reasons)})" if local.reasons else "")
    )

    for status in report.remotes:
        lines.append(f"\npin-status: {status.name} (remote)")
        if not status.reachable:
            lines.append(f"  VERDICT       unreachable  ({'; '.join(status.reasons)})")
            continue
        lines.append(f"  vendor HEAD   {(status.vendor_head or 'none')[:12]}")
        lines.append(
            f"  pin           {status.pinned_ref or 'none'} "
            f"-> {(status.pinned_sha or '?')[:12]}"
        )
        lines.append(f"  bigcherry     {(status.bigcherry_head or 'none')[:12]}")
        lines.append(
            f"  VERDICT       {status.verdict}"
            + (f"  ({'; '.join(status.reasons)})" if status.reasons else "")
        )

    if report.remotes:
        if report.converged == True:  # noqa: E712 -- tri-state: None = no remotes
            lines.append("\nAGGREGATE  converged")
        elif report.converged == False:  # noqa: E712
            lines.append(
                f"\nAGGREGATE  DIVERGED  ({'; '.join(report.aggregate_reasons)})"
            )
    return "\n".join(lines)


def cmd_pin_status(args: argparse.Namespace) -> int:
    """RE48: name the pin state. Reads only -- never mutates."""
    try:
        repo_paths, cfg, trees = _pin_status_paths()
    except Exception as exc:  # ConfigError, RecipeError, OSError
        print(f"pin-status: config error: {exc}", file=sys.stderr)
        return 2

    if args.remote:
        trees = [t for t in trees if t.name == args.remote]
        if not trees:
            print(
                f"pin-status: no configured tree named {args.remote!r}", file=sys.stderr
            )
            return 2
    elif not args.all_remotes:
        trees = []

    try:
        report = pin_status.build_report(repo_paths, tuple(trees))
    except Exception as exc:
        print(f"pin-status: {exc}", file=sys.stderr)
        return 1

    if args.json:
        document = {
            "local": {
                "pinned_ref": report.local.pinned_ref,
                "pinned_sha": report.local.pinned_sha,
                "vendor_head": report.local.vendor_head,
                "vendor_tags": list(report.local.vendor_tags),
                "vendor_modified": report.local.vendor_modified,
                "vendor_untracked": report.local.vendor_untracked,
                "marker": report.local.marker.to_json()
                if report.local.marker
                else None,
                "marker_state": report.local.marker_state,
                "records": report.local.records,
                "descriptors": list(report.local.descriptors),
                "bigcherry_head": report.local.bigcherry_head,
                "verdict": report.local.verdict,
                "reasons": list(report.local.reasons),
            },
            "remotes": [
                {
                    "name": s.name,
                    "reachable": s.reachable,
                    "vendor_head": s.vendor_head,
                    "pinned_ref": s.pinned_ref,
                    "pinned_sha": s.pinned_sha,
                    "bigcherry_head": s.bigcherry_head,
                    "verdict": s.verdict,
                    "reasons": list(s.reasons),
                }
                for s in report.remotes
            ],
            "converged": report.converged,
            "aggregate_reasons": list(report.aggregate_reasons),
        }
        print(json.dumps(document, indent=2, sort_keys=True))
    else:
        print(_render_pin_status(report, trees))

    if args.complete:
        failures = pin_status.complete_failures(report, tuple(trees))
        document = {"pass": not failures, "reasons": failures}
        if args.json:
            pass  # JSON already emitted above
        else:
            print(f"\nCOMPLETION  {'PASS' if not failures else 'FAIL'}")
            for reason in failures:
                print(f"  - {reason}")
        return 1 if failures else 0

    if args.strict:
        failures = pin_status.strict_failure(report)
        if failures:
            for reason in failures:
                print(f"pin-status --strict FAIL: {reason}", file=sys.stderr)
            return 1
        if report.local.verdict == pin_status.VERDICT_MID_REBASE:
            print(
                "WARNING: tree is mid-rebase (a declared bump is in "
                "flight); proceeding only because the pipeline's source "
                "identity is revision-bound.",
                file=sys.stderr,
            )
        return 0
    return 0


def _overlay_relative_paths() -> list[Path]:
    """Paths the overlay writes, relative to the checkout root."""
    return [
        source.relative_to(paths.SRC_OVERLAY)
        for source in sorted(paths.SRC_OVERLAY.rglob("*"))
        if source.is_file()
    ]


def cmd_build_new(args: argparse.Namespace) -> int:
    """RE21: `build`'s new-engine implementation -- parse -> plan -> run_campaign
    -> render. No legacy tree-state/checkout mutation of any kind; this only
    ever touches ``context.work_root`` and a dedicated ArtifactStore, exactly
    like ``campaign-build``/``re14_real_run`` before it.

    Exit codes (RE21/RE22): 2 for invalid/unsupported request syntax
    (argparse itself raises this for any legacy-only flag, since this parser
    simply does not define them); 1 if one or more planned lanes execute and
    fail; 0 only if every planned lane succeeds.
    """
    from . import config as campaign_config
    from .artifacts import ArtifactStore
    from .campaign_lane import smoke_environment_for_hip_devices
    from .campaign_planner import (
        CampaignPlannerError,
        CampaignRequest,
        plan,
        run_campaign,
    )
    from .context import ProjectContext
    from .runtime_smoke import RuntimeSmokeSpec

    if sum(bool(x) for x in (args.lane, args.all, args.profile)) != 1:
        print(
            "build: pass exactly one of --profile, --all, or --lane (repeatable)",
            file=sys.stderr,
        )
        return 2

    context = ProjectContext.resolve(
        work_root=None, upstream_repo=Path(args.llama_root) if args.llama_root else None
    )
    try:
        cfg = campaign_config.load(context.config_path)
    except campaign_config.ConfigError as exc:
        print(f"build: {exc}", file=sys.stderr)
        return 2

    store = ArtifactStore(context.work_root / "artifacts-store")

    selectors: tuple[campaign_config.CampaignLaneSelector, ...] = ()
    profile_name = None
    if args.all:
        profile_name = "standard"
    elif args.profile:
        profile_name = args.profile
    else:
        parsed: list[campaign_config.CampaignLaneSelector] = []
        for raw in args.lane:
            parts = raw.split(":")
            if len(parts) != 3:
                print(
                    f"build: --lane {raw!r} must be SOURCE:BUILD:PLATFORM",
                    file=sys.stderr,
                )
                return 2
            parsed.append(campaign_config.CampaignLaneSelector(*parts))
        selectors = tuple(parsed)

    # RE48 preflight: never start a lane from a tree whose pin state is
    # drift / uncommitted-transition / unresolvable (fail closed, no
    # bypass). mid-rebase is allowed with a warning: the pipeline's source
    # identity is revision-bound. A campaign-only tree with no git checkout
    # at the clone source has nothing to guard (its lanes bind their own
    # source identity). Placed AFTER the exit-2 request-syntax checks above
    # so a malformed request reports its syntax error, not the tree state.
    try:
        pin_repo_paths = pin_status.RepoPaths(
            repo_root=context.project_root,
            llama_root=context.upstream_repo,
            releases_dir=context.project_root / "releases",
            artifacts_dir=context.project_root / "artifacts",
        )
        pin_failures, pin_mid_rebase = pin_status.campaign_preflight(
            context.upstream_repo, pin_repo_paths
        )
    except Exception as exc:
        print(f"build: pin preflight error: {exc}", file=sys.stderr)
        return 1
    if pin_failures:
        for reason in pin_failures:
            print(f"build: pin preflight FAIL: {reason}", file=sys.stderr)
        print(
            "build: run `bigcherry pin-status` for the full report and see "
            "docs/reference/PIN_BUMP.md (fail closed)",
            file=sys.stderr,
        )
        return 1
    if pin_mid_rebase:
        print(
            "WARNING: tree is mid-rebase (a declared bump is in flight); "
            "proceeding because the pipeline's source identity is "
            "revision-bound.",
            file=sys.stderr,
        )

    architectures = tuple(args.arch.split(",")) if args.arch else ()
    inventory = Path(args.inventory) if args.inventory else None
    winners = Path(args.winners) if args.winners else None
    validation = RuntimeSmokeSpec(model_path=Path(args.model)) if args.model else None
    inputs_by_build = {}
    validation_by_build = {}
    for build_name, build_cfg in cfg.builds.items():
        needed = build_cfg.needs
        provided = {}
        if inventory is not None and "inventory" in needed:
            provided["inventory"] = inventory
        if winners is not None and "promoted-winners" in needed:
            provided["promoted-winners"] = winners
        if provided:
            inputs_by_build[build_name] = tuple(sorted(provided.items()))
        if validation is not None:
            validation_by_build[build_name] = validation

    request = CampaignRequest(
        selectors=selectors,
        profile_name=profile_name,
        architectures=architectures,
        inputs_by_build=inputs_by_build,
        validation_by_build=validation_by_build,
        binary_relative_path=args.binary_relative_path,
        c_compiler=args.c_compiler,
        cxx_compiler=args.cxx_compiler,
        smoke_environment=smoke_environment_for_hip_devices(args.hip_visible_devices),
        experiment=args.experiment,
    )
    try:
        lanes = plan(request, cfg)
    except CampaignPlannerError as exc:
        print(f"build: {exc}", file=sys.stderr)
        return 2

    results = run_campaign(
        lanes, cfg=cfg, context=context, store=store, run_id=args.run_id
    )

    failed = 0
    for lid in sorted(results):
        result = results[lid]
        if isinstance(result, Exception):
            print(f"{lid}: FAILED -- {result}", file=sys.stderr)
            failed += 1
        else:
            print(
                f"{lid}: ok build_plan_id={result.build_plan_id} "
                f"workload_id={result.workload_id}"
            )
    if failed:
        print(f"build: {failed}/{len(results)} lane(s) failed", file=sys.stderr)
        return 1
    return 0


def _add_selection_args(parser: argparse.ArgumentParser) -> None:
    """The patch-selection flags, shared by every command that selects."""
    parser.add_argument(
        "--recipe",
        default=None,
        choices=recipes.names() or None,
        help="named build definition from config/recipes.toml (default: all patches)",
    )
    parser.add_argument(
        "--groups",
        default=None,
        help="comma-separated patch groups, overriding the recipe's "
        "(e.g. 'core'). Empty string selects none.",
    )
    parser.add_argument(
        "--states",
        default=None,
        help=f"comma-separated patch states, overriding the recipe's "
        f"({', '.join(patchset.STATES)}).",
    )


def _resolve_selection(
    args: argparse.Namespace,
) -> tuple[frozenset[str] | None, frozenset[str] | None, str]:
    """Patch selection from ``--recipe``, with ``--groups``/``--states`` on top.

    The explicit flags override the recipe axis by axis rather than replacing
    it, so ``--recipe release --states untested`` is a one-off question about
    a known configuration instead of a configuration of its own.
    """
    groups = states = None
    label_parts = []

    if getattr(args, "recipe", None):
        recipe = recipes.get(args.recipe)
        groups, states = recipe.groups, recipe.states
        label_parts.append(f"recipe={recipe.name} ref={recipe.ref}")

    override_groups = patchset.parse_filter(getattr(args, "groups", None))
    override_states = patchset.parse_filter(getattr(args, "states", None))
    if override_groups is not None:
        groups = override_groups
        label_parts.append("groups overridden")
    if override_states is not None:
        states = override_states
        label_parts.append("states overridden")

    def show(value: frozenset[str] | None) -> str:
        if value is None:
            return "all"
        return ",".join(sorted(value)) or "none"

    label_parts.append(f"groups={show(groups)} states={show(states)}")
    return groups, states, "  ".join(label_parts)


def cmd_patches(args: argparse.Namespace) -> int:
    """Show every patch, its metadata, and whether a selection takes it.

    --kind/--backend/--origin filter against patches/catalog.toml (RE30
    phase 1's declarative metadata) -- the metadata substitute for browsing
    a physical folder split. RE41 decided patches/ stays flat indefinitely
    because this filtering already answers the "which patches form the
    framework / are HIP vs Vulkan / came from an external fork" questions a
    directory move would otherwise exist to answer, with zero path churn.

    RE39: reads patches/ and patches/catalog.toml exactly once via a single
    CatalogSnapshot, instead of the two independent scans (patchset.describe()
    + patch_catalog.load_catalog()) this command used to make.
    """
    try:
        snapshot = patch_catalog.build_snapshot()
    except ValueError as exc:
        print(f"patches: could not load patches/catalog.toml: {exc}", file=sys.stderr)
        return 2
    if not snapshot.modules:
        print("no patches found", file=sys.stderr)
        return 1

    try:
        groups, states, label = _resolve_selection(args)
    except recipes.RecipeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    catalog_filter_active = bool(args.kind or args.backend or args.origin)

    root = paths.llama_root(args.llama_root)
    print(f"selection: {label}")
    print(f"checkout:  {root}")
    if catalog_filter_active:
        print(
            f"catalog:   kind={args.kind or 'any'} backend={args.backend or 'any'} "
            f"origin={args.origin or 'any'}"
        )
    print()

    rows, problems, selected = [], [], 0
    for module in snapshot.modules:
        entry = snapshot.entry_for(module.patch_id)
        if catalog_filter_active:
            if entry is None:
                continue
            if args.kind and entry.kind != args.kind:
                continue
            if args.backend and entry.backend != args.backend:
                continue
            if args.origin and entry.origin != args.origin:
                continue

        taken = (groups is None or module.group in groups) and (
            states is None or module.state in states
        )
        selected += taken

        note = ""
        if module.upstream:
            landed = patchset.upstream_landed(module.upstream, root)
            if landed:
                note = f"upstream {module.upstream[:8]} landed -- redundant here"
            elif landed is None:
                note = f"upstream {module.upstream[:8]} unknown"
            else:
                note = f"upstream {module.upstream[:8]} not in this checkout"

        if module.state not in patchset.STATES:
            problems.append(
                f"{module.patch_id}: STATE={module.state!r} is not one of "
                f"{', '.join(patchset.STATES)} -- no recipe will select it"
            )

        catalog_label = f"{entry.kind}/{entry.backend}" if entry is not None else ""
        rows.append(
            (
                "[x]" if taken else "[ ]",
                module.patch_id,
                module.group,
                module.state,
                catalog_label,
                note,
            )
        )

    if not rows:
        print("no patches match the given --kind/--backend/--origin filter")
        return 0

    widths = [max(len(r[i]) for r in rows) for i in range(5)]
    for mark, name, group, state, catalog_label, note in rows:
        line = (
            f"{mark} {name:<{widths[1]}}  {group:<{widths[2]}}  "
            f"{state:<{widths[3]}}  {catalog_label:<{widths[4]}}"
        )
        print(f"{line}  {note}".rstrip())

    print(
        f"\n{selected} of {len(rows)} shown selected"
        + (
            ""
            if not catalog_filter_active
            else f" ({len(snapshot.modules)} total in catalog)"
        )
    )
    for problem in problems:
        print(f"warning: {problem}", file=sys.stderr)
    return 1 if problems else 0


def cmd_patch_status(args: argparse.Namespace) -> int:
    """EC19: computed plan/patch lifecycle status -- see patch_lifecycle.py
    for exactly which signals are real vs reported as unknown."""
    statuses = patch_lifecycle.compute_all()
    if args.item:
        statuses = {k: v for k, v in statuses.items() if k == args.item}
        if not statuses:
            print(
                f"no lifecycle signal found for plan-item {args.item!r}",
                file=sys.stderr,
            )
            return 1
    if not statuses:
        print(
            "no plan-items with any tracked/materialized/contracted signal found",
            file=sys.stderr,
        )
        return 1
    print(patch_lifecycle.render_table(statuses))
    return 0


def cmd_patch_explain(args: argparse.Namespace) -> int:
    """RE43 (external patch-management review, Section 16): everything
    known about one patch -- source, plan, requires/conflicts, which
    recipes/experiments select it, state, content hash, files touched."""
    try:
        snapshot = patch_catalog.build_snapshot()
    except ValueError as exc:
        print(
            f"patch explain: could not load patches/catalog.toml: {exc}",
            file=sys.stderr,
        )
        return 2
    from . import config as campaign_config

    try:
        cfg = campaign_config.load(paths.RECIPES)
    except (campaign_config.ConfigError, OSError):
        cfg = None
    try:
        info = patch_catalog.explain(args.patch_id, snapshot, cfg)
    except KeyError as exc:
        print(f"patch explain: {exc}", file=sys.stderr)
        return 1
    print(patch_catalog.render_explanation(info))
    return 0


def cmd_patch_graph(args: argparse.Namespace) -> int:
    """RE43 (external patch-management review, Section 16): textual
    REQUIRES/CONFLICTS dependency topology. --roots restricts the view to
    a real dependency closure (patchset.expand_composition); with no
    roots, shows every patch that has any requires/conflicts edge."""
    try:
        snapshot = patch_catalog.build_snapshot()
    except ValueError as exc:
        print(
            f"patch graph: could not load patches/catalog.toml: {exc}", file=sys.stderr
        )
        return 2
    roots = tuple(args.roots or ())
    try:
        print(patch_catalog.dependency_graph(snapshot, roots=roots))
    except ValueError as exc:
        print(f"patch graph: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_patch_verify_evidence(args: argparse.Namespace) -> int:
    """HI83: report which STATE='validated' patches have a current,
    qualifying patch_validation_evidence record -- purely observational,
    not wired into apply/build (see patch_catalog.py's module docstring
    and plan item HI83's notes for why hard enforcement is deliberately
    deferred)."""
    from . import config as campaign_config

    cfg = campaign_config.load(paths.RECIPES)
    modules = patchset.catalog()

    if args.patch_id is not None:
        modules = [module for module in modules if module.patch_id == args.patch_id]
        if not modules:
            print(f"unknown patch {args.patch_id!r}", file=sys.stderr)
            return 1

    patch_ids = [module.patch_id for module in modules]
    statuses = patch_catalog.validation_evidence_statuses(
        patch_ids, pinned_ref=cfg.pinned,
        allow_legacy_grandfather=not args.no_legacy_grandfather,
    )

    if args.json:
        print(json.dumps(
            {
                patch_id: {
                    "status": status.status,
                    "problems": list(status.problems),
                    "campaign_digests": list(status.campaign_digests),
                }
                for patch_id, status in sorted(statuses.items())
            },
            indent=2, sort_keys=True,
        ))
    else:
        for patch_id, status in sorted(statuses.items()):
            print(f"{patch_id}: {status.status}")
            for problem in status.problems:
                print(f"  - {problem}")

    return 0 if all(status.ok for status in statuses.values()) else 1


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
    if args.generated_root:
        forwarded += ["--generated-root", args.generated_root]
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


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report migration assumptions without modifying source or build state."""
    return doctor.main(as_json=args.json)


# --------------------------------------------------------------------- main


def cmd_replay_inspect(args: argparse.Namespace) -> int:
    """HI15/HI16: registry + cache inspection through the real C++ loader."""
    from . import replay_inspect

    tool = replay_inspect.find_tool(args.tool)
    report = replay_inspect.run_tool(
        tool,
        cache=Path(args.cache) if args.cache else None,
        interpreter=args.tool_interpreter or None,
    )
    if args.manifest:
        report["manifest"] = replay_inspect.manifest_agreement(
            report, Path(args.manifest)
        )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(replay_inspect.format_report(report))
    # The C++ exit code is the report's own classification; carry it through
    # so the tool composes as a gate (0 ok, 1 registry anomaly, 3 cache
    # rejected, 4 loaded-but-nothing-usable).
    return report["_exit"]


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
        "--ref",
        default=None,
        help="tag, branch or sha to check out (e.g. b1234), or 'latest' for "
        "the newest upstream release. Overrides --recipe.",
    )
    pull.add_argument(
        "--recipe",
        default=None,
        choices=recipes.names() or None,
        help="take the ref from this recipe in config/recipes.toml",
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
    _add_selection_args(apply_cmd)
    apply_cmd.set_defaults(func=cmd_apply)

    patches_cmd = sub.add_parser(
        "patches",
        help="list patches with group, state and upstream status",
    )
    patches_cmd.add_argument("--llama-root", default=None)
    _add_selection_args(patches_cmd)
    patches_cmd.add_argument(
        "--kind",
        default=None,
        choices=patch_catalog.KINDS,
        help="filter to patches/catalog.toml's kind (framework|upstream-backport|"
        "enhancement) -- the metadata substitute for a physical folder split "
        "(RE41: patches/ stays flat, this filter is the browsability answer)",
    )
    patches_cmd.add_argument(
        "--backend",
        default=None,
        choices=patch_catalog.BACKENDS,
        help="filter to patches/catalog.toml's backend (hip|vulkan|agnostic)",
    )
    patches_cmd.add_argument(
        "--origin",
        default=None,
        choices=patch_catalog.ORIGINS,
        help="filter to patches/catalog.toml's origin (local|upstream-commit|"
        "upstream-pr|external-fork)",
    )
    patches_cmd.set_defaults(func=cmd_patches)

    patch_status_cmd = sub.add_parser(
        "patch-status",
        help="EC19: computed plan/patch/contract lifecycle status, not "
        "hand-maintained prose -- source-pinned/materialized/build-state/"
        "contracted per RD/EX/HI plan item",
    )
    patch_status_cmd.add_argument(
        "--item",
        default=None,
        help="show only this plan-item (e.g. RD21) instead of every item "
        "with any signal",
    )
    patch_status_cmd.set_defaults(func=cmd_patch_status)

    patch_explain_cmd = sub.add_parser(
        "patch-explain",
        help="RE43: everything known about one patch -- source, plan, "
        "requires/conflicts, which recipes/experiments select it, "
        "state, content hash, files touched",
    )
    patch_explain_cmd.add_argument(
        "patch_id", help="e.g. 1217_rd44_graph_opt_default_rdna35"
    )
    patch_explain_cmd.set_defaults(func=cmd_patch_explain)

    patch_graph_cmd = sub.add_parser(
        "patch-graph",
        help="RE43: textual REQUIRES/CONFLICTS dependency topology",
    )
    patch_graph_cmd.add_argument(
        "--roots",
        action="append",
        default=None,
        help="restrict to this patch's real dependency closure (repeatable); "
        "omit to show every patch with any requires/conflicts edge",
    )
    patch_graph_cmd.set_defaults(func=cmd_patch_graph)

    patch_verify_evidence_cmd = sub.add_parser(
        "patch-verify-evidence",
        help="HI83: verify that every STATE='validated' patch has current "
        "matching validation evidence (observational only -- not wired "
        "into apply/build)",
    )
    patch_verify_evidence_cmd.add_argument(
        "patch_id", nargs="?", default=None,
        help="check only this patch instead of every patch in the catalog",
    )
    patch_verify_evidence_cmd.add_argument("--json", action="store_true")
    patch_verify_evidence_cmd.add_argument(
        "--no-legacy-grandfather", action="store_true",
        help="require real HI83 evidence; do not accept the one-time legacy baseline",
    )
    patch_verify_evidence_cmd.set_defaults(func=cmd_patch_verify_evidence)

    sources.register(sub)

    repin = sub.add_parser(
        "repin", help="move config/recipes.toml's pin to the newest upstream release"
    )
    repin.add_argument(
        "--ref",
        default=None,
        help="pin to this ref instead of querying for the newest release",
    )
    repin.set_defaults(func=cmd_repin)

    pin_status_cmd = sub.add_parser(
        "pin-status",
        help=(
            "name the pin state of this tree (and configured remotes): "
            "consistent / mid-rebase / drift; --strict = pipeline preflight, "
            "--complete = bump-completion gate"
        ),
    )
    mode = pin_status_cmd.add_mutually_exclusive_group()
    mode.add_argument(
        "--strict",
        action="store_true",
        help="pipeline preflight: fail on drift/unavailable/"
        "unresolvable-pin/uncommitted-transition of the "
        "gated tree",
    )
    mode.add_argument(
        "--complete",
        action="store_true",
        help="bump-completion gate: every required tree "
        "reachable, converged and consistent",
    )
    pin_status_cmd.add_argument(
        "--json", action="store_true", help="machine-readable report"
    )
    remotes = pin_status_cmd.add_mutually_exclusive_group()
    remotes.add_argument(
        "--remote", default=None, help="probe only this configured tree"
    )
    remotes.add_argument(
        "--all-remotes", action="store_true", help="probe every configured tree"
    )
    pin_status_cmd.set_defaults(func=cmd_pin_status)

    replay_inspect_cmd = sub.add_parser(
        "replay-inspect",
        help=(
            "HI15/HI16: inspect the registry and a replay cache through the "
            "real C++ loader; --manifest adds catalog<->registry agreement"
        ),
    )
    replay_inspect_cmd.add_argument(
        "cache",
        nargs="?",
        default=None,
        help="replay cache to inspect (omit for registry only)",
    )
    replay_inspect_cmd.add_argument(
        "--manifest",
        default=None,
        help="build manifest JSON; check catalog<->registry agreement",
    )
    replay_inspect_cmd.add_argument(
        "--tool",
        default=None,
        help="path to the hip-autotune-inspect binary "
        "(default: $BIGCHERRY_INSPECT_TOOL, then build dirs)",
    )
    replay_inspect_cmd.add_argument(
        "--tool-interpreter",
        nargs="*",
        default=None,
        metavar="ARG",
        help="command prefix for invoking the tool (e.g. a python script "
        "stand-in); the tool path is appended to it",
    )
    replay_inspect_cmd.add_argument(
        "--json", action="store_true", help="machine-readable report"
    )
    replay_inspect_cmd.set_defaults(func=cmd_replay_inspect)

    # RE21/RE23: `build` is the multi-lane planner/runner (RE18) and nothing
    # else -- a canonical-v2 interface only, never a translation layer for
    # the retired mutable-checkout recipe model (--recipe/--groups/--states/
    # --variant-set/--force/--target selected compat recipes and mutated
    # patch-selection axes on one shared checkout; canonical v2 sources
    # instead name exact patch sets directly, each build isolated by
    # content-addressed identity). Any such flag is simply not defined on
    # this parser, so argparse itself rejects it with exit 2 ("unrecognized
    # arguments") -- fail closed, never silent routing anywhere.
    # --inventory/--winners DO translate (they distribute to whichever
    # standard lanes declare that need, per Build.needs), since they are
    # canonical v2 concepts (CampaignLaneExecutionSpec.inputs), not legacy
    # ones. The `legacy-build` compatibility/diagnostic command that used
    # to exist alongside this was deleted in RE23 once the cutover's
    # objective compatibility gates (RE15's real-hardware acceptance chain,
    # both platform suites, RE24's adversarial matrix) were all satisfied.
    new_build_cmd = sub.add_parser(
        "build",
        help="build via the multi-lane campaign engine (canonical v2 identities only)",
    )
    new_build_cmd.add_argument("--llama-root", default=None)
    new_build_cmd.add_argument("--source", default="bigcherry")
    new_build_cmd.add_argument(
        "--profile",
        default=None,
        help="named campaign profile from config/recipes.toml's [campaign.<name>] "
        "(e.g. 'standard')",
    )
    new_build_cmd.add_argument(
        "--lane",
        action="append",
        default=None,
        metavar="SOURCE:BUILD:PLATFORM",
        help="explicit lane selector (repeatable); alternative to --profile, "
        "not combinable with it",
    )
    new_build_cmd.add_argument(
        "--all",
        action="store_true",
        help="build the canonical standard profile -- shorthand for --profile standard",
    )
    new_build_cmd.add_argument(
        "--arch",
        default=None,
        help="comma-separated architectures, overriding each lane's "
        "platform.targets (must be a non-empty subset)",
    )
    new_build_cmd.add_argument(
        "--inventory",
        default=None,
        help="signature inventory JSON, distributed to any planned lane "
        'whose build declares needs = ["inventory", ...]',
    )
    new_build_cmd.add_argument(
        "--winners",
        default=None,
        help="promoted-winners JSONL, distributed to any planned lane whose "
        'build declares needs including "promoted-winners"',
    )
    new_build_cmd.add_argument(
        "--model",
        default=None,
        help="gguf model path -- if given, every planned lane runs a real "
        "runtime-smoke validation against it",
    )
    new_build_cmd.add_argument(
        "--hip-visible-devices",
        default="0",
        help="only meaningful together with --model",
    )
    new_build_cmd.add_argument(
        "--binary-relative-path",
        default="bin/llama-bench",
        help="which binary each planned lane builds/publishes as its "
        "primary artifact, e.g. 'bin/llama-server' for a real "
        "production/deployment build -- matches campaign-build's own "
        "flag of the same name (re14_real_run.py); every lane in the "
        "request gets the same value, there is no per-lane override",
    )
    new_build_cmd.add_argument("--c-compiler", default=None)
    new_build_cmd.add_argument("--cxx-compiler", default=None)
    new_build_cmd.add_argument("--run-id", default=None)
    new_build_cmd.add_argument(
        "--experiment",
        default=None,
        help="name of a [experiment.<name>] entry in config/recipes.toml (an exact "
        "extra patch list) -- for benching one experimental patch in "
        "isolation against the source's normal patch-set, e.g. "
        "'--source bigcherry-native --experiment rd19-only'",
    )
    new_build_cmd.set_defaults(func=cmd_build_new)

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
    generate.add_argument(
        "--generated-root",
        default=None,
        help="build-local directory for generated compile inputs",
    )
    generate.add_argument("--dry-run", action="store_true")
    generate.add_argument(
        "--force", action="store_true", help="generate even against an unpatched tree"
    )
    generate.set_defaults(func=cmd_generate)

    status = sub.add_parser("status", help="show checkout and release status")
    status.set_defaults(func=cmd_status)

    experiment_cmd = sub.add_parser(
        "experiment-contract",
        help="Experiment Contract validate/list/plan/run/report (EC01-EC11) -- "
        "not to be confused with `experiment`, HI47's managed bundle runner",
    )
    experiment_sub = experiment_cmd.add_subparsers(
        dest="experiment_command", required=True
    )

    def _add_contracts_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--contracts",
            default=None,
            help="path to the contract registry TOML (default: config/experiment-contracts.toml)",
        )

    def _add_lane_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--config",
            default=None,
            help="path to recipes.toml (default: config/recipes.toml)",
        )
        p.add_argument(
            "--source",
            required=True,
            help="config.Source name to expand the contract against",
        )
        p.add_argument("--build", required=True, help="config.Build name")
        p.add_argument("--platform", required=True, help="config.Platform name")

    experiment_validate = experiment_sub.add_parser(
        "validate",
        help="schema-check a contract (or every contract) without running anything",
    )
    _add_contracts_arg(experiment_validate)
    experiment_validate.add_argument("contract_id", nargs="?", default=None)
    experiment_validate.set_defaults(func=cmd_experiment_validate)

    experiment_list = experiment_sub.add_parser(
        "list", help="list every registered contract"
    )
    _add_contracts_arg(experiment_list)
    experiment_list.set_defaults(func=cmd_experiment_list)

    experiment_plan = experiment_sub.add_parser(
        "plan", help="show the campaign lanes a contract would expand into (dry-run)"
    )
    _add_contracts_arg(experiment_plan)
    _add_lane_args(experiment_plan)
    experiment_plan.add_argument("contract_id")
    experiment_plan.set_defaults(func=cmd_experiment_plan)

    experiment_run = experiment_sub.add_parser(
        "run", help="execute a contract's full lane set through run_campaign()"
    )
    _add_contracts_arg(experiment_run)
    _add_lane_args(experiment_run)
    experiment_run.add_argument("contract_id")
    experiment_run.set_defaults(func=cmd_experiment_run)

    experiment_report = experiment_sub.add_parser(
        "report", help="render a contract's report from a stored evidence JSON file"
    )
    _add_contracts_arg(experiment_report)
    experiment_report.add_argument("contract_id")
    experiment_report.add_argument(
        "--evidence-file",
        required=True,
        help="JSON with correctness_gate/aggregated_effects/generalisation_result "
        "(and optionally promotion_gate) keys",
    )
    experiment_report.set_defaults(func=cmd_experiment_report)

    doctor_cmd = sub.add_parser(
        "doctor", help="audit migration assumptions and identity inputs"
    )
    doctor_cmd.add_argument("--json", action="store_true")
    doctor_cmd.set_defaults(func=cmd_doctor)

    tune_journal_cmd = sub.add_parser(
        "tune-journal", help="crash-safe tuning journal status/compaction (HI48)"
    )
    tune_journal_cmd.set_defaults(
        func=lambda args: _tune_journal_main(args.tune_journal_args)
    )
    tune_journal_cmd.add_argument("tune_journal_args", nargs=argparse.REMAINDER)

    tune_promote_cmd = sub.add_parser(
        "tune-promote",
        help="apply experiment-wide BH promotion to fresh-confirmation evidence (HI34)",
    )
    tune_promote_cmd.add_argument("measurements")
    tune_promote_cmd.add_argument("--output", required=True)
    tune_promote_cmd.add_argument(
        "--dispatch-db",
        required=True,
        help="dispatch database (schema 6+) this measurements file was ingested "
        "into, e.g. via `bigcherry inventory tuning` -- required so a "
        "non-native winner's CPU-reference correctness evidence (HI67) "
        "can be checked as a hard AND with the statistical promotion "
        "criteria",
    )
    tune_promote_cmd.add_argument("--q", type=float, default=0.05)
    tune_promote_cmd.add_argument("--threshold-pct", type=float, default=1.0)
    tune_promote_cmd.add_argument("--resamples", type=int, default=10_000)
    tune_promote_cmd.set_defaults(
        func=lambda args: _tune_promote_main(
            [
                args.measurements,
                "--output",
                args.output,
                "--dispatch-db",
                args.dispatch_db,
                "--q",
                str(args.q),
                "--threshold-pct",
                str(args.threshold_pct),
                "--resamples",
                str(args.resamples),
            ]
        )
    )

    tune_null_fdr_cmd = sub.add_parser(
        "tune-null-fdr",
        help="deterministic global-null BH simulation, for auditing the promotion gate",
    )
    tune_null_fdr_cmd.add_argument("--output", required=True)
    tune_null_fdr_cmd.add_argument("--experiments", type=int, default=5000)
    tune_null_fdr_cmd.add_argument("--hypotheses", type=int, required=True)
    tune_null_fdr_cmd.add_argument("--q", type=float, default=0.05)
    tune_null_fdr_cmd.add_argument("--seed", type=int, required=True)
    tune_null_fdr_cmd.set_defaults(
        func=lambda args: _tune_null_fdr_main(
            [
                "--output",
                args.output,
                "--experiments",
                str(args.experiments),
                "--hypotheses",
                str(args.hypotheses),
                "--q",
                str(args.q),
                "--seed",
                str(args.seed),
            ]
        )
    )

    experiment_cmd = sub.add_parser(
        "experiment", help="managed experiment bundle: run or validate (HI47)"
    )
    experiment_cmd.set_defaults(
        func=lambda args: _experiment_main(args.experiment_args)
    )
    experiment_cmd.add_argument("experiment_args", nargs=argparse.REMAINDER)

    # RE14: the new, content-addressed, isolated-worktree campaign path,
    # registered as a real subcommand rather than remaining only a
    # standalone script -- not yet the default execution path (that flip
    # is RE14 step 7, gated on further negative-case coverage). Legacy
    # `build` above is completely untouched by this and remains the normal
    # path until that flip happens.
    campaign_build_cmd = sub.add_parser(
        "campaign-build",
        help="RE14: build via the new isolated/content-addressed campaign "
        "path (not yet the default -- see `build` for the normal path)",
    )
    campaign_build_cmd.set_defaults(
        func=lambda args: _campaign_build_main(args.campaign_build_args)
    )
    campaign_build_cmd.add_argument("campaign_build_args", nargs=argparse.REMAINDER)

    from . import compare_tunes as _compare_tunes

    compare = sub.add_parser(
        "compare-tunes", help="compare two current tuning runs by signature"
    )
    compare.add_argument("before")
    compare.add_argument("after")
    compare.add_argument(
        "--record", default=None, help="record JSONL for call-weighted impact"
    )
    compare.add_argument("--output", default=None, help="JSON report path")

    def _run_compare(args):
        try:
            result = _compare_tunes.compare(
                Path(args.before),
                Path(args.after),
                record=Path(args.record) if args.record else None,
            )
        except (OSError, ValueError, _compare_tunes.CompareError) as exc:
            print(f"invalid: {exc}", file=sys.stderr)
            return 1
        rendered = json.dumps(result, indent=2, sort_keys=True)
        if args.output:
            Path(args.output).write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return 0

    compare.set_defaults(func=_run_compare)

    ab = sub.add_parser(
        "ab-benchmark",
        help="paired, interleaved native-versus-replay end-to-end benchmark",
    )
    ab.add_argument("--cache", required=True)
    ab.add_argument("--output", required=True)
    ab.add_argument("--pairs", type=int, default=3)
    ab.add_argument("--schedule-seed", type=int, default=0)
    ab.add_argument("--structured", action="store_true")
    ab.add_argument("--practical-threshold-pct", type=float, default=1.0)
    ab.add_argument("--decision-grade", action="store_true")
    ab.add_argument("--settle-seconds", type=float, default=20.0)
    ab.add_argument("--cwd", default=None)
    ab.add_argument("--metric", action="append", default=[])
    ab.add_argument("--lower-is-better", action="append", default=[])
    ab.add_argument("--stock-binary", default=None)
    ab.add_argument("--stock-cmake-cache", default=None)
    ab.add_argument("--patched-cmake-cache", default=None)
    ab.add_argument("command", nargs=argparse.REMAINDER)
    ab.set_defaults(
        func=lambda args: _ab_benchmark_main(
            [
                "--cache",
                args.cache,
                "--output",
                args.output,
                "--pairs",
                str(args.pairs),
                "--schedule-seed",
                str(args.schedule_seed),
                "--practical-threshold-pct",
                str(args.practical_threshold_pct),
                *(["--structured"] if args.structured else []),
                *(["--decision-grade"] if args.decision_grade else []),
                "--settle-seconds",
                str(args.settle_seconds),
                *(["--cwd", args.cwd] if args.cwd else []),
                *(item for spec in args.metric for item in ["--metric", spec]),
                *(
                    item
                    for name in args.lower_is_better
                    for item in ["--lower-is-better", name]
                ),
                *(["--stock-binary", args.stock_binary] if args.stock_binary else []),
                *(
                    ["--stock-cmake-cache", args.stock_cmake_cache]
                    if args.stock_cmake_cache
                    else []
                ),
                *(
                    ["--patched-cmake-cache", args.patched_cmake_cache]
                    if args.patched_cmake_cache
                    else []
                ),
                "--",
                *args.command,
            ]
        )
    )

    validate_release_cmd = sub.add_parser(
        "probe-release",
        help="probe patch compatibility against a ref in an isolated checkout (HI46)",
    )
    validate_release_cmd.add_argument("--run-id", required=True)
    validate_release_cmd.add_argument("--staging-root", default=None)
    validate_release_cmd.add_argument("--ref", default="master")
    validate_release_cmd.add_argument("--recipe", default="bigcherry")
    validate_release_cmd.add_argument("--inventory", default=None)
    validate_release_cmd.set_defaults(
        func=lambda args: _validate_release_main(
            [
                "--run-id",
                args.run_id,
                "--ref",
                args.ref,
                "--recipe",
                args.recipe,
                *(["--inventory", args.inventory] if args.inventory else []),
                *(["--staging-root", args.staging_root] if args.staging_root else []),
            ]
        )
    )

    validate_ref_cmd = sub.add_parser(
        "validate-ref",
        help="alias for the isolated patch/build compatibility probe (HI46)",
    )
    validate_ref_cmd.add_argument("--run-id", required=True)
    validate_ref_cmd.add_argument("--staging-root", default=None)
    validate_ref_cmd.add_argument("--ref", default="master")
    validate_ref_cmd.add_argument("--recipe", default="bigcherry")
    validate_ref_cmd.add_argument("--inventory", default=None)
    validate_ref_cmd.add_argument("--promoted-winners", default=None)
    validate_ref_cmd.set_defaults(
        func=lambda args: _validate_release_main(
            [
                "--run-id",
                args.run_id,
                "--ref",
                args.ref,
                "--recipe",
                args.recipe,
                *(["--inventory", args.inventory] if args.inventory else []),
                *(
                    ["--promoted-winners", args.promoted_winners]
                    if args.promoted_winners
                    else []
                ),
                *(["--staging-root", args.staging_root] if args.staging_root else []),
            ]
        )
    )

    rank_replay_cmd = sub.add_parser(
        "rank-replay",
        help="report/replay ranking-policy decisions recorded in a measurements file (HI50)",
    )
    rank_replay_cmd.add_argument("measurements")
    rank_replay_cmd.add_argument(
        "--dispatch", help="full per-policy candidate detail for one dispatch"
    )
    rank_replay_cmd.add_argument(
        "--verify-parity",
        action="store_true",
        help="assert the production policy's pick matches provisional_winner",
    )
    rank_replay_cmd.add_argument(
        "--policy-module",
        help="registry name, dotted module path, or .py file of a "
        "not-yet-installed policy to replay alongside the recorded ones",
    )
    rank_replay_cmd.add_argument("--output", help="write the JSON report here too")
    rank_replay_cmd.add_argument(
        "--json", action="store_true", help="print JSON instead of a text summary"
    )
    rank_replay_cmd.set_defaults(
        func=lambda args: _rank_replay_main(
            [
                args.measurements,
                *(["--dispatch", args.dispatch] if args.dispatch else []),
                *(["--verify-parity"] if args.verify_parity else []),
                *(
                    ["--policy-module", args.policy_module]
                    if args.policy_module
                    else []
                ),
                *(["--output", args.output] if args.output else []),
                *(["--json"] if args.json else []),
            ]
        )
    )

    resource = sub.add_parser(
        "resource-report", help="parse and policy-check a compiler resource stream"
    )
    resource.add_argument("raw")
    resource.add_argument("--symbol-map", required=True)
    resource.add_argument("--output", required=True)
    resource.add_argument("--compiler-family", default="clang")
    resource.add_argument("--compiler-major", type=int, required=True)
    resource.add_argument("--compiler-version", required=True)
    resource.add_argument("--architecture", required=True)
    resource.add_argument("--source-revision", required=True)
    resource.add_argument("--manifest-hash", required=True)
    resource.add_argument("--reject-lds-gt", type=int, default=None)
    resource.add_argument("--warn-occupancy-lt", type=float, default=None)
    resource.set_defaults(
        func=lambda args: _resource_report_main(
            [
                args.raw,
                "--symbol-map",
                args.symbol_map,
                "--output",
                args.output,
                "--compiler-family",
                args.compiler_family,
                "--compiler-major",
                str(args.compiler_major),
                "--compiler-version",
                args.compiler_version,
                "--architecture",
                args.architecture,
                "--source-revision",
                args.source_revision,
                "--manifest-hash",
                args.manifest_hash,
                *(
                    ["--reject-lds-gt", str(args.reject_lds_gt)]
                    if args.reject_lds_gt is not None
                    else []
                ),
                *(
                    ["--warn-occupancy-lt", str(args.warn_occupancy_lt)]
                    if args.warn_occupancy_lt is not None
                    else []
                ),
            ]
        )
    )

    binsize = sub.add_parser(
        "candidate-binary-size",
        help="per-candidate device .text size from a built HIP library",
    )
    binsize.add_argument("library")
    binsize.add_argument("--manifest", required=True)
    binsize.add_argument("--output", required=True)
    binsize.add_argument("--workdir", default=None)
    binsize.add_argument("--symbol-map-dir", default=None)
    binsize.add_argument("--objdump", default=None)
    binsize.add_argument("--readelf", default=None)
    binsize.add_argument("--allow-unresolved", action="store_true")
    binsize.set_defaults(
        func=lambda args: _candidate_binary_size_main(
            [
                args.library,
                "--manifest",
                args.manifest,
                "--output",
                args.output,
                *(["--workdir", args.workdir] if args.workdir else []),
                *(
                    ["--symbol-map-dir", args.symbol_map_dir]
                    if args.symbol_map_dir
                    else []
                ),
                *(["--objdump", args.objdump] if args.objdump else []),
                *(["--readelf", args.readelf] if args.readelf else []),
                *(["--allow-unresolved"] if args.allow_unresolved else []),
            ]
        )
    )

    from . import report as _report

    _report.build_parser(sub)

    from . import impact as _impact
    from . import kernel_fraction as _kernel_fraction

    _impact.build_parser(sub)
    _kernel_fraction.build_parser(sub)

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

    # Hot list: rank observed signatures by estimated time contribution
    # (HI24 steps 5-6), consumed by GGML_HIP_TUNE_HOT_SIGNATURES.
    inv_hot = inv_sub.add_parser(
        "hot-list", help="Rank observed signatures by estimated time contribution"
    )
    inv_hot.add_argument("record", help="JSONL written by GGML_HIP_DISPATCH_DB")
    inv_hot.add_argument(
        "--measurements",
        default=None,
        help="a previous tune's .measurements.jsonl; upgrades the ranking "
        "from calls x est_bytes to calls x native_median_us",
    )
    inv_hot.add_argument(
        "--output", default=None, help="hot list to write (default: alongside)"
    )
    inv_hot.set_defaults(func=lambda args: cmd_inventory(args, subcmd="hot-list"))

    return parser


def cmd_inventory(args: argparse.Namespace, *, subcmd: str) -> int:
    """Dispatch to inventory record/tuning subcommand."""
    from pathlib import Path

    from . import inventory as inv_mod

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

    elif subcmd == "hot-list":
        record_path = Path(args.record)
        if not record_path.is_file():
            print(f"no such record file: {record_path}", file=sys.stderr)
            return 2
        try:
            record = inv_mod.read_jsonl(record_path)
        except inv_mod.RecordError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        output = (
            Path(args.output) if args.output else record_path.with_suffix(".hot")
        )
        measurements = Path(args.measurements) if args.measurements else None
        summary = inv_mod.write_hot_list(record, output, measurements=measurements)

        print(f"wrote {output}")
        print(f"  {summary['signatures']} signature(s), basis {summary['basis']}")
        for row in summary["rows"][:5]:
            print(
                f"  {row['rank']:>3}  {row['signature'][:16]}  "
                f"{row['calls']:>8} calls  {row['share_pct']:6.2f}%  "
                f"cum {row['cum_share_pct']:6.2f}%"
            )
        return 0

    else:
        # Backward compat: positional arg means record mode (no subcommand)
        return cmd_inventory(args, subcmd="record")


def _tune_journal_main(argv: list[str]) -> int:
    from . import tune_journal

    return tune_journal.main(argv)


def _tune_promote_main(argv: list[str]) -> int:
    from . import tune_promotion

    return tune_promotion.main(argv)


def _tune_null_fdr_main(argv: list[str]) -> int:
    from . import tune_promotion

    return tune_promotion.null_fdr_main(argv)


def _experiment_main(argv: list[str]) -> int:
    from . import experiment_bundle

    return experiment_bundle.main(argv)


def _ab_benchmark_main(argv: list[str]) -> int:
    from . import ab_benchmark

    return ab_benchmark.main(argv)


def _campaign_build_main(argv: list[str]) -> int:
    from . import re14_real_run

    # REMAINDER captures a leading "--" (needed so the outer parser doesn't
    # try to consume flags like --upstream-repo itself) literally as part
    # of argv -- strip it before forwarding, same as ab_benchmark.main()
    # already does for its own REMAINDER-captured command.
    if argv[:1] == ["--"]:
        argv = argv[1:]
    return re14_real_run.main(argv)


def _resource_report_main(argv: list[str]) -> int:
    from . import resource_report

    return resource_report.main(argv)


def _validate_release_main(argv: list[str]) -> int:
    from . import release_validate

    return release_validate.main(argv)


def _rank_replay_main(argv: list[str]) -> int:
    from . import rank_replay

    return rank_replay.main(argv)


def _candidate_binary_size_main(argv: list[str]) -> int:
    from . import candidate_binary_size

    return candidate_binary_size.main(argv)


# ------------------------------------------------------------- experiment (EC11)


def _load_contract_registry(args: argparse.Namespace):
    from . import experiment_contract as ec

    contracts_path = (
        Path(args.contracts) if args.contracts else paths.EXPERIMENT_CONTRACTS
    )
    return ec, ec.load_contracts(contracts_path)


def cmd_experiment_validate(args: argparse.Namespace) -> int:
    """Schema-check every contract in the registry (or just --contract-id,
    if named) without running anything -- EC01/EC02's own validation, the
    cheap check before any real campaign work (same discipline as
    `patcher.apply_all(dry_run=True)` before a real build)."""
    from . import experiment_contract as ec

    try:
        _, registry = _load_contract_registry(args)
    except ec.ExperimentContractError as exc:
        print(f"experiment-contract validate: {exc}", file=sys.stderr)
        return 1
    ids = [args.contract_id] if args.contract_id else sorted(registry.contracts)
    if args.contract_id and args.contract_id not in registry.contracts:
        print(
            f"experiment-contract validate: no such contract {args.contract_id!r}",
            file=sys.stderr,
        )
        return 1
    for contract_id in ids:
        contract = registry[contract_id]
        print(
            f"  [ OK ] {contract_id}: {contract.title} (hash {contract.contract_hash})"
        )
    return 0


def cmd_experiment_list(args: argparse.Namespace) -> int:
    from . import experiment_contract as ec

    try:
        _, registry = _load_contract_registry(args)
    except ec.ExperimentContractError as exc:
        print(f"experiment-contract list: {exc}", file=sys.stderr)
        return 1
    if not registry.contracts:
        print("  (no contracts registered)")
        return 0
    for contract_id in sorted(registry.contracts):
        contract = registry[contract_id]
        target_label = (
            f"{contract.target.kind}:{contract.target.family}"
            if contract.target.family is not None
            else contract.target.kind
        )
        print(
            f"  {contract_id:<20} target={target_label:<20} "
            f"source={contract.source.source_id} -- {contract.title}"
        )
    return 0


def cmd_experiment_plan(args: argparse.Namespace) -> int:
    """Show what campaign lanes a contract would expand into (EC03),
    without executing them -- a dry-run, matching this project's existing
    --dry-run conventions."""
    from . import config as campaign_config
    from . import experiment_contract as ec
    from .campaign_planner import CampaignPlannerError, expand_contract

    try:
        _, registry = _load_contract_registry(args)
        contract = registry[args.contract_id]
    except (ec.ExperimentContractError, KeyError) as exc:
        print(f"experiment-contract plan: {exc}", file=sys.stderr)
        return 1
    context_config_path = Path(args.config) if args.config else paths.RECIPES
    try:
        cfg = campaign_config.load(context_config_path)
        lanes = expand_contract(
            contract,
            cfg=cfg,
            source_name=args.source,
            build_name=args.build,
            platform_name=args.platform,
        )
    except (campaign_config.ConfigError, CampaignPlannerError) as exc:
        print(f"experiment-contract plan: {exc}", file=sys.stderr)
        return 1
    print(f"  contract {contract.id}: {len(lanes)} lane(s)")
    for lane in lanes:
        detail = f"role={lane.role}"
        if lane.workload_tag:
            detail += f" workload={lane.workload_tag}"
        if lane.model_ref:
            detail += f" model={lane.model_ref}"
        if lane.boundary_dimension:
            detail += f" {lane.boundary_dimension}={lane.boundary_value}"
        print(
            f"    {lane.source_name}:{lane.build_name}:{lane.platform_name} ({detail})"
        )
    return 0


def cmd_experiment_run(args: argparse.Namespace) -> int:
    """Execute a contract's full lane set (EC03/EC04) through
    run_campaign() -- the real materialize/generate/build/smoke
    orchestration, identical to `bigcherry build`'s own engine. Does NOT
    yet run the correctness/aggregation/promotion chain (EC06-EC09) --
    those consume real benchmark measurements this command does not
    itself capture; wire a real comparison harness (see
    tools/bigcherry/re15_acceptance_run.py's stage-by-stage pattern) on
    top of this lane execution to close that gap."""
    from . import config as campaign_config
    from . import experiment_contract as ec
    from .artifacts import ArtifactStore
    from .campaign_planner import CampaignPlannerError, expand_contract, run_campaign
    from .context import ProjectContext

    try:
        _, registry = _load_contract_registry(args)
        contract = registry[args.contract_id]
    except (ec.ExperimentContractError, KeyError) as exc:
        print(f"experiment-contract run: {exc}", file=sys.stderr)
        return 1
    context = ProjectContext.resolve(work_root=None)
    try:
        cfg = campaign_config.load(Path(args.config) if args.config else paths.RECIPES)
        lanes = expand_contract(
            contract,
            cfg=cfg,
            source_name=args.source,
            build_name=args.build,
            platform_name=args.platform,
        )
    except (campaign_config.ConfigError, CampaignPlannerError) as exc:
        print(f"experiment-contract run: {exc}", file=sys.stderr)
        return 1
    store = ArtifactStore(context.work_root / "artifacts-store")
    results = run_campaign(lanes, cfg=cfg, context=context, store=store)
    failed = 0
    for lane_id, result in results.items():
        if isinstance(result, Exception):
            failed += 1
            print(f"  [FAIL] {lane_id}: {type(result).__name__}: {result}")
        else:
            print(f"  [ OK ] {lane_id}: binary={result.binary_ref.artifact_id}")
    return 1 if failed else 0


def cmd_experiment_report(args: argparse.Namespace) -> int:
    """Render EC10's report from a stored evidence JSON file (produced by
    a caller that ran EC06/EC07/EC08/EC09 against real measurements --
    this command does not itself run anything, matching how `report`
    stays a pure rendering step over already-collected evidence)."""
    from . import experiment_contract as ec

    try:
        _, registry = _load_contract_registry(args)
        contract = registry[args.contract_id]
    except (ec.ExperimentContractError, KeyError) as exc:
        print(f"experiment-contract report: {exc}", file=sys.stderr)
        return 1
    try:
        evidence = json.loads(Path(args.evidence_file).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"experiment-contract report: cannot read {args.evidence_file}: {exc}",
            file=sys.stderr,
        )
        return 1
    correctness_gate = evidence.get("correctness_gate", {})
    aggregated_effects = evidence.get("aggregated_effects", {})
    promotion_gate = evidence.get("promotion_gate") or ec.evaluate_promotion_gate(
        contract,
        correctness_gate=correctness_gate,
        aggregated_effects=aggregated_effects,
        generalisation_result=evidence.get("generalisation_result"),
    )
    print(
        ec.render_report(
            contract,
            correctness_gate=correctness_gate,
            aggregated_effects=aggregated_effects,
            promotion_gate=promotion_gate,
            generalisation_result=evidence.get("generalisation_result"),
        )
    )
    return 0 if promotion_gate.get("passed") else 1


def main(argv: list[str] | None = None) -> int:
    _configure_output()
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


def _configure_output() -> None:
    """Keep diagnostic output printable on Windows' legacy code pages.

    Reports intentionally use a few Unicode layout glyphs.  Preserve the
    console's selected encoding, but replace characters it cannot represent
    instead of allowing a diagnostic command to fail while printing it.
    Captured/test streams may not implement ``reconfigure``; those are left
    untouched.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="backslashreplace")


if __name__ == "__main__":
    raise SystemExit(main())

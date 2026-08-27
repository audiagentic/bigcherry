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
import importlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

from . import __version__
from .core import paths
from .patch import catalog as patch_catalog
from .patch import apply as patcher
from .patch import patchset
from .release import pin_status
from . import pin_transition
from . import recipes
from .release import records as releases
from .source import audit as source_audit
from .source import sources
from .source import upstream
from .release import pin as _release_pin
from .cli.diagnostics import cmd_check, cmd_doctor, cmd_status

_source_cli = importlib.import_module("bigcherry.cli.source")
cmd_audit = _source_cli.cmd_audit
cmd_pull = _source_cli.cmd_pull

_patch_cli = importlib.import_module("bigcherry.cli.patch")
cmd_apply = _patch_cli.cmd_apply
cmd_patch_explain = _patch_cli.cmd_patch_explain
cmd_patch_graph = _patch_cli.cmd_patch_graph
cmd_patch_lint = _patch_cli.cmd_patch_lint
cmd_patch_status = _patch_cli.cmd_patch_status
cmd_patch_validate = _patch_cli.cmd_patch_validate
cmd_patch_verify_evidence = _patch_cli.cmd_patch_verify_evidence
cmd_patches = _patch_cli.cmd_patches
_tuning_cli = importlib.import_module("bigcherry.cli.tuning")
cmd_generate = _tuning_cli.cmd_generate
cmd_replay_inspect = _tuning_cli.cmd_replay_inspect
cmd_inventory = _tuning_cli.cmd_inventory

_build_cli = importlib.import_module("bigcherry.cli.build")
cmd_build_new = _build_cli.cmd_build_new

_experiment_cli = importlib.import_module("bigcherry.cli.experiment")
cmd_experiment_validate = _experiment_cli.cmd_experiment_validate
cmd_experiment_list = _experiment_cli.cmd_experiment_list
cmd_experiment_plan = _experiment_cli.cmd_experiment_plan
cmd_experiment_run = _experiment_cli.cmd_experiment_run
cmd_experiment_report = _experiment_cli.cmd_experiment_report

# TR09: build_parser/_legacy_main/cmd_repin/cmd_pin_status/_configure_output now
# live in bigcherry.cli.main -- the canonical parser-assembly/entrypoint home.
# Re-exported here (same pattern as the cmd_* handlers above) because existing
# tests resolve them as bigcherry.__main__.build_parser et al.
_main_cli = importlib.import_module("bigcherry.cli.main")
build_parser = _main_cli.build_parser
_legacy_main = _main_cli._legacy_main
cmd_repin = _main_cli.cmd_repin
cmd_pin_status = _main_cli.cmd_pin_status
_configure_output = _main_cli._configure_output

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


# -------------------------------------------------------------------- audit


# -------------------------------------------------------------------- apply


def _copy_overlay(
    root: Path, *, dry_run: bool, backup: dict[str, str | None] | None = None,
    sim_texts: dict[str, str] | None = None,
) -> list[str]:
    """Mirror ``src/`` onto the checkout. Returns the paths written.

    ``backup`` (adversarial-review follow-up, patch-rebase-check design):
    when supplied, every touched target's ORIGINAL content (or ``None`` if
    it didn't exist) is recorded before the write, keyed by the same
    relative path returned in ``written`` -- so a caller whose subsequent
    anchored-patch pass fails can restore the overlay to its pre-apply
    state instead of leaving a half-applied tree (overlay written, patches
    rolled back) that ``apply_all()``'s own transaction never covered,
    since it only ever snapshotted the anchored-patch targets it itself
    writes, not this function's writes.

    ``sim_texts`` (adversarial-review follow-up, dry-run/apply parity gap):
    when supplied, every touched target's POST-write content is recorded
    here regardless of ``dry_run`` -- so a caller can feed it to
    ``apply_all(..., initial_texts=sim_texts)`` and have an anchored edit
    that targets an overlay-added file see the same bytes in a dry run
    that it would see in a real apply. Without this, a dry-run trial reads
    such a target straight off disk (pre-overlay) and silently diverges
    from what real apply would actually do.
    """
    written: list[str] = []
    for source in sorted(paths.SRC_OVERLAY.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(paths.SRC_OVERLAY)
        target = root / relative
        text = source.read_text(encoding="utf-8")
        if target.is_file() and target.read_text(encoding="utf-8") == text:
            continue
        relative_str = str(relative).replace("\\", "/")
        if backup is not None and relative_str not in backup:
            backup[relative_str] = target.read_text(encoding="utf-8") if target.is_file() else None
        if sim_texts is not None:
            sim_texts[relative_str] = text
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8", newline="")
        written.append(relative_str)
    return written


def _restore_overlay(root: Path, backup: dict[str, str | None]) -> None:
    """Undo `_copy_overlay()`'s writes using its own captured backup.

    Mirrors patch/apply.py's own rollback discipline: never let one file's
    restore failure stop the others, and never follow a symlink planted at
    the target path since the backup was taken.
    """
    import os

    for relative_str, original in backup.items():
        target = root / relative_str
        try:
            if original is None:
                if target.is_file():
                    target.unlink()
            else:
                nofollow = getattr(os, "O_NOFOLLOW", 0)
                fd = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_TRUNC | nofollow, 0o644)
                with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                    handle.write(original)
        except OSError:
            pass


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
    #
    # Adversarial-review follow-up: overlay writes were never covered by
    # apply_all()'s own transaction -- that function only ever snapshots
    # and rolls back the anchored-patch targets IT writes, so a subsequent
    # anchored-patch failure previously left the overlay's writes on disk
    # with no rollback, even though the overall selection reports ok=False.
    # Capture the overlay's own backup here and restore it on the same
    # failure condition, matching apply_all()'s all-or-nothing contract for
    # the WHOLE selection, not just its own half of it.
    overlay_backup: dict[str, str | None] = {}
    overlay_sim: dict[str, str] = {}
    written = _copy_overlay(root, dry_run=dry_run, backup=overlay_backup, sim_texts=overlay_sim)
    patches = patchset.load_patches(groups=groups, states=states)
    results = patcher.apply_all(patches, root, dry_run=dry_run, initial_texts=overlay_sim)
    ok = all(r.ok for r in results)
    if not ok and not dry_run and overlay_backup:
        _restore_overlay(root, overlay_backup)
    intended_tree_state = recipes.tree_state_key(
        record.release_tag or record.revision, groups, states
    )
    selection_changed = record.tree_state != intended_tree_state
    tree_mutated = bool(written) or any(result.changed for result in results)

    if ok:
        verb = "would write" if dry_run else "wrote"
        print(f"overlay: {verb} {len(written)} file(s)")
    else:
        verb = "not written (dry run)" if dry_run else "rolled back"
        print(f"overlay: {verb} -- patches failed")
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


def _legacy_resolve_pin_sha(ref: str) -> str:
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


def _legacy_cmd_repin(args: argparse.Namespace) -> int:
    """Move the pin to the newest upstream release and declare the transition.

    RE48/RV78: repin now writes the pin-transition marker (releases/
    pin-transition.json) atomically with the recipes.toml rewrite. Commit
    both together -- the committed marker is what makes the in-flight state
    'mid-rebase' instead of 'drift' (see docs/reference/build/PIN_BUMP.md)."""
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

    # RD95 (exact tracked-commit ancestry) + RD99 (manually-annotated
    # upstream-equivalent ancestry): advisory ancestry/redundancy report.
    # Planning cards stay in the MCP single-writer domain; this reports
    # now-baseline tracked changes but changes no state. A report failure
    # must never invalidate an otherwise-successful RE48 transition, so it
    # is wrapped broadly.
    try:
        report = sources.baseline_candidates_at_pin(target)
        sources.print_baseline_candidates(report)
    except Exception as exc:  # noqa: BLE001 -- report is advisory only
        print(
            f"ancestry gate: report unavailable: {exc}; "
            "pin move remains valid and no commits are asserted redundant",
            file=sys.stderr,
        )

    print(
        "next: COMMIT recipes.toml + the marker together, then "
        "python -m bigcherry pull --recipe <name>"
    )
    return 0


def _legacy_pin_status_paths() -> tuple[pin_status.RepoPaths, object, list]:
    """Local RepoPaths + config trees + the llama root, from the real config."""
    from .core import config as campaign_config

    cfg = campaign_config.load(paths.RECIPES)
    repo_paths = pin_status.RepoPaths(
        repo_root=paths.REPO_ROOT,
        llama_root=paths.llama_root(),
        releases_dir=paths.REPO_ROOT / "releases",
        artifacts_dir=paths.REPO_ROOT / "artifacts",
    )
    return repo_paths, cfg, list(cfg.trees)


def _legacy_render_pin_status(report, trees) -> str:
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


def _legacy_cmd_pin_status(args: argparse.Namespace) -> int:
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


def _resolve_pin_sha(ref: str) -> str:
    return _release_pin.resolve_pin_sha(ref)


def _pin_status_paths():
    return _release_pin.pin_status_paths()


def _render_pin_status(report, trees) -> str:
    return _release_pin.render_pin_status(report, trees)


def main(argv: list[str] | None = None) -> int:
    """Delegate the package entrypoint to the canonical CLI bootstrap."""
    from importlib import import_module

    cli_main = cast(Any, import_module("bigcherry.cli.main").main)
    return int(cli_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())

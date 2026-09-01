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
from typing import TYPE_CHECKING, Any, cast

from . import __version__
from .core import paths
from .patch import catalog as patch_catalog
from .patch import apply as patcher
from .patch import patchset
from .patch import rebase as patch_rebase_module

if TYPE_CHECKING:
    from .patch.selection import CliPatchSelection
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
        # `text` is LF-only (universal-newline translation of src) and gets
        # written verbatim (newline=""). Comparing against a universal-
        # newline read of `target` would translate a stale CRLF target to
        # LF first, read as "already matches", and skip the write forever
        # -- leaving CRLF bytes on disk that a real write would have
        # normalized. Decode raw bytes directly (no translation) so that's
        # detected as needing a rewrite. See rebase.py's
        # `_write_overlay_snapshot` for the same fix, applied there first
        # (found live during the b10502->b10680 bump).
        # (Path.read_text(newline=...) is Python 3.13+ only -- Brutus runs
        # 3.12 -- so this reads bytes and decodes rather than using it.)
        if target.is_file() and target.read_bytes().decode("utf-8") == text:
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


def _apply_exact_selection(
    root: Path,
    selection: "CliPatchSelection",
    *,
    force: bool = False,
    dry_run: bool = False,
) -> bool:
    """Install the overlay and apply one exact ``--source``-resolved patch
    selection. True if all placed.

    Real safety requirements this satisfies (all found necessary by gpt's
    review, not hypothetical): (1) verifies the LIVE vendor HEAD (a real,
    immutable commit SHA) matches the source's expected ref before any
    mutation -- a moved pin must not silently apply the wrong composition;
    (2) re-resolves the canonical source immediately before mutating and
    requires patch_set_id + patch_ids to be unchanged from the selection
    passed in -- a TOCTOU guard for this shared, multi-agent repo, where
    config/recipes.toml or the patch registry could genuinely change
    between CLI startup and this call; (3) never installs the overlay for
    a source whose overlay flag is False (llama-native, vulkan-stock are
    real overlay=false sources today) and fails closed if that flag was
    never resolved, rather than defaulting to "install it anyway"."""
    from .patch import selection as patch_selection

    live_revision = patch_rebase_module._git(root, "rev-parse", "HEAD")
    # selection.source_ref may be a movable symbolic ref (e.g. the pin tag
    # "b10705"), not yet a commit SHA -- resolve it in the SAME local repo
    # before comparing, or a correctly-pinned checkout would always look
    # mismatched (a real bug caught by testing this against the live
    # project instead of only synthetic fixtures).
    try:
        expected_revision = patch_rebase_module._git(
            root, "rev-parse", f"{selection.source_ref}^{{commit}}"
        )
    except Exception as exc:  # noqa: BLE001 -- surfaces as a clear apply-time error
        print(
            f"apply: --source {selection.source_name!r}'s ref "
            f"{selection.source_ref!r} does not resolve in this checkout: {exc}",
            file=sys.stderr,
        )
        return False
    if expected_revision != live_revision:
        print(
            f"apply: --source {selection.source_name!r} expects upstream "
            f"revision {selection.source_ref!r} ({expected_revision}), but "
            f"the live checkout is at {live_revision!r} -- pull first, or "
            "re-run patch-rebase-check",
            file=sys.stderr,
        )
        return False

    fresh = patch_selection._resolve_exact_selection(selection.source_name)
    if fresh.patch_set_id != selection.patch_set_id or fresh.patch_ids != selection.patch_ids:
        print(
            f"apply: --source {selection.source_name!r}'s composition changed "
            "since it was resolved (config/recipes.toml or the patch "
            "registry moved concurrently) -- re-run to pick up the current "
            "composition",
            file=sys.stderr,
        )
        return False
    selection = fresh

    if selection.overlay is None:
        print(
            f"apply: --source {selection.source_name!r}'s overlay flag was "
            "never resolved -- refusing to guess whether to install it",
            file=sys.stderr,
        )
        return False

    record = _record_for(root)
    if not force and not record.audit.get("passed"):
        print(
            "refusing to patch a tree that has not passed a strict audit.\n"
            "  run `python -m bigcherry audit` first, or pass --force.",
            file=sys.stderr,
        )
        return False

    overlay_backup: dict[str, str | None] = {}
    overlay_sim: dict[str, str] = {}
    written: list[str] = []
    if selection.overlay:
        written = _copy_overlay(root, dry_run=dry_run, backup=overlay_backup, sim_texts=overlay_sim)

    resolved = patchset.resolve_exact(selection.patch_ids, allow_rejected=False)
    patches = patchset.load_resolved(resolved)
    results = patcher.apply_all(patches, root, dry_run=dry_run, initial_texts=overlay_sim)
    ok = all(r.ok for r in results)
    if not ok and not dry_run and overlay_backup:
        _restore_overlay(root, overlay_backup)

    intended_tree_state = selection.tree_state_key(live_revision)
    selection_changed = record.tree_state != intended_tree_state
    tree_mutated = bool(written) or any(result.changed for result in results)

    if ok:
        verb = "would write" if dry_run else "wrote"
        print(f"overlay: {verb} {len(written)} file(s)" if selection.overlay else "overlay: none (source overlay=false)")
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
        "python -m bigcherry pull --source <name>"
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

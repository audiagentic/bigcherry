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
from . import recipes
from . import releases
from . import source_audit
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


def cmd_pull(args: argparse.Namespace) -> int:
    root = paths.llama_root(args.llama_root)

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
            print(f"cleared {len(stale)} stale git lock(s) from an earlier "
                  f"interrupted run")

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
    record.advance_to("audited" if good else "broken")
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

    Shared by ``apply`` and by ``build``, which re-applies when it has to flip
    the tree between recipes -- so both paths produce the same tree and the
    same record, rather than build growing a second, subtly different apply.
    """
    record = _record_for(root)
    if not force and not record.audit.get("passed"):
        print("refusing to patch a tree that has not passed a strict audit.\n"
              "  run `python -m bigcherry audit` first, or pass --force.",
              file=sys.stderr)
        return False

    # An anchored edit extends an overlay source; install overlays before
    # resolving anchors so a fresh upstream clone follows the same path.
    written = _copy_overlay(root, dry_run=dry_run)
    patches = patchset.load_patches(groups=groups, states=states)
    results = patcher.apply_all(patches, root, dry_run=dry_run)
    ok = all(r.ok for r in results)

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
        record.advance_to("patched" if ok else "broken")
        if not ok:
            record.notes = "patches failed: " + ", ".join(
                record.patches["failed_edits"])
        elif record.notes.startswith("patches failed:"):
            record.notes = ""
        # Key what the tree now carries, so `build` can tell whether it needs
        # a reset or can compile against this selection as-is.
        record.tree_state = recipes.tree_state_key(
            record.release_tag or record.revision, groups, states) if ok else ""
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


def cmd_repin(args: argparse.Namespace) -> int:
    """Move the pin to the newest upstream release."""
    try:
        target = args.ref or upstream.latest_release()
        old = recipes.repin(target)
    except (upstream.UpstreamError, recipes.RecipeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if old == target:
        print(f"already pinned to {target}")
        return 0
    print(f"pinned: {old} -> {target}")
    print("recipes following the pin now build from it; recipes naming their "
          "own ref are unchanged.")
    print("next: python -m bigcherry pull --recipe <name>")
    return 0


def _build_dir(recipe: recipes.Recipe, build: recipes.Build) -> Path:
    """One directory per (recipe, build). The variants are mutually exclusive
    at compile time, so they must never share a configure cache."""
    return paths.REPO_ROOT / "build" / f"{recipe.name}-{build.name}"


#: Files ``generate`` writes into the checkout. Listed explicitly rather than
#: globbed: the overlay puts similarly-named sources in the same directory,
#: and a glob that swept those up would delete the dispatch layer itself.
GENERATED_IN_TREE = (
    "hip-autotune-registry.inc",
    "hip-autotune-build-hash.h",
    "hip-autotune-arch.h",
    "hip-autotune-manifest.json",
    "hip-autotune-build-descriptor.json",
    "hip-autotune-mmvq-instances.inc",
)


def _overlay_relative_paths() -> list[Path]:
    """Paths the overlay writes, relative to the checkout root."""
    return [source.relative_to(paths.SRC_OVERLAY)
            for source in sorted(paths.SRC_OVERLAY.rglob("*"))
            if source.is_file()]


def _reset_tree(root: Path, ref: str, *, dry_run: bool) -> int:
    """Return the checkout to pristine upstream at ``ref``.

    Deliberately not ``git clean``. Everything this project adds is known --
    tracked edits come out with ``git checkout``, and the untracked files are
    the overlay walk plus the generated list above. Sweeping with ``git clean``
    would also take unrelated untracked files, which on this SMB share
    includes ``.fuse_hidden*`` litter and anything the operator left there.
    """
    removed = 0
    targets = [root / rel for rel in _overlay_relative_paths()]
    targets += [paths.cuda_dir(root) / name for name in GENERATED_IN_TREE]
    for target in targets:
        if target.is_file():
            removed += 1
            if not dry_run:
                target.unlink()

    if not dry_run:
        # Tracked files back to the ref. Scoped to the worktree so it cannot
        # move HEAD if the ref were ever a branch name.
        _run(["git", "-C", str(root), "checkout", "--force", ref, "--", "."])
    return removed


def _ensure_tree_state(
        root: Path,
        recipe: recipes.Recipe,
        *,
        dry_run: bool,
        force: bool,
        assume_state: str | None = None,
) -> bool:
    """Make the checkout match what ``recipe`` needs. True if it now does.

    Skips entirely when the tree already carries this exact selection, which
    is what keeps the default set to one flip instead of three: two of its
    three recipes want the same patches, so only the unpatched one costs a
    rebuild.

    ``assume_state`` lets a dry run carry the state it would have produced
    into the next recipe, so the preview reports the flips a real run would
    do rather than one per recipe.
    """
    wanted = recipes.tree_state_key(recipe.ref, recipe.groups, recipe.states)
    if assume_state is not None:
        current, clean = assume_state, True
    else:
        record = _record_for(root)
        current = record.tree_state
        clean = bool(record.patches.get("applied_cleanly"))
    if current == wanted and clean:
        print(f"    tree already at {wanted} -- no reset")
        return True

    print(f"    tree {current or 'unknown'} -> {wanted}: "
          f"resetting to {recipe.ref} and re-applying")
    removed = _reset_tree(root, recipe.ref, dry_run=dry_run)
    print(f"    removed {removed} overlay/generated file(s)")
    if dry_run:
        return True

    ok = _apply_selection(root, recipe.groups, recipe.states, force=force)
    record = _record_for(root)
    record.tree_state = wanted if ok else ""
    record.save()
    return ok


def _verify_tree(root: Path, needs_patches: bool) -> list[str]:
    """Preconditions for building. Returns the reasons it cannot proceed.

    Checked once, before anything is configured or generated, because the
    whole point of the pipeline is that a broken tree stops it rather than
    producing binaries nobody can explain.
    """
    record = _record_for(root)
    problems = []

    if not record.audit.get("passed"):
        failed = record.audit.get("failed_checks") or []
        detail = f": {', '.join(failed)}" if failed else " (never run)"
        problems.append(f"source audit has not passed{detail}")

    if needs_patches:
        if record.stage not in ("patched", "generated", "built", "tested",
                                "tuned", "validated"):
            problems.append(
                f"tree is at stage {record.stage!r}, not patched -- run `apply`")
        broken = record.patches.get("failed_edits") or []
        if broken:
            # Patch drift against a new release is the expected failure here,
            # and the fix is to update the named anchor -- so name them.
            problems.append(
                f"{len(broken)} patch edit(s) did not apply: {', '.join(broken)}")
    return problems


def _generate_for(
        build: recipes.Build, root: Path, *,
        variant_set: str | None,
        inventory: str | None,
        winners: str | None,
        dry_run: bool,
) -> list[str]:
    """The `generate` invocation this build needs, as a command.

    Generation is per-variant and rewrites a single shared file in the
    checkout (`ggml-cuda/hip-autotune-registry.inc`), so it must run
    immediately before the build that consumes it. Two variants cannot share
    one registry, which is also why these builds must stay sequential.
    """
    chosen = variant_set or build.variant_set
    if not chosen:
        return []
    cmd = [sys.executable, "-m", "bigcherry", "generate", "--variant-set", chosen]
    if root != paths.llama_root(None):
        cmd += ["--llama-root", str(root)]
    if inventory:
        cmd += ["--inventory", inventory]
    if winners:
        cmd += ["--winners", winners]
    if dry_run:
        cmd += ["--dry-run"]
    return cmd


def _cmake_configure_args(
        recipe: recipes.Recipe,
        build: recipes.Build,
        platform: recipes.Platform,
        root: Path,
        build_dir: Path,
        *,
        variant_set: str | None = None,
        inventory: str | None = None,
) -> list[str]:
    options = {
        "CMAKE_BUILD_TYPE": "Release",
        **platform.options,
        **build.options,
        "AMDGPU_TARGETS": platform.targets,
    }
    # CLI override beats the build's declared set, so a custom run does not
    # need a recipe of its own.
    chosen = variant_set or build.variant_set
    if chosen:
        options["GGML_HIP_AUTOTUNE_VARIANT_SET"] = chosen
        # Only meaningful alongside a variant set. A stock build has no
        # dispatch layer, and handing it autotune options would describe a
        # binary that cannot use them.
        if inventory:
            options["GGML_HIP_AUTOTUNE_SIGNATURE_FILE"] = str(
                Path(inventory).resolve())
    if platform.c_compiler:
        options["CMAKE_C_COMPILER"] = platform.c_compiler
    if platform.cxx_compiler:
        options["CMAKE_CXX_COMPILER"] = platform.cxx_compiler
    return [
        "cmake", "-S", str(root), "-B", str(build_dir), "-G", "Ninja",
        *(f"-D{key}={value}" for key, value in sorted(options.items())),
    ]


def _build_one_recipe(
        config: recipes.Config,
        recipe: recipes.Recipe,
        args: argparse.Namespace,
        root: Path,
) -> tuple[int, int]:
    """Build every variant of one recipe. Returns (attempted, failed)."""
    platform = config.platform_for(recipe)
    names = args.build or list(recipe.builds)
    if not names:
        raise recipes.RecipeError(
            f"recipe {recipe.name!r} lists no builds; pass --build explicitly")
    selected = [config.build(name) for name in names]

    print(f"    platform {platform.name} [{platform.targets}]")
    print(f"    builds   {', '.join(b.name for b in selected)}")

    missing = [b.name for b in selected
               if b.needs == "inventory" and not args.inventory]
    if missing:
        raise recipes.RecipeError(
            f"recipe {recipe.name!r}: builds {', '.join(missing)} need "
            f"--inventory <inv.json> (produced by a record run)")

    failed = 0
    for build in selected:
        build_dir = _build_dir(recipe, build)
        generate = _generate_for(
            build, root, variant_set=args.variant_set,
            inventory=args.inventory, winners=args.winners,
            dry_run=args.dry_run)
        configure = _cmake_configure_args(
            recipe, build, platform, root, build_dir,
            variant_set=args.variant_set, inventory=args.inventory)
        compile_cmd = ["cmake", "--build", str(build_dir), "-j"]
        if args.target:
            compile_cmd += ["--target", *args.target]

        print(f"\n=== {recipe.name}/{build.name} -> {build_dir} ===")
        if build.description:
            print(f"    {build.description}")
        if args.dry_run:
            for cmd in (generate, configure, compile_cmd):
                if cmd:
                    print("    " + " ".join(cmd))
            continue
        try:
            if generate:
                _run(generate)
            _run(configure)
            _run(compile_cmd)
        except subprocess.CalledProcessError as exc:
            # Keep going: one variant failing should still leave the others
            # available, and the summary names every casualty.
            print(f"    FAILED ({exc.returncode})", file=sys.stderr)
            failed += 1
    return len(selected), failed


def cmd_build(args: argparse.Namespace) -> int:
    """Verify the tree, then generate and build every variant requested.

    This is a pipeline, not a one-shot compile: it checks the tree is sound,
    regenerates the candidate registry for each variant, and builds them all.
    """
    try:
        config = recipes.load_config()
        if args.all:
            chosen = [r for r in config.recipes.values() if r.default]
            if not chosen:
                raise recipes.RecipeError(
                    "no recipes are marked `default = true` for --all")
        else:
            chosen = [config.recipe(name) for name in (args.recipe or [])]
            if not chosen:
                raise recipes.RecipeError("pass --recipe <name> or --all")
    except recipes.RecipeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    root = paths.llama_root(args.llama_root)

    # Group recipes that need the same tree so it is flipped once, not once
    # per recipe. The default set is three recipes but two states: `upstream`
    # wants no patches, the two bigcherry recipes want the same ones.
    chosen.sort(key=lambda r: (
        recipes.tree_state_key(r.ref, r.groups, r.states), r.name))

    # Verify once, up front. A recipe that applies no patches (stock upstream)
    # still needs a clean audit, but must not be blocked on patches it does
    # not want.
    needs_patches = any(r.groups != frozenset() for r in chosen)
    problems = _verify_tree(root, needs_patches)
    if problems and not args.force:
        print("refusing to build:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print("\nfix the tree (see PATCH_DRIFT triage after a failed apply), "
              "or pass --force.", file=sys.stderr)
        return 2
    if problems:
        print("warning: building a tree with unresolved problems (--force):")
        for problem in problems:
            print(f"  - {problem}")
        print()

    attempted = failures = 0
    simulated: str | None = _record_for(root).tree_state if args.dry_run else None
    for recipe in chosen:
        try:
            print(f"\n### recipe {recipe.name} (ref {recipe.ref})")
            if not _ensure_tree_state(
                    root, recipe, dry_run=args.dry_run, force=args.force,
                    assume_state=simulated):
                print(f"    tree could not be prepared -- skipping "
                      f"{recipe.name}", file=sys.stderr)
                failures += 1
                continue
            if args.dry_run:
                simulated = recipes.tree_state_key(
                    recipe.ref, recipe.groups, recipe.states)
            did, bad = _build_one_recipe(config, recipe, args, root)
        except recipes.RecipeError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        attempted += did
        failures += bad

    if args.dry_run:
        return 0
    print()
    if failures:
        print(f"{failures} of {attempted} variant(s) failed", file=sys.stderr)
        return 1
    print(f"built {attempted} variant(s) across {len(chosen)} recipe(s)")
    return 0


def _add_selection_args(parser: argparse.ArgumentParser) -> None:
    """The patch-selection flags, shared by every command that selects."""
    parser.add_argument(
        "--recipe",
        default=None,
        choices=recipes.names() or None,
        help="named build definition from recipes.toml (default: all patches)",
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
    """Show every patch, its metadata, and whether a selection takes it."""
    infos = patchset.describe()
    if not infos:
        print("no patches found", file=sys.stderr)
        return 1

    try:
        groups, states, label = _resolve_selection(args)
    except recipes.RecipeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    root = paths.llama_root(args.llama_root)
    print(f"selection: {label}")
    print(f"checkout:  {root}")
    print()

    rows, problems, selected = [], [], 0
    for info in infos:
        taken = ((groups is None or info.group in groups)
                 and (states is None or info.state in states))
        selected += taken

        note = ""
        if info.upstream:
            landed = patchset.upstream_landed(info.upstream, root)
            if landed is True:
                note = f"upstream {info.upstream[:8]} landed -- redundant here"
            elif landed is False:
                note = f"upstream {info.upstream[:8]} not in this checkout"
            else:
                note = f"upstream {info.upstream[:8]} unknown"

        if not info.state_valid:
            problems.append(
                f"{info.name}: STATE={info.state!r} is not one of "
                f"{', '.join(patchset.STATES)} -- no recipe will select it")

        rows.append(("[x]" if taken else "[ ]", info.name, info.group,
                     info.state, note))

    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    for mark, name, group, state, note in rows:
        line = (f"{mark} {name:<{widths[1]}}  {group:<{widths[2]}}  "
                f"{state:<{widths[3]}}")
        print(f"{line}  {note}".rstrip())

    print(f"\n{selected} of {len(infos)} selected")
    for problem in problems:
        print(f"warning: {problem}", file=sys.stderr)
    return 1 if problems else 0


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
        "--ref",
        default=None,
        help="tag, branch or sha to check out (e.g. b1234), or 'latest' for "
             "the newest upstream release. Overrides --recipe.",
    )
    pull.add_argument(
        "--recipe",
        default=None,
        choices=recipes.names() or None,
        help="take the ref from this recipe in recipes.toml",
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
    patches_cmd.set_defaults(func=cmd_patches)

    repin = sub.add_parser(
        "repin", help="move recipes.toml's pin to the newest upstream release")
    repin.add_argument(
        "--ref", default=None,
        help="pin to this ref instead of querying for the newest release")
    repin.set_defaults(func=cmd_repin)

    build_cmd = sub.add_parser(
        "build", help="configure and build the variants a recipe asks for")
    build_cmd.add_argument("--llama-root", default=None)
    build_cmd.add_argument(
        "--recipe", action="append", default=None, choices=recipes.names() or None,
        help="recipe to build (repeatable)")
    build_cmd.add_argument(
        "--all", action="store_true",
        help="build every recipe marked `default = true` -- the standard "
             "comparison set")
    build_cmd.add_argument(
        "--build", action="append", default=None,
        help="build only this variant (repeatable); default is the recipe's list")
    build_cmd.add_argument(
        "--target", action="append", default=None,
        help="cmake target to build (repeatable), e.g. ggml-hip")
    from . import autotune_schema as _variant_schema
    build_cmd.add_argument(
        "--variant-set", default=None, choices=_variant_schema.VARIANT_SETS,
        help="candidate set to compile in, overriding the build's own")
    build_cmd.add_argument(
        "--inventory", default=None,
        help="signature inventory JSON from a record run; required by builds "
             "declaring needs = \"inventory\"")
    build_cmd.add_argument(
        "--winners", default=None,
        help="winners JSONL from a tuning run, for replay-slim generation")
    build_cmd.add_argument(
        "--force", action="store_true",
        help="build despite a failed audit or unapplied patches")
    build_cmd.add_argument(
        "--dry-run", action="store_true",
        help="print the cmake commands without running them")
    build_cmd.set_defaults(func=cmd_build)

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

    tune_journal_cmd = sub.add_parser(
        "tune-journal", help="crash-safe tuning journal status/compaction (HI48)")
    tune_journal_cmd.set_defaults(
        func=lambda args: _tune_journal_main(args.tune_journal_args))
    tune_journal_cmd.add_argument("tune_journal_args", nargs=argparse.REMAINDER)

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


def _tune_journal_main(argv: list[str]) -> int:
    from . import tune_journal
    return tune_journal.main(argv)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

"""RE14 step 3/4: a scripted, repeatable legacy-vs-new parity comparison.

Round 9 answered RE14's core question ("does the new path produce the same
build as the legacy path?") with a real but MANUAL comparison -- separate ad
hoc SSH commands, run by hand, with the result recorded in prose. This script
automates that same sequence so it can be re-run on demand (a real toolchain
change, a recipe edit, a future RDNA target) rather than re-derived by hand
each time.

Two arms, kept in fully separate output locations exactly as RE14 step 3
requires (the legacy arm mutates the shared vendor checkout in place, as
legacy always does; the new arm never touches that checkout):

  legacy: apply the recipe's patches to the real shared checkout, generate
      the candidate catalog with an EXPLICIT --arch (see below for why this
      matters), then configure+compile into a scratch build directory that
      is never the recipe's real build/<recipe>-<build> directory.

  new: python -m bigcherry.re14_real_run, the already-proven real
      materialize -> generate -> build -> runtime-smoke -> reuse path,
      invoked as a subprocess so this script and that proof harness cannot
      drift into two different definitions of "the new path".

Why generate needs an explicit --arch here specifically: round 9 found that
the legacy build orchestrator's internal generate step (__main__._generate_for)
never forwards --arch to the `generate` CLI it shells out to, so it silently
falls back to that CLI's own --arch=all default -- a real, pre-existing
legacy-path bug (the legacy catalog has always been broader than what it
actually compiles for) that is explicitly out of scope to fix here per RE14's
"legacy cmd_build stays untouched" rule. Comparing against that default would
make every run report a spurious candidate-set mismatch that has nothing to
do with cutover correctness. This script sidesteps it the same way round 9's
manual investigation did: call `generate` directly with the same --arch the
new path is given, then compile with the resulting in-tree catalog rather
than going through the legacy build orchestrator's own (bugged) generate call
a second time.

binary_hash is allowed to differ by default: round 9 diagnosed the residual
raw-byte difference conclusively as embedded absolute build-tree paths (295
of them, found via `strings`), differing only because the two arms compile
from different tree locations -- exactly the isolation property RE14 exists
to establish, not a functional divergence. Pass --allow to widen the gate
further for a deliberately-scoped comparison; leave everything else at the
default (empty) to keep the real acceptance bar RE14 step 4 describes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from pathlib import Path

from . import paths
from . import recipes as recipes_module
from .artifacts import ArtifactStore
from .parity import ParityError, check_parity
from .parity_loaders import load_legacy_arm, load_new_arm


def _run(cmd: list[str]) -> None:
    print("    $ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_legacy_arm(
    *, recipe: str, build: str, inventory: Path, arch: str,
    c_compiler: str | None, cxx_compiler: str | None,
    binary_target: str, build_dir: Path,
) -> tuple:
    """Returns ``(CampaignArm, platform_name)``. The platform name is
    returned, not just the arm, so ``main()`` can pass the SAME platform to
    ``run_new_arm`` instead of letting it silently default to whatever
    ``re14_real_run.py``'s own --platform default happens to be -- a real
    gpt-auto-agent review finding: --recipe was threaded to the legacy arm
    only, so a non-default recipe naming a different platform would compare
    two arms built for different platforms while still reporting a result.
    """
    # Imported lazily and used only for its cmake-argument construction --
    # not to invoke any of the legacy orchestration (_build_one_recipe,
    # _generate_for) that carries the --arch bug this function exists to
    # sidestep.
    from . import __main__ as legacy_cli

    root = paths.llama_root(None)
    # recipes.load_config(), not config.load(): __main__.py's legacy
    # orchestration (and _cmake_configure_args, reused below) is built on
    # recipes.Config's .recipe()/.build()/.platform_for() accessors, a
    # different module from the v2-only config.Config the new path uses.
    cfg = recipes_module.load_config()
    recipe_obj = cfg.recipe(recipe)
    build_obj = cfg.build(build)
    platform_obj = cfg.platform_for(recipe_obj)
    variant_set = build_obj.variant_set or "workload-max"

    print(f"=== legacy arm: apply recipe={recipe!r} to the real shared checkout ===")
    _run([sys.executable, "-m", "bigcherry", "--llama-root", str(root),
          "apply", "--recipe", recipe])

    print(f"=== legacy arm: generate --arch {arch!r} (explicit, sidesteps the "
          f"known _generate_for '--arch all' default -- see module docstring) ===")
    _run([sys.executable, "-m", "bigcherry", "--llama-root", str(root),
          "generate", "--variant-set", variant_set, "--arch", arch,
          "--inventory", str(inventory)])

    print(f"=== legacy arm: cmake configure+build -> {build_dir} ===")
    build_dir.mkdir(parents=True, exist_ok=True)
    configure = legacy_cli._cmake_configure_args(
        recipe_obj, build_obj, platform_obj, root, build_dir,
        variant_set=variant_set, inventory=str(inventory),
        c_compiler=c_compiler, cxx_compiler=cxx_compiler)
    _run(configure)
    _run(["cmake", "--build", str(build_dir), "-j", "--target", binary_target])

    manifest_path = paths.cuda_dir(root) / "hip-autotune-manifest.json"
    descriptor_path = paths.cuda_dir(root) / "hip-autotune-build-descriptor.json"
    binary_path = build_dir / "bin" / binary_target
    arm = load_legacy_arm(
        "legacy", manifest_path=manifest_path, descriptor_path=descriptor_path,
        binary_path=binary_path)
    return arm, platform_obj.name


def run_new_arm(
    *, upstream_repo: Path, inventory: Path, arch: str, model: Path,
    run_id: str, work_root: Path | None, hip_visible_devices: str,
    split_mode: str, binary_target: str, source: str, build: str, platform: str,
):
    cmd = [
        sys.executable, "-m", "bigcherry.re14_real_run",
        "--upstream-repo", str(upstream_repo), "--inventory", str(inventory),
        "--arch", arch, "--model", str(model), "--run-id", run_id,
        "--hip-visible-devices", hip_visible_devices, "--split-mode", split_mode,
        "--binary-relative-path", f"bin/{binary_target}",
        "--source", source, "--build", build, "--platform", platform,
    ]
    if work_root:
        cmd += ["--work-root", str(work_root)]
    print("=== new arm: python -m bigcherry.re14_real_run ===")
    print("    $ " + " ".join(cmd))
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(completed.stdout)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr)

    summary = None
    prefix = "RE14_PARITY_RESULT_JSON: "
    for line in completed.stdout.splitlines():
        if line.startswith(prefix):
            summary = json.loads(line[len(prefix):])
    if summary is None:
        raise RuntimeError(
            "re14_real_run.py exited 0 but did not print its "
            "RE14_PARITY_RESULT_JSON summary line -- cannot load its "
            "artifacts back for comparison"
        )

    store = ArtifactStore(Path(summary["store_root"]))
    return load_new_arm(
        "new", store=store, manifest_relative=summary["manifest_relative"],
        binary_relative=summary["binary_relative"],
        manifest_content_hash=summary["manifest_content_hash"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", default="bigcherry")
    parser.add_argument("--build", default="tune")
    parser.add_argument("--upstream-repo", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--arch", required=True,
                         help="given identically to both arms -- see module docstring")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--c-compiler", default=None)
    parser.add_argument("--cxx-compiler", default=None)
    parser.add_argument("--binary-target", default="llama-bench")
    parser.add_argument("--legacy-build-dir", type=Path,
                         default=paths.REPO_ROOT / "build" / "re14-parity-check")
    parser.add_argument("--work-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--hip-visible-devices", default="0")
    parser.add_argument("--split-mode", default="none")
    parser.add_argument(
        "--allow", action="append", default=[],
        help="an additional allowed-difference field name (repeatable); "
             "binary_hash is always allowed (see module docstring)")
    args = parser.parse_args(argv)

    run_id = args.run_id or uuid.uuid4().hex[:12]
    print(f"=== RE14 parity run {run_id} ===")

    legacy_arm, platform_name = run_legacy_arm(
        recipe=args.recipe, build=args.build, inventory=args.inventory,
        arch=args.arch, c_compiler=args.c_compiler, cxx_compiler=args.cxx_compiler,
        binary_target=args.binary_target, build_dir=args.legacy_build_dir)

    new_arm = run_new_arm(
        upstream_repo=args.upstream_repo, inventory=args.inventory, arch=args.arch,
        model=args.model, run_id=run_id, work_root=args.work_root,
        hip_visible_devices=args.hip_visible_devices, split_mode=args.split_mode,
        binary_target=args.binary_target,
        # --recipe names a source by the same string in this project's
        # recipes.toml convention (both configs are parsed from the same
        # file); --build is already literally shared between the two
        # config loaders. platform_name comes from the recipe itself, not
        # a separately-guessable default, so the two arms cannot silently
        # diverge on which platform they were built for.
        source=args.recipe, build=args.build, platform=platform_name)

    allowed = frozenset({"binary_hash", *args.allow})
    try:
        report = check_parity(
            legacy_arm, new_arm, label=f"{args.recipe}/{args.build}",
            allowed_differences=allowed)
    except ParityError as exc:
        print(f"PARITY FAILED:\n{exc}", file=sys.stderr)
        return 1

    print(f"PARITY OK: {report.label} (allowed differences: {sorted(allowed)})")
    if report.missing_candidates or report.extra_candidates:
        print(f"  missing from new: {sorted(report.missing_candidates)}")
        print(f"  extra in new:     {sorted(report.extra_candidates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

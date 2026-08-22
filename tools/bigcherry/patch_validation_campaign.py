"""HI81 (informal): unattended patch-validation campaign.

Given only a patch module name and a model path, materializes an isolated,
content-addressed source worktree for that one patch (patch_source_
isolation.py, HI82 -- never the shared vendor/llama.cpp tree), builds the
tune-mode/replay-mode/stock-baseline trees from it, then runs the full
record->tune->promote->export->replay->bench->report smoke campaign
(tools/bigcherry/e2e_smoke_campaign.py) against it -- the same process used
to validate HI30/HI31 this session, generalized to any single untested
patch so a human only has to choose the patch and the model.

This intentionally does NOT decide whether the patch is good -- it produces
the same report.md/bench.json/measurements.jsonl a human (or a follow-up
GPT review) reads to make that call. It also does not touch git or the
patch's catalog state (validated/rejected/untested) -- promoting a patch
out of "untested" is a separate, deliberate decision.

Usage:
    python -m bigcherry.patch_validation_campaign \\
        --patch 1204_rd08_q6k_mmvq_vdr2 \\
        --model G:/models/qwen3.5-2b/Qwen_Qwen3.5-2B-Q4_K_M.gguf \\
        --hip-path H:/.../vendor/rocm/7.1 --amdgpu-targets gfx1100 \\
        --manifest H:/.../artifacts/<rev>/hip-autotune-manifest.json \\
        --workdir C:/scratch/patch-1204-qwen2b

Safe to re-run: source materialization and every build/campaign stage below
reuse existing output where present (patch_source_isolation.py's manifest-
verified worktree reuse, cmake incremental builds, e2e_smoke_campaign.py's
per-stage resume-check).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bigcherry.build_identity import capture_build_identity

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LLAMA_CPP_SRC = REPO_ROOT / "vendor" / "llama.cpp"
CMAKE_GENERATOR = "Ninja"


class PatchCampaignError(RuntimeError):
    pass


def _print(msg: str) -> None:
    print(f"[patch-campaign] {msg}", flush=True)


def _hip_env(hip_path: Path) -> dict[str, str]:
    """Environment for cmake configure/build subprocesses with the chosen
    vendored ROCm toolchain (tools/rocm-env.ps1/.sh) explicit in-process,
    rather than assuming the invoking shell already sourced it -- this tool
    is meant to run unattended/backgrounded, where that assumption doesn't
    hold (hit for real: `find_package(hip)` failed with CMAKE_PREFIX_PATH
    unset when this campaign was launched from a plain shell)."""
    import os

    env = os.environ.copy()
    env["ROCM_PATH"] = str(hip_path)
    env["HIP_PATH"] = str(hip_path)
    env["CMAKE_PREFIX_PATH"] = os.pathsep.join(
        p for p in (str(hip_path), env.get("CMAKE_PREFIX_PATH", "")) if p
    )
    env["PATH"] = os.pathsep.join([str(hip_path / "bin"), env.get("PATH", "")])
    return env


def generate_registry(*, source: Path, amdgpu_targets: str, generated_dir: Path) -> None:
    """Run `bigcherry generate` against an isolated worktree so ggml-hip's
    CMakeLists finds hip-autotune-registry.inc + template-instances/ there.

    Real gap found via HI82 isolation testing: the pre-isolation campaign
    tool never called this itself -- it worked only because the shared
    vendor/llama.cpp tree already had a stale in-tree registry.inc left
    over from earlier manual `bigcherry generate` runs. A fresh isolated
    worktree has no such leftover, so cmake failed with "hip-autotune-
    registry.inc is missing" the first time this ran for real. Writing to
    an explicit --generated-root (not the in-tree default) matches the
    "campaign builds use an out-of-tree generated directory" contract
    already documented in ggml/src/ggml-hip/CMakeLists.txt.

    --force is required: `bigcherry generate`'s "unpatched tree" guard
    checks a ReleaseRecord keyed by git revision (releases/<rev>.json),
    which knows nothing about an isolated worktree's own out-of-band
    patcher.apply_all() run -- patch_source_isolation.materialize_source()
    is itself the real proof the tree is patched (it raises on any failed
    edit), so this bypass is sound, not a shortcut around a real check."""
    import subprocess

    generated_dir.mkdir(parents=True, exist_ok=True)
    _print(f"generating autotune registry into {generated_dir} ...")
    args = [
        sys.executable, "-m", "bigcherry", "--llama-root", str(source),
        "generate", "--arch", amdgpu_targets, "--generated-root", str(generated_dir),
        "--force",
    ]
    result = subprocess.run(
        args, cwd=str(REPO_ROOT / "tools"), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True,
    )
    if result.returncode != 0:
        raise PatchCampaignError(f"bigcherry generate failed:\n{result.stdout}")


def build_tree(
    *, name: str, extra_cmake_args: list[str], hip_path: Path,
    amdgpu_targets: str, workdir: Path, targets: list[str], source: Path,
) -> Path:
    """cmake configure (if not already configured) + build the given
    targets. Returns the build tree's bin/ directory.

    `source` is an isolated, content-addressed worktree from
    patch_source_isolation.materialize_source() -- never the shared
    vendor/llama.cpp working tree (HI82: that sharing is exactly the
    contamination risk this module was rewritten to close)."""
    import subprocess

    build_dir = workdir / name
    log_dir = workdir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    is_windows = sys.platform == "win32"
    clang = hip_path / "bin" / ("clang.exe" if is_windows else "clang")
    clangxx = hip_path / "bin" / ("clang++.exe" if is_windows else "clang++")

    env = _hip_env(hip_path)

    if not (build_dir / "CMakeCache.txt").exists():
        _print(f"configuring {name} ...")
        args = [
            "cmake", "-S", str(source), "-B", str(build_dir), "-G", CMAKE_GENERATOR,
            "-DCMAKE_BUILD_TYPE=Release", "-DGGML_HIP=ON",
            f"-DAMDGPU_TARGETS={amdgpu_targets}",
            f"-DCMAKE_C_COMPILER={clang}", f"-DCMAKE_CXX_COMPILER={clangxx}",
            f"-DCMAKE_PREFIX_PATH={hip_path}",
            # HI82 item 7: build_identity.post_build_verify() needs real
            # compiled command lines, not just CMakeCache.txt's configured
            # intent -- CMAKE_HIP_FLAGS is proven this session (HI81) to
            # silently not reach the compiler on Windows/Ninja+Clang.
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            *extra_cmake_args,
        ]
        log_path = log_dir / f"{name}-configure.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            result = subprocess.run(args, stdout=log_file, stderr=subprocess.STDOUT, env=env)
        if result.returncode != 0:
            raise PatchCampaignError(f"{name} configure failed (see {log_path})")

    for target in targets:
        _print(f"building {name} ({target}) ...")
        log_path = log_dir / f"{name}-build-{target}.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            result = subprocess.run(
                ["cmake", "--build", str(build_dir), "--target", target, "-j"],
                stdout=log_file, stderr=subprocess.STDOUT, env=env,
            )
        if result.returncode != 0:
            tail = "\n".join(
                log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
            )
            raise PatchCampaignError(f"{name} build ({target}) failed:\n{tail}")
    _print(f"{name}: OK")
    return build_dir / "bin"


def ensure_stock_baseline(
    *, hip_path: Path, amdgpu_targets: str, workdir: Path, stock_src: Path,
) -> Path:
    """Build a genuinely unpatched llama.cpp worktree at stock_src (from
    patch_source_isolation.materialize_stock_source(), a real git worktree
    pinned to base_revision with zero patches applied) for a stock
    comparison arm. A patch under test never touches this tree."""
    import subprocess

    is_windows = sys.platform == "win32"
    clang = hip_path / "bin" / ("clang.exe" if is_windows else "clang")
    clangxx = hip_path / "bin" / ("clang++.exe" if is_windows else "clang++")
    build_dir = workdir / "stock"
    log_dir = workdir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = _hip_env(hip_path)
    if not (build_dir / "CMakeCache.txt").exists():
        _print("configuring stock baseline ...")
        args = [
            "cmake", "-S", str(stock_src), "-B", str(build_dir), "-G", CMAKE_GENERATOR,
            "-DCMAKE_BUILD_TYPE=Release", "-DGGML_HIP=ON",
            f"-DAMDGPU_TARGETS={amdgpu_targets}",
            f"-DCMAKE_C_COMPILER={clang}", f"-DCMAKE_CXX_COMPILER={clangxx}",
            f"-DCMAKE_PREFIX_PATH={hip_path}",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        ]
        log_path = log_dir / "stock-configure.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            result = subprocess.run(args, stdout=log_file, stderr=subprocess.STDOUT, env=env)
        if result.returncode != 0:
            raise PatchCampaignError(f"stock configure failed (see {log_path})")
    for target in ("llama-bench",):
        log_path = log_dir / f"stock-build-{target}.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            result = subprocess.run(
                ["cmake", "--build", str(build_dir), "--target", target, "-j"], env=env,
                stdout=log_file, stderr=subprocess.STDOUT,
            )
        if result.returncode != 0:
            raise PatchCampaignError(f"stock build ({target}) failed (see {log_path})")
    _print("stock: OK")
    return build_dir / "bin"


def run(args: argparse.Namespace) -> int:
    import os

    workdir: Path = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)

    # e2e_smoke_campaign.Campaign launches llama-server via `dict(os.environ)`
    # (this process's own environment), not through _hip_env() -- that helper
    # only covers the cmake configure/build subprocesses above. Without the
    # ROCm bin dir on PATH here, the HIP runtime DLLs are unresolved at
    # process launch (Windows exit code 0xC0000135 / STATUS_DLL_NOT_FOUND --
    # hit for real running this tool headless/backgrounded, where no
    # interactive shell had already sourced tools/rocm-env.ps1|.sh).
    os.environ["ROCM_PATH"] = str(args.hip_path)
    os.environ["HIP_PATH"] = str(args.hip_path)
    os.environ["PATH"] = os.pathsep.join(
        [str(args.hip_path / "bin"), os.environ.get("PATH", "")]
    )

    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from bigcherry import patch_source_isolation as psi  # noqa: E402

    worktree_root: Path = args.worktree_root
    base_revision = psi.git_head(LLAMA_CPP_SRC)
    _print(f"materializing isolated source for {args.patch} @ {base_revision[:12]} ...")
    patched_src = psi.materialize_source(
        base_repo=LLAMA_CPP_SRC, worktree_root=worktree_root,
        patch_module=args.patch, base_revision=base_revision,
    )
    _print(f"patch source: {patched_src}")
    stock_src = psi.materialize_stock_source(
        base_repo=LLAMA_CPP_SRC, worktree_root=worktree_root, base_revision=base_revision,
    )
    _print(f"stock source: {stock_src}")

    # Build trees are keyed by --build-root, not --workdir: build_tree()/
    # ensure_stock_baseline() skip configure when CMakeCache.txt exists and
    # `cmake --build` is incremental, so a build tree is reusable across
    # runs on this machine+arch as long as its SOURCE (now an isolated,
    # content-addressed worktree, not the shared vendor/llama.cpp tree --
    # HI82) hasn't changed identity. --workdir (record/tune/promote/replay/
    # bench/report output) is what needs to be fresh per patch+model.
    build_root: Path = (args.build_root or workdir) / patched_src.name

    # One shared out-of-tree registry serves both the tune and replay builds
    # of this same patched source -- both need it (ggml-hip/CMakeLists.txt
    # gates on GGML_HIP_AUTOTUNE OR GGML_HIP_DISPATCH_REPLAY), and it is
    # pure generated-from-source content, not build-mode-specific.
    generated_dir = build_root / "generated"
    generate_registry(
        source=patched_src, amdgpu_targets=args.amdgpu_targets, generated_dir=generated_dir,
    )

    exe = ".exe" if sys.platform == "win32" else ""
    build_env = _hip_env(args.hip_path)

    tune_cmake_args = [
        "-DGGML_HIP_AUTOTUNE=ON", "-DGGML_HIP_AUTOTUNE_RECORD=ON",
        "-DGGML_HIP_ROUTING_TRANSFORM=ON",
        f"-DGGML_HIP_AUTOTUNE_GENERATED_DIR={generated_dir}",
    ]
    tune_bin = build_tree(
        name="tune", hip_path=args.hip_path, amdgpu_targets=args.amdgpu_targets,
        workdir=build_root, targets=["llama-server", "llama-bench"], source=patched_src,
        extra_cmake_args=tune_cmake_args,
    )
    # HI82 item 7: refuse to hand a build to Campaign() until its actual
    # compiled command lines are proven to match configured intent -- a
    # build that silently lost a flag (the HI81 shape) must never reach
    # benchmarking. Raises BuildIdentityError uncaught, which is the
    # intended fail-closed behavior: no partial/best-effort campaign runs
    # against an unverified build.
    tune_build_identity = capture_build_identity(
        build_root / "tune", source_root=patched_src, source_tree=patched_src.name,
        architecture=args.amdgpu_targets, build_mode="tune",
        requested_cmake_args=tune_cmake_args, generator=CMAKE_GENERATOR, build_env=build_env,
    )
    _print(f"tune build identity: {tune_build_identity.identity_digest[:12]}")

    replay_cmake_args = [
        "-DGGML_HIP_DISPATCH_REPLAY=ON",
        f"-DGGML_HIP_AUTOTUNE_GENERATED_DIR={generated_dir}",
    ]
    replay_bin = build_tree(
        name="replay", hip_path=args.hip_path, amdgpu_targets=args.amdgpu_targets,
        workdir=build_root, targets=["llama-server", "llama-bench"], source=patched_src,
        extra_cmake_args=replay_cmake_args,
    )
    replay_build_identity = capture_build_identity(
        build_root / "replay", source_root=patched_src, source_tree=patched_src.name,
        architecture=args.amdgpu_targets, build_mode="replay",
        requested_cmake_args=replay_cmake_args, generator=CMAKE_GENERATOR, build_env=build_env,
    )
    _print(f"replay build identity: {replay_build_identity.identity_digest[:12]}")

    stock_build_root = (args.build_root or workdir) / stock_src.name
    stock_bin = ensure_stock_baseline(
        hip_path=args.hip_path, amdgpu_targets=args.amdgpu_targets,
        workdir=stock_build_root, stock_src=stock_src,
    )
    stock_build_identity = capture_build_identity(
        stock_build_root / "stock", source_root=stock_src, source_tree=stock_src.name,
        architecture=args.amdgpu_targets, build_mode="stock",
        requested_cmake_args=[], generator=CMAKE_GENERATOR, build_env=build_env,
    )
    _print(f"stock build identity: {stock_build_identity.identity_digest[:12]}")

    from bigcherry.e2e_smoke_campaign import Campaign, CampaignError  # noqa: E402

    campaign = Campaign(
        model=args.model,
        tune_server=tune_bin / f"llama-server{exe}",
        replay_server=replay_bin / f"llama-server{exe}",
        manifest=args.manifest,
        workdir=workdir / "campaign",
        stock_bench=stock_bin / f"llama-bench{exe}",
        tune_bench=tune_bin / f"llama-bench{exe}",
        replay_bench=replay_bin / f"llama-bench{exe}",
        bench_prompt=args.bench_prompt, bench_gen=args.bench_gen,
        bench_repetitions=args.bench_repetitions,
    )
    try:
        campaign.run()
    except CampaignError as exc:
        _print(f"CAMPAIGN FAILED: {exc}")
        return 1

    report_path = workdir / "campaign" / "report.md"
    _print(f"done -- report: {report_path}")
    print(report_path.read_text(encoding="utf-8"))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bigcherry patch-validation-campaign")
    parser.add_argument("--patch", required=True,
                         help="patch module name under patches/, e.g. 1204_rd08_q6k_mmvq_vdr2")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--hip-path", required=True, type=Path)
    parser.add_argument("--amdgpu-targets", required=True, help="e.g. gfx1100 or gfx1201")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path,
                         help="per-run campaign output (record/tune/promote/replay/bench/report)")
    parser.add_argument("--build-root", type=Path, default=None,
                         help="shared build-tree location (tune/replay/stock), reused across "
                              "multiple patch+model runs on this machine+arch; defaults to "
                              "--workdir (no reuse) if omitted")
    parser.add_argument("--worktree-root", type=Path, default=Path(r"C:\bc-worktrees")
                         if sys.platform == "win32" else Path.home() / "bc-worktrees",
                         help="content-addressed isolated source worktrees "
                              "(patch_source_isolation.py, HI82) live here, one per "
                              "(base_revision, patch, framework-baseline) identity")
    parser.add_argument("--bench-prompt", type=int, default=512)
    parser.add_argument("--bench-gen", type=int, default=128)
    parser.add_argument("--bench-repetitions", type=int, default=5)
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

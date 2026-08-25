"""HI76: compile-only check for both HIP autotune build configurations.

The offline pytest suite only does source-text pattern matching, never
compilation. This session found a production-breaking compile bug
(hip-autotune-dispatch.cu referencing a transform-only type outside its
GGML_HIP_ROUTING_TRANSFORM guard) that broke every GGML_HIP_DISPATCH_REPLAY=ON
build -- the actual production configuration -- and it had apparently never
been compiled before. This script configures and builds ggml-hip for BOTH
build configurations, no GPU or model required, cheap enough to run before
landing any change touching hip-autotune-*.cu/.cuh/.h:

  tune-mode:   GGML_HIP_AUTOTUNE=ON, GGML_HIP_AUTOTUNE_RECORD=ON,
               GGML_HIP_ROUTING_TRANSFORM=ON
  replay-only: GGML_HIP_DISPATCH_REPLAY=ON (GGML_HIP_AUTOTUNE off -- these
               two are cmake-mutually-exclusive, so this is necessarily a
               second configure+build, not a flag flip on one tree)

Usage:
    python -m bigcherry.compile_check --hip-path H:/.../vendor/rocm/7.1 \\
        --amdgpu-targets gfx1100 --workdir C:/bcw-check
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
LLAMA_CPP_SRC = REPO_ROOT / "vendor" / "llama.cpp"

CONFIGS = {
    "tune-mode": [
        "-DGGML_HIP_AUTOTUNE=ON",
        "-DGGML_HIP_AUTOTUNE_RECORD=ON",
        "-DGGML_HIP_ROUTING_TRANSFORM=ON",
    ],
    "replay-only": [
        "-DGGML_HIP_DISPATCH_REPLAY=ON",
    ],
}


class CompileCheckError(RuntimeError):
    pass


def _run(args: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(
            args, stdout=log_file, stderr=subprocess.STDOUT, text=True,
        )
    if result.returncode != 0:
        tail = "\n".join(
            log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        )
        raise CompileCheckError(
            f"{' '.join(args)} failed (see {log_path}):\n{tail}"
        )


def check_one(
    name: str, extra_cmake_args: list[str], *, hip_path: Path,
    amdgpu_targets: str, workdir: Path, target: str,
) -> None:
    build_dir = workdir / name
    log_dir = workdir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    clang = hip_path / "bin" / (
        "clang.exe" if sys.platform == "win32" else "clang"
    )
    clangxx = hip_path / "bin" / (
        "clang++.exe" if sys.platform == "win32" else "clang++"
    )
    print(f"[compile-check] configuring {name} ...", flush=True)
    _run(
        [
            "cmake", "-S", str(LLAMA_CPP_SRC), "-B", str(build_dir), "-G", "Ninja",
            "-DCMAKE_BUILD_TYPE=Release", "-DGGML_HIP=ON",
            f"-DAMDGPU_TARGETS={amdgpu_targets}",
            f"-DCMAKE_C_COMPILER={clang}", f"-DCMAKE_CXX_COMPILER={clangxx}",
            *extra_cmake_args,
        ],
        log_dir / f"{name}-configure.log",
    )
    print(f"[compile-check] building {name} ({target}) ...", flush=True)
    _run(
        ["cmake", "--build", str(build_dir), "--target", target, "-j"],
        log_dir / f"{name}-build.log",
    )
    print(f"[compile-check] {name}: OK", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bigcherry compile-check")
    parser.add_argument("--hip-path", required=True, type=Path,
                         help="ROCm/HIP install root (bin/clang{,++} live under it)")
    parser.add_argument("--amdgpu-targets", required=True,
                         help="e.g. gfx1100, or gfx1100;gfx1201;gfx1030")
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--target", default="ggml-hip",
                         help="build target (default: ggml-hip, the fastest "
                              "way to surface a compile error)")
    parser.add_argument(
        "--only", choices=sorted(CONFIGS), action="append",
        help="restrict to one config (repeatable); default: both",
    )
    args = parser.parse_args(argv)

    configs = args.only or list(CONFIGS)
    failures: list[str] = []
    for name in configs:
        try:
            check_one(
                name, CONFIGS[name], hip_path=args.hip_path,
                amdgpu_targets=args.amdgpu_targets, workdir=args.workdir,
                target=args.target,
            )
        except CompileCheckError as exc:
            print(f"[compile-check] {name}: FAILED\n{exc}", file=sys.stderr)
            failures.append(name)

    if failures:
        print(f"[compile-check] FAILED: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"[compile-check] all configurations OK: {', '.join(configs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

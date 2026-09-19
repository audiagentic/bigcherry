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
    python -m bigcherry.patch.validation_campaign \\
        --patch <patch-id> \\
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
import dataclasses
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from array import array
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from bigcherry.build.builds import (
    CompletedBuildEvidence,
    capture_completed_build_evidence,
)
from bigcherry.campaign.bench_runner import (  # noqa: F401
    BENCH_RUNNER_ROOT, BenchRunnerError, run_bench_runner_server_bench,
)
from bigcherry.experiment import contract as experiment_contract
from bigcherry.experiment.attestation import (
    ExecutionAttestation,
    ExecutionIdentity,
    compare_execution_identity,
)
from bigcherry.experiment.server_execution import AttestedServerSession
from bigcherry.patch.validation import (
    ArtifactRef,
    ValidationContext,
    ValidationPlan,
    ValidationResult,
    Verdict,
)
from bigcherry.patch.validation_producer import (
    FatTargetPlan,
    ProducerBuildPair,
    ProducerContext,
    ProducerDeviceContext,
    ProducerPairedBenchmarkOutcome,
    ProducerResult,
    ProducerSelection,
    ValidationProducerError,
    resolve_producer,
    validate_producer_cli_compatibility,
    validate_producer_inputs,
    validate_producer_result,
)

# GPT review (req_8429aa8e0d35496e, 2026-09-11): every RD73 server session
# governed by this module's HIP-only selector contract (require_device_
# visibility()/DeviceVisibility, see PNRO17) must explicitly unset an
# inherited ROCR_VISIBLE_DEVICES, not merely avoid setting one -- ambient
# env still reaches ServerRunner.launch() (dict(os.environ) + env_unset,
# then env_overrides) otherwise, reproducing the double-filtering bug.
_ROCR_VISIBLE_DEVICES_UNSET: tuple[str, ...] = ("ROCR_VISIBLE_DEVICES",)


def _hip_only(env_overrides: "dict[str, str] | None") -> "dict[str, str] | None":
    """Defense in depth alongside env_unset=_ROCR_VISIBLE_DEVICES_UNSET:
    strip ROCR_VISIBLE_DEVICES from a caller-supplied overrides dict too,
    so a future direct caller passing it through selector_env/env_overrides
    cannot reintroduce the double-filtering hazard via override-ordering
    (ServerRunner.launch() applies env_unset BEFORE env_overrides)."""
    if not env_overrides:
        return None
    return {k: v for k, v in env_overrides.items() if k != "ROCR_VISIBLE_DEVICES"}
from bigcherry.patch.activation import ActivationEvidence, verdict, write_activation_json

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
LLAMA_CPP_SRC = REPO_ROOT / "vendor" / "llama.cpp"
CMAKE_GENERATOR = "Ninja"


class PatchCampaignError(RuntimeError):
    pass


def _print(msg: str) -> None:
    print(f"[patch-campaign] {msg}", flush=True)


def _write_bound_artifact(run_dir: Path, name: str, payload: object) -> dict[str, str]:
    target = run_dir / "artifacts" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return {
        "path": target.relative_to(run_dir).as_posix(),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }


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


def resolve_selected_device_execution_identity(
    *, expected_arch: str, host_devices=None,
) -> tuple[ExecutionIdentity, dict[str, str]]:
    """PRBE111 shared helper: resolve the real, host-configured
    ExecutionIdentity (architecture + verified PCI locator) and env-override
    dict for the ONE physical device the ambient HIP_VISIBLE_DEVICES
    currently selects.

    Required, fail-closed semantics (never guessed, never partial):
    - HIP_VISIBLE_DEVICES must be explicitly set in the environment.
    - It must select exactly one device (a single numeric index).
    - That index must be a real, configured Device in this host's
      inventory (config/environment.toml).
    - The configured Device must have a real, verified `locator`.
    - The configured Device's `arch` must equal `expected_arch` -- a
      selector pointing at the wrong physical architecture is a caller
      bug, not something to silently accept.

    Any server-based (llama-server) producer needing real execution
    attestation should go through this helper rather than constructing
    its own ExecutionIdentity, so the "never guess a locator" discipline
    stays in exactly one place.
    """
    import os

    from bigcherry.core import environment as bc_environment

    raw = os.environ.get("HIP_VISIBLE_DEVICES")
    if not raw:
        raise PatchCampaignError(
            "resolve_selected_device_execution_identity: HIP_VISIBLE_DEVICES "
            "must be explicitly set (an unset/ambient-default device list "
            "cannot be trusted for a real-hardware measurement)"
        )
    selectors = [part for part in raw.split(",") if part != ""]
    if len(selectors) != 1:
        raise PatchCampaignError(
            f"resolve_selected_device_execution_identity: HIP_VISIBLE_DEVICES={raw!r} "
            "must select exactly one device for a server-based producer"
        )
    try:
        index = int(selectors[0])
    except ValueError as exc:
        raise PatchCampaignError(
            f"resolve_selected_device_execution_identity: HIP_VISIBLE_DEVICES={raw!r} "
            "is not a numeric device index"
        ) from exc

    if host_devices is None:
        host_devices = bc_environment.load_default().host().devices
    matches = [d for d in host_devices if d.index == index]
    if not matches:
        raise PatchCampaignError(
            f"resolve_selected_device_execution_identity: HIP_VISIBLE_DEVICES "
            f"selects index {index}, which is not a configured device in "
            f"config/environment.toml (known indices: {sorted(d.index for d in host_devices)})"
        )
    device = matches[0]
    if device.locator is None:
        raise PatchCampaignError(
            f"resolve_selected_device_execution_identity: device index {index} "
            f"({device.arch}) has no verified locator in config/environment.toml -- "
            "add one (via real `rocm-smi --showbus` output, never invented) before "
            "using it with a server-based attestation producer"
        )
    if device.arch != expected_arch:
        raise PatchCampaignError(
            f"resolve_selected_device_execution_identity: device index {index} is "
            f"configured as {device.arch!r}, but {expected_arch!r} was requested -- "
            "HIP_VISIBLE_DEVICES selects the wrong physical architecture"
        )

    identity = ExecutionIdentity(
        backend="ROCm", architectures=(device.arch,), locators=(device.locator,),
    )
    selector_env = {"HIP_VISIBLE_DEVICES": str(index)}
    return identity, selector_env


def _requested_cmake_args(amdgpu_targets: str, extra_cmake_args: list[str]) -> list[str]:
    """Identity-relevant CMake intent shared by configure and post-build
    verification -- ONE definition so the two can never silently drift
    apart (capture_completed_build_evidence() checks these same values
    against the resolved CMakeCache.txt)."""
    return [
        "-DCMAKE_BUILD_TYPE=Release", "-DGGML_HIP=ON",
        f"-DAMDGPU_TARGETS={amdgpu_targets}", *extra_cmake_args,
    ]


def _full_requested_cmake_args(
    *, hip_path: Path, amdgpu_targets: str, extra_cmake_args: list[str],
) -> list[str]:
    """Every -D value actually supplied to `cmake` configure, AND what
    capture_completed_build_evidence() is told was requested -- the two
    must describe the same build intent, or the verifier's "requested vs
    resolved cache" check is comparing against an incomplete picture."""
    is_windows = sys.platform == "win32"
    clang = hip_path / "bin" / ("clang.exe" if is_windows else "clang")
    clangxx = hip_path / "bin" / ("clang++.exe" if is_windows else "clang++")

    args = [
        *_requested_cmake_args(amdgpu_targets, extra_cmake_args),
        f"-DCMAKE_C_COMPILER={clang}", f"-DCMAKE_CXX_COMPILER={clangxx}",
        f"-DCMAKE_PREFIX_PATH={hip_path}", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    ]

    if is_windows:
        # ROCm's clang driver invokes lld-link, which (like MSVC link.exe)
        # embeds a wall-clock PE timestamp by default -- every relink
        # produces different binary bytes even with byte-identical inputs.
        # /Brepro makes lld-link derive that field from content instead,
        # so a genuine no-op relink is byte-reproducible. Found for real
        # via HI82 item 9: two back-to-back identical campaign runs
        # produced different runtime_bundle_hash values purely from this
        # (diagnosed with GPT, req_cc5af49494fe457a).
        args += [
            "-DCMAKE_EXE_LINKER_FLAGS=-Wl,/Brepro",
            "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,/Brepro",
            "-DCMAKE_MODULE_LINKER_FLAGS=-Wl,/Brepro",
        ]

    return args


_CONFIGURE_REQUEST_SCHEMA_VERSION = 1


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _configure_request_document(*, source: Path, cmake_args: list[str]) -> dict[str, object]:
    return {
        "schema_version": _CONFIGURE_REQUEST_SCHEMA_VERSION,
        "source": str(source.resolve()), "generator": CMAKE_GENERATOR,
        "cmake_args": list(cmake_args),
    }


def _configure_request_matches(*, build_dir: Path, expected: dict[str, object]) -> bool:
    """Whether build_dir's existing CMake cache was configured by the exact
    same request as `expected` -- the real fix for the old, too-broad "skip
    configure whenever CMakeCache.txt exists" check (which could silently
    reuse a stale configuration across differently-parameterized
    invocations) without paying for a full reconfigure on every single run
    (which itself can dirty Ninja's dependency graph unnecessarily)."""
    cache = build_dir / "CMakeCache.txt"
    request_path = build_dir / "bigcherry-configure-request.json"
    if not cache.is_file() or not request_path.is_file():
        return False
    try:
        recorded = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return recorded == expected


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
    generated_proof_callback=None,
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

    env = _hip_env(hip_path)
    cmake_args = _full_requested_cmake_args(
        hip_path=hip_path, amdgpu_targets=amdgpu_targets, extra_cmake_args=extra_cmake_args,
    )
    configure_request = _configure_request_document(source=source, cmake_args=cmake_args)
    configure_request_path = build_dir / "bigcherry-configure-request.json"

    if generated_proof_callback is not None:
        generated_proof_callback("preconfigure", build_dir)

    # Reconfigure only when the REQUEST actually changed -- the old "skip
    # whenever CMakeCache.txt exists" check could silently reuse a stale
    # configuration across differently-parameterized invocations; the
    # opposite extreme, always reconfiguring, was tried this session and
    # found to needlessly perturb Ninja's dependency graph on every run
    # (compounding with autotune_catalog's now-fixed unconditional compile-
    # input rewrites to make byte-stable resume unreachable in practice).
    if _configure_request_matches(build_dir=build_dir, expected=configure_request):
        _print(f"{name}: configure request unchanged; reusing CMake cache")
    else:
        _print(f"configuring {name} ...")
        args = ["cmake", "-S", str(source), "-B", str(build_dir), "-G", CMAKE_GENERATOR, *cmake_args]
        log_path = log_dir / f"{name}-configure.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            result = subprocess.run(args, stdout=log_file, stderr=subprocess.STDOUT, env=env)
        if result.returncode != 0:
            raise PatchCampaignError(f"{name} configure failed (see {log_path})")
        _atomic_write_json(configure_request_path, configure_request)

    if generated_proof_callback is not None:
        generated_proof_callback("postconfigure-precompile", build_dir)

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
    if generated_proof_callback is not None:
        generated_proof_callback("postcompile", build_dir)
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

    build_dir = workdir / "stock"
    log_dir = workdir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = _hip_env(hip_path)
    cmake_args = _full_requested_cmake_args(
        hip_path=hip_path, amdgpu_targets=amdgpu_targets, extra_cmake_args=[],
    )
    configure_request = _configure_request_document(source=stock_src, cmake_args=cmake_args)
    configure_request_path = build_dir / "bigcherry-configure-request.json"

    if _configure_request_matches(build_dir=build_dir, expected=configure_request):
        _print("stock: configure request unchanged; reusing CMake cache")
    else:
        _print("configuring stock baseline ...")
        args = [
            "cmake", "-S", str(stock_src), "-B", str(build_dir), "-G", CMAKE_GENERATOR,
            *cmake_args,
        ]
        log_path = log_dir / "stock-configure.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            result = subprocess.run(args, stdout=log_file, stderr=subprocess.STDOUT, env=env)
        if result.returncode != 0:
            raise PatchCampaignError(f"stock configure failed (see {log_path})")
        _atomic_write_json(configure_request_path, configure_request)
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


def _trace_probe_env(*, hip_path: Path, disable_fusion: bool) -> dict[str, str]:
    env = _hip_env(hip_path)
    # A parent campaign/shell must not accidentally carry a dispatch mode
    # or an earlier trace/fusion setting into this isolated probe.
    for key in list(env):
        if key.startswith("GGML_HIP_DISPATCH_") or key.startswith("BIGCHERRY_"):
            env.pop(key, None)
    env["BIGCHERRY_PATCH_TRACE"] = "1"
    env["GGML_HIP_DISPATCH_MODE"] = "native"
    if disable_fusion:
        env["GGML_CUDA_DISABLE_FUSION"] = "1"
    else:
        env.pop("GGML_CUDA_DISABLE_FUSION", None)
    return env


# VA21 real-hardware finding: a llama.cpp binary whose ROCm/HIP device
# init fails silently falls back to CPU execution, but still prints a
# normal-looking llama-bench table under a "backend: ROCm" label with
# real-looking (if CPU-speed) numbers -- a benchmark/trigger-probe run's
# exit code and printed metrics alone cannot be trusted as proof that it
# actually executed on the GPU. Fail closed: require the real, positive
# "ggml_cuda_init: found N ROCm devices" line and reject any known
# ROCm-init-failure signature, rather than accepting output that never
# actually touched the GPU (which is exactly what let RD08's real
# subject_hit=false confound with a genuine dispatch-routing question on
# 2026-09-01 -- gfx1030's HIP runtime failed to detect the device at all,
# and nothing caught it before the trigger check quietly "ran" and
# reported a negative).
_ROCM_INIT_FAILURE_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"failed to initialize ROCm"),
    re.compile(r"no ROCm-capable device is detected"),
    re.compile(r"hipErrorNoDevice"),
)
_ROCM_INIT_SUCCESS_PATTERN = re.compile(r"ggml_cuda_init:\s*found\s+(\d+)\s+ROCm device")


def _require_real_gpu_execution(stdout: str, stderr: str, *, context: str) -> None:
    """Fail closed unless the combined output carries real, positive
    evidence of ROCm/HIP device initialization (not merely the absence of
    an error) -- see the module comment above this function for why
    absence-of-failure is not itself sufficient evidence here."""
    combined = f"{stdout}\n{stderr}"
    for pattern in _ROCM_INIT_FAILURE_PATTERNS:
        if pattern.search(combined):
            raise PatchCampaignError(
                f"{context}: ROCm/HIP device initialization failed -- real GPU execution "
                f"cannot be confirmed (matched {pattern.pattern!r} in the process output)"
            )
    match = _ROCM_INIT_SUCCESS_PATTERN.search(combined)
    if match is None or int(match.group(1)) < 1:
        raise PatchCampaignError(
            f"{context}: no real GPU execution evidence in the process output -- expected a "
            "'ggml_cuda_init: found N ROCm devices' line; a benchmark/probe that silently ran "
            "CPU-only must never be accepted as real hardware evidence"
        )


def _run_one_trace_probe(
    *, name: str, binary: Path, model: Path, hip_path: Path, workdir: Path,
    bench_prompt: int, bench_gen: int, disable_fusion: bool,
    extra_flags: tuple[str, ...] = (),
    env_overrides: Mapping[str, str] | None = None,
    env_unset: tuple[str, ...] = (),
) -> str:
    import subprocess

    binary = Path(binary)
    model = Path(model)
    if not binary.is_file():
        raise PatchCampaignError(f"activation probe binary does not exist: {binary}")
    if not model.is_file():
        raise PatchCampaignError(f"activation probe model does not exist: {model}")

    # VA21 real-hardware finding (2026-09-01): llama-bench.cpp itself gates
    # ggml's log level on its OWN --verbose flag (GGML_LOG_LEVEL_DEBUG if
    # verbose else GGML_LOG_LEVEL_ERROR) -- without it, BOTH GGML_LOG_INFO
    # and GGML_LOG_WARN are filtered before this probe's whole reason for
    # existing (observing a log-based activation marker) ever has a chance.
    # Confirmed directly: the exact same binary/env only emits
    # BIGCHERRY_PATCH_HIT with --verbose present. This function's entire
    # purpose is reading ggml log output, so it must always request it.
    command = [
        str(binary.resolve()), "-m", str(model.resolve()),
        "-p", str(bench_prompt), "-n", str(bench_gen), "-r", "1", "-ngl", "99", "--verbose",
        *extra_flags,
    ]
    env = _trace_probe_env(hip_path=hip_path, disable_fusion=disable_fusion)
    # PA36 RD13/1206 migration (GPT req_760c0fe82d7b4609 MAJOR): a
    # producer trace probe must run on the SAME physical device as the
    # correctness measurement -- apply the selected device's HIP-only
    # selector overrides, then its unsets LAST (the sanctioned selector
    # always wins; never ambient visibility).
    if env_overrides:
        env.update(env_overrides)
    for key in env_unset:
        env.pop(key, None)

    log_dir = workdir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"activation-{name}.log"

    completed = subprocess.run(
        command, cwd=workdir, env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        encoding="utf-8", errors="replace",
    )
    combined = completed.stdout + "\n" + completed.stderr
    log_path.write_text(
        f"command: {command!r}\n"
        f"GGML_CUDA_DISABLE_FUSION={env.get('GGML_CUDA_DISABLE_FUSION')!r}\n\n"
        f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise PatchCampaignError(
            f"activation probe {name!r} failed with exit code {completed.returncode}; "
            f"see {log_path}"
        )
    _require_real_gpu_execution(
        completed.stdout, completed.stderr, context=f"activation probe {name!r}",
    )
    return combined


def run_trace_activation_probes(
    *, marker_regex: str | None, description: str | None,
    binary: Path, model: Path, hip_path: Path, workdir: Path,
    bench_prompt: int, bench_gen: int,
    env_overrides: Mapping[str, str] | None = None,
    env_unset: tuple[str, ...] = (),
) -> tuple[ActivationEvidence, dict[str, object]] | None:
    """Run positive + fusion-disabled negative-control activation probes.

    Returns None for patches which do not use the shared trace-marker
    mechanism.

    Positive: BIGCHERRY_PATCH_TRACE=1 -- expected marker MUST occur to
    prove the patch's path executed.

    Negative control: BIGCHERRY_PATCH_TRACE=1 + GGML_CUDA_DISABLE_FUSION=1
    -- marker MUST NOT occur. A marker that survives the negative control
    is not trustworthy activation evidence (it would fire regardless of
    whether the specific fusion this patch adds actually ran) and is
    classified unobservable rather than executed.
    """
    if not marker_regex or not description:
        return None

    pattern = re.compile(marker_regex)

    _print(f"activation probe: {description} (positive)")
    positive_output = _run_one_trace_probe(
        name="positive", binary=binary, model=model, hip_path=hip_path, workdir=workdir,
        bench_prompt=bench_prompt, bench_gen=bench_gen, disable_fusion=False,
        env_overrides=env_overrides, env_unset=env_unset,
    )
    positive_hit = pattern.search(positive_output) is not None

    _print(f"activation probe: {description} (fusion-disabled control)")
    negative_output = _run_one_trace_probe(
        name="fusion-disabled", binary=binary, model=model, hip_path=hip_path, workdir=workdir,
        bench_prompt=bench_prompt, bench_gen=bench_gen, disable_fusion=True,
        env_overrides=env_overrides, env_unset=env_unset,
    )
    negative_hit = pattern.search(negative_output) is not None

    if negative_hit:
        evidence = ActivationEvidence(
            status="unobservable", mechanism="BIGCHERRY_PATCH_TRACE two-probe control",
            detail=(
                f"{description}: marker was present even with GGML_CUDA_DISABLE_FUSION=1. "
                "The marker therefore does not uniquely prove execution of the intended "
                "fusion path."
            ),
        )
    elif positive_hit:
        evidence = ActivationEvidence(
            status="executed", mechanism="BIGCHERRY_PATCH_TRACE two-probe control",
            detail=(
                f"{description}: expected marker was observed with fusion enabled and "
                "was absent with GGML_CUDA_DISABLE_FUSION=1."
            ),
        )
    else:
        evidence = ActivationEvidence(
            status="not_executed", mechanism="BIGCHERRY_PATCH_TRACE two-probe control",
            detail=(
                f"{description}: expected marker was absent from the positive probe and "
                "remained absent in the fusion-disabled control. This model/workload did not "
                "prove execution of the patch path."
            ),
        )

    detail: dict[str, object] = {
        "description": description, "marker_regex": marker_regex,
        "positive": {
            "BIGCHERRY_PATCH_TRACE": "1", "GGML_CUDA_DISABLE_FUSION": None,
            "marker_observed": positive_hit, "log": "logs/activation-positive.log",
        },
        "negative_control": {
            "BIGCHERRY_PATCH_TRACE": "1", "GGML_CUDA_DISABLE_FUSION": "1",
            "marker_observed": negative_hit, "log": "logs/activation-fusion-disabled.log",
        },
    }
    return evidence, detail


def compute_contract_correctness_gate(
    contract: experiment_contract.ExperimentContract | None,
    named_results: "dict[str, object] | None" = None,
) -> dict[str, object] | None:
    """VA14 final slice (GPT session ses_5bbee8ce5c9a4265, req_75c09f14757640af):
    delegates to the real, native `experiment_contract.evaluate_correctness_gate()`
    instead of always reporting BLOCKED -- a real per-named-check evidence
    producer now exists for at least one contract (RD08's
    require_rd08_correctness_evidence(), orchestrated by
    run_rd08_contract_correctness() below). Returns None when there is no
    bound contract at all, or the contract declares no required correctness
    checks (a pure-performance contract passes trivially).

    ``named_results`` must be real ``CorrectnessResult``s keyed by check
    name, never a generic --correctness-evidence summary standing in for a
    specific named check (GPT round 8, req_84fca34f83064678 -- that
    confusion is exactly what this signature change prevents: there is no
    longer a `correctness_summary` parameter to misuse this way). Missing
    results (None, or a check simply absent from the dict) are reported via
    evaluate_correctness_gate()'s own missing_checks -- caught here as a
    BLOCKED-shaped dict only when the gate would otherwise raise for
    receiving literally zero results against a contract that requires some
    (its own hard-fail-on-truly-empty-input behavior); any check present in
    ``named_results`` is judged on its own passed/failed value."""
    if contract is None or not contract.correctness.required_checks:
        return None
    from bigcherry.experiment import contract as experiment_contract

    try:
        return experiment_contract.evaluate_correctness_gate(contract, dict(named_results or {}))
    except experiment_contract.ExperimentContractError:  # pi-lens-ignore: no-bare-except
        required_checks = contract.correctness.required_checks
        return {
            "passed": False,
            "status": "blocked",
            "required_checks": list(required_checks),
            "missing_checks": list(required_checks),
            "failed_checks": [],
            "results": {},
            "detail": (
                f"contract requires correctness check(s) {list(required_checks)!r}; no "
                "per-named-check evidence was supplied for this run"
            ),
        }


def assert_validation_subject_parity(
    control_build_evidence: CompletedBuildEvidence,
    validation_subject_build_evidence: CompletedBuildEvidence,
    *, patch_id: str,
) -> None:
    """VA14-B (GPT session ses_5bbee8ce5c9a4265, req_cb50258c7f4c40f1): the
    validation-subject build must be a real build-parity match to control --
    same requested cmake args, same effective configure, same effective
    build id -- so a measured RD08 lane effect can be attributed to the
    patch alone, never to an accidental build-option drift between the two
    binaries. Deliberately does NOT compare runtime_bundle_hash,
    compile_verification_id, or full campaign_identity(): those are
    correctly source-content-sensitive, and control/subject sources
    legitimately differ (that is the whole point of the comparison)."""
    control_configure = control_build_evidence.effective_configure
    subject_configure = validation_subject_build_evidence.effective_configure
    if control_configure != subject_configure:
        raise PatchCampaignError(
            f"{patch_id}: validation-subject build is not configure-parity with control -- "
            f"control={control_configure!r} subject={subject_configure!r}"
        )
    control_id = control_build_evidence.effective_build_id
    subject_id = validation_subject_build_evidence.effective_build_id
    if control_id != subject_id:
        raise PatchCampaignError(
            f"{patch_id}: validation-subject build_id {subject_id!r} does not match "
            f"control build_id {control_id!r} despite matching effective_configure -- "
            "refusing to run parity-dependent RD08 lanes against a non-parity build"
        )


def rd08_validation_lane_commands(
    *, control_binary: Path, subject_binary: Path, model: Path, workload: str,
    extra_flags: tuple[str, ...] = (),
) -> tuple[list[str], list[str]]:
    """VA14-B: the real, minimal llama-bench command pair for one RD08 lane
    -- control_command, subject_command -- differing only by binary path,
    consistent with metric_for_workload()'s decode->tg128/prefill->pp512
    mapping (decode: -p 0 -n 128; prefill: -p 512 -n 0). ``extra_flags``
    (VA06: e.g. ("-sm", "tensor") for a multi-GPU model) is appended
    after the workload shape/-ngl flags -- empty by default, so RD08's
    own existing behavior is unchanged."""
    if workload == "decode":
        workload_flags = ["-p", "0", "-n", "128"]
    elif workload == "prefill":
        workload_flags = ["-p", "512", "-n", "0"]
    else:
        raise PatchCampaignError(f"rd08 lane: no llama-bench flag mapping for workload {workload!r}")
    control_command = [
        str(control_binary), "-m", str(model), *workload_flags, "-ngl", "99", *extra_flags,
    ]
    subject_command = [
        str(subject_binary), "-m", str(model), *workload_flags, "-ngl", "99", *extra_flags,
    ]
    return control_command, subject_command


def run_rd08_validation_lanes(
    *, contract: object, control_binary: Path, subject_binary: Path, model: Path,
    model_ref: str, hip_path: Path, run_dir: Path,
    control_build_identity: dict[str, object], subject_build_identity: dict[str, object],
    pairs: int = 3,
) -> dict[str, object]:
    """VA14-B: execute RD08's real positive (decode) and control (prefill)
    lanes via experiment/execution.py's paired runner, persist raw
    stdout/stderr + paired measurements + bootstrap stats alongside the
    computed LaneEffects, and return a bound artifact reference plus the
    LaneEffect list -- so a caller can both persist real evidence and feed
    aggregate_contract_effects() without re-running anything.

    Deliberately does NOT touch correctness/promotion/eligibility: this is
    execution + evidence persistence only, per GPT's explicit VA14-B scope
    (bit_identical producer integration, trigger/promotion composition,
    evaluate_promotion_gate(), and the eligibility cutover are deferred).

    PVPS02 step 2: now a compatibility wrapper over run_paired_llama_
    benchmark() -- same argv shape (rd08_validation_lane_commands' own
    extra_flags is always empty on this call path, so runtime_args=()
    reproduces it exactly), same env sanitization, same raw-log capture;
    only the LaneEffect/Contract-evidence composition below is RD08-
    specific and stays here."""
    from bigcherry.experiment import contract as experiment_contract
    from bigcherry.experiment import execution as experiment_execution

    outcome = run_paired_llama_benchmark(
        control_binary=control_binary, subject_binary=subject_binary, model=model,
        hip_path=hip_path, pairs=pairs, log_context="rd08",
    )
    decode_run, prefill_run = outcome.runs["decode"], outcome.runs["prefill"]
    decode_control_cmd = outcome.commands["decode"]["control"]
    decode_subject_cmd = outcome.commands["decode"]["subject"]
    prefill_control_cmd = outcome.commands["prefill"]["control"]
    prefill_subject_cmd = outcome.commands["prefill"]["subject"]
    raw_logs = outcome.raw_logs
    effects = [
        experiment_execution.lane_effect_from_run("positive", "tg128", decode_run),
        experiment_execution.lane_effect_from_run("control", "pp512", prefill_run),
    ]
    positive_ref = experiment_contract.evidence_ref_for_lane(
        contract, role="positive", workload_tag="decode", model_ref=model_ref,
    )
    control_ref = experiment_contract.evidence_ref_for_lane(
        contract, role="control", workload_tag="prefill", model_ref=model_ref,
    )
    lane_evidence = {
        "contract_id": contract.id,
        "model_ref": model_ref, "model_path": str(model),
        "validation_build_identities": {
            "control": control_build_identity, "subject": subject_build_identity,
        },
        "raw_logs": raw_logs,
        "lanes": {
            "positive": {
                "metric": "tg128", "contract_evidence": positive_ref.document(),
                "control_command": decode_control_cmd, "subject_command": decode_subject_cmd,
                "runs": list(decode_run.runs), "stats": decode_run.stats,
            },
            "control": {
                "metric": "pp512", "contract_evidence": control_ref.document(),
                "control_command": prefill_control_cmd, "subject_command": prefill_subject_cmd,
                "runs": list(prefill_run.runs), "stats": prefill_run.stats,
            },
        },
    }
    artifact_ref = _write_bound_artifact(run_dir, "validation-lanes.json", lane_evidence)
    return {"artifact": artifact_ref, "effects": effects}


def _load_rd08_correctness_module() -> object:
    """Dynamically load the real, already-reviewed RD08 correctness
    producer (patches/1204_rd08_q6k_mmvq_vdr2/validation/rd08_correctness.py)
    -- orchestrated here, never reimplemented (that module is the
    authoritative 5-shape x 3-seed exact-digest proof; this caller's job is
    build/execute/persist plumbing only)."""
    module_path = (
        REPO_ROOT / "patches" / "1204_rd08_q6k_mmvq_vdr2" / "validation" / "rd08_correctness.py"
    )
    if not module_path.is_file():
        raise PatchCampaignError(f"rd08 correctness producer not found at {module_path}")
    import importlib.util

    spec = importlib.util.spec_from_file_location("_bigcherry_rd08_correctness", module_path)
    if spec is None or spec.loader is None:
        raise PatchCampaignError(f"cannot load rd08 correctness producer at {module_path}")
    module = importlib.util.module_from_spec(spec)
    # VA15 real-hardware finding: module_from_spec() does not register the
    # module in sys.modules -- @dataclass (Rd08Shape, ShapeSeedComparison)
    # resolves its owning module via sys.modules[cls.__module__] during
    # decoration, so without this the decorator crashes with
    # AttributeError: 'NoneType' object has no attribute '__dict__'.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_rd08_contract_correctness(
    *, base_revision: str, hip_path: Path, amdgpu_targets: str, worktree_root: Path,
    build_root: Path, build_env: dict[str, str], run_dir: Path, _module: object | None = None,
) -> dict[str, object]:
    """VA14 final slice: RD08's real bit-identical correctness producer,
    orchestrated. materialize_rd08_variants() builds its OWN isolated
    VDR2-subject/VDR1-control worktrees -- a source-level A/B distinct from
    this campaign's control/validation-subject trees -- which this function
    then builds symmetrically (extra_cmake_args=[], matching each other)
    and hands to require_rd08_correctness_evidence(), the authoritative
    5-shape x 3-seed exact-digest proof.

    Only Rd08CorrectnessError (a real, specific correctness failure) is
    caught and turned into passed=False; materialization/build/
    infrastructure errors remain hard campaign errors, never silently
    downgraded to a correctness result. ``_module`` is injectable for
    hardware-free testing; defaults to the real dynamically-loaded
    producer."""
    from bigcherry.experiment import contract as experiment_contract
    from bigcherry.patch import source as psi

    rd08_correctness = _module or _load_rd08_correctness_module()
    subject_src, control_src = rd08_correctness.materialize_rd08_variants(
        base_repo=LLAMA_CPP_SRC, worktree_root=worktree_root, base_revision=base_revision,
    )
    exe = ".exe" if sys.platform == "win32" else ""
    correctness_build_root = build_root / "rd08-correctness"

    subject_bin = build_tree(
        name="rd08-correctness-subject", hip_path=hip_path, amdgpu_targets=amdgpu_targets,
        workdir=correctness_build_root, targets=["test-backend-ops"], source=subject_src,
        extra_cmake_args=[],
    )
    control_bin = build_tree(
        name="rd08-correctness-control", hip_path=hip_path, amdgpu_targets=amdgpu_targets,
        workdir=correctness_build_root, targets=["test-backend-ops"], source=control_src,
        extra_cmake_args=[],
    )
    cmake_args = _full_requested_cmake_args(
        hip_path=hip_path, amdgpu_targets=amdgpu_targets, extra_cmake_args=[],
    )
    subject_build_evidence = capture_completed_build_evidence(
        correctness_build_root / "rd08-correctness-subject", source_root=subject_src,
        architecture=amdgpu_targets, binary=subject_bin / f"test-backend-ops{exe}",
        requested_cmake_args=cmake_args, build_env=build_env,
    )
    control_build_evidence = capture_completed_build_evidence(
        correctness_build_root / "rd08-correctness-control", source_root=control_src,
        architecture=amdgpu_targets, binary=control_bin / f"test-backend-ops{exe}",
        requested_cmake_args=cmake_args, build_env=build_env,
    )

    # VA15 real-hardware finding: correctness_evidence.run_test_backend_ops()
    # calls subprocess.run(argv, env=run_env) with run_env built from
    # scratch ({} plus its own explicit keys) -- an explicit env=
    # REPLACES the child's environment rather than extending it, so
    # HIP_VISIBLE_DEVICES/ROCR_VISIBLE_DEVICES set on this process are
    # silently dropped and the subprocess sees every GPU again, crashing
    # against an architecture-restricted build exactly like the earlier
    # multi-GPU segfault this campaign already worked around. Wrap the
    # runner to restore the real ambient environment underneath whatever
    # run_test_backend_ops() explicitly sets (which must still win).
    def _correctness_runner(argv, **kwargs):
        env = {**os.environ, **(kwargs.pop("env", None) or {})}
        return subprocess.run(argv, env=env, **kwargs)

    # GPT review (req_3c98f154389148bb, 2026-09-11): run every (shape,
    # seed) pair exactly once via the non-raising collector -- not once
    # for the gate and again for diagnostics, which would double real
    # hardware time -- and derive the SAME pass/fail semantics
    # require_rd08_correctness_evidence() would have raised on (first
    # non-.ok row) from those results, while retaining every row's full
    # numerical metrics (err/max_abs/threshold/n, not just a bool) even
    # when the run fails early. The gate itself is unchanged: still
    # exact per-row .ok (bit-identical digest equality); this only stops
    # discarding the rows that already ran successfully before a later
    # failure, or that ran cleanly alongside one.
    all_rows = rd08_correctness.collect_all_rd08_correctness_rows(
        subject_binary=subject_bin / f"test-backend-ops{exe}",
        control_binary=control_bin / f"test-backend-ops{exe}",
        runner=_correctness_runner,
    )
    rows_doc = [rd08_correctness.row_to_diagnostic_dict(r) for r in all_rows]
    failing = next((r for r in all_rows if not r.ok), None)
    if failing is None:
        bit_identical_result = experiment_contract.CorrectnessResult(
            check="bit_identical", passed=True,
            detail=f"{len(all_rows)} (shape,seed) pairs bit-identical",
        )
    else:
        bit_identical_result = experiment_contract.CorrectnessResult(
            check="bit_identical", passed=False,
            detail=(
                f"RD08 correctness evidence failed for shape={failing.shape_name!r} "
                f"seed={failing.seed}: subject_status={failing.subject_status} "
                f"control_status={failing.control_status} "
                f"subject_input_digest={failing.subject_digest.digest if failing.subject_digest else None} "
                f"control_input_digest={failing.control_digest.digest if failing.control_digest else None} "
                f"subject_output_digest={failing.subject_metric.backend1_digest if failing.subject_metric else None} "
                f"control_output_digest={failing.control_metric.backend1_digest if failing.control_metric else None}"
            ),
        )

    # PRBE104 (2026-09-13): RD08's contract was deliberately revised from
    # requiring bit_identical to backend_reference (GPT req_8163325eb9c545ea)
    # -- VDR=2 intentionally changes accumulation grouping, so exact digest
    # equality was never the scientifically appropriate bar; each row's own
    # subject_metric.err/threshold (already computed by the NMSE comparison
    # every row performs regardless of the exact-digest result) IS the real
    # backend_reference evidence. Derive it from the SAME rows, never a
    # second correctness-checking engine -- bit_identical is preserved above
    # as the real, now-non-gating diagnostic fact PRBE103 established.
    numeric_rows = [r for r in all_rows if r.subject_metric is not None]
    if not numeric_rows:
        backend_reference_result = experiment_contract.CorrectnessResult(
            check="backend_reference", passed=False,
            detail="no rows produced a subject_metric to evaluate",
        )
    else:
        worst = max(numeric_rows, key=lambda r: r.subject_metric.err)
        over_threshold = worst.subject_metric.err > worst.subject_metric.threshold
        backend_reference_result = experiment_contract.CorrectnessResult(
            check="backend_reference", passed=not over_threshold,
            detail=(
                f"{len(numeric_rows)} rows, worst subject err={worst.subject_metric.err} "
                f"vs threshold={worst.subject_metric.threshold} "
                f"(shape={worst.shape_name!r} seed={worst.seed})"
            ),
        )

    correctness_doc = {
        "bit_identical": {"passed": bit_identical_result.passed, "detail": bit_identical_result.detail},
        "backend_reference": {"passed": backend_reference_result.passed, "detail": backend_reference_result.detail},
        "subject_source_tree": psi.git_worktree_tree(subject_src),
        "control_source_tree": psi.git_worktree_tree(control_src),
        "subject_build_identity": subject_build_evidence.campaign_identity(),
        "control_build_identity": control_build_evidence.campaign_identity(),
        "rows": rows_doc,
    }
    artifact_ref = _write_bound_artifact(run_dir, "rd08-correctness.json", correctness_doc)
    return {
        "results": {
            "bit_identical": bit_identical_result,
            "backend_reference": backend_reference_result,
        },
        "artifact": artifact_ref,
        "subject_build_identity": subject_build_evidence.campaign_identity(),
        "control_build_identity": control_build_evidence.campaign_identity(),
    }




def run_rd30_correctness_check(
    *,
    base_revision: str,
    hip_path: Path,
    amdgpu_targets: str,
    worktree_root: Path,
    build_root: Path,
    build_env: dict[str, str],
    run_dir: Path,
    seeds: tuple[int, ...] = (1, 2, 3),
    _source_module: object | None = None,
    _evidence_module: object | None = None,
    _signature_module: object | None = None,
    _runner=None,
) -> dict[str, object]:
    """RD30 exact-output correctness producer.

    Control is the normal BigCherry source plus the deterministic
    test-backend-ops evidence chain; subject is the same composition plus
    1237. No partial/reverse edit is used -- RD30's six edit sites form one
    atomic implementation, and the correct control is "RD30 absent",
    exactly like RD13, not a bespoke partial-revert worktree like RD08's
    VDR1/VDR2 flip.

    Each arm runs the same exact 256-expert MUL_MAT_ID --test-file shapes
    in native dispatch mode. The bit-identical gate requires, for every
    shape/seed:
      * successful execution in both arms,
      * identical deterministic routing (leaf_2 digest -- requires
        patches/1236_hi105_deterministic_mul_mat_id_ids for two independent
        process invocations to see identical full-expert-range routing),
      * identical CPU-reference output digest,
      * identical output element count, and
      * identical HIP backend output bytes (backend1_digest).

    backend_reference is returned as an additional diagnostic result; the
    RD30 contract (config/experiment-contracts.toml's
    RD30-MOE-MMQ-COMPACT-GRID) gates on bit_identical only, since this is
    a pure launch-configuration change with no accumulation-order
    difference.
    """
    import math as _math

    from bigcherry.experiment import contract as experiment_contract

    if _source_module is None:
        from bigcherry.patch import source as psi
    else:
        psi = _source_module

    if _evidence_module is None:
        from bigcherry.tuning import correctness_evidence as correctness_evidence
    else:
        correctness_evidence = _evidence_module

    if _signature_module is None:
        from bigcherry.tuning import signature_mapping
    else:
        signature_mapping = _signature_module

    targets = tuple(
        target.strip()
        for target in amdgpu_targets.replace(",", ";").split(";")
        if target.strip()
    )
    if targets != ("gfx1100",):
        raise PatchCampaignError(
            "RD30 correctness is scoped to gfx1100 exactly; "
            f"got AMDGPU_TARGETS={amdgpu_targets!r}"
        )
    if not seeds or any(seed == 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise PatchCampaignError(
            "RD30 correctness requires a non-empty set of unique nonzero seeds"
        )

    evidence_patches = (
        "1222_hi67_deterministic_test_backend_ops_seed",
        "1223_hi67_machine_readable_correctness_metrics",
        "1236_hi105_deterministic_mul_mat_id_ids",
    )
    subject_patch = "1237_rd30_moe_mmq_compact_grid"

    control_revision, control_composition = psi.resolve_source_composition(
        "bigcherry",
        extra_patches=evidence_patches,
        base_ref=base_revision,
        base_repo=LLAMA_CPP_SRC,
    )
    subject_revision, subject_composition = psi.resolve_source_composition(
        "bigcherry",
        extra_patches=(*evidence_patches, subject_patch),
        base_ref=base_revision,
        base_repo=LLAMA_CPP_SRC,
    )
    if control_revision != subject_revision:
        raise PatchCampaignError(
            "RD30 correctness: control and subject resolved different base revisions"
        )

    control_src = psi.materialize_composition(
        base_repo=LLAMA_CPP_SRC,
        worktree_root=worktree_root / "rd30-correctness-control",
        resolved_revision=control_revision,
        composition=control_composition,
        overlay_root=psi.REPO_ROOT / "src",
        requested_revision=base_revision,
    )
    subject_src = psi.materialize_composition(
        base_repo=LLAMA_CPP_SRC,
        worktree_root=worktree_root / "rd30-correctness-subject",
        resolved_revision=subject_revision,
        composition=subject_composition,
        overlay_root=psi.REPO_ROOT / "src",
        requested_revision=base_revision,
    )

    exe = ".exe" if sys.platform == "win32" else ""
    correctness_build_root = build_root / "rd30-correctness"
    architecture_tag = amdgpu_targets.replace(";", "_").replace(",", "_")
    control_name = f"rd30-correctness-control-{architecture_tag}"
    subject_name = f"rd30-correctness-subject-{architecture_tag}"

    control_bin_dir = build_tree(
        name=control_name,
        hip_path=hip_path,
        amdgpu_targets=amdgpu_targets,
        workdir=correctness_build_root,
        targets=["test-backend-ops"],
        source=control_src,
        extra_cmake_args=[],
    )
    subject_bin_dir = build_tree(
        name=subject_name,
        hip_path=hip_path,
        amdgpu_targets=amdgpu_targets,
        workdir=correctness_build_root,
        targets=["test-backend-ops"],
        source=subject_src,
        extra_cmake_args=[],
    )
    control_binary = control_bin_dir / f"test-backend-ops{exe}"
    subject_binary = subject_bin_dir / f"test-backend-ops{exe}"

    cmake_args = _full_requested_cmake_args(
        hip_path=hip_path,
        amdgpu_targets=amdgpu_targets,
        extra_cmake_args=[],
    )
    control_build_evidence = capture_completed_build_evidence(
        correctness_build_root / control_name,
        source_root=control_src,
        architecture=amdgpu_targets,
        binary=control_binary,
        requested_cmake_args=cmake_args,
        build_env=build_env,
    )
    subject_build_evidence = capture_completed_build_evidence(
        correctness_build_root / subject_name,
        source_root=subject_src,
        architecture=amdgpu_targets,
        binary=subject_binary,
        requested_cmake_args=cmake_args,
        build_env=build_env,
    )

    op_names = signature_mapping.load_ggml_op_names(control_src)
    type_names = signature_mapping.load_ggml_type_names(control_src)

    def _enum_id(names: dict[int, str], wanted: str, kind: str) -> int:
        for value, name in names.items():
            if name.upper() == wanted.upper():
                return int(value)
        raise PatchCampaignError(
            f"RD30 correctness: {kind} enum {wanted!r} "
            "is absent from materialized source"
        )

    op_mul_mat_id = _enum_id(op_names, "MUL_MAT_ID", "ggml_op")
    type_f32 = _enum_id(type_names, "F32", "ggml_type")

    # 32 routed tokens is deliberately > upstream MMVQ_MAX_BATCH_SIZE (8).
    # With n_expert=256 on gfx1100, the production selector therefore
    # reaches MoE MMQ rather than the decode MMVQ path. ne1[1]==1 is the
    # real gate/up broadcast shape and remains inside RD30's nsamples_y==1
    # scope.
    shape_specs = (
        ("q4_k-moe-prefill32", "Q4_K"),
        ("q8_0-moe-prefill32", "Q8_0"),
    )

    scratch_dir = run_dir / "scratch" / "rd30-correctness"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    mapped_shapes: list[dict[str, object]] = []
    for shape_name, weight_type in shape_specs:
        signature = {
            "op": op_mul_mat_id,
            "flags": 0x0F,  # contiguous src0/src1/dst + HAS_IDS
            "ne0": [2048, 256, 256, 1],
            "ne1": [2048, 1, 32, 1],
            "ned": [256, 8, 32, 1],
            "n_expert": 256,
            "n_expert_used": 8,
            "src0_type": _enum_id(type_names, weight_type, "ggml_type"),
            "src1_type": type_f32,
            "dst_type": type_f32,
        }
        line, target_tensor, digest_tensor = (
            signature_mapping.signature_to_mul_mat_id_test_file_line(
                signature,
                vendor_root=control_src,
            )
        )
        test_file = scratch_dir / f"{shape_name}.txt"
        test_file.write_text(line + "\n", encoding="utf-8")
        mapped_shapes.append(
            {
                "name": shape_name,
                "weight_type": weight_type,
                "signature": signature,
                "test_file": test_file,
                "target_tensor": target_tensor,
                "digest_tensor": digest_tensor,
            }
        )

    runner = _runner or subprocess.run

    def _correctness_runner(argv, **kwargs):
        # correctness_evidence supplies a minimal env dict. Preserve
        # ambient HIP_VISIBLE_DEVICES/ROCR_VISIBLE_DEVICES underneath its
        # explicit deterministic/dispatch variables.
        env = {**os.environ, **(kwargs.pop("env", None) or {})}
        return runner(argv, env=env, **kwargs)

    def _finite_or_none(value: float) -> float | None:
        return float(value) if _math.isfinite(float(value)) else {}

    rows: list[dict[str, object]] = []
    for shape in mapped_shapes:
        for seed in seeds:
            control = correctness_evidence.collect_native_seed_evidence(
                control_binary,
                test_file=shape["test_file"],
                target_tensor=shape["target_tensor"],
                digest_tensor=shape["digest_tensor"],
                seed=seed,
                runner=_correctness_runner,
            )
            subject = correctness_evidence.collect_native_seed_evidence(
                subject_binary,
                test_file=shape["test_file"],
                target_tensor=shape["target_tensor"],
                digest_tensor=shape["digest_tensor"],
                seed=seed,
                runner=_correctness_runner,
            )

            control_backend_ok = (
                control.native_execution_status == "ok"
                and _math.isfinite(control.e_n_nmse)
                and _math.isfinite(control.threshold_t)
                and control.e_n_nmse <= control.threshold_t
            )
            subject_backend_ok = (
                subject.native_execution_status == "ok"
                and _math.isfinite(subject.e_n_nmse)
                and _math.isfinite(subject.threshold_t)
                and subject.e_n_nmse <= subject.threshold_t
            )

            bit_identical = (
                control.native_execution_status == "ok"
                and subject.native_execution_status == "ok"
                and control.reference_digest == subject.reference_digest
                and control.reference_output_digest is not None
                and control.reference_output_digest
                == subject.reference_output_digest
                and control.native_output_digest is not None
                and control.native_output_digest == subject.native_output_digest
                and control.output_nels is not None
                and control.output_nels == subject.output_nels
            )

            rows.append(
                {
                    "shape": shape["name"],
                    "weight_type": shape["weight_type"],
                    "seed": seed,
                    "control_status": control.native_execution_status,
                    "subject_status": subject.native_execution_status,
                    "control_ids_digest": control.reference_digest,
                    "subject_ids_digest": subject.reference_digest,
                    "control_output_digest": control.native_output_digest,
                    "subject_output_digest": subject.native_output_digest,
                    "control_reference_output_digest":
                        control.reference_output_digest,
                    "subject_reference_output_digest":
                        subject.reference_output_digest,
                    "control_output_nels": control.output_nels,
                    "subject_output_nels": subject.output_nels,
                    "control_nmse": _finite_or_none(control.e_n_nmse),
                    "subject_nmse": _finite_or_none(subject.e_n_nmse),
                    "control_threshold":
                        _finite_or_none(control.threshold_t),
                    "subject_threshold":
                        _finite_or_none(subject.threshold_t),
                    "control_max_abs":
                        _finite_or_none(control.max_abs_native),
                    "subject_max_abs":
                        _finite_or_none(subject.max_abs_native),
                    "backend_reference_ok":
                        control_backend_ok and subject_backend_ok,
                    "bit_identical": bit_identical,
                }
            )

    first_exact_failure = next(
        (row for row in rows if not row["bit_identical"]),
        None,
    )
    if first_exact_failure is None:
        bit_identical_result = experiment_contract.CorrectnessResult(
            check="bit_identical",
            passed=True,
            detail=(
                f"{len(rows)} RD30 256-expert MUL_MAT_ID "
                "(shape,seed) pairs produced byte-identical HIP outputs"
            ),
        )
    else:
        bit_identical_result = experiment_contract.CorrectnessResult(
            check="bit_identical",
            passed=False,
            detail=(
                "RD30 exact-output mismatch for "
                f"shape={first_exact_failure['shape']!r} "
                f"seed={first_exact_failure['seed']}: "
                f"control_status={first_exact_failure['control_status']} "
                f"subject_status={first_exact_failure['subject_status']} "
                "ids_equal="
                f"{first_exact_failure['control_ids_digest'] == first_exact_failure['subject_ids_digest']} "
                "reference_equal="
                f"{first_exact_failure['control_reference_output_digest'] == first_exact_failure['subject_reference_output_digest']} "
                "output_equal="
                f"{first_exact_failure['control_output_digest'] == first_exact_failure['subject_output_digest']} "
                "nels_equal="
                f"{first_exact_failure['control_output_nels'] == first_exact_failure['subject_output_nels']}"
            ),
        )

    first_backend_failure = next(
        (row for row in rows if not row["backend_reference_ok"]),
        None,
    )
    backend_reference_result = experiment_contract.CorrectnessResult(
        check="backend_reference",
        passed=first_backend_failure is None,
        detail=(
            f"{len(rows)} subject/control rows stayed within each emitted "
            "backend-reference threshold"
            if first_backend_failure is None
            else (
                "RD30 backend-reference failure for "
                f"shape={first_backend_failure['shape']!r} "
                f"seed={first_backend_failure['seed']}: "
                f"control_nmse={first_backend_failure['control_nmse']} "
                f"control_threshold={first_backend_failure['control_threshold']} "
                f"subject_nmse={first_backend_failure['subject_nmse']} "
                f"subject_threshold={first_backend_failure['subject_threshold']}"
            )
        ),
    )

    artifact_doc = {
        "schema_version": 1,
        "contract_id": "RD30-MOE-MMQ-COMPACT-GRID",
        "check": "bit_identical",
        "passed": bit_identical_result.passed,
        "base_revision": base_revision,
        "architecture": "gfx1100",
        "mechanism": (
            "test-backend-ops MUL_MAT_ID native subject/control "
            "backend1_digest equality"
        ),
        "evidence_patches": list(evidence_patches),
        "subject_patch": subject_patch,
        "seeds": list(seeds),
        "shapes": [
            {
                "name": shape["name"],
                "weight_type": shape["weight_type"],
                "signature": shape["signature"],
                "target_tensor": shape["target_tensor"],
                "digest_tensor": shape["digest_tensor"],
            }
            for shape in mapped_shapes
        ],
        "control_source_tree": psi.git_worktree_tree(control_src),
        "subject_source_tree": psi.git_worktree_tree(subject_src),
        "control_build_identity":
            control_build_evidence.campaign_identity(),
        "subject_build_identity":
            subject_build_evidence.campaign_identity(),
        "rows": rows,
    }
    artifact_ref = _write_bound_artifact(
        run_dir,
        "rd30-correctness.json",
        artifact_doc,
    )
    return {
        "results": {
            "bit_identical": bit_identical_result,
            "backend_reference": backend_reference_result,
        },
        "artifact": artifact_ref,
        "rows": rows,
    }








def run_rd08_contract_trigger(
    *, marker_regex: str, control_binary: Path, subject_binary: Path, model: Path,
    hip_path: Path, workdir: Path, run_dir: Path, bench_prompt: int = 0, bench_gen: int = 128,
) -> dict[str, object]:
    """VA14 final slice: RD08's real trigger proof. Unlike the generic
    activation probe (tune binary + GGML_CUDA_DISABLE_FUSION=1 as its
    negative control, which proves nothing about RD08's specific MMVQ
    marker), this runs the SAME decode command against the validation
    control binary (which never has the RD08 patch applied at all -- a
    genuine negative) and the validation-subject binary (the parity-built
    patched binary), both with BIGCHERRY_PATCH_TRACE=1 via the existing
    trace-probe machinery (_run_one_trace_probe)."""
    from bigcherry.experiment import execution as experiment_execution

    pattern = re.compile(marker_regex)
    subject_log = _run_one_trace_probe(
        name="rd08-trigger-subject", binary=subject_binary, model=model, hip_path=hip_path,
        workdir=workdir, bench_prompt=bench_prompt, bench_gen=bench_gen, disable_fusion=False,
    )
    control_log = _run_one_trace_probe(
        name="rd08-trigger-control", binary=control_binary, model=model, hip_path=hip_path,
        workdir=workdir, bench_prompt=bench_prompt, bench_gen=bench_gen, disable_fusion=False,
    )
    subject_hit = pattern.search(subject_log) is not None
    control_hit = pattern.search(control_log) is not None
    subject_te = experiment_execution.trigger_evidence_from_marker_probe(
        lane_id="rd08-decode-subject", role="positive", positive_hit=subject_hit,
    )
    control_te = experiment_execution.trigger_evidence_from_marker_probe(
        lane_id="rd08-decode-control", role="control", positive_hit=control_hit,
    )
    subject_log_path = "logs/activation-rd08-trigger-subject.log"
    control_log_path = "logs/activation-rd08-trigger-control.log"
    subject_log_ref = {
        "path": subject_log_path,
        "sha256": hashlib.sha256((run_dir / subject_log_path).read_bytes()).hexdigest(),
    }
    control_log_ref = {
        "path": control_log_path,
        "sha256": hashlib.sha256((run_dir / control_log_path).read_bytes()).hexdigest(),
    }
    trigger_doc = {
        "marker_regex": marker_regex, "subject_hit": subject_hit, "control_hit": control_hit,
        "positive": {
            "lane_id": subject_te.lane_id, "candidate_launches": subject_te.candidate_launches,
            "artifact": subject_log_ref,
        },
        "control": {
            "lane_id": control_te.lane_id, "candidate_launches": control_te.candidate_launches,
            "artifact": control_log_ref,
        },
    }
    artifact_ref = _write_bound_artifact(run_dir, "rd08-trigger.json", trigger_doc)
    return {
        "evidence": [subject_te, control_te], "artifact": artifact_ref,
        "subject_hit": subject_hit, "control_hit": control_hit,
        # GPT round 4 (req_4544a9240b6d45df): the real subject/control probe
        # logs _run_one_trace_probe() already writes to run_dir/"logs"/... --
        # exposed here (paths relative to run_dir) so the caller can bind
        # them as the RD08-authoritative activation evidence (adapter
        # trace-marker check AND the record's top-level activation field),
        # instead of the generic tune-binary/fusion-disabled probe, which is
        # not a valid negative control for RD08's MMVQ marker.
        "subject_log_path": subject_log_path, "control_log_path": control_log_path,
        "subject_log_artifact": subject_log_ref, "control_log_artifact": control_log_ref,
    }


def run_rd08_contract_qualification(
    *, contract: object, descriptor: object, base_revision: str,
    control_binary: Path, subject_binary: Path, model: Path, model_ref: str,
    marker_regex: str, hip_path: Path, amdgpu_targets: str, worktree_root: Path,
    build_root: Path, build_env: dict[str, str], run_dir: Path,
    control_build_identity: dict[str, object], subject_build_identity: dict[str, object],
    pairs: int = 3,
) -> dict[str, object]:
    """VA14 final slice: the authoritative RD08 full-qualification path
    (``--run-rd08-contract``). Composes real lane execution + real
    per-named correctness + real trigger proof into
    evaluate_promotion_gate()'s verdict -- this is the ONLY path allowed to
    produce contract promotion/eligibility; ``run_rd08_validation_lanes()``
    alone (``--run-rd08-lanes``) stays diagnostic-only."""
    from bigcherry.experiment import contract as experiment_contract

    lanes = run_rd08_validation_lanes(
        contract=contract, control_binary=control_binary, subject_binary=subject_binary,
        model=model, model_ref=model_ref, hip_path=hip_path, run_dir=run_dir,
        control_build_identity=control_build_identity, subject_build_identity=subject_build_identity,
        pairs=pairs,
    )
    correctness = run_rd08_contract_correctness(
        base_revision=base_revision, hip_path=hip_path, amdgpu_targets=amdgpu_targets,
        worktree_root=worktree_root, build_root=build_root, build_env=build_env, run_dir=run_dir,
    )
    trigger = run_rd08_contract_trigger(
        marker_regex=marker_regex, control_binary=control_binary, subject_binary=subject_binary,
        model=model, hip_path=hip_path, workdir=run_dir, run_dir=run_dir,
    )
    correctness_gate = compute_contract_correctness_gate(contract, correctness["results"])
    aggregated_effects = experiment_contract.aggregate_contract_effects(
        contract, lanes["effects"], target_metric="tg128",
    )
    trigger_proof = experiment_contract.evaluate_trigger_proof(trigger["evidence"])
    # GPT round 4 (req_4544a9240b6d45df): evaluate_trigger_proof() only
    # checks positive-role lanes (by design -- EC18 scopes control lanes
    # out, since most contracts' control lanes should NOT trigger). RD08's
    # negative control is a real, separate claim this gate must still
    # enforce here: if the control (unpatched) binary ALSO shows the
    # marker, the negative control itself is invalid, and no promotion
    # verdict can be trusted regardless of what the positive lane showed.
    if trigger["control_hit"]:
        trigger_proof = {
            "passed": False,
            "reasons": list(trigger_proof.get("reasons") or []) + [
                "control-role lane observed the target marker -- the negative "
                "control is invalid, so trigger proof cannot be trusted"
            ],
            "checked_lanes": trigger_proof.get("checked_lanes", 0),
            "untriggered_lanes": list(trigger_proof.get("untriggered_lanes") or []),
        }
    promotion = experiment_contract.evaluate_promotion_gate(
        contract, correctness_gate=correctness_gate, aggregated_effects=aggregated_effects,
        trigger_proof=trigger_proof,
    )
    qualification_doc = {
        "contract_id": contract.id, "contract_hash": contract.contract_hash,
        "lanes_artifact": lanes["artifact"], "correctness_artifact": correctness["artifact"],
        "trigger_artifact": trigger["artifact"],
        "correctness_gate": correctness_gate, "aggregated_effects": aggregated_effects,
        "trigger_proof": trigger_proof, "promotion": promotion,
    }
    artifact_ref = _write_bound_artifact(run_dir, "contract-qualification.json", qualification_doc)
    return {
        "lanes": lanes, "correctness": correctness, "trigger": trigger,
        "correctness_gate": correctness_gate, "aggregated_effects": aggregated_effects,
        "trigger_proof": trigger_proof, "promotion": promotion, "artifact": artifact_ref,
    }


_PAIRED_BENCH_WORKLOAD_FLAGS: dict[str, tuple[str, ...]] = {
    "decode": ("-p", "0", "-n", "128"),
    "prefill": ("-p", "512", "-n", "0"),
}
_PAIRED_BENCH_METRIC_NAME: dict[str, str] = {"decode": "tg128", "prefill": "pp512"}
_PAIRED_BENCH_METRIC_PATTERN: dict[str, "re.Pattern[str]"] = {
    "decode": re.compile(r"tg128\s*\|\s*([0-9.]+)"),
    "prefill": re.compile(r"pp512\s*\|\s*([0-9.]+)"),
}


def _paired_llama_bench_command(
    binary: Path, model: Path, workload: str, *,
    patch_args: tuple[str, ...] = (), runtime_args: tuple[str, ...] = (),
) -> list[str]:
    """PVPS02 step 2: the one llama-bench command shape both RD04 and
    RD08's producers build. ``patch_args`` land BEFORE -ngl (RD04's
    historical position for its -fa/-ctk/-ctv flags); ``runtime_args``
    land AFTER -ngl (RD08's historical position for e.g. -sm tensor
    topology flags via rd08_validation_lane_commands' extra_flags) --
    kept as two distinct insertion points, not one undifferentiated list,
    specifically so each caller's exact historical argv order is
    reproducible byte-for-byte."""
    if workload not in _PAIRED_BENCH_WORKLOAD_FLAGS:
        raise PatchCampaignError(
            f"paired llama-bench: no flag mapping for workload {workload!r}"
        )
    return [
        str(binary), "-m", str(model), *_PAIRED_BENCH_WORKLOAD_FLAGS[workload],
        *patch_args, "-ngl", "99", *runtime_args,
    ]


@dataclass(frozen=True)
class PairedBenchmarkOutcome:
    """PVPS02 step 2: the result of running one or more paired llama-bench
    workloads (decode/prefill) for one control/subject binary pair."""

    runs: dict[str, "experiment_execution.PairedLaneRun"]
    commands: dict[str, dict[str, list[str]]]
    raw_logs: list[dict[str, object]]


# PVPS02 step 3: the closed vocabulary of recognized benchmark executors.
# One entry today (the shared paired-llama-bench primitive); a genuinely
# new execution shape (e.g. a server-based benchmark) gets a new name
# here, never a silent branch on patch id.
BENCHMARK_EXECUTORS: frozenset[str] = frozenset({"paired-llama-bench-v1"})

# GPT review (req_e608313764834497, 2026-09-11): resolve_benchmark_wiring()
# validated wiring.executor against BENCHMARK_EXECUTORS but nothing actually
# dispatched on it -- _run_performance_benchmark() always called
# run_paired_llama_benchmark() directly, harmless with one executor but a
# silent-wrong-implementation trap the moment a second is added. Populated
# after run_paired_llama_benchmark() is defined, below.
BENCHMARK_EXECUTOR_FUNCS: dict[str, object] = {}


@dataclass(frozen=True)
class BenchmarkWiring:
    """PVPS02 step 3: one patch's resolved performance-benchmark wiring,
    read from its validation.toml -- never a hardcoded per-patch branch."""

    executor: str
    patch_args: tuple[str, ...]


def resolve_benchmark_wiring(
    descriptor: "patch_registry.PatchDescriptor", *, root: "str | Path | None" = None,
) -> BenchmarkWiring:
    """A patch is generically benchmarkable iff exactly one of its
    REQUIRED capability="performance" checks declares a recognized
    benchmark-executor in its validation.toml config. Zero, more than
    one, or an unrecognized executor is a configuration error -- fail
    closed, never guess which check/executor was meant.

    ``root`` matches build_plan_for_patch()'s own parameter (default:
    the real patches/ tree) -- exposed here purely so this can be
    unit-tested against isolated fixtures without touching real patches."""
    from bigcherry.patch import validation as patch_validation

    plan = patch_validation.build_plan_for_patch(descriptor, root=root)
    if plan is None:
        raise PatchCampaignError(
            f"{descriptor.patch_id}: no validation plan (no bound contract and no "
            "validation.toml adapter) -- cannot resolve benchmark wiring"
        )
    wired = [
        check for check in plan.checks_for("performance")
        if check.required and check.config.get("benchmark-executor") is not None
    ]
    if not wired:
        raise PatchCampaignError(
            f"{descriptor.patch_id}: no required performance check declares a "
            "benchmark-executor in validation.toml"
        )
    if len(wired) > 1:
        raise PatchCampaignError(
            f"{descriptor.patch_id}: {len(wired)} required performance checks declare a "
            "benchmark-executor -- ambiguous wiring, exactly one is required"
        )
    check = wired[0]
    executor = check.config["benchmark-executor"]
    if executor not in BENCHMARK_EXECUTORS:
        raise PatchCampaignError(
            f"{descriptor.patch_id}: unknown benchmark-executor {executor!r} "
            f"(known: {sorted(BENCHMARK_EXECUTORS)})"
        )
    extra_args = check.config.get("benchmark-extra-args", [])
    if not isinstance(extra_args, list) or not all(isinstance(a, str) for a in extra_args):
        raise PatchCampaignError(
            f"{descriptor.patch_id}: benchmark-extra-args must be a list of strings"
        )
    return BenchmarkWiring(executor=executor, patch_args=tuple(extra_args))


# PVPS02 step 5 (hardened after GPT review req_e608313764834497, 2026-09-11:
# tensor-2 selected 2 devices but never passed llama-bench's -sm tensor flag,
# so the dual-GPU cell silently ran with llama-bench's default split mode
# instead of the declared topology): the closed vocabulary of benchmark
# topologies a models.toml entry can declare, owning BOTH how many real
# devices it needs AND which runtime args select that split mode -- so a
# topology can never again own a device count without the args that make
# that count meaningful.
BENCHMARK_TOPOLOGIES: dict[str, "BenchmarkTopology"] = {}


@dataclass(frozen=True)
class BenchmarkTopology:
    device_count: int
    runtime_args: tuple[str, ...]


BENCHMARK_TOPOLOGIES["single"] = BenchmarkTopology(device_count=1, runtime_args=())
BENCHMARK_TOPOLOGIES["tensor-2"] = BenchmarkTopology(device_count=2, runtime_args=("-sm", "tensor"))

# Preserved for any external/test code still keying off device count alone.
BENCHMARK_TOPOLOGY_DEVICE_COUNT: dict[str, int] = {
    name: topo.device_count for name, topo in BENCHMARK_TOPOLOGIES.items()
}


@dataclass(frozen=True)
class ResolvedBenchmarkModel:
    """PVPS02 step 5: one config/models.toml entry resolved against a real
    host's model root -- real file path, real verified size, real
    topology-derived device count and runtime args."""

    id: str
    path: Path
    size_bytes: int
    topology: str

    @property
    def device_count(self) -> int:
        return BENCHMARK_TOPOLOGIES[self.topology].device_count

    @property
    def runtime_args(self) -> tuple[str, ...]:
        return BENCHMARK_TOPOLOGIES[self.topology].runtime_args


def resolve_benchmark_model(
    model_id: str, *, model_root: Path, registry_path: "Path | None" = None,
) -> ResolvedBenchmarkModel:
    """Resolve a model id from config/models.toml into a real, verified
    file path + benchmark topology (docs/planning/active/
    patching-validation-package-standard/PVPS02.md). Fails closed on: an
    unknown id, a missing file, a real-vs-declared size mismatch (wrong
    file/quantisation), or a model with no declared benchmark-topology --
    deliberately never defaults an undeclared topology to "single", since
    that would silently under-provision a model that actually needs 2+
    devices (e.g. tierL-qwen27b-q8)."""
    import tomllib

    from bigcherry.core import paths as bc_paths

    resolved_registry = registry_path if registry_path is not None else bc_paths.MODELS
    raw = tomllib.loads(resolved_registry.read_text(encoding="utf-8"))
    entries = {
        entry["id"]: entry for entry in raw.get("models", [])
        if isinstance(entry, dict) and entry.get("id")
    }
    entry = entries.get(model_id)
    if entry is None:
        raise PatchCampaignError(
            f"unknown benchmark model id {model_id!r} (not in {resolved_registry})"
        )
    topology = entry.get("benchmark-topology")
    if topology not in BENCHMARK_TOPOLOGIES:
        raise PatchCampaignError(
            f"{model_id}: benchmark-topology must be one of "
            f"{sorted(BENCHMARK_TOPOLOGIES)}, got {topology!r} -- a models.toml "
            "entry with no (or an unrecognized) benchmark-topology is not eligible for the "
            "generic performance-benchmark matrix"
        )
    path_value = entry.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise PatchCampaignError(f"{model_id}: models.toml entry has no valid 'path'")
    model_path = model_root / path_value
    if not model_path.is_file():
        raise PatchCampaignError(f"{model_id}: model file not found at {model_path}")
    # GPT review (req_e608313764834497, 2026-09-11): size-bytes verification
    # must be fail-closed, not skip-if-absent-or-malformed -- a missing or
    # non-integer size-bytes silently bypassed the "wrong file/quantisation"
    # check this exists for, contradicting PVPS02's explicit requirement of
    # real file-size verification.
    declared_size = entry.get("size-bytes")
    if not isinstance(declared_size, int) or isinstance(declared_size, bool) or declared_size <= 0:
        raise PatchCampaignError(
            f"{model_id}: models.toml entry must declare a positive integer size-bytes "
            f"(got {declared_size!r})"
        )
    real_size = model_path.stat().st_size
    if real_size != declared_size:
        raise PatchCampaignError(
            f"{model_id}: real file size {real_size} does not match models.toml's declared "
            f"size-bytes={declared_size} at {model_path} -- wrong file or quantisation?"
        )
    return ResolvedBenchmarkModel(id=model_id, path=model_path, size_bytes=real_size, topology=topology)


def resolve_device_pool(
    device_map: dict[str, tuple[str, ...]], architecture: str, device_count: int,
) -> tuple[str, ...]:
    """PVPS02 step 5: --device-map defines an ORDERED device pool per
    architecture; an N-device cell consumes the first N ids from that
    architecture's pool. Order is load-bearing (which physical card ends
    up in which tensor-split slot), never sorted/reordered. Fails closed
    if the architecture has no mapping at all, or its pool has fewer
    devices than the cell needs -- never silently infers a device index
    from the architecture name."""
    pool = device_map.get(architecture)
    if pool is None:
        raise PatchCampaignError(
            f"--device-map has no entry for architecture {architecture!r}"
        )
    if len(pool) < device_count:
        raise PatchCampaignError(
            f"--device-map {architecture}={','.join(pool)} exposes only {len(pool)} "
            f"device(s), this cell needs {device_count}"
        )
    return pool[:device_count]


def parse_device_map(entries: list[str]) -> dict[str, tuple[str, ...]]:
    """Parse repeated --device-map ARCH=ID[,ID...] CLI values into an
    ordered-pool-per-architecture dict. Fails closed on a malformed entry
    or the same architecture given twice (ambiguous which pool wins)."""
    device_map: dict[str, tuple[str, ...]] = {}
    for entry in entries:
        if "=" not in entry:
            raise PatchCampaignError(
                f"--device-map {entry!r} is malformed -- expected ARCH=ID[,ID...]"
            )
        arch, _, ids_raw = entry.partition("=")
        arch = arch.strip()
        if not arch:
            raise PatchCampaignError(f"--device-map {entry!r}: architecture must not be blank")
        if arch in device_map:
            raise PatchCampaignError(f"--device-map: architecture {arch!r} given more than once")
        ids = tuple(i.strip() for i in ids_raw.split(","))
        if not ids_raw or any(not i for i in ids):
            raise PatchCampaignError(
                f"--device-map {entry!r}: device id list must not be empty or contain a "
                "blank entry"
            )
        device_map[arch] = ids
    return device_map


def run_paired_llama_benchmark(
    *, control_binary: Path, subject_binary: Path, model: Path, hip_path: Path,
    workloads: tuple[str, ...] = ("decode", "prefill"),
    patch_args: tuple[str, ...] = (), runtime_args: tuple[str, ...] = (),
    pairs: int = 3, log_context: str,
    env_overrides: dict[str, str] | None = None,
    env_unset: tuple[str, ...] = (),
    execution_identity: "object | None" = None,
) -> PairedBenchmarkOutcome:
    """PVPS02 step 2/4: the shared execution shape behind
    run_rd08_validation_lanes() (and the 1202/RD04 patch-local producer's
    paired benchmark, via ProducerRuntime.run_paired_llama_benchmark()) --
    a pure, semantics-preserving extraction of the duplicated
    clean-env/runner/command/raw-log/paired-run logic
    (docs/planning/active/patching-validation-package-standard/PVPS02.md).
    The remaining public caller (run_rd08_validation_lanes) is kept as a
    compatibility wrapper that does its own result-shaping
    (validation-lanes.json + LaneEffects) -- its callers/tests see no
    behavior change.

    ``env_overrides`` (step 4): applied on top of the sanitized/stripped
    environment for THIS call's own subprocesses only -- never mutates
    the real process-global os.environ, so a multi-cell caller (the
    generic matrix) can give each cell its own HIP_VISIBLE_DEVICES
    without any risk of one cell's selector leaking into the next.
    ROCR_VISIBLE_DEVICES is never a real selector here (see below) --
    a caller must not pass it in env_overrides either; if one does, it
    is stripped again, not honored. ``execution_identity`` (step 4):
    passed straight through to run_paired_lane(), which attests every
    measured process before accepting its metric -- proves architecture
    + device COUNT, not physical card identity (llama-bench's own ROCm
    attestation never carries a device locator; see DeviceVisibility's
    docstring)."""
    from bigcherry.experiment import execution as experiment_execution
    from bigcherry.campaign.benchmark import sanitize_environment

    clean_env = sanitize_environment(_hip_env(hip_path), mode="stock")
    for key in list(clean_env):
        if key.startswith("BIGCHERRY_") or key == "GGML_CUDA_DISABLE_FUSION":
            clean_env.pop(key, None)
    if env_overrides:
        clean_env.update(env_overrides)
    # GPT review (req_8429aa8e0d35496e / follow-up on req_d1ef22d846854960,
    # 2026-09-11): sanitize_environment() does not strip ROCR_VISIBLE_DEVICES,
    # so an ambient value inherited from the invoking shell would otherwise
    # reach this measured process even though this module's whole selector
    # contract is HIP-only now (require_device_visibility()/DeviceVisibility,
    # see PNRO17) -- reproducing the exact double-filtering bug that
    # contract exists to prevent. Stripped AFTER env_overrides is applied
    # (not just before) so a caller cannot reintroduce the hazard by
    # explicitly passing ROCR_VISIBLE_DEVICES in env_overrides either --
    # this key has no purpose in this harness any more, full stop.
    clean_env.pop("ROCR_VISIBLE_DEVICES", None)
    # RD58 (PA36 migration #4, dev-gpt-agent req_82fbbafe52c0472d
    # Q3/Q6): a multi-GPU producer (device=None) can name additional
    # env vars to unset beyond the always-stripped RROC/BIGCHERRY/
    # GGML_CUDA_DISABLE_FUSION set. Applied AFTER env_overrides so a
    # caller cannot reintroduce a stripped key by passing it in
    # env_overrides.
    for key in env_unset:
        clean_env.pop(key, None)

    raw_logs: list[dict[str, object]] = []

    def _make_runner(workload: str):
        def _runner(command: list[str]) -> "experiment_execution.RunnerOutput":
            completed = subprocess.run(
                command, capture_output=True, text=True, check=False, env=clean_env,
            )
            raw_logs.append({
                "workload": workload, "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout, "stderr": completed.stderr,
            })
            if completed.returncode == 0:
                _require_real_gpu_execution(
                    completed.stdout, completed.stderr,
                    context=f"{log_context} {workload} lane ({Path(command[0]).name})",
                )
            return experiment_execution.RunnerOutput(
                returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr,
            )
        return _runner

    runs: dict[str, "experiment_execution.PairedLaneRun"] = {}
    commands: dict[str, dict[str, list[str]]] = {}
    for workload in workloads:
        control_cmd = _paired_llama_bench_command(
            control_binary, model, workload, patch_args=patch_args, runtime_args=runtime_args,
        )
        subject_cmd = _paired_llama_bench_command(
            subject_binary, model, workload, patch_args=patch_args, runtime_args=runtime_args,
        )
        runs[workload] = experiment_execution.run_paired_lane(
            metric=_PAIRED_BENCH_METRIC_NAME[workload],
            control_command=control_cmd, subject_command=subject_cmd,
            pattern=_PAIRED_BENCH_METRIC_PATTERN[workload], pairs=pairs,
            runner=_make_runner(workload), execution_identity=execution_identity,
        )
        commands[workload] = {"control": control_cmd, "subject": subject_cmd}

    return PairedBenchmarkOutcome(runs=runs, commands=commands, raw_logs=raw_logs)


BENCHMARK_EXECUTOR_FUNCS["paired-llama-bench-v1"] = run_paired_llama_benchmark


# --------------------------------------------------------- PA36-F producer runtime


@dataclass(frozen=True)
class CampaignProducerRuntime:
    """The one concrete ``ProducerRuntime`` implementation (PA36-F step 1,
    GPT design req_8ec9b90c05f84a30). Patch-local producer modules never
    import this module directly -- they receive an instance of this class
    through ``ProducerContext.runtime`` instead, which is the sanctioned
    dependency-direction seam validation_producer.py's module docstring
    establishes.

    Every method here is a thin, faithful wrapper around the existing
    generic primitives already living in this module
    (``build_tree()``/``capture_completed_build_evidence()``/
    ``resolve_selected_device_execution_identity()``/
    ``run_paired_llama_benchmark()``/``source.resolve_source_composition()``
    /``source.materialize_composition()``) -- nothing here reimplements
    them."""

    repo_root: Path
    patch_id: str
    base_revision: str
    workdir: Path
    hip_path: Path
    fat_targets: FatTargetPlan
    run_dir: Path

    def build_pair(
        self,
        *,
        targets: tuple[str, ...],
        primary_target: str,
        common_extra_patches: tuple[str, ...] = (),
        baseline_source: str = "bigcherry",
        control_extra_cmake_args: tuple[str, ...] = (),
        subject_extra_cmake_args: tuple[str, ...] = (),
        require_parity: bool = False,
    ) -> ProducerBuildPair:
        """The one authority for the PA36 build-once-fat-multiarch rule:
        exactly one control build and one subject build, both at the same
        target set, regardless of how many devices or architectures the
        producer will later run against.

        ``targets`` is the authoritative, REQUIRED target set (validated
        through FatTargetPlan, so its non-empty/no-duplicate invariants hold;
        a producer may request a fat gfx1100;gfx1201;gfx1030 set while the
        outer run names one execution arch). ``common_extra_patches``
        resolves into BOTH arms (control = baseline + common; subject =
        baseline + common + focal) so a producer whose correctness pair
        needs supplementary evidence patches in both arms passes them here
        instead of materializing its own pair below the build authority
        (dev-gpt-agent req_98777b7a51f84820 + review req_052817cb66d14bc1)."""
        # targets and primary_target are REQUIRED (GPT review
        # req_052817cb66d14bc1): the pre-existing API already required both,
        # so silent defaults would only add a path where a future producer
        # accidentally builds the wrong fat plan or the wrong binary. The
        # producer names the explicit target set (typically
        # ctx.fat_targets.targets); FatTargetPlan validates it, and its
        # cmake_value is the exact AMDGPU_TARGETS string. The binary build
        # list still comes from primary_target -- build_tree() itself only
        # ever builds the single requested target for a producer build pair
        # (a producer wanting multiple binaries calls build_pair() once per
        # binary set, never widens this one call).
        target_plan = FatTargetPlan(targets=targets)
        from bigcherry.patch import source as psi

        control_revision, control_composition = psi.resolve_source_composition(
            baseline_source, focal=None, extra_patches=common_extra_patches,
            base_ref=self.base_revision, base_repo=LLAMA_CPP_SRC,
        )
        subject_revision, subject_composition = psi.resolve_source_composition(
            baseline_source, focal=self.patch_id, extra_patches=common_extra_patches,
            base_ref=self.base_revision, base_repo=LLAMA_CPP_SRC,
        )
        if control_revision != subject_revision:
            raise PatchCampaignError(
                f"{self.patch_id}: build_pair() control/subject resolved different "
                f"base revisions ({control_revision!r} vs {subject_revision!r})"
            )

        control_src = psi.materialize_composition(
            base_repo=LLAMA_CPP_SRC, worktree_root=self.workdir / "control",
            resolved_revision=control_revision, composition=control_composition,
            overlay_root=psi.REPO_ROOT / "src", requested_revision=self.base_revision,
        )
        subject_src = psi.materialize_composition(
            base_repo=LLAMA_CPP_SRC, worktree_root=self.workdir / "subject",
            resolved_revision=subject_revision, composition=subject_composition,
            overlay_root=psi.REPO_ROOT / "src", requested_revision=self.base_revision,
        )

        exe = ".exe" if sys.platform == "win32" else ""
        build_root = self.workdir / "builds"
        # Directory NAME must not contain ';' -- target_plan.cmake_value is the
        # correct CMake AMDGPU_TARGETS *value* (semicolon-joined, as CMake
        # list syntax requires), but a build directory literally named with
        # embedded semicolons breaks CMake's own internal argument handling
        # (real failure found on real hardware, PA39 real-hardware acceptance
        # attempt #3b: "execute_process given unknown argument 'gfx1201'"
        # during compiler-id detection, because CMake treats ';' in certain
        # internal strings as its own list separator). Use a '+'-joined slug
        # for the directory name only; the actual cmake invocation still
        # receives the real semicolon-joined value.
        target_slug = "+".join(target_plan.targets)
        control_name = f"{self.patch_id}-control-{target_slug}"
        subject_name = f"{self.patch_id}-subject-{target_slug}"

        control_bin = build_tree(
            name=control_name, hip_path=self.hip_path,
            amdgpu_targets=target_plan.cmake_value, workdir=build_root,
            targets=[primary_target], source=control_src,
            extra_cmake_args=list(control_extra_cmake_args),
        )
        subject_bin = build_tree(
            name=subject_name, hip_path=self.hip_path,
            amdgpu_targets=target_plan.cmake_value, workdir=build_root,
            targets=[primary_target], source=subject_src,
            extra_cmake_args=list(subject_extra_cmake_args),
        )

        build_env = _hip_env(self.hip_path)
        control_binary = control_bin / f"{primary_target}{exe}"
        subject_binary = subject_bin / f"{primary_target}{exe}"
        control_cmake_args = _full_requested_cmake_args(
            hip_path=self.hip_path, amdgpu_targets=target_plan.cmake_value,
            extra_cmake_args=list(control_extra_cmake_args),
        )
        subject_cmake_args = _full_requested_cmake_args(
            hip_path=self.hip_path, amdgpu_targets=target_plan.cmake_value,
            extra_cmake_args=list(subject_extra_cmake_args),
        )
        control_build_evidence = capture_completed_build_evidence(
            build_root / control_name, source_root=control_src,
            architecture=target_plan.targets, binary=control_binary,
            requested_cmake_args=control_cmake_args, build_env=build_env,
        )
        subject_build_evidence = capture_completed_build_evidence(
            build_root / subject_name, source_root=subject_src,
            architecture=target_plan.targets, binary=subject_binary,
            requested_cmake_args=subject_cmake_args, build_env=build_env,
        )

        if require_parity:
            assert_validation_subject_parity(
                control_build_evidence,
                subject_build_evidence,
                patch_id=self.patch_id,
            )
        return ProducerBuildPair(
            base_revision=control_revision,
            control_source=control_src, subject_source=subject_src,
            control_composition=tuple(control_composition),
            subject_composition=tuple(subject_composition),
            control_bin=control_binary, subject_bin=subject_binary,
            validation_build_identities={
                "control": control_build_evidence.campaign_identity(),
                "subject": subject_build_evidence.campaign_identity(),
            },
        )

    def device_contexts(
        self, *, device_map: Mapping[str, tuple[int, ...]],
    ) -> tuple[ProducerDeviceContext, ...]:
        """The only producer-facing device selector (PA36-F step 1):
        reuses ``resolve_selected_device_execution_identity()``'s real
        device-inventory verification per explicit index rather than
        mutating ambient HIP_VISIBLE_DEVICES, and always returns the
        HIP-only env shape (``env_overrides={"HIP_VISIBLE_DEVICES":
        str(index)}``, ``env_unset=("ROCR_VISIBLE_DEVICES",)``) -- no
        ambient-only selector and no ROCR/HIP double-filtering (PNRO17)."""
        contexts: list[ProducerDeviceContext] = []
        for architecture, indices in device_map.items():
            for index in indices:
                identity, _selector_env, locator = self._resolve_one_device(
                    architecture, index,
                )
                contexts.append(
                    ProducerDeviceContext(
                        architecture=architecture, device_index=index,
                        execution_identity=identity,
                        env_overrides={"HIP_VISIBLE_DEVICES": str(index)},
                        env_unset=_ROCR_VISIBLE_DEVICES_UNSET,
                        locator=locator,
                    )
                )
        return tuple(contexts)

    def _resolve_one_device(
        self, architecture: str, index: int,
    ) -> tuple[ExecutionIdentity, dict[str, str], str]:
        """Resolve exactly ONE device index against the real host
        inventory -- the same fail-closed checks
        ``resolve_selected_device_execution_identity()`` performs from
        ``HIP_VISIBLE_DEVICES``, applied here to an explicit index
        instead of reading ambient environment, so a multi-device
        producer never has to mutate process-global env to select each
        device in turn."""
        from bigcherry.core import environment as bc_environment

        host_devices = bc_environment.load_default().host().devices
        matches = [d for d in host_devices if d.index == index]
        if not matches:
            raise PatchCampaignError(
                f"{self.patch_id}: device_contexts() index {index} is not a "
                f"configured device in config/environment.toml (known indices: "
                f"{sorted(d.index for d in host_devices)})"
            )
        device = matches[0]
        if device.locator is None:
            raise PatchCampaignError(
                f"{self.patch_id}: device_contexts() index {index} "
                f"({device.arch}) has no verified locator in config/environment.toml"
            )
        if device.arch != architecture:
            raise PatchCampaignError(
                f"{self.patch_id}: device_contexts() index {index} is configured "
                f"as {device.arch!r}, but {architecture!r} was requested"
            )
        # NOT locators=(device.locator,): the observed side of this
        # comparison comes from parse_rocm_attestation() parsing a plain
        # llama-bench/llama-perplexity stdout device banner ("Device 0:
        # AMD Radeon Graphics, gfx1201 (0x1201), ..."), which has no PCI
        # locator at all and hardcodes ObservedDevice.locator=None --
        # structurally, not as a bug in that parser (llama.cpp does not
        # print one). Passing a real verified locator here made
        # compare_execution_identity() require a match this observation
        # channel can never supply, so every real-hardware run through
        # this generic --validation-producer device_contexts() path was
        # guaranteed to fail closed at the first paired-benchmark call
        # (found on real hardware, PA39 real-hardware acceptance attempt
        # #3c). The two other ExecutionIdentity(...) call sites in this
        # same file (~4931, ~5952) already omit locators for the exact
        # same reason -- this now matches that established precedent
        # rather than inventing a new, weaker guarantee: device identity
        # here rests on architecture match + the HIP_VISIBLE_DEVICES
        # env-scoping this method already returns below, same as those
        # sibling call sites. Known residual gap (documented, not fixed
        # here, matching this item's existing gfx1100-coverage-gap
        # documentation pattern for RD06/RD07): for RD07's gfx1100 arm,
        # which has TWO real physical devices (index 0 and 1, both
        # architecture gfx1100), this evidence channel cannot distinguish
        # which of the two actually ran -- that relies on
        # HIP_VISIBLE_DEVICES scoping alone, unverified in the
        # attestation record, for gfx1100 specifically. RD05/RD06 use
        # gfx1201 only, the sole device of that architecture on this
        # host, so the gap does not apply to them.
        identity = ExecutionIdentity(
            backend="ROCm", architectures=(device.arch,),
        )
        return identity, {"HIP_VISIBLE_DEVICES": str(index)}, device.locator

    def write_artifact(self, *, name: str, payload: JsonObject):
        return _write_bound_artifact_ref(self.run_dir, name, payload)

    def write_text_artifact(self, *, name: str, text: str) -> ArtifactRef:
        return _write_bound_text_artifact_ref(self.run_dir, name, text)

    def run_paired_llama_benchmark(
        self,
        *,
        control_binary: Path,
        subject_binary: Path,
        model: Path,
        workloads: tuple[str, ...] = ("decode", "prefill"),
        patch_args: tuple[str, ...] = (),
        runtime_args: tuple[str, ...] = (),
        pairs: int = 3,
        log_context: str,
        device: ProducerDeviceContext | None = None,
        env_overrides: Mapping[str, str] | None = None,
        env_unset: tuple[str, ...] = (),
    ) -> ProducerPairedBenchmarkOutcome:
        # RD58 (PA36 migration #4, dev-gpt-agent
        # req_ecb4b77a4c4e4bdd MAJOR #2): the env selector authority
        # is the DEVICE, not the caller. When device != None, reject
        # selector keys (HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES)
        # in the caller's env_overrides/env_unset (the caller could
        # otherwise replace the device selector or delete it, while
        # execution_identity still describes the device object), and
        # apply the device selector LAST (so it wins over any
        # non-selector collision). When device is None (RD58
        # multi-GPU), preserve the explicit ambient
        # HIP_VISIBLE_DEVICES (reject it in caller env_overrides so
        # it is not replaced); the caller may set non-selector
        # overrides (e.g. GGML_CUDA_REGISTER_HOST=1) and unset
        # ROCR_VISIBLE_DEVICES.
        _SELECTOR_KEYS = frozenset(
            {"HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"}
        )
        if device is not None:
            for _key in (env_overrides or {}):
                if _key in _SELECTOR_KEYS:
                    raise PatchCampaignError(
                        f"run_paired_llama_benchmark: caller "
                        f"env_overrides must not set selector key "
                        f"{_key!r} when a device context is supplied "
                        "(the device owns the selector)"
                    )
            for _key in env_unset:
                if _key in _SELECTOR_KEYS:
                    raise PatchCampaignError(
                        f"run_paired_llama_benchmark: caller "
                        f"env_unset must not unset selector key "
                        f"{_key!r} when a device context is supplied "
                        "(the device owns the selector)"
                    )
            # Caller's non-selector overrides first, device selector
            # LAST (device wins on any collision).
            merged_overrides: dict[str, str] = {}
            if env_overrides:
                merged_overrides.update(env_overrides)
            merged_overrides.update(_hip_only(dict(device.env_overrides)))
        else:
            # device=None (RD58 multi-GPU): preserve the explicit
            # ambient HIP_VISIBLE_DEVICES (reject it in caller
            # env_overrides so it is not replaced); the caller may
            # set non-selector overrides + unset ROCR_VISIBLE_DEVICES.
            for _key in (env_overrides or {}):
                if _key == "HIP_VISIBLE_DEVICES":
                    raise PatchCampaignError(
                        "run_paired_llama_benchmark: caller "
                        "env_overrides must not set "
                        "HIP_VISIBLE_DEVICES when device is None "
                        "(the explicit ambient selector is preserved)"
                    )
            # RD58 (PA36 migration #4, dev-gpt-agent
            # req_2c5e7a0230914eab MAJOR): also reject
            # HIP_VISIBLE_DEVICES in env_unset -- a multi-GPU
            # producer could otherwise delete the authoritative
            # ambient selector via env_unset=("HIP_VISIBLE_DEVICES",
            # ), defeating the preflight and running with
            # unrestricted visibility. ROCR_VISIBLE_DEVICES remains
            # allowed/expected in env_unset.
            for _key in env_unset:
                if _key == "HIP_VISIBLE_DEVICES":
                    raise PatchCampaignError(
                        "run_paired_llama_benchmark: caller "
                        "env_unset must not unset "
                        "HIP_VISIBLE_DEVICES when device is None "
                        "(the explicit ambient selector is preserved)"
                    )
            merged_overrides = (
                dict(env_overrides) if env_overrides else {}
            )
        execution_identity = device.execution_identity if device is not None else None
        outcome = run_paired_llama_benchmark(
            control_binary=control_binary, subject_binary=subject_binary, model=model,
            hip_path=self.hip_path, workloads=workloads, patch_args=patch_args,
            runtime_args=runtime_args, pairs=pairs, log_context=log_context,
            env_overrides=merged_overrides or None,
            env_unset=env_unset,
            execution_identity=execution_identity,
        )
        return ProducerPairedBenchmarkOutcome(
            runs=outcome.runs, commands=outcome.commands, raw_logs=tuple(outcome.raw_logs),
        )


def _write_bound_artifact_ref(run_dir: Path, name: str, payload: JsonObject):
    """``_write_bound_artifact()`` returns a plain ``{"path", "sha256"}``
    dict; ``ProducerRuntime.write_artifact()`` must return the real typed
    ``ArtifactRef`` a ``ValidationResult.artifacts``/``validate_producer_
    result()`` can bind -- this wraps the former into the latter without
    duplicating the write logic."""
    ref = _write_bound_artifact(run_dir, name, payload)
    return ArtifactRef(name=name, path=ref["path"], sha256=ref["sha256"])


def _write_bound_text_artifact_ref(run_dir: Path, name: str, text: str) -> ArtifactRef:
    """``write_artifact()`` serializes a JSON payload; ``write_text_artifact()``
    writes raw text VERBATIM (RD12's raw per-arm activation logs are not JSON,
    and re-serializing them would change the exact bytes the trace-marker
    validator re-reads). Same bound-artifact contract as write_artifact():
    returns the typed ArtifactRef a ValidationResult.artifacts /
    validate_producer_result() can bind."""
    target = run_dir / "artifacts" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    # Byte-verbatim: a text-mode write would translate newlines on Windows
    # (\n -> \r\n) and change the exact bytes (and sha256) the trace-marker
    # validator re-reads (GPT review req_052817cb66d14bc1).
    data = text.encode("utf-8")
    target.write_bytes(data)
    return ArtifactRef(
        name=name,
        path=target.relative_to(run_dir).as_posix(),
        sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    )


@dataclass(frozen=True)
class ProducerEvidenceBindingContext:
    """The shared-code identity facts a standard_campaign="run" producer's
    semantic evidence gets bound into (PA36 sub-slice 2, dev-gpt-agent
    req_2ecda033763949a9 T3). The producer supplies ONLY semantic
    measurements; the binder owns every canonical identity field -- patch
    identity, source tree, campaign identity digest -- and writes the
    root-level evidence artifacts under run_dir."""

    run_dir: Path
    patch_id: str
    patch_path: Path
    base_revision: str
    patched_source_tree: str
    campaign_identity_digest: str
    gpu_architectures: tuple[str, ...]


@dataclass(frozen=True)
class BoundProducerEvidence:
    """The evaluation context AFTER a producer's evidence was bound (T3):
    the same ValidationContext shape with trace/correctness/performance
    evidence replaced by their canonical bound forms, plus the root
    correctness document and activation disposition the executor must
    carry into make_record()."""

    validation_context: ValidationContext
    correctness: dict[str, object] | None
    activation_disposition: str | None


def _bind_producer_correctness(
    semantic: JsonObject | None,
    *,
    binding: ProducerEvidenceBindingContext,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    """Bind a producer's semantic correctness measurement into the
    canonical correctness document (T3). The producer supplies EXACTLY
    {disposition, mechanism, detail} -- no identity fields, which the
    shared binder owns: the root correctness.json under run_dir, its
    artifact binding, and the schema/patch/source/campaign identity."""
    from bigcherry.patch import evidence as patch_validation_evidence

    if semantic is None:
        return None, {}

    values = dict(semantic)
    expected = {"disposition", "mechanism", "detail"}
    if set(values) != expected:
        raise PatchCampaignError(
            "producer correctness must contain exactly "
            "{'disposition','mechanism','detail'}"
        )

    disposition = values["disposition"]
    mechanism = values["mechanism"]
    detail = values["detail"]
    if disposition not in ("passed", "failed"):
        raise PatchCampaignError(
            f"producer correctness disposition must be passed/failed, got {disposition!r}"
        )
    if not isinstance(mechanism, str) or not mechanism:
        raise PatchCampaignError("producer correctness mechanism must be non-empty")
    if not isinstance(detail, str):
        raise PatchCampaignError("producer correctness detail must be a string")

    document = {
        "schema_version": patch_validation_evidence.CORRECTNESS_SCHEMA_VERSION,
        "patch_id": binding.patch_id,
        "patch_validation_subject_digest":
            patch_validation_evidence.patch_validation_subject_digest(
                binding.patch_path
            ),
        "base_revision": binding.base_revision,
        "patched_source_tree": binding.patched_source_tree,
        "campaign_identity_digest": binding.campaign_identity_digest,
        "gpu_architectures": list(binding.gpu_architectures),
        "disposition": disposition,
        "mechanism": mechanism,
        "detail": detail,
    }

    path = binding.run_dir / "correctness.json"
    _atomic_write_json(path, document)
    artifact = {
        "path": path.relative_to(binding.run_dir).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    return document, {"artifact": artifact}


def _producer_owned_artifact_ref(
    artifact: object,
    role: str,
    *,
    declared: frozenset[str],
    emitted: frozenset[str],
) -> dict[str, object]:
    """Manifest-ownership gate on producer-supplied artifact refs
    (dev-gpt-agent req_f5ba56f4088e4742 finding #1): the ref must be
    exactly {path, sha256}, the path exactly 'artifacts/<basename>', and
    the basename must appear BOTH in the producer's declared manifest
    (spec.artifact_names) and in result.emitted_artifacts.
    _artifact_is_bound() alone proves only path containment + SHA --
    without this gate a producer could write an undeclared file and
    smuggle it into the persisted record through trace/performance
    evidence."""
    if not isinstance(artifact, Mapping):
        raise PatchCampaignError(
            f"producer {role} artifact reference must be an object"
        )
    if set(artifact) != {"path", "sha256"}:
        raise PatchCampaignError(
            f"producer {role} artifact reference must contain exactly "
            f"path/sha256, got {sorted(artifact)!r}"
        )
    path = artifact["path"]
    sha256 = artifact["sha256"]
    if not isinstance(path, str) or not isinstance(sha256, str):
        raise PatchCampaignError(
            f"producer {role} artifact reference path/sha256 must be strings"
        )
    if "\\" in path:
        raise PatchCampaignError(
            f"producer {role} artifact path must use forward slashes: {path!r}"
        )
    parts = path.split("/")
    if len(parts) != 2 or parts[0] != "artifacts" or not parts[1]:
        raise PatchCampaignError(
            f"producer {role} artifact path must be exactly "
            f"'artifacts/<basename>', got {path!r}"
        )
    basename = parts[1]
    if basename not in declared:
        raise PatchCampaignError(
            f"producer {role} artifact {basename!r} is not declared in the "
            f"producer manifest; declared: {sorted(declared)!r}"
        )
    if basename not in emitted:
        raise PatchCampaignError(
            f"producer {role} artifact {basename!r} was not claimed in "
            f"result.emitted_artifacts"
        )
    return {"path": path, "sha256": sha256}


def _bind_producer_trace_evidence(
    trace_evidence: JsonObject | None,
    *,
    validation_plan: ValidationPlan,
    declared_artifacts: frozenset[str],
    emitted_artifacts: frozenset[str],
) -> dict[str, object]:
    """Bind a producer's trace observations into the canonical
    trace_evidence shape (T3). The producer owns ONLY the positive/negative
    artifact refs; the plan owns the marker semantics -- marker_regex is
    injected from the single declared trace-marker activation check, and a
    producer-supplied regex is rejected. Every artifact ref is additionally
    manifest-gated (declared AND emitted) before evaluation."""
    if trace_evidence is None:
        return {}

    trace_specs = tuple(
        spec
        for spec in validation_plan.checks
        if spec.capability == "activation" and spec.validator == "trace-marker"
    )
    if len(trace_specs) != 1:
        raise PatchCampaignError(
            "producer supplied trace evidence but validation plan does not "
            "declare exactly one trace-marker activation check"
        )

    marker = trace_specs[0].config.get("marker-regex")
    if not isinstance(marker, str) or not marker:
        raise PatchCampaignError(
            "trace-marker activation check has no non-empty marker-regex"
        )

    raw = dict(trace_evidence)
    if set(raw) != {"positive", "negative"}:
        raise PatchCampaignError(
            "producer trace evidence must contain exactly positive/negative"
        )

    bound: dict[str, object] = {}
    for role in ("positive", "negative"):
        observation = raw[role]
        if not isinstance(observation, Mapping):
            raise PatchCampaignError(
                f"producer trace evidence {role} must be an object"
            )
        if set(observation) != {"artifact"}:
            raise PatchCampaignError(
                f"producer trace evidence {role} must contain exactly 'artifact'"
            )
        bound[role] = {
            "marker_regex": marker,
            "artifact": _producer_owned_artifact_ref(
                observation["artifact"],
                f"trace {role}",
                declared=declared_artifacts,
                emitted=emitted_artifacts,
            ),
        }
    return bound


def _bind_producer_result_evidence(
    result: ProducerResult,
    *,
    validation_plan: ValidationPlan,
    validation_context: ValidationContext,
    binding: ProducerEvidenceBindingContext | None,
    declared_artifacts: frozenset[str],
) -> BoundProducerEvidence:
    """One generic post-producer evidence-binding pass (T3/T4): trace,
    performance, correctness, and activation evidence are replaced by
    their canonical bound forms so the fallback validators -- and
    compute_verdict() -- see exactly what gets persisted. When ``binding``
    is None (self-contained "skip" producers) the context passes through
    unchanged except for producer-supplied evidence dicts. Every
    producer-supplied artifact ref is manifest-gated against
    ``declared_artifacts`` AND ``result.emitted_artifacts``
    (req_f5ba56f4088e4742 finding #1)."""
    trace_evidence = (
        _bind_producer_trace_evidence(
            result.trace_evidence,
            validation_plan=validation_plan,
            declared_artifacts=declared_artifacts,
            emitted_artifacts=result.emitted_artifacts,
        )
        if result.trace_evidence is not None
        else validation_context.trace_evidence
    )
    performance_evidence = (
        dict(result.performance_evidence)
        if result.performance_evidence is not None
        else validation_context.performance_evidence
    )
    if "artifact" in performance_evidence:
        performance_evidence = {
            **performance_evidence,
            "artifact": _producer_owned_artifact_ref(
                performance_evidence["artifact"],
                "performance",
                declared=declared_artifacts,
                emitted=result.emitted_artifacts,
            ),
        }

    correctness: dict[str, object] | None = None
    correctness_evidence = validation_context.correctness_evidence
    activation_disposition: str | None = None

    if binding is not None:
        correctness, correctness_evidence = _bind_producer_correctness(
            result.correctness,
            binding=binding,
        )

        activation_evidence = result.activation_evidence
        if activation_evidence is not None:
            if not isinstance(activation_evidence, ActivationEvidence):
                raise PatchCampaignError(
                    "ProducerResult.activation_evidence must be ActivationEvidence or None"
                )
            activation_disposition = verdict(
                activation_evidence,
                correctness_passed=None,
            )
            write_activation_json(
                binding.run_dir / "activation.json",
                activation_evidence,
                activation_disposition,
                extra={
                    "campaign_identity_digest": binding.campaign_identity_digest,
                },
            )

    return BoundProducerEvidence(
        validation_context=dataclasses.replace(
            validation_context,
            trace_evidence=trace_evidence,
            correctness_evidence=correctness_evidence,
            performance_evidence=performance_evidence,
        ),
        correctness=correctness,
        activation_disposition=activation_disposition,
    )


@dataclass(frozen=True)
class ProducerExecution:
    """The full result of running one producer through the generic
    dispatcher (PA36-F step 3): the typed producer result plus every
    downstream value ``make_record()`` needs, computed exactly once so
    the caller never has to re-derive them. ``bound_correctness`` and
    ``activation_disposition`` are the T3/T4 bound evidence values the
    caller persists -- never re-derived."""

    selection: ProducerSelection
    result: ProducerResult
    evaluated: Mapping[str, ValidationResult]
    verdict: Verdict
    contract_verdicts: Mapping[str, JsonObject]
    bound_correctness: JsonObject | None = None
    activation_disposition: str | None = None
    # trace_probe="run": the dispatcher's own scaffold-probe activation
    # evidence (the producer returns none in that mode).
    activation_evidence: ActivationEvidence | None = None


def _run_producer_trace_probes(
    *,
    result: ProducerResult,
    producer_context: ProducerContext,
    validation_plan: ValidationPlan,
    run_dir: Path,
    bench_prompt: int,
    bench_gen: int,
) -> tuple[ActivationEvidence, dict[str, object]]:
    """trace_probe="run" (dev-gpt-agent req_110d0729beb44d8b Q3): the
    dispatcher itself runs the scaffold's generic two-probe activation
    probe against the standard-campaign scaffold's SUBJECT llama-bench --
    the producer returns NO activation/trace evidence in this mode (both
    are rejected fail-closed), and this probe's ActivationEvidence plus
    its bound log refs are the record's canonical activation evidence.
    The marker comes from the single declared trace-marker check; the
    description uses the standard campaign's fallback
    (``<patch_id> activation``)."""
    trace_specs = tuple(
        spec
        for spec in validation_plan.checks
        if spec.capability == "activation" and spec.validator == "trace-marker"
    )
    if len(trace_specs) != 1:
        raise PatchCampaignError(
            "trace_probe='run' requires the validation plan to declare "
            "exactly one trace-marker activation check"
        )
    marker = trace_specs[0].config.get("marker-regex")
    if not isinstance(marker, str) or not marker:
        raise PatchCampaignError(
            "trace_probe='run': trace-marker activation check has no "
            "non-empty marker-regex"
        )
    if result.activation_evidence is not None or result.trace_evidence is not None:
        raise PatchCampaignError(
            "trace_probe='run' producers must not supply their own "
            "activation/trace evidence -- the dispatcher runs the "
            "scaffold's generic probe"
        )
    if producer_context.model is None:
        raise PatchCampaignError(
            "trace_probe='run' requires a model (the probe is a real "
            "llama-bench run)"
        )
    subject_binaries = producer_context.validation_binaries.get("subject")
    if (
        not isinstance(subject_binaries, Mapping)
        or "llama-bench" not in subject_binaries
    ):
        raise PatchCampaignError(
            "trace_probe='run' requires the standard-campaign scaffold "
            "subject llama-bench binary"
        )
    # PA36 RD13/1206 migration (GPT req_760c0fe82d7b4609 MAJOR): the
    # activation probe must run on the SAME physical device as the
    # correctness measurement -- resolve exactly one device for the
    # invocation architecture and thread its HIP-only selector env
    # (never ambient visibility).
    run_architecture = producer_context.fat_targets.targets[0]
    devices = producer_context.runtime.device_contexts(
        device_map=producer_context.device_map,
    )
    device_matches = [d for d in devices if d.architecture == run_architecture]
    if len(device_matches) != 1:
        raise PatchCampaignError(
            f"trace_probe='run': expected exactly one device mapped for "
            f"run architecture {run_architecture!r}, got {len(device_matches)}; "
            "--device-map must select one real device for it"
        )
    probe_device = device_matches[0]
    probe = run_trace_activation_probes(
        marker_regex=marker,
        description=f"{producer_context.patch_id} activation",
        binary=subject_binaries["llama-bench"], model=producer_context.model,
        hip_path=producer_context.hip_path, workdir=run_dir,
        bench_prompt=bench_prompt, bench_gen=bench_gen,
        env_overrides=dict(probe_device.env_overrides),
        env_unset=probe_device.env_unset,
    )
    if probe is None:
        raise PatchCampaignError(
            "trace_probe='run': the trace probe unexpectedly returned None "
            "(marker_regex and description were both non-empty)"
        )
    return probe


def execute_validation_producer(
    *,
    patch_dir: Path,
    producer_id: str,
    provided_inputs: Mapping[str, str],
    producer_context: ProducerContext,
    validation_plan: ValidationPlan,
    validation_context: ValidationContext,
    correctness_evidence_requested: bool,
    performance_benchmark_requested: bool,
    selection: ProducerSelection | None = None,
    evidence_binding_context: ProducerEvidenceBindingContext | None = None,
    bench_prompt: int = 512,
    bench_gen: int = 128,
) -> ProducerExecution:
    """The one generic entry point that executes a selected patch-local
    validation producer end to end (PA36-F step 3, GPT design
    req_8ec9b90c05f84a30). ``validation_campaign.py`` knows how to run
    the producer this resolves; it never knows which patch/RD it is --
    all patch identity lives in ``patch_dir``/``producer_id`` and the
    already-constructed ``producer_context``.

    Exact sequence (per GPT's design, section 3):
    1. resolve_producer()
    2. assert selection.spec.patch_id == producer_context.patch_id
    3. validate_producer_inputs()
    4. validate_producer_cli_compatibility()
    5. dataclasses.replace(producer_context, inputs=validated_inputs)
    6. result = selection.producer(context) -- any exception fails closed
       as PatchCampaignError; a partial/half-built result is never used.
    7. validate_producer_result()
    8. build producer_results = {check_id: validation_result}
    9. for every plan check in order: producer-supplied result, else
       evaluate_check()
    10. verdict = compute_verdict()
    11. contract_verdicts from every check_result carrying a disposition
    12. return ProducerExecution(...)
    """
    from bigcherry.patch import validation as patch_validation

    selection = selection or resolve_producer(
        patch_dir=patch_dir,
        producer_id=producer_id,
    )

    if selection.spec.patch_id != producer_context.patch_id:
        raise ValidationProducerError(
            f"execute_validation_producer: resolved producer patch_id "
            f"{selection.spec.patch_id!r} does not match "
            f"producer_context.patch_id {producer_context.patch_id!r}"
        )

    validated_inputs = validate_producer_inputs(selection.spec, provided_inputs)
    validate_producer_cli_compatibility(
        selection.spec,
        correctness_evidence_requested=correctness_evidence_requested,
        performance_benchmark_requested=performance_benchmark_requested,
    )

    context = dataclasses.replace(producer_context, inputs=validated_inputs)

    try:
        result = selection.producer(context)
    except Exception as exc:
        raise PatchCampaignError(
            f"{selection.spec.patch_id}/{selection.spec.producer_id}: producer raised: {exc}"
        ) from exc
    if not isinstance(result, ProducerResult):
        raise PatchCampaignError(
            f"{selection.spec.patch_id}/{selection.spec.producer_id}: producer returned "
            f"{type(result).__name__}, not ProducerResult"
        )

    validate_producer_result(
        selection.spec, result, plan=validation_plan, context=validation_context,
    )

    # trace_probe="run" (dev-gpt-agent req_110d0729beb44d8b Q3): the
    # dispatcher itself runs the scaffold's generic two-probe activation
    # probe; the producer supplies no activation/trace evidence in that
    # mode. The probe's ActivationEvidence + bound log refs replace the
    # context's trace_evidence BEFORE the T3/T4 binding pass, so the
    # trace-marker fallback validator evaluates the real probe output.
    probe_evidence: ActivationEvidence | None = None
    probe_disposition: str | None = None
    if selection.spec.trace_probe == "run":
        if evidence_binding_context is None:
            raise PatchCampaignError(
                "trace_probe='run' requires a standard campaign (the "
                "scaffold subject llama-bench + a bound run_dir)"
            )
        probe_evidence, probe_detail = _run_producer_trace_probes(
            result=result,
            producer_context=context,
            validation_plan=validation_plan,
            run_dir=evidence_binding_context.run_dir,
            bench_prompt=bench_prompt,
            bench_gen=bench_gen,
        )
        from bigcherry.patch.activation import (
            verdict as _activation_verdict,
        )
        probe_disposition = _activation_verdict(
            probe_evidence, correctness_passed=None
        )
        write_activation_json(
            evidence_binding_context.run_dir / "activation.json",
            probe_evidence, probe_disposition,
            extra={
                "campaign_identity_digest":
                    evidence_binding_context.campaign_identity_digest,
                "trace_probe": probe_detail,
            },
        )
        _print(f"activation: {probe_evidence.status} ({probe_evidence.mechanism})")

        def _probe_log(
            detail: dict[str, object], role: str, key: str,
        ) -> str:
            observation = detail[role]
            if not isinstance(observation, Mapping):
                raise PatchCampaignError(
                    f"trace probe detail {role!r} must be an object"
                )
            value = observation[key]
            if not isinstance(value, str):
                raise PatchCampaignError(
                    f"trace probe detail {role}.{key} must be a string"
                )
            return value

        def _bind_producer_log(relative_log_path: str) -> dict[str, str]:
            target = (evidence_binding_context.run_dir / relative_log_path).resolve()
            return {
                "path": relative_log_path,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }

        validation_context = dataclasses.replace(
            validation_context,
            trace_evidence={
                "positive": {
                    "marker_regex": probe_detail["marker_regex"],
                    "artifact": _bind_producer_log(
                        _probe_log(probe_detail, "positive", "log")
                    ),
                },
                "negative": {
                    "marker_regex": probe_detail["marker_regex"],
                    "artifact": _bind_producer_log(
                        _probe_log(probe_detail, "negative_control", "log")
                    ),
                },
            },
        )

    # T3/T4 (dev-gpt-agent req_2ecda033763949a9): generic post-producer
    # evidence binding -- the fallback validators see the canonical bound
    # evidence, not the raw producer dicts, before compute_verdict().
    bound = _bind_producer_result_evidence(
        result,
        validation_plan=validation_plan,
        validation_context=validation_context,
        binding=evidence_binding_context,
        declared_artifacts=selection.spec.artifact_names,
    )

    producer_results: dict[str, ValidationResult] = {
        record.check_id: record.validation_result for record in result.check_results
    }
    evaluated: dict[str, ValidationResult] = {}
    for spec in validation_plan.checks:
        if spec.check_id in producer_results:
            evaluated[spec.check_id] = producer_results[spec.check_id]
        else:
            evaluated[spec.check_id] = patch_validation.evaluate_check(
                spec, bound.validation_context,
            )

    verdict = patch_validation.compute_verdict(validation_plan, evaluated)

    contract_verdicts: dict[str, JsonObject] = {}
    for record in result.check_results:
        if record.disposition is not None:
            (contract_id,) = record.contract_ids
            contract_verdicts[contract_id] = record.disposition

    return ProducerExecution(
        selection=selection, result=result, evaluated=evaluated, verdict=verdict,
        contract_verdicts=contract_verdicts,
        bound_correctness=bound.correctness,
        activation_disposition=(
            bound.activation_disposition
            if bound.activation_disposition is not None
            else probe_disposition
        ),
        activation_evidence=probe_evidence,
    )


# --------------------------------------------------------- PA36-F step 5: CLI


def _parse_validation_producer_selector(value: str) -> tuple[str, str]:
    """``PATCH/PRODUCER_ID`` -> ``(patch_id, producer_id)``. Uses
    ``rsplit("/", 1)`` (per GPT design section 5) so a producer_id can never
    itself be misread as containing a ``/`` -- only the patch component
    could (it never does today, but the split direction is the one that
    stays correct if it ever did)."""
    if "/" not in value:
        raise PatchCampaignError(
            f"--validation-producer {value!r} must be PATCH/PRODUCER_ID"
        )
    patch_id, producer_id = value.rsplit("/", 1)
    if not patch_id or not producer_id:
        raise PatchCampaignError(
            f"--validation-producer {value!r}: patch and producer id must both be non-empty"
        )
    return patch_id, producer_id


def _parse_producer_inputs(values: Iterable[str]) -> dict[str, str]:
    """Repeated ``--producer-input NAME=VALUE`` -> ``{name: value}``. Fails
    closed on a missing ``=``, an empty name, or the same name given
    twice -- ambiguity is never silently resolved by last-one-wins."""
    inputs: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise PatchCampaignError(
                f"--producer-input {raw!r} must be NAME=VALUE"
            )
        name, _, value = raw.partition("=")
        if not name:
            raise PatchCampaignError(f"--producer-input {raw!r}: name must not be empty")
        if name in inputs:
            raise PatchCampaignError(f"--producer-input: duplicate name {name!r}")
        inputs[name] = value
    return inputs


def _parse_producer_device_map(entries: list[str]) -> dict[str, tuple[int, ...]]:
    """Same ARCH=ID[,ID...] shape as ``parse_device_map()``, but a
    producer's ``ProducerRuntime.device_contexts()`` addresses real devices
    by integer index (``bigcherry.core.environment`` device inventory), not
    the opaque string ids the legacy ``--run-performance-benchmark`` device
    pool uses -- so this fails closed on a non-integer id instead of
    passing an unusable string through."""
    raw_map = parse_device_map(entries)
    device_map: dict[str, tuple[int, ...]] = {}
    for architecture, ids in raw_map.items():
        try:
            device_map[architecture] = tuple(int(i) for i in ids)
        except ValueError as exc:
            raise PatchCampaignError(
                f"--device-map {architecture}={','.join(ids)}: device ids must be integers "
                "for --validation-producer"
            ) from exc
    return device_map


def _producer_file_identity(path: Path) -> dict[str, object]:
    """GPT review req_7a72896b609a48b5 BLOCKER #1: the real file facts
    ({path, size, sha256}) of a producer input, bound into the campaign
    identity. A producer run that depends on model/corpus BYTES must show
    that dependency in its identity -- two runs with different model or
    corpus bytes never share a campaign identity. Fails closed when the
    file is missing."""
    if not path.is_file():
        raise PatchCampaignError(f"producer input file missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _run_validation_producer(
    args: argparse.Namespace, *, producer_id: str, provided_inputs: Mapping[str, str],
) -> int:
    """PA36-F step 5 entry point: resolve+execute one patch-local producer
    through the generic dispatcher end to end, entirely outside the legacy
    tune/replay/stock campaign-build flow ``run()`` otherwise always runs --
    every build this path performs goes through
    ``CampaignProducerRuntime.build_pair()`` instead (the PA36 build-once-
    fat-multiarch authority), mirroring how ``_run_performance_benchmark()``
    is already its own self-contained path rather than a branch bolted onto
    the legacy one.

    Binding this producer's typed result into a persisted, tracked
    ``patch_validation_evidence`` record (the shape GPT's design section 3
    sketches for a caller) is real per-patch MIGRATION work -- it requires
    that migration's own campaign-build identity domain decisions (what
    ``build_identities`` even means for a patch with no tune/replay/stock
    build) and is explicitly out of PA36-F's scope (see the atomic
    migration sequence). This prints the verdict and writes the generic,
    typed execution outcome as a workdir-local JSON artifact -- the same
    diagnostic-artifact pattern ``--run-performance-benchmark`` already
    uses for ``performance-matrix.json`` -- and exits 0 iff the plan is
    eligible.
    """
    import os

    from bigcherry.core import paths as bc_paths
    from bigcherry.core import config as campaign_config
    from bigcherry.patch import evidence as patch_validation_evidence
    from bigcherry.patch import registry as patch_registry
    from bigcherry.patch import source as psi
    from bigcherry.patch import validation as patch_validation
    from bigcherry.patch import validation_policy as patch_validation_policy

    os.environ["ROCM_PATH"] = str(args.hip_path)
    os.environ["HIP_PATH"] = str(args.hip_path)
    os.environ["PATH"] = os.pathsep.join(
        [str(args.hip_path / "bin"), os.environ.get("PATH", "")]
    )

    registry = patch_registry.load_registry(bc_paths.PATCHES)
    descriptor = registry.get(args.patch)
    cfg = campaign_config.load(bc_paths.RECIPES)

    validation_plan = patch_validation_policy.require_execution_package(
        descriptor, root=bc_paths.PATCHES,
    )
    if validation_plan is None:
        raise PatchCampaignError(
            f"{args.patch}: --validation-producer requires a resolvable validation plan"
        )

    # Plural-aware (VA17): a --validation-producer patch may bind more than
    # one Experiment Contract (e.g. 1203's RD05/RD06/RD07) -- the singular
    # load_contract_for_descriptor()/.experiment_contract compatibility path
    # fails closed (PatchRegistryError) for exactly that case, so this
    # generic dispatcher must use the plural loader, never the singular one.
    bound_contracts = patch_validation.load_contracts_for_descriptor(descriptor)

    workdir: Path = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    patch_dir = bc_paths.PATCHES / args.patch

    # Resolve before expensive scaffold work because manifest policy owns
    # whether a standard campaign is required.
    selection = resolve_producer(
        patch_dir=patch_dir,
        producer_id=producer_id,
    )
    if selection.spec.patch_id != args.patch:
        raise ValidationProducerError(
            f"producer patch_id {selection.spec.patch_id!r} does not match "
            f"requested patch {args.patch!r}"
        )

    # Fail fast before builds. execute_validation_producer() deliberately
    # repeats these authoritative gates for direct callers.
    validate_producer_inputs(selection.spec, provided_inputs)
    validate_producer_cli_compatibility(
        selection.spec,
        correctness_evidence_requested=args.correctness_evidence is not None,
        performance_benchmark_requested=bool(args.run_performance_benchmark),
    )

    amdgpu_targets = args.amdgpu_targets
    fat_targets = FatTargetPlan(
        targets=tuple(amdgpu_targets.split(";")) if amdgpu_targets else (),
    )
    device_map = _parse_producer_device_map(list(args.device_map or ()))
    
    # PA36 (dev-gpt-agent req_19c0ea3d2d3f40db): validate requested AND
    # measured architectures against the union of bound contracts'
    # scope.architectures before producer execution. This is the generic,
    # contract-authority guard -- not a per-patch hardcoded check.
    if fat_targets.targets or device_map:
        contract_architectures: set[str] = set()
        for _contract in bound_contracts:
            if _contract.scope.architectures:
                contract_architectures.update(_contract.scope.architectures)
        if contract_architectures:
            # Validate requested (fat) architectures
            if fat_targets.targets:
                requested_archs = set(fat_targets.targets)
                unsupported_requested = requested_archs - contract_architectures
                if unsupported_requested:
                    raise PatchCampaignError(
                        f"{args.patch}: requested architectures "
                        f"{sorted(unsupported_requested)} are not in the "
                        f"bound contract scope {sorted(contract_architectures)}"
                    )
            # Validate measured (device_map) architectures
            if device_map:
                measured_archs = set(device_map.keys())
                unsupported_measured = measured_archs - contract_architectures
                if unsupported_measured:
                    raise PatchCampaignError(
                        f"{args.patch}: measured architectures "
                        f"{sorted(unsupported_measured)} are not in the "
                        f"bound contract scope {sorted(contract_architectures)}"
                    )
                # Require measured architectures to be a subset of fat targets
                # so a selected device cannot run against a binary not built
                # for that architecture
                if fat_targets.targets:
                    not_in_fat = measured_archs - set(fat_targets.targets)
                    if not_in_fat:
                        raise PatchCampaignError(
                            f"{args.patch}: measured architectures "
                            f"{sorted(not_in_fat)} are not in the requested "
                            f"fat targets {sorted(fat_targets.targets)}"
                        )

    scaffold: StandardCampaignScaffold | None = None
    evidence_binding: ProducerEvidenceBindingContext | None = None
    campaign_identity_digest: str | None = None

    if selection.spec.standard_campaign == "run":
        # RD58 (PA36 migration #4, dev-gpt-agent req_82fbbafe52c0472d
        # Q4): generic pre-scaffold GPU-count preflight. A bound
        # contract that declares scope.gpu_count.minimum (RD58 is
        # currently the only one) is enforced BEFORE the expensive
        # 5-build scaffold: fail closed if HIP_VISIBLE_DEVICES does
        # not declare enough distinct selector tokens. Generic, not an
        # RD58-specific branch.
        required_gpu_count: int | None = None
        for _contract in bound_contracts:
            if _contract.scope.gpu_count is not None:
                _minimum = _contract.scope.gpu_count.minimum
                if _minimum is not None:
                    required_gpu_count = (
                        _minimum if required_gpu_count is None
                        else max(required_gpu_count, _minimum)
                    )
        if required_gpu_count is not None:
            from bigcherry.experiment.execution import (
                require_device_visibility,
            )

            require_device_visibility(
                context=f"{args.patch} pre-scaffold GPU-count preflight",
                minimum_count=required_gpu_count,
            )
        baseline_source = getattr(args, "baseline_source", "bigcherry")
        scaffold = _build_standard_campaign_scaffold(
            patch_id=args.patch,
            base_ref=cfg.pinned,
            baseline_source=baseline_source,
            hip_path=args.hip_path,
            amdgpu_targets=args.amdgpu_targets,
            workdir=workdir,
            worktree_root=args.worktree_root,
            build_root=args.build_root,
        )

        base_revision = scaffold.base_revision
        run_dir = workdir / "campaign"
        run_dir.mkdir(parents=True, exist_ok=True)

        control_tree = psi.git_worktree_tree(scaffold.control_source)
        subject_tree = psi.git_worktree_tree(scaffold.subject_source)
        patch_file = registry.root / descriptor.implementation_path
        patch_digest = psi.patch_implementation_digest(args.patch)

        model_identity = (
            _producer_file_identity(Path(args.model))
            if args.model is not None
            else None
        )
        corpus_identity = (
            _producer_file_identity(Path(args.producer_corpus))
            if args.producer_corpus is not None
            else None
        )
        if model_identity is None and corpus_identity is None:
            # RD12 pilot path: no model/corpus inputs -- the model-free
            # digest stays bit-identical to the approved pilot records.
            campaign_identity_digest = (
                patch_validation_evidence.model_free_campaign_identity_digest(
                    patch_name=args.patch,
                    patch_digest=patch_digest,
                    patched_source_tree=subject_tree,
                    gpu_architecture=args.amdgpu_targets,
                    campaign_build_identities=scaffold.campaign_build_identities,
                    base_revision=base_revision,
                )
            )
        else:
            # Input-bound identity (GPT review req_7a72896b609a48b5
            # BLOCKER #1): the producer's real model/corpus file facts are
            # hashed in, so two runs that differ in model or corpus bytes
            # never share a campaign identity.
            campaign_identity_digest = (
                patch_validation_evidence.producer_campaign_identity_digest(
                    patch_name=args.patch,
                    patch_digest=patch_digest,
                    patched_source_tree=subject_tree,
                    gpu_architecture=args.amdgpu_targets,
                    campaign_build_identities=scaffold.campaign_build_identities,
                    base_revision=base_revision,
                    model=model_identity,
                    corpus=corpus_identity,
                )
            )

        build_evidence = {
            "control": {
                "build_id":
                    scaffold.control_build_evidence.effective_build_id,
                "source_tree": control_tree,
                "architecture": args.amdgpu_targets,
                "options":
                    scaffold.control_build_evidence.effective_configure,
                "compile_commands": _write_bound_artifact(
                    run_dir,
                    "build/control-compile-commands.json",
                    scaffold.control_build_evidence.verification.to_dict(),
                ),
                "runtime_bundle": _write_bound_artifact(
                    run_dir,
                    "build/control-runtime-bundle.json",
                    scaffold.control_build_evidence.runtime_artifacts,
                ),
            },
            "subject": {
                "build_id":
                    scaffold.validation_subject_build_evidence.effective_build_id,
                "source_tree": subject_tree,
                "architecture": args.amdgpu_targets,
                "options":
                    scaffold.validation_subject_build_evidence.effective_configure,
                "compile_commands": _write_bound_artifact(
                    run_dir,
                    "build/subject-compile-commands.json",
                    scaffold.validation_subject_build_evidence.verification.to_dict(),
                ),
                "runtime_bundle": _write_bound_artifact(
                    run_dir,
                    "build/subject-runtime-bundle.json",
                    scaffold.validation_subject_build_evidence.runtime_artifacts,
                ),
            },
        }

        apply_evidence = {
            "control": {
                "verified": True,
                "idempotent": scaffold.control_idempotent,
                "artifact": _write_bound_artifact(
                    run_dir,
                    "apply/control.json",
                    {
                        "source_tree": control_tree,
                        "composition": list(scaffold.control_composition),
                    },
                ),
            },
            "subject": {
                "verified": True,
                "idempotent": scaffold.subject_idempotent,
                "artifact": _write_bound_artifact(
                    run_dir,
                    "apply/subject.json",
                    {
                        "source_tree": subject_tree,
                        "composition": list(scaffold.subject_composition),
                    },
                ),
            },
        }

        package_root = (
            registry.root / descriptor.package_root
            if descriptor.package_root is not None
            else {}
        )
        validation_context = patch_validation.ValidationContext(
            descriptor=descriptor,
            base_revision=base_revision,
            control_source=scaffold.control_source,
            subject_source=scaffold.subject_source,
            stock_source=scaffold.stock_source,
            package_root=package_root,
            control_tree=control_tree,
            subject_tree=subject_tree,
            build_identities={
                "control":
                    scaffold.control_build_evidence.effective_build_id,
                "subject":
                    scaffold.validation_subject_build_evidence.effective_build_id,
            },
            build_evidence=build_evidence,
            apply_evidence=apply_evidence,
            architecture=args.amdgpu_targets,
            model=str(args.model) if args.model is not None else None,
            contracts=bound_contracts,
            contract_hashes={
                c.id: c.contract_hash for c in bound_contracts
            },
            run_dir=run_dir,
            register_artifact=
                patch_validation.make_default_register_artifact(run_dir),
            trace_evidence={},
            correctness_evidence={},
            performance_evidence={},
        )

        evidence_binding = ProducerEvidenceBindingContext(
            run_dir=run_dir,
            patch_id=args.patch,
            patch_path=patch_file,
            base_revision=base_revision,
            patched_source_tree=subject_tree,
            campaign_identity_digest=campaign_identity_digest,
            gpu_architectures=fat_targets.targets,
        )
        scaffold_validation_ids = (
            scaffold.scaffold_validation_build_identities
        )
        _scaffold_exe = ".exe" if sys.platform == "win32" else ""
        scaffold_validation_binaries = {
            "control": {
                "llama-bench": scaffold.control_bin / f"llama-bench{_scaffold_exe}",
                "llama-server": scaffold.control_bin / f"llama-server{_scaffold_exe}",
            },
            "subject": {
                "llama-bench":
                    scaffold.validation_subject_bin / f"llama-bench{_scaffold_exe}",
                "llama-server":
                    scaffold.validation_subject_bin / f"llama-server{_scaffold_exe}",
            },
        }
    else:
        # Preserve current self-contained producer semantics.
        base_revision = cfg.pinned
        run_dir = workdir / "producer" / producer_id
        run_dir.mkdir(parents=True, exist_ok=True)
        validation_context = patch_validation.ValidationContext(
            descriptor=descriptor,
            base_revision=base_revision,
            control_source=None,
            subject_source=None,
            contracts=bound_contracts,
            contract_hashes={
                c.id: c.contract_hash for c in bound_contracts
            },
        )
        scaffold_validation_ids = {}
        scaffold_validation_binaries = {}

    runtime = CampaignProducerRuntime(
        repo_root=REPO_ROOT,
        patch_id=args.patch,
        base_revision=base_revision,
        workdir=args.worktree_root,
        hip_path=args.hip_path,
        fat_targets=fat_targets,
        run_dir=run_dir,
    )
    producer_context = ProducerContext(
        repo_root=REPO_ROOT,
        patch_dir=patch_dir,
        workdir=workdir,
        campaign_id=f"{args.patch}/{producer_id}",
        base_revision=base_revision,
        hip_path=args.hip_path,
        fat_targets=fat_targets,
        model=args.model,
        corpus=args.producer_corpus,
        build_env=_hip_env(args.hip_path),
        inputs={},
        validation_build_identities=scaffold_validation_ids,
        patch_id=args.patch,
        device_map=device_map,
        runtime=runtime,
        validation_binaries=scaffold_validation_binaries,
    )

    execution = execute_validation_producer(
        patch_dir=patch_dir,
        producer_id=producer_id,
        provided_inputs=provided_inputs,
        producer_context=producer_context,
        validation_plan=validation_plan,
        validation_context=validation_context,
        correctness_evidence_requested=args.correctness_evidence is not None,
        performance_benchmark_requested=bool(args.run_performance_benchmark),
        selection=selection,
        evidence_binding_context=evidence_binding,
        bench_prompt=args.bench_prompt,
        bench_gen=args.bench_gen,
    )

    # GPT review req_7a72896b609a48b5 BLOCKER #2: the real contract-
    # correctness gate over the producer's typed named results. The bound
    # contract is the authority on which named checks are required; the
    # producer only measures and reports. No producer results -> no gate
    # (the RD12 pilot record shape stays unchanged).
    contract_correctness_gate: dict[str, object] | None = None
    named = execution.result.contract_correctness_results
    if named:
        # Fail closed on the plural case (GPT re-review
        # req_6e79607f075c479b): the dispatcher is plural-aware, but the
        # named-result gate currently binds to a single contract's
        # authority. Rather than silently picking bound_contracts[0] when
        # several are bound, reject the ambiguous case. Duplicate check
        # names are rejected too -- the {check: result} mapping below
        # would otherwise silently overwrite one result with another.
        if len(bound_contracts) != 1:
            raise PatchCampaignError(
                "contract_correctness_results currently requires exactly "
                "one bound contract"
            )
        if len({result.check for result in named}) != len(named):
            raise PatchCampaignError(
                "contract_correctness_results contains duplicate check "
                "names"
            )
        contract_correctness_gate = compute_contract_correctness_gate(
            bound_contracts[0],
            {result.check: result for result in named},
        )
    producer_check_results = {
        check_id: asdict(result)
        for check_id, result in execution.evaluated.items()
    }
    if contract_correctness_gate is not None:
        producer_check_results = {
            **producer_check_results,
            "_contract_correctness_gate": contract_correctness_gate,
        }

    record_path: Path | None = None
    validation_contract_verdicts = None
    if scaffold is not None:
        assert campaign_identity_digest is not None

        # RD58 (PA36 migration #4, dev-gpt-agent req_82fbbafe52c0472d
        # Q6): the typed producer->dispatcher promotion channel. The
        # producer supplies per-contract lane_effects (real LaneEffect
        # objects) + target_metric; the dispatcher owns
        # aggregate_contract_effects() + evaluate_promotion_gate() (the
        # producer never computes a gate itself). No
        # promotion_lane_effects -> {} (every bound contract persists
        # its explicit BLOCKED "no promotion result produced" verdict,
        # exactly as before for non-promoting producers).
        bound_by_id = {c.id: c for c in bound_contracts}
        # RD58 (PA36 migration #4, dev-gpt-agent
        # req_918c7e1be6614f84 invariant): the contract-ID keysets
        # of promotion_lane_effects, promotion_target_metric, and
        # promotion_trigger_evidence must be identical (a
        # per-contract entry in one without a matching entry in
        # another is a producer bug, not a silent omission).
        _lane_keys = set(execution.result.promotion_lane_effects)
        _metric_keys = set(execution.result.promotion_target_metric)
        _trigger_keys = set(
            execution.result.promotion_trigger_evidence
        )
        if _lane_keys != _metric_keys or _lane_keys != _trigger_keys:
            raise PatchCampaignError(
                "promotion channel keysets must be identical: "
                f"lane_effects={sorted(_lane_keys)}, "
                f"target_metric={sorted(_metric_keys)}, "
                f"trigger_evidence={sorted(_trigger_keys)}"
            )
        producer_contract_promotions: dict[str, dict[str, object]] = {}
        for contract_id, lane_effects in (
            execution.result.promotion_lane_effects.items()
        ):
            contract = bound_by_id.get(contract_id)
            if contract is None:
                raise PatchCampaignError(
                    f"promotion_lane_effects for unknown contract "
                    f"{contract_id!r}"
                )
            target_metric = execution.result.promotion_target_metric.get(
                contract_id
            )
            if target_metric is None:
                raise PatchCampaignError(
                    f"promotion_lane_effects for {contract_id!r} requires "
                    "promotion_target_metric"
                )
            # RD58 (PA36 migration #4, dev-gpt-agent
            # req_ecb4b77a4c4e4bdd MAJOR #3): a producer that
            # supplies promotion lanes MUST have a matching
            # evaluated contract correctness gate -- missing
            # correctness evidence is BLOCKED/absent evidence, not a
            # measured promotion FAIL.
            if contract_correctness_gate is None:
                raise PatchCampaignError(
                    f"promotion_lane_effects for {contract_id!r} "
                    "requires an evaluated contract correctness gate "
                    "(the producer supplied promotion lanes but no "
                    "contract_correctness_results)"
                )
            # RD58 (PA36 migration #4, dev-gpt-agent
            # req_ecb4b77a4c4e4bdd BLOCKER): a producer that
            # supplies promotion lanes MUST supply trigger evidence --
            # a contract PASS without trigger proof would be a
            # fail-OPEN (the target code path may never have run).
            trigger_evidence = (
                execution.result.promotion_trigger_evidence.get(
                    contract_id
                )
            )
            if not trigger_evidence:
                raise PatchCampaignError(
                    f"promotion_lane_effects for {contract_id!r} "
                    "requires promotion_trigger_evidence (a contract "
                    "PASS without trigger proof is fail-OPEN)"
                )
            trigger_proof = experiment_contract.evaluate_trigger_proof(
                list(trigger_evidence)
            )
            # RD58 (PA36 migration #4, dev-gpt-agent
            # req_ecb4b77a4c4e4bdd MAJOR #3): require the
            # lane/metric keysets to match -- a mismatch would
            # silently aggregate the wrong lanes.
            lane_metrics = {e.metric for e in lane_effects}
            if target_metric not in lane_metrics:
                raise PatchCampaignError(
                    f"promotion_lane_effects for {contract_id!r}: "
                    f"target_metric {target_metric!r} not in lane "
                    f"metrics {sorted(lane_metrics)}"
                )
            aggregated = experiment_contract.aggregate_contract_effects(
                contract, list(lane_effects), target_metric=target_metric
            )
            # RD73 (PA36 migration #5, dev-gpt-agent req_a232ff7fb1f045db):
            # session aggregation (dispatcher-owned). Under a session
            # policy, the gain bound is established across repeated
            # SESSIONS, not from the pairs inside this one run. Fold the
            # prior sessions' persisted lane effects together with the one
            # just measured and re-aggregate over all of them.
            if (
                contract.acceptance.effect_evidence_policy
                == "session_ci95_threshold_bound_v1"
            ):
                from bigcherry.patch import evidence as patch_validation_evidence
                prior_records = patch_validation_evidence.load_records(
                    args.patch
                )
                # GPT round 6 BLOCKER: schema-v4 persists contracts as a
                # list of {"id": ..., "hash": ...}, not a mapping.
                matching_records = [
                    r for r in prior_records
                    if any(
                        entry.get("id") == contract_id
                        and entry.get("hash") == contract.contract_hash
                        for entry in r.get("contracts", [])
                        if isinstance(entry, dict)
                    )
                ]
                # GPT round 6 BLOCKER: use fat_targets.targets (not
                # ctx.amdgpu_targets which doesn't exist in this scope)
                session_archs = list(fat_targets.targets)
                # Build the current-session stub
                this_session = {
                    "gpu_architectures": session_archs,
                    "lane_effects": [
                        {
                            "role": e.role,
                            "metric": e.metric,
                            "pair_ratios": list(e.pair_ratios),
                        }
                        for e in lane_effects
                    ],
                }
                gain_field = (
                    "end_to_end_gain_pct"
                    if contract.acceptance.end_to_end_gain_pct is not None
                    else "target_kernel_gain_pct"
                )
                aggregated = dict(aggregated)
                aggregated.update(
                    experiment_contract.aggregate_session_effects(
                        [*matching_records, this_session],
                        field=gain_field,
                        role="positive",
                        metric=target_metric,
                        architectures=session_archs,
                    )
                )
            # RD73 (PA36 migration #5, dev-gpt-agent req_a232ff7fb1f045db):
            # compute the resource gate from the producer's
            # promotion_resource_results. A resource-bound contract
            # without resource evidence must fail closed.
            resource_gate = None
            if contract.acceptance.resource_limits:
                resource_results = (
                    execution.result.promotion_resource_results.get(
                        contract_id
                    )
                )
                if not resource_results:
                    raise PatchCampaignError(
                        f"promotion_lane_effects for {contract_id!r} "
                        "requires promotion_resource_results (a "
                        "resource-bound contract without resource "
                        "evidence must fail closed)"
                    )
                # Build a {metric: result} mapping (evaluate_resource_gate
                # expects a dict, not a list). Reject duplicate metrics.
                resource_map: dict[str, experiment_contract.ResourceResult] = {}
                for rr in resource_results:
                    if rr.metric in resource_map:
                        raise PatchCampaignError(
                            f"promotion_resource_results for "
                            f"{contract_id!r}: duplicate metric "
                            f"{rr.metric!r}"
                        )
                    resource_map[rr.metric] = rr
                resource_gate = experiment_contract.evaluate_resource_gate(
                    contract, resource_map
                )
            producer_contract_promotions[contract_id] = (
                experiment_contract.evaluate_promotion_gate(
                    contract,
                    correctness_gate=contract_correctness_gate,
                    aggregated_effects=aggregated,
                    trigger_proof=trigger_proof,
                    resource_gate=resource_gate,
                )
            )
        # RD58 (PA36 migration #4, dev-gpt-agent
        # req_ecb4b77a4c4e4bdd additional invariant): persist the
        # exact promotion_lane_effects used for the verdict in the
        # record's lane_effects -- otherwise the contract promotion
        # is derived from ephemeral measurements that cannot be
        # audited from committed evidence.
        promotion_lane_json = tuple(
            asdict(effect)
            for effects in execution.result.promotion_lane_effects.values()
            for effect in effects
        )
        combined_lane_effects = (
            execution.result.lane_effects + promotion_lane_json
        )

        # Contract promotions (evaluate_promotion_gate() results) are a
        # different semantic type from the producer's check dispositions
        # (execution.contract_verdicts); the promotion APIs must never be
        # fed dispositions (req_f5ba56f4088e4742 finding #2).
        validation_contracts, validation_contract_verdicts = (
            build_contract_evidence_for_persistence(
                validation_plan.contracts,
                producer_contract_promotions,
            )
        )

        validation_record = patch_validation_evidence.make_record(
            patch_id=args.patch,
            patch_path=registry.root / descriptor.implementation_path,
            patch_implementation_digest=
                psi.patch_implementation_digest(args.patch),
            base_ref=cfg.pinned,
            base_revision=scaffold.base_revision,
            framework_baseline_digest=
                psi.composition_digest(scaffold.subject_composition),
            patched_source_tree=
                psi.git_worktree_tree(scaffold.subject_source),
            gpu_architectures=args.amdgpu_targets,
            activation_evidence=(
                execution.activation_evidence
                if execution.activation_evidence is not None
                else execution.result.activation_evidence
            ),
            activation_disposition=execution.activation_disposition,
            correctness=execution.bound_correctness,
            campaign_identity_digest=campaign_identity_digest,
            build_identities=scaffold.campaign_build_identities,
            # Deliberately producer-owned isolated pair, NOT scaffold pair.
            validation_build_identities=
                execution.result.validation_build_identities,
            campaign_workdir=run_dir,
            producer_artifact_names=
                execution.selection.spec.artifact_names,
            check_results=producer_check_results,
            validation_eligible=compute_persisted_validation_eligible(
                descriptor,
                execution.verdict,
                producer_contract_promotions,
                activation_disposition=
                    execution.activation_disposition,
                correctness=(
                    dict(execution.bound_correctness)
                    if execution.bound_correctness is not None
                    else {}
                ),
            ),
            lane_effects=combined_lane_effects,
            representation=descriptor.representation,
            validation_implementation_digest=descriptor.validation_digest,
            contracts=validation_contracts,
            contract_verdicts=validation_contract_verdicts,
            baseline_composition={
                "source": getattr(
                    args, "baseline_source", "bigcherry"
                ),
                "base_revision": scaffold.base_revision,
                "patches": list(scaffold.control_composition),
            },
            control_composition={
                "base_revision": scaffold.base_revision,
                "patches": list(scaffold.control_composition),
            },
            subject_composition={
                "base_revision": scaffold.base_revision,
                "patches": list(scaffold.subject_composition),
            },
            control_tree=
                psi.git_worktree_tree(scaffold.control_source),
            subject_tree=
                psi.git_worktree_tree(scaffold.subject_source),
            stock_tree=
                psi.git_worktree_tree(scaffold.stock_source),
        )
        record_path = patch_validation_evidence.write_record(
            validation_record
        )

    outcome_doc = {
        "patch_id": args.patch,
        "producer_id": producer_id,
        "eligible": execution.verdict.eligible,
        "reasons": list(execution.verdict.reasons),
        "blocked": execution.verdict.blocked,
        "errors": list(execution.verdict.errors),
        "check_results": producer_check_results,
        "contract_verdicts": (
            validation_contract_verdicts
            if scaffold is not None
            else dict(execution.contract_verdicts)
        ),
        "validation_build_identities":
            dict(execution.result.validation_build_identities),
        "evidence_record":
            str(record_path) if record_path is not None else None,
    }
    outcome_path = run_dir / "producer-execution.json"
    _atomic_write_json(outcome_path, outcome_doc)

    _print(
        f"validation producer {args.patch}/{producer_id}: "
        f"{'eligible' if execution.verdict.eligible else 'ineligible'} "
        f"({len(execution.verdict.reasons)} blocking reasons) -- "
        f"{outcome_path}"
    )

    # Success means the requested producer execution and, when required,
    # tracked evidence persistence completed. Eligibility is evidence,
    # not process success (dev-gpt-agent req_2ecda033763949a9 T5).
    return 0


# PA35: real verification producer for patch 1000_rdna4_mmq_q2k_q6k_fix.
# Its own README documents an unverified gap (2026-09-11 process audit):
# the patch's claimed Q6_K 1.90x / Q2_K 28.2x gains are entirely upstream
# PR #25940's own reported numbers -- no independent measurement on this
# project's own hardware exists. Those upstream numbers are themselves
# test-backend-ops `perf -o MUL_MAT` FLOP/s throughput at n=512, not
# llama-bench token/s -- so the exact-shape microbenchmark below
# reproduces the upstream claim's own methodology, and the llama-bench
# lane is a separate, model-level analogue (prefill ~= matrix n=512),
# not a re-statement of the same number.
_PATCH1000_ID = "1000_rdna4_mmq_q2k_q6k_fix"
_PATCH1000_QUANT_NAMES = {
    "Q2_K": "q2_K",
    "Q6_K": "q6_K",
}
_PATCH1000_PERF_TIME_PATTERN = re.compile(
    r"MUL_MAT\([^)]*\):\s+\d+\s+runs\s+-\s+([0-9.]+)\s+us/run"
)


def _patch1000_backend_ops_command(
    binary: Path,
    quant: str,
    *,
    mode: str,
) -> list[str]:
    try:
        type_name = _PATCH1000_QUANT_NAMES[quant]
    except KeyError:
        raise PatchCampaignError(
            f"patch1000: unsupported quant {quant!r}; "
            f"expected one of {tuple(_PATCH1000_QUANT_NAMES)}"
        ) from None

    if mode == "perf":
        # The exact n=512 MUL_MAT shape corresponding to upstream PR
        # #25940's Q2_K/Q6_K n=512 comparison.
        params = f"type_a={type_name},type_b=f32,m=4096,n=512,k=14336"
    elif mode == "test":
        # Correctness mode filters all real MUL_MAT cases for this
        # quant/F32 activation pair rather than assuming the perf-only
        # 4096x512x14336 case is also registered in the correctness corpus.
        params = f"type_a={type_name},type_b=f32"
    else:
        raise PatchCampaignError(
            f"patch1000: unsupported test-backend-ops mode {mode!r}"
        )

    return [str(binary), mode, "-o", "MUL_MAT", "-p", params]


def _patch1000_run_env(
    hip_path: Path,
    env_overrides: dict[str, str],
) -> dict[str, str]:
    from bigcherry.campaign.benchmark import sanitize_environment

    env = sanitize_environment(_hip_env(hip_path), mode="stock")
    for key in list(env):
        if key.startswith("BIGCHERRY_") or key == "GGML_CUDA_DISABLE_FUSION":
            env.pop(key, None)
    env.update(env_overrides)
    # ROCR_VISIBLE_DEVICES + HIP_VISIBLE_DEVICES to the same index
    # double-filters to zero devices (STANDARDIZED_PATCH_VALIDATION_
    # CRITERIA.md's known trap) -- HIP_VISIBLE_DEVICES alone is sufficient.
    env.pop("ROCR_VISIBLE_DEVICES", None)
    return env


def run_patch1000_backend_ops_perf(
    *,
    control_binary: Path,
    subject_binary: Path,
    quant: str,
    hip_path: Path,
    env_overrides: dict[str, str],
    pairs: int = 5,
    execution_identity: object | None = None,
) -> dict[str, object]:
    """Paired exact-shape reproduction of PR #25940's n=512 claim."""
    from bigcherry.experiment import execution as experiment_execution

    env = _patch1000_run_env(hip_path, env_overrides)
    experiment_execution.require_device_visibility(
        context=f"patch1000 {quant} backend-ops perf",
        env=env,
        exact_count=1,
    )

    control_command = _patch1000_backend_ops_command(control_binary, quant, mode="perf")
    subject_command = _patch1000_backend_ops_command(subject_binary, quant, mode="perf")

    raw_logs: list[dict[str, object]] = []

    def runner(command: list[str]) -> "experiment_execution.RunnerOutput":
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False, env=env,
        )
        raw_logs.append({
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        })
        if completed.returncode == 0:
            _require_real_gpu_execution(
                completed.stdout, completed.stderr,
                context=f"patch1000 {quant} backend-ops perf ({Path(command[0]).name})",
            )
        return experiment_execution.RunnerOutput(
            returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr,
        )

    paired = experiment_execution.run_paired_lane(
        metric="time_us", control_command=control_command, subject_command=subject_command,
        pattern=_PATCH1000_PERF_TIME_PATTERN, pairs=pairs, runner=runner,
        lower_is_better=True, execution_identity=execution_identity,
    )

    stats = dict(paired.stats)
    # lower_is_better=True's effect_pct is control_time/subject_time - 1;
    # convert to a direct speedup multiplier comparable with 28.2x/1.90x.
    effect_pct = float(stats["geometric_effect_pct"])
    ci_low_pct = float(stats["ci95_low_pct"])
    ci_high_pct = float(stats["ci95_high_pct"])
    stats["geometric_speedup_x"] = 1.0 + effect_pct / 100.0
    stats["ci95_low_x"] = 1.0 + ci_low_pct / 100.0
    stats["ci95_high_x"] = 1.0 + ci_high_pct / 100.0

    return {
        "quant": quant,
        "shape": {"op": "MUL_MAT", "type_b": "f32", "m": 4096, "n": 512, "k": 14336},
        "control_command": control_command,
        "subject_command": subject_command,
        "run": asdict(paired),
        "stats": stats,
        "raw_logs": raw_logs,
    }


def run_patch1000_backend_ops_correctness(
    *,
    binary: Path,
    quant: str,
    hip_path: Path,
    env_overrides: dict[str, str],
    log_context: str,
    execution_identity: object | None = None,
) -> dict[str, object]:
    """Real CPU-reference/NMSE MUL_MAT correctness check for one arm."""
    from bigcherry.experiment import attestation
    from bigcherry.experiment import execution as experiment_execution

    env = _patch1000_run_env(hip_path, env_overrides)
    experiment_execution.require_device_visibility(
        context=log_context, env=env, exact_count=1,
    )
    command = _patch1000_backend_ops_command(binary, quant, mode="test")

    completed = subprocess.run(command, capture_output=True, text=True, check=False, env=env)
    if completed.returncode != 0:
        raise PatchCampaignError(
            f"{log_context}: test-backend-ops exited "
            f"{completed.returncode}; stderr={completed.stderr[-1000:]!r}"
        )

    _require_real_gpu_execution(completed.stdout, completed.stderr, context=log_context)

    if execution_identity is not None:
        attestation.require_execution_identity(
            execution_identity,
            attestation.parse_rocm_attestation(completed.stdout + "\n" + completed.stderr),
            context=log_context,
        )

    return {
        "passed": True,
        "quant": quant,
        "mechanism": "test-backend-ops MUL_MAT CPU-reference/NMSE",
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def run_patch1000_verification(
    *,
    base_revision: str,
    hip_path: Path,
    build_root: Path,
    q2k_model: Path,
    q6k_model: Path,
    devices: Mapping[str, str],
    architectures: tuple[str, ...] = ("gfx1100", "gfx1201", "gfx1030"),
    pairs: int = 5,
    _source_module=None,
    _build_func=None,
    _perf_func=None,
    _correctness_func=None,
    _bench_func=None,
    _recipes_path: Path | None = None,
) -> dict[str, object]:
    """PA35 audit producer for patch 1000.

    Builds stock/control/subject ONCE as fat multi-arch trees, then varies
    only HIP_VISIBLE_DEVICES during execution (STANDARDIZED_PATCH_
    VALIDATION_CRITERIA.md's "build once, run per-device" rule).

      stock   = pinned upstream, no overlay/patches
      control = BigCherry overlay + serving-core, patch 1000 absent
      subject = control + patch 1000

    The subject composition is additionally required to equal the current
    named bigcherry-serving-base composition -- if config later gains another
    upstream-fix, fail closed rather than silently changing this audit's
    meaning.

    PA29 cutover (GPT design review req_964ec5fc21c14848): this is live
    producer code, not PA26/PA30 historical comparison apparatus, so it
    migrates with every other real consumer -- framework -> serving-core,
    bigcherry-native -> bigcherry-serving-base.
    """
    import tomllib

    from bigcherry.experiment import execution as experiment_execution
    from bigcherry.patch import source as real_source

    if pairs < 3:
        raise PatchCampaignError("patch1000: pairs must be >= 3")
    if not architectures or len(set(architectures)) != len(architectures):
        raise PatchCampaignError("patch1000: architectures must be non-empty and unique")

    missing_devices = [a for a in architectures if a not in devices]
    if missing_devices:
        raise PatchCampaignError(
            "patch1000: missing device selectors for " + ", ".join(missing_devices)
        )

    for model in (q2k_model, q6k_model):
        if not model.is_file():
            raise PatchCampaignError(f"patch1000: model file does not exist: {model}")

    psi = _source_module or real_source
    build_func = _build_func or build_tree
    perf_func = _perf_func or run_patch1000_backend_ops_perf
    correctness_func = _correctness_func or run_patch1000_backend_ops_correctness
    bench_func = _bench_func or run_paired_llama_benchmark

    recipes_path = _recipes_path or (REPO_ROOT / "config" / "recipes.toml")
    recipes = tomllib.loads(recipes_path.read_text(encoding="utf-8"))
    try:
        serving_core_ids = tuple(recipes["patch-set"]["serving-core"]["patches"])
    except (KeyError, TypeError):
        raise PatchCampaignError(
            "patch1000: recipes.toml has no valid patch-set.serving-core"
        ) from None

    if _PATCH1000_ID in serving_core_ids:
        raise PatchCampaignError("patch1000: focal patch unexpectedly appears in serving-core")

    stock_revision, stock_composition = psi.resolve_source_composition(
        "llama-native", base_ref=base_revision, base_repo=LLAMA_CPP_SRC,
    )
    control_revision, control_composition = psi.resolve_source_composition(
        "llama-native", extra_patches=serving_core_ids, base_ref=base_revision, base_repo=LLAMA_CPP_SRC,
    )
    subject_revision, subject_composition = psi.resolve_source_composition(
        "llama-native", extra_patches=(*serving_core_ids, _PATCH1000_ID),
        base_ref=base_revision, base_repo=LLAMA_CPP_SRC,
    )
    production_revision, production_composition = psi.resolve_source_composition(
        "bigcherry-serving-base", base_ref=base_revision, base_repo=LLAMA_CPP_SRC,
    )

    revisions = {stock_revision, control_revision, subject_revision, production_revision}
    if len(revisions) != 1:
        raise PatchCampaignError("patch1000: validation arms resolved different base revisions")

    if subject_composition != production_composition:
        raise PatchCampaignError(
            "patch1000: serving-core + focal no longer equals the current bigcherry-serving-base "
            "composition; update this audit producer rather than silently including/"
            "excluding another upstream fix"
        )

    source_root = build_root / "patch1000-sources"
    stock_src = psi.materialize_composition(
        base_repo=LLAMA_CPP_SRC, worktree_root=source_root / "stock",
        resolved_revision=stock_revision, composition=stock_composition,
        overlay_root=None, requested_revision=base_revision,
    )
    control_src = psi.materialize_composition(
        base_repo=LLAMA_CPP_SRC, worktree_root=source_root / "control",
        resolved_revision=control_revision, composition=control_composition,
        overlay_root=psi.REPO_ROOT / "src", requested_revision=base_revision,
    )
    subject_src = psi.materialize_composition(
        base_repo=LLAMA_CPP_SRC, worktree_root=source_root / "subject",
        resolved_revision=subject_revision, composition=subject_composition,
        overlay_root=psi.REPO_ROOT / "src", requested_revision=base_revision,
    )

    # Load-bearing PA35 rule: these builds are outside the per-device loop.
    fat_targets = ";".join(architectures)
    build_workdir = build_root / "patch1000-builds"
    build_args = {
        "hip_path": hip_path,
        "amdgpu_targets": fat_targets,
        "workdir": build_workdir,
        "targets": ["test-backend-ops", "llama-bench"],
        "extra_cmake_args": [],
    }

    arm_sources = {"stock": stock_src, "control": control_src, "subject": subject_src}
    arm_bins: dict[str, Path] = {}
    for arm, source in arm_sources.items():
        arm_bins[arm] = build_func(name=f"patch1000-{arm}", source=source, **build_args)

    exe = ".exe" if sys.platform == "win32" else ""
    backend_ops = {arm: bin_dir / f"test-backend-ops{exe}" for arm, bin_dir in arm_bins.items()}
    llama_bench = {arm: bin_dir / f"llama-bench{exe}" for arm, bin_dir in arm_bins.items()}

    comparisons = {
        "stock_vs_control": ("stock", "control"),
        "control_vs_subject": ("control", "subject"),
        "stock_vs_subject": ("stock", "subject"),
    }
    models = {"Q2_K": q2k_model, "Q6_K": q6k_model}

    cells: dict[str, object] = {}
    for architecture in architectures:
        selector = devices[architecture]
        selector_env = {"HIP_VISIBLE_DEVICES": selector}
        visibility = experiment_execution.require_device_visibility(
            context=f"patch1000 {architecture}", env=selector_env, exact_count=1,
        )
        expected_execution = ExecutionIdentity(backend="ROCm", architectures=(architecture,))

        arch_doc: dict[str, object] = {
            "architecture": architecture,
            "device_visibility": visibility.document(),
            "correctness": {},
            "microbenchmark": {},
            "llama_bench": {},
        }

        for quant in ("Q2_K", "Q6_K"):
            correctness_doc: dict[str, object] = {}
            for arm in ("control", "subject"):
                correctness_doc[arm] = correctness_func(
                    binary=backend_ops[arm], quant=quant, hip_path=hip_path,
                    env_overrides=selector_env,
                    log_context=f"patch1000 {architecture} {quant} {arm} correctness",
                    execution_identity=expected_execution,
                )
            arch_doc["correctness"][quant] = correctness_doc

            perf_doc: dict[str, object] = {}
            for comparison, (left, right) in comparisons.items():
                perf_doc[comparison] = perf_func(
                    control_binary=backend_ops[left], subject_binary=backend_ops[right],
                    quant=quant, hip_path=hip_path, env_overrides=selector_env, pairs=pairs,
                    execution_identity=expected_execution,
                )
            arch_doc["microbenchmark"][quant] = perf_doc

            bench_doc: dict[str, object] = {}
            for comparison, (left, right) in comparisons.items():
                outcome = bench_func(
                    control_binary=llama_bench[left], subject_binary=llama_bench[right],
                    model=models[quant], hip_path=hip_path, workloads=("prefill", "decode"),
                    pairs=pairs, log_context=f"patch1000 {architecture} {quant} {comparison}",
                    env_overrides=selector_env, execution_identity=expected_execution,
                )
                bench_doc[comparison] = asdict(outcome)
            arch_doc["llama_bench"][quant] = bench_doc

        cells[architecture] = arch_doc

    return {
        "schema_version": 1,
        "producer": "patch1000-verification-v1",
        "patch": _PATCH1000_ID,
        "base_revision": base_revision,
        "compiled_targets": list(architectures),
        "amdgpu_targets": fat_targets,
        "pairs": pairs,
        "models": {
            quant: {"path": str(model), "size_bytes": model.stat().st_size}
            for quant, model in models.items()
        },
        "compositions": {
            "stock": list(stock_composition),
            "control": list(control_composition),
            "subject": list(subject_composition),
        },
        "cells": cells,
    }


def collect_lane_effect_records(
    *, rd08_qualification: "dict[str, object] | None",
    rd73_qualification: "dict[str, object] | None",
) -> list[dict[str, object]]:
    """RV99: the per-lane measurements to persist in the validation record.

    The record used to keep identity, provenance and verdicts but not the
    numbers those verdicts came from -- per-lane effects and their
    ``pair_ratios`` existed only under ``artifacts/``, which is gitignored. An
    interval therefore could not be re-derived, re-aggregated across sessions,
    re-analysed under a new estimator, or audited from committed evidence.

    Normalises the two shapes that actually carry a paired-lane measurement
    into one:

      * RD73's contract qualification returns ``LaneEffect`` dataclasses;
      * RD08's lanes carry ``block_bootstrap_effect()``'s own stats dict
        (``PairedLaneRun.stats``).

    Both already contain the ratio vector -- this only decides to KEEP it.
    ``pair_ratios`` is normalised to a list so a re-read record serialises
    identically to the one that was written (JSON has no tuple), keeping
    ``record_digest`` stable across a load/store round trip.
    """
    records: list[dict[str, object]] = []

    def _add(role: str, metric: str, source: object) -> None:
        if source is None:
            return
        raw = dataclasses.asdict(source) if dataclasses.is_dataclass(source) else dict(source)
        ratios = raw.get("pair_ratios") or ()
        entry: dict[str, object] = {
            "role": raw.get("role") or role,
            "metric": raw.get("metric") or metric,
            "pair_ratios": [float(value) for value in ratios],
        }
        for field in _LANE_EFFECT_FIELDS:
            if raw.get(field) is not None:
                entry[field] = raw[field]
        records.append(entry)

    if rd73_qualification is not None:
        _add("positive", "mtp_wall_tps", rd73_qualification["mtp"].get("effect"))
        _add("control", "decode_tps", rd73_qualification["decode_control"].get("effect"))
    if rd08_qualification is not None:
        for role, lane in (rd08_qualification.get("lanes") or {}).items():
            if isinstance(lane, Mapping):
                _add(role, str(lane.get("metric") or ""), lane.get("stats"))
    return records


def compute_persisted_validation_eligible(
    descriptor: object, validation_verdict: object | None,
    contract_promotions: "dict[str, dict[str, object]] | None",
    *, activation_disposition: str | None, correctness: "dict[str, object] | None",
) -> bool | None:
    """VA14 final slice (GPT req_75c09f14757640af): a bound-contract patch
    is eligible_for_validated_state only when BOTH the adapter verdict
    (compute_verdict() -- validation.toml's own checks) AND every one of
    the patch's bound Experiment Contracts have a passing
    evaluate_promotion_gate() result in ``contract_promotions`` (keyed by
    contract id). Uses the plural ``descriptor.experiment_contracts`` --
    never the singular ``.experiment_contract`` compatibility property,
    which raises for a multi-contract patch. A patch with NO bound contract
    is unaffected -- the adapter verdict alone is the only qualification
    such a patch ever claims, exactly as before.

    RV95: this predicate must also require what evidence.py's
    ``_record_qualifies()`` requires of the record's OWN top-level
    activation/correctness fields, because the two are read as answering the
    same question and previously did not. --run-rd73-contract populated the
    adapter verdict and the contract promotion but left activation/
    correctness at disposition="unknown", so this returned True while
    verify_validated_patch() rejected the very same record with "activation
    is not executed+activation-verified; correctness did not pass". The
    campaign then printed "STATE='validated' eligible: yes" for a record no
    verifier would accept -- a fail-OPEN disagreement in a system whose
    whole contract is to fail closed.

    Keeping the two predicates in sync structurally (rather than by
    convention) is the point: a producer that cannot populate these fields
    now reports ineligible, which is the safe direction. The literals below
    are deliberately the same ones evidence.py:745-754 tests."""
    if not descriptor.experiment_contracts:
        if validation_verdict is None:
            return None
        return validation_verdict.eligible
    if validation_verdict is None or not validation_verdict.eligible:
        return False
    if activation_disposition != "activation-verified":
        return False
    if not isinstance(correctness, Mapping) or correctness.get("disposition") != "passed":
        return False
    promotions = contract_promotions or {}
    return all(
        promotions.get(contract_id, {}).get("passed") is True  # pi-lens-ignore: no-identity-operator-on-literals
        for contract_id in descriptor.experiment_contracts
    )


def build_contract_evidence_for_persistence(
    plan_contracts: "tuple[object, ...]", contract_promotions: "dict[str, dict[str, object]] | None",
) -> "tuple[list[dict[str, str]], dict[str, dict[str, object]]]":
    """VA18 persistence plumbing: derive make_record()'s plural
    ``contracts``/``contract_verdicts`` arguments from
    ``validation_plan.contracts`` (never the singular
    ``descriptor.experiment_contract`` compatibility property, which fails
    closed for a real multi-contract patch) and the existing
    ``contract_promotions`` dict (populated only by --run-rd08-contract
    today). A bound contract with no produced promotion result gets an
    explicit BLOCKED verdict ({"passed": False, "status": "blocked", ...})
    -- never an inferred PASS."""
    promotions = contract_promotions or {}
    contracts = [
        {"id": binding.contract_id, "hash": binding.contract_hash} for binding in plan_contracts
    ]
    verdicts = {
        binding.contract_id: (
            {"passed": bool(promotion.get("passed")), "status": promotion.get("status"), "detail": promotion}
            if (promotion := promotions.get(binding.contract_id)) is not None
            else {"passed": False, "status": "blocked", "detail": {"reasons": ["no promotion result produced"]}}
        )
        for binding in plan_contracts
    }
    return contracts, verdicts


_RD73_RESOURCE_PREFIX = "BIGCHERRY_RD73_RESOURCE"
_RD73_RESOURCE_PATTERN = re.compile(r"BIGCHERRY_RD73_RESOURCE graph_cache_entries=(\d+)\s*$")


def parse_rd73_resource_telemetry(text: str) -> tuple[int, ...]:
    """VA06: pure parser for RD73's opt-in graph-cache-entry telemetry
    (BIGCHERRY_RD73_RESOURCE_TRACE=1, common.cuh's cuda_graph() insertion
    site) -- extracts every real ``graph_cache_entries=N`` reading from a
    process's combined stdout/stderr, in emission order. Returns an empty
    tuple when the patch never emitted (e.g. the control binary, which
    has no RD73 telemetry code at all). Fails closed: any line carrying
    the ``BIGCHERRY_RD73_RESOURCE`` prefix that doesn't match the exact
    expected shape is treated as corrupt evidence, not silently ignored --
    even when other lines in the same text parse cleanly."""
    readings = []
    for line in text.splitlines():
        if _RD73_RESOURCE_PREFIX not in line:
            continue
        match = _RD73_RESOURCE_PATTERN.search(line)
        if match is None:
            raise PatchCampaignError(
                f"rd73 resource telemetry: malformed BIGCHERRY_RD73_RESOURCE line: {line!r}"
            )
        readings.append(int(match.group(1)))
    return tuple(readings)


def peak_rd73_resource_result(
    subject_readings: "tuple[int, ...] | list[int]",
    control_readings: "tuple[int, ...] | list[int] | None" = None,
) -> "experiment_contract.ResourceResult":
    """VA06: reduce raw telemetry readings into the real
    ``ResourceResult(metric="graph_cache_entries", unit="count", ...)``
    evaluate_resource_gate() checks against. Fails closed (raises) on
    missing/malformed evidence -- an empty subject reading set means the
    real telemetry was never observed at all, which must never silently
    read as a zero/passing measurement."""
    from bigcherry.experiment import contract as experiment_contract

    if not subject_readings:
        raise PatchCampaignError(
            "rd73 resource evidence: no graph_cache_entries readings observed -- the "
            "subject binary's telemetry (BIGCHERRY_RD73_RESOURCE_TRACE=1) never emitted"
        )
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in subject_readings):
        raise PatchCampaignError(
            f"rd73 resource evidence: malformed subject reading(s) in {subject_readings!r}"
        )
    control_value = None
    if control_readings:
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in control_readings):
            raise PatchCampaignError(
                f"rd73 resource evidence: malformed control reading(s) in {control_readings!r}"
            )
        control_value = float(max(control_readings))
    return experiment_contract.ResourceResult(
        metric="graph_cache_entries", unit="count",
        subject_value=float(max(subject_readings)), control_value=control_value,
    )


def evaluate_rd73_activation_evidence(
    *, marker_regex: str, control_log_path: Path, subject_log_path: Path, run_dir: Path,
) -> dict[str, object]:
    """VA06 (user redirect, 2026-09-01): RD73's real subject-hit/control-miss
    activation evidence, read from the control/subject llama-server LOG
    FILES run_rd73_mtp_server_lane() already produced (BIGCHERRY_PATCH_TRACE=1
    is always set on those servers) -- no separate llama-bench probe.
    llama-bench itself has proven unworkable for RD73's real 27B/dual-GPU/
    -sm-tensor config on real hardware (repeated crashes: OOM under
    resource contention, --fit argument-parse errors), and a second probe
    would be redundant anyway: the MTP servers already ran the patched/
    control binaries under real repeated traffic. Mirrors RD08's own
    control-vs-subject-binary negative control (never the generic tune-
    binary/GGML_CUDA_DISABLE_FUSION mechanism, which is invalid for RD73's
    graph-cache-key marker for the same reason RD08's own docstring
    already establishes)."""
    pattern = re.compile(marker_regex)
    subject_text = Path(subject_log_path).read_text(encoding="utf-8", errors="replace")
    control_text = Path(control_log_path).read_text(encoding="utf-8", errors="replace")
    subject_hit = pattern.search(subject_text) is not None
    control_hit = pattern.search(control_text) is not None
    subject_rel = Path(subject_log_path).relative_to(run_dir).as_posix()
    control_rel = Path(control_log_path).relative_to(run_dir).as_posix()
    doc = {
        "marker_regex": marker_regex, "subject_hit": subject_hit, "control_hit": control_hit,
        "positive": {
            "artifact": {
                "path": subject_rel,
                "sha256": hashlib.sha256(Path(subject_log_path).read_bytes()).hexdigest(),
            },
        },
        "control": {
            "artifact": {
                "path": control_rel,
                "sha256": hashlib.sha256(Path(control_log_path).read_bytes()).hexdigest(),
            },
        },
    }
    artifact_ref = _write_bound_artifact(run_dir, "rd73-activation.json", doc)
    return {
        "subject_hit": subject_hit, "control_hit": control_hit, "artifact": artifact_ref,
        "subject_log_path": subject_rel, "control_log_path": control_rel,
        # VA23: the per-log bound refs, so the campaign can build the
        # positive/negative trace_evidence that _builtin_trace_marker()
        # requires. It re-reads both logs and re-verifies the marker itself,
        # so this exposes evidence for independent checking rather than
        # asserting a result -- subject_hit/control_hit above are NOT what
        # the validator trusts.
        "positive": {"artifact": doc["positive"]["artifact"], "marker_regex": marker_regex},
        "negative": {"artifact": doc["control"]["artifact"], "marker_regex": marker_regex},
    }


def run_rd73_mtp_server_lane(
    *, control_binary: Path, subject_binary: Path, model: Path, corpus_path: Path, run_dir: Path,
    expected_execution: ExecutionIdentity,
    host: str = "127.0.0.1", control_port: int = 18080, subject_port: int = 18081,
    spec_draft_n_max: int = 4,
    n_predict: int = 128, warmup_pairs: int = 2, measured_pairs: int = 10,
    selector_env: "dict[str, str] | None" = None,
) -> dict[str, object]:
    """VA06 next slice: RD73's paired control/subject mtp_verify
    performance lane over a real llama-server HTTP harness (GPT scoping,
    session ses_89a3ef2b02b94469, req_a25bb805975c43c0/req corrected):
    upstream llama-bench does not support speculative/MTP flags at all,
    so unlike RD08's simple paired-subprocess lanes, this reuses
    tuning/server_runner.py's ServerRunner for real process lifecycle
    (launch/health-check/shutdown) and bench/server_completion.py's real
    request/metrics machinery for each measured sample.

    Target metric is wall_tps (client-measured, real request-to-response
    wall-clock throughput) -- deliberately NOT predicted_tps, which is
    the server's own self-reported decode timing and can exclude HTTP/
    queueing overhead; per GPT direction, "the number an end user
    actually experiences" is what this contract's end_to_end_gain_pct
    must measure.

    Reuses experiment/execution.py's run_paired_lane() (RD08's own
    alternating-order + block-bootstrap statistics engine) via a
    synthetic-stdout adapter rather than duplicating that statistics
    code: each paired-lane "command" is a control/subject arm tag, and
    the injected runner performs one real HTTP completion request against
    the already-launched server for that arm, encoding the real wall_tps
    it measured into a parseable stdout line. warmup_pairs real paired
    requests execute first (cold-cache discipline, matching
    server_completion.run_session()'s own pattern) and are never fed into
    the paired statistics; only the following measured_pairs are.

    Every per-request record (including generated ``content``) is
    retained and returned for RD73's separate bit-identical correctness
    lane to consume -- this function does not itself judge correctness.
    Fails closed: a request with no usable wall_tps raises
    PatchCampaignError immediately, never silently drops a sample."""
    from bigcherry.bench import server_completion as sc
    from bigcherry.experiment import execution as experiment_execution

    prompts, corpus_sha256 = sc.load_corpus(corpus_path)
    metric_pattern = re.compile(r"BIGCHERRY_RD73_MTP wall_tps=([0-9.]+)")

    # Real llama-server CLI flags only (verified against vendor/llama.cpp's
    # own common/arg.cpp -- an earlier draft of this function invented
    # "--spec-n-max"/"--spec-draft-k"/"--spec-draft-v", none of which
    # exist; the real flag is --spec-draft-n-max, and there is no
    # separate draft-cache-type flag this lane needs to set (the
    # production dual-XTX/27B baseline profile leaves cache types at
    # their defaults too). -sm tensor is REQUIRED for this 27B model on
    # 2x gfx1100 -- the default -sm layer understates throughput by
    # roughly 2-10x (a real, previously-confirmed production finding).
    # --fit off is ALSO required alongside -sm tensor for llama-SERVER
    # specifically: llama.cpp's automatic device-memory-fit feature
    # (default on) raises "llama_params_fit is not implemented for
    # SPLIT_MODE_TENSOR" and aborts (common/fit.cpp) -- a real hardware
    # crash found running this exact lane on Brutus. llama-BENCH (used
    # by RD73's other lanes) does not register this flag at all --
    # passing --fit to it is itself a hard error ("invalid parameter for
    # argument: --fit"), also found on real hardware -- so it must never
    # be added to those lanes' extra_flags.
    server_args = (
        "--parallel", "1", "--metrics", "-sm", "tensor", "--fit", "off",
        "--spec-type", "draft-mtp", "--spec-draft-n-max", str(spec_draft_n_max),
    )
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # User redirect (2026-09-01, real hardware finding): control and
    # subject servers must NEVER run concurrently for this model --
    # each needs ~13GB/GPU under -sm tensor split, and two full copies
    # exceed the 24.5GB/GPU Brutus dual-XTX cards (a real cudaMalloc
    # out-of-memory abort, confirmed on hardware). So each single
    # measured/warmup request gets its own fresh server: launch, one
    # request, shut down -- alternating control/subject in the same
    # order run_paired_lane already calls them, preserving the real
    # alternating-order/thermal-drift discipline this project's own
    # prior production benchmarking found necessary (a non-alternating
    # "all control then all subject" design previously produced a real,
    # since-corrected measurement artifact on this exact model/hardware
    # -- see patches/1233.../README.md's "Historical evidence" section).
    #
    # PVPS02 step 7 (2026-09-11): ``selector_env`` carries the already-
    # validated HIP_VISIBLE_DEVICES value from
    # run_rd73_contract_qualification()'s single require_device_visibility()
    # call (validate ONCE, not per-lane/per-request) -- ServerRunner
    # starts each server from ambient env then applies these overrides,
    # so this is enough to make the selector explicit for every server
    # this lane launches without touching global os.environ.
    rd73_env = {"BIGCHERRY_PATCH_TRACE": "1", "BIGCHERRY_RD73_RESOURCE_TRACE": "1"}
    if selector_env:
        rd73_env.update(selector_env)
    # GPT review follow-up (req_d1ef22d846854960, 2026-09-11): defense in
    # depth alongside env_unset=_ROCR_VISIBLE_DEVICES_UNSET below -- a
    # future direct caller passing ROCR_VISIBLE_DEVICES via selector_env
    # must not be able to reintroduce it through env_overrides ordering
    # (ServerRunner.launch() applies env_unset BEFORE env_overrides).
    rd73_env.pop("ROCR_VISIBLE_DEVICES", None)
    ports = {"control": control_port, "subject": subject_port}
    binaries = {"control": control_binary, "subject": subject_binary}
    per_request_logs: dict[str, list[Path]] = {"control": [], "subject": []}

    sampling = sc.SamplingConfig(temperature=1.0, top_p=0.95, top_k=20)
    session_kwargs = dict(
        corpus_id=corpus_path.stem, corpus_sha256=corpus_sha256, bigcherry_revision="rd73-va06",
        llama_pin="", llama_revision="", model_id=str(model), server_argv=server_args,
        spec_type="draft-mtp", spec_n_max=spec_draft_n_max,
        # SessionConfig's spec_draft_k/spec_draft_v fields are provenance
        # labels only (there is no real --spec-draft-k/--spec-draft-v
        # llama-server flag); "default" records that this lane leaves the
        # draft cache type at its build default, matching the production
        # dual-XTX/27B baseline profile, which does not override it either.
        spec_draft_k="default", spec_draft_v="default",
        sampling=sampling, n_predict=n_predict, order_seed=12345,
    )
    configs = {
        "control": sc.SessionConfig(session_id="rd73-mtp-control", **session_kwargs),
        "subject": sc.SessionConfig(session_id="rd73-mtp-subject", **session_kwargs),
    }

    request_records: dict[str, list[dict[str, object]]] = {"control": [], "subject": []}
    request_counters = {"control": 0, "subject": 0}

    def _runner(command: list[str]) -> "experiment_execution.RunnerOutput":
        arm = command[-1]
        index = request_counters[arm]
        request_counters[arm] += 1
        log_path = logs_dir / f"rd73-mtp-{arm}-server-{index}.log"
        per_request_logs[arm].append(log_path)
        session = AttestedServerSession(
            binary=binaries[arm], model=model, expected=expected_execution,
            host=host, port=ports[arm],
            extra_args=server_args, log_path=log_path, env_overrides=rd73_env,
            env_unset=_ROCR_VISIBLE_DEVICES_UNSET,
        )
        with session:
            transport = sc.HttpTransport(f"http://{host}:{ports[arm]}")
            sc.validate_server(transport)
            prompt = prompts[index % len(prompts)]
            record = sc.run_request(transport, prompt, configs[arm], pass_number=1, order_index=index)
        request_records[arm].append(record)
        if not isinstance(record.get("wall_tps"), (int, float)):
            raise PatchCampaignError(
                f"rd73 mtp lane ({arm}, request {index}): no usable wall_tps in the "
                f"real completion response -- refusing to feed a missing sample into "
                f"the paired statistics"
            )
        return experiment_execution.RunnerOutput(
            returncode=0, stdout=f"BIGCHERRY_RD73_MTP wall_tps={record['wall_tps']}\n", stderr="",
        )

    for _ in range(warmup_pairs):
        _runner(["rd73-mtp-lane", "control"])
        _runner(["rd73-mtp-lane", "subject"])

    paired_run = experiment_execution.run_paired_lane(
        metric="mtp_wall_tps", control_command=["rd73-mtp-lane", "control"],
        subject_command=["rd73-mtp-lane", "subject"], pattern=metric_pattern,
        pairs=measured_pairs, runner=_runner,
    )

    # Concatenate each arm's per-request server logs into one combined
    # log file, so downstream evidence readers (correctness/activation
    # investigation, manual debugging) see one file per arm as before,
    # even though each request used its own fresh process.
    combined_log_paths: dict[str, Path] = {}
    for arm in ("control", "subject"):
        combined_path = logs_dir / f"rd73-mtp-{arm}-server.log"
        combined_path.write_text(
            "".join(p.read_text(encoding="utf-8", errors="replace") for p in per_request_logs[arm]),
            encoding="utf-8",
        )
        combined_log_paths[arm] = combined_path

    effect = experiment_execution.lane_effect_from_run("positive", "mtp_wall_tps", paired_run)
    doc = {
        "metric": "mtp_wall_tps", "stats": paired_run.stats,
        "warmup_pairs": warmup_pairs, "measured_pairs": measured_pairs,
        "control_requests": request_records["control"], "subject_requests": request_records["subject"],
    }
    artifact_ref = _write_bound_artifact(run_dir, "rd73-mtp-lane.json", doc)
    return {
        "effect": effect, "artifact": artifact_ref, "stats": paired_run.stats,
        "control_requests": request_records["control"], "subject_requests": request_records["subject"],
        "control_log_path": combined_log_paths["control"], "subject_log_path": combined_log_paths["subject"],
    }


# VA26: run_bench_runner_server_bench() and its constants moved to
# campaign/bench_runner.py -- the documented server-bench harness is not
# patch-specific, and the qualification matrix needs it without importing
# patch internals. Imported below; no alias is kept here.
def run_rd73_decode_control_lane(
    *, control_binary: Path, subject_binary: Path, model: Path, run_dir: Path,
    expected_execution: ExecutionIdentity,
    host: str = "127.0.0.1", control_port: int = 18082, subject_port: int = 18083,
    pairs: int = 3, extra_flags: tuple[str, ...] = ("-sm", "tensor", "--fit", "off"),
    selector_env: "dict[str, str] | None" = None,
) -> dict[str, object]:
    """VA06 (user redirect, 2026-09-01): RD73's decode control lane --
    launches real, plain (non-speculative) control/subject llama-server
    processes (ServerRunner, matching run_rd73_mtp_server_lane()'s
    lifecycle pattern) and drives each paired measurement via the
    documented Brutus bench runner (run_bench_runner_server_bench(),
    "tg128" config) rather than a raw llama-bench subprocess -- see that
    function's docstring for why llama-bench itself is unworkable here.
    Reuses experiment/execution.py's run_paired_lane() via the same
    synthetic-stdout adapter pattern as the MTP lane, rather than
    duplicating its alternating-order + block-bootstrap statistics."""
    from bigcherry.experiment import execution as experiment_execution

    metric_pattern = re.compile(r"BIGCHERRY_RD73_DECODE tg128_tps=([0-9.]+)")
    server_args = ("--parallel", "1", *extra_flags)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Real hardware finding (2026-09-01): control and subject servers
    # must never run concurrently for this model -- each needs
    # ~13GB/GPU under -sm tensor, exceeding the 24.5GB/GPU cards
    # together (a real cudaMalloc OOM abort). One fresh server per
    # single bench-runner call, alternating arms, mirrors
    # run_rd73_mtp_server_lane()'s same fix.
    ports = {"control": control_port, "subject": subject_port}
    binaries = {"control": control_binary, "subject": subject_binary}
    request_counters = {"control": 0, "subject": 0}
    raw_metrics: dict[str, list[dict[str, float]]] = {"control": [], "subject": []}

    def _runner(command: list[str]) -> "experiment_execution.RunnerOutput":
        arm = command[-1]
        index = request_counters[arm]
        request_counters[arm] += 1
        session = AttestedServerSession(
            binary=binaries[arm], model=model, expected=expected_execution,
            host=host, port=ports[arm],
            extra_args=server_args, log_path=logs_dir / f"rd73-decode-{arm}-server-{index}.log",
            env_overrides=_hip_only(selector_env),
            env_unset=_ROCR_VISIBLE_DEVICES_UNSET,
        )
        with session:
            metrics = run_bench_runner_server_bench(
                server_url=f"http://{host}:{ports[arm]}", bench_configs="tg128", repetitions=1,
            )
        raw_metrics[arm].append(metrics)
        if "tg128_tps" not in metrics:
            raise PatchCampaignError(
                f"rd73 decode control lane ({arm}): bench runner produced no tg128_tps "
                f"metric (got {sorted(metrics)})"
            )
        return experiment_execution.RunnerOutput(
            returncode=0, stdout=f"BIGCHERRY_RD73_DECODE tg128_tps={metrics['tg128_tps']}\n", stderr="",
        )

    decode_run = experiment_execution.run_paired_lane(
        metric="tg128", control_command=["rd73-decode-lane", "control"],
        subject_command=["rd73-decode-lane", "subject"], pattern=metric_pattern,
        pairs=pairs, runner=_runner,
    )

    effect = experiment_execution.lane_effect_from_run("control", "tg128", decode_run)
    doc = {
        "metric": "tg128", "stats": decode_run.stats,
        "control_raw_metrics": raw_metrics["control"], "subject_raw_metrics": raw_metrics["subject"],
        "runs": list(decode_run.runs),
    }
    artifact_ref = _write_bound_artifact(run_dir, "rd73-decode-control.json", doc)
    return {"effect": effect, "artifact": artifact_ref, "stats": decode_run.stats}


def evaluate_rd73_resource_evidence(
    *, subject_log_path: Path, run_dir: Path,
) -> dict[str, object]:
    """VA06 (user redirect, 2026-09-01): RD73's real graph-cache-entries
    resource evidence, read from the subject llama-server LOG FILE
    run_rd73_mtp_server_lane() already produced
    (BIGCHERRY_RD73_RESOURCE_TRACE=1 is always set on that server) --
    no separate llama-bench probe. Subject-only (GPT's phase-1 scoping:
    the contract's resource_limits only bounds max_value, so no paired
    control reading is needed). Parses every real graph_cache_entries=N
    reading (parse_rd73_resource_telemetry(), fails closed on any
    malformed line) and reduces to a peak ResourceResult
    (peak_rd73_resource_result())."""
    subject_text = Path(subject_log_path).read_text(encoding="utf-8", errors="replace")
    readings = parse_rd73_resource_telemetry(subject_text)
    result = peak_rd73_resource_result(readings)
    subject_rel = Path(subject_log_path).relative_to(run_dir).as_posix()
    doc = {
        "readings": list(readings), "peak": result.subject_value,
        "artifact": {
            "path": subject_rel,
            "sha256": hashlib.sha256(Path(subject_log_path).read_bytes()).hexdigest(),
        },
    }
    artifact_ref = _write_bound_artifact(run_dir, "rd73-resource.json", doc)
    return {"result": result, "artifact": artifact_ref, "readings": readings}


def run_rd73_resource_burst_session(
    *, subject_binary: Path, model: Path, corpus_path: Path, run_dir: Path,
    expected_execution: ExecutionIdentity,
    host: str = "127.0.0.1", port: int = 18084, burst_requests: int = 20, n_predict: int = 32,
    selector_env: "dict[str, str] | None" = None,
) -> dict[str, object]:
    """VA06 (real hardware finding, 2026-09-01): RD73's graph-cache-entries
    resource evidence needs a real accumulated-cache burst -- repeated
    requests against ONE long-lived subject server (matching the
    contract's own documented methodology: "a fixed repeated-shape MTP
    completion burst", patches/1233.../README.md). This is NOT compatible
    with run_rd73_mtp_server_lane()'s per-request server restart (needed
    there to avoid a real control+subject concurrent-VRAM OOM): a fresh
    process resets the in-memory graph cache every single request, so
    that lane's own combined logs would only ever show a trivial
    peak (~1), never the real accumulated cache size the contract's
    max_value=800 bound was calibrated against (subject peak 651 under
    VA06's original characterization run). This session is subject-only
    (no concurrent control server), so it needs no restart discipline --
    launch once, drive burst_requests real repeated requests against the
    SAME live process, read the resulting log, shut down."""
    from bigcherry.bench import server_completion as sc

    prompts, _ = sc.load_corpus(corpus_path)
    burst_prompt = prompts[0]
    server_args = (
        "--parallel", "1", "--metrics", "-sm", "tensor", "--fit", "off",
        "--spec-type", "draft-mtp", "--spec-draft-n-max", "4",
    )
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "rd73-resource-burst-subject-server.log"
    rd73_env = {"BIGCHERRY_PATCH_TRACE": "1", "BIGCHERRY_RD73_RESOURCE_TRACE": "1"}
    if selector_env:
        rd73_env.update(selector_env)
    # GPT review follow-up (req_d1ef22d846854960, 2026-09-11): defense in
    # depth alongside env_unset=_ROCR_VISIBLE_DEVICES_UNSET below -- a
    # future direct caller passing ROCR_VISIBLE_DEVICES via selector_env
    # must not be able to reintroduce it through env_overrides ordering
    # (ServerRunner.launch() applies env_unset BEFORE env_overrides).
    rd73_env.pop("ROCR_VISIBLE_DEVICES", None)
    session = AttestedServerSession(
        binary=subject_binary, model=model, expected=expected_execution,
        host=host, port=port,
        extra_args=server_args, log_path=log_path, env_overrides=rd73_env,
        env_unset=_ROCR_VISIBLE_DEVICES_UNSET,
    )
    sampling = sc.SamplingConfig(temperature=1.0, top_p=0.95, top_k=20)
    config = sc.SessionConfig(
        session_id="rd73-resource-burst", corpus_id=corpus_path.stem, corpus_sha256="",
        bigcherry_revision="rd73-va06", llama_pin="", llama_revision="", model_id=str(model),
        server_argv=server_args, spec_type="draft-mtp", spec_n_max=4,
        spec_draft_k="default", spec_draft_v="default", sampling=sampling,
        n_predict=n_predict, order_seed=12345,
    )
    with session:
        transport = sc.HttpTransport(f"http://{host}:{port}")
        sc.validate_server(transport)
        for index in range(burst_requests):
            sc.run_request(transport, burst_prompt, config, pass_number=1, order_index=index)

    return evaluate_rd73_resource_evidence(subject_log_path=log_path, run_dir=run_dir)


class Rd73CorrectnessError(PatchCampaignError):
    """RD73's bit-identical correctness check found a real content
    mismatch -- distinct from PatchCampaignError's other, infrastructure-
    level failure modes only in name (still fails the campaign)."""


def evaluate_rd73_mtp_correctness(
    *, control_requests: list[dict[str, object]], subject_requests: list[dict[str, object]],
    run_dir: Path,
) -> dict[str, object]:
    """VA06 next slice: RD73's bit-identical correctness check, evaluated
    from run_rd73_mtp_server_lane()'s already-retained per-request
    ``content`` fields -- reuses the exact same real MTP requests already
    executed for the performance lane; never launches a second server
    lane just for correctness. Pairs control/subject requests by
    order_index (both arms ran the identical corpus/order, so index
    alignment is real pairing, not a coincidence) and requires EXACT
    string equality -- no trimming/normalization/tolerance. Fails closed
    on a mismatch, missing/non-string content, or an unpairable
    (differently-sized) record set; never silently skips a bad pair."""
    if len(control_requests) != len(subject_requests):
        raise Rd73CorrectnessError(
            f"rd73 correctness: control has {len(control_requests)} request(s) but subject "
            f"has {len(subject_requests)} -- cannot pair records for comparison"
        )
    rows: list[dict[str, object]] = []
    mismatches: list[str] = []
    for control_record, subject_record in zip(control_requests, subject_requests):
        control_index = control_record.get("order_index")
        subject_index = subject_record.get("order_index")
        if control_index != subject_index:
            raise Rd73CorrectnessError(
                f"rd73 correctness: control/subject request order_index mismatch "
                f"({control_index!r} vs {subject_index!r}) -- records are not aligned"
            )
        control_content = control_record.get("content")
        subject_content = subject_record.get("content")
        if not isinstance(control_content, str) or not isinstance(subject_content, str):
            raise Rd73CorrectnessError(
                f"rd73 correctness: request order_index={control_index!r} has non-string "
                f"content (control={type(control_content).__name__}, "
                f"subject={type(subject_content).__name__}) -- cannot compare"
            )
        ok = control_content == subject_content
        if not ok:
            mismatches.append(f"order_index={control_index!r}")
        rows.append({"order_index": control_index, "ok": ok})
    if mismatches:
        raise Rd73CorrectnessError(
            f"rd73 correctness: {len(mismatches)} request(s) mismatched: {', '.join(mismatches)}"
        )
    # VA23: "ops" is what _builtin_backend_ops() matches against the check's
    # declared config. patch 1233's validation.toml declares
    # ops = ["RD73_MTP_BIT_IDENTICAL"] for its correctness check, so the
    # producer must emit that exact identifier or the validator cannot tell
    # this artifact apart from any other correctness evidence. "passed" stays
    # the real comparison outcome; only the identifier is added.
    doc = {
        "check": "bit_identical", "passed": True, "rows": rows,
        "ops": ["RD73_MTP_BIT_IDENTICAL"],
    }
    artifact_ref = _write_bound_artifact(run_dir, "rd73-correctness.json", doc)
    return {"artifact": artifact_ref, "rows": rows}


def run_rd73_contract_qualification(
    *, contract: object, control_server_binary: Path, subject_server_binary: Path, model: Path,
    marker_regex: str, corpus_path: Path, run_dir: Path,
    # VA24: decode_pairs raised 3 -> 10 to match measured_pairs. min_paired_rounds
    # is a minimum VALIDITY requirement for every interval used to establish an
    # acceptance bound, so a control decision taken on 3 rounds violates exactly
    # what a contract declaring 10 claims. The previous asymmetry meant RD73
    # demanded 10 rounds to prove its own gain while accepting 3 to prove it had
    # broken nothing. "Controls need less evidence" is not defensible as a
    # general rule: required sample size depends on variance, distance from the
    # acceptance boundary, and the estimator -- not on lane role (dev-gpt-agent,
    # req_875d13b29a204075). Costs ~5 extra minutes on a ~15-minute
    # qualification, measured.
    decode_pairs: int = 10, warmup_pairs: int = 2, measured_pairs: int = 10,
    # RV99: the patch's already-committed validation records, each a prior
    # measurement SESSION. Required, not defaulted: under a session policy a
    # caller that forgets them silently under-counts sessions and the gate
    # reports "collect more" for ever. An empty tuple is the honest value for
    # a first session, and is meaningless under a non-session policy.
    prior_session_records: "Iterable[Mapping[str, object]]",
    # Required: session aggregation must pool only same-hardware sessions,
    # and this is the run's own architecture.
    amdgpu_targets: str,
) -> dict[str, object]:
    """VA06 next slice: the authoritative RD73 full-qualification path
    (``--run-rd73-contract``), mirroring RD08's own
    run_rd08_contract_qualification() result/schema/promotion semantics
    (real lane execution + real correctness + real trigger proof, composed
    via evaluate_promotion_gate()) -- no RD73-specific parallel gate model.
    Every threshold comes from ``contract`` itself
    (aggregate_contract_effects() / evaluate_resource_gate() /
    evaluate_promotion_gate()); nothing here hardcodes a number.

    User redirect (2026-09-01): every lane now runs entirely over real
    llama-server processes driven by HTTP requests / the documented
    Brutus bench runner -- never a raw llama-bench subprocess, which
    proved unworkable for RD73's real 27B/dual-GPU/-sm-tensor config on
    real hardware. The MTP performance lane runs first; its own
    control/subject server log files (BIGCHERRY_PATCH_TRACE=1 is always
    set on them) are the real source for activation evidence. Resource
    evidence uses a SEPARATE subject-only burst session
    (run_rd73_resource_burst_session()): the MTP lane restarts a fresh
    server per single request (a real hardware constraint -- control and
    subject cannot run concurrently, each needs ~13GB/GPU and two copies
    exceed the 24.5GB/GPU cards), which resets the in-memory graph cache
    every request and would make a peak reading trivial/meaningless;
    the burst session's one long-lived process gets the real
    accumulated-cache reading the contract's resource bound needs.
    Decode control similarly launches its own control/subject servers
    one-request-at-a-time (never concurrently) for the same VRAM
    reason."""
    from bigcherry.experiment import contract as experiment_contract
    from bigcherry.experiment.execution import (
        DeviceVisibilityError as _DeviceVisibilityError,
        require_device_visibility as _require_device_visibility,
    )

    # VA25: RD73's dual-XTX qualification is always `-sm tensor` across 2
    # homogeneous devices of this run's own architecture -- the exact
    # topology every RD73 server lane's docstring already documents as a
    # real hardware constraint (concurrent control+subject exceeds VRAM,
    # etc.). Constructed once here, not per-lane, so all three lanes
    # attest against the identical expectation.
    expected_execution = ExecutionIdentity(
        backend="ROCm", architectures=(amdgpu_targets, amdgpu_targets),
    )

    # PVPS02 step 7 (2026-09-11): validate the real HIP_VISIBLE_DEVICES
    # selector ONCE here (the same fail-closed contract RD58/the generic
    # matrix use), then copy that validated value into every lane's own
    # server-session env overrides below -- ServerRunner starts each
    # server from ambient env then applies overrides, so this makes the
    # selector explicit for every process this qualification launches
    # without ever mutating global os.environ. ExecutionIdentity above
    # remains the separate architecture/device-count attestation this
    # was already doing; selector validation and attestation are
    # deliberately two different checks, not merged into one.
    #
    # Real-hardware finding (2026-09-11, PNRO17): only HIP_VISIBLE_DEVICES
    # is set here -- also setting ROCR_VISIBLE_DEVICES to the same value
    # actively breaks non-prefix-from-0 device selection (confirmed on
    # real gfx1100/gfx1201/gfx1030 hardware; see require_device_visibility()
    # and DeviceVisibility's own docstrings for the full mechanism).
    try:
        rd73_visibility = _require_device_visibility(
            context=f"--run-rd73-contract ({amdgpu_targets})", exact_count=2,
        )
    except _DeviceVisibilityError as exc:
        raise PatchCampaignError(str(exc)) from exc
    rd73_selector_env = {"HIP_VISIBLE_DEVICES": rd73_visibility.hip_visible_devices}

    mtp = run_rd73_mtp_server_lane(
        control_binary=control_server_binary, subject_binary=subject_server_binary, model=model,
        corpus_path=corpus_path, run_dir=run_dir, expected_execution=expected_execution,
        warmup_pairs=warmup_pairs, measured_pairs=measured_pairs,
        selector_env=rd73_selector_env,
    )
    activation = evaluate_rd73_activation_evidence(
        marker_regex=marker_regex, control_log_path=mtp["control_log_path"],
        subject_log_path=mtp["subject_log_path"], run_dir=run_dir,
    )
    resource = run_rd73_resource_burst_session(
        subject_binary=subject_server_binary, model=model, corpus_path=corpus_path, run_dir=run_dir,
        expected_execution=expected_execution, selector_env=rd73_selector_env,
    )
    decode_control = run_rd73_decode_control_lane(
        control_binary=control_server_binary, subject_binary=subject_server_binary, model=model,
        run_dir=run_dir, expected_execution=expected_execution, pairs=decode_pairs,
        selector_env=rd73_selector_env,
    )
    # A real content mismatch (or a missing/non-string/unpaired record) is a
    # genuine correctness RESULT, not an infrastructure failure -- it must
    # flow into correctness_gate/promotion as passed=False, never abort the
    # whole qualification run (mirrors RD08's Rd08CorrectnessError handling
    # in run_rd08_contract_correctness()).
    try:
        correctness = evaluate_rd73_mtp_correctness(
            control_requests=mtp["control_requests"], subject_requests=mtp["subject_requests"],
            run_dir=run_dir,
        )
        correctness_result = experiment_contract.CorrectnessResult(check="bit_identical", passed=True)
    except Rd73CorrectnessError as exc:
        correctness = {"artifact": None, "rows": [], "error": str(exc)}
        correctness_result = experiment_contract.CorrectnessResult(
            check="bit_identical", passed=False, detail=str(exc),
        )
    correctness_gate = compute_contract_correctness_gate(contract, {"bit_identical": correctness_result})
    aggregated_effects = experiment_contract.aggregate_contract_effects(
        contract, [mtp["effect"], decode_control["effect"]], target_metric="mtp_wall_tps",
    )
    # RV99: under a session policy the gain bound is established across
    # repeated SESSIONS, not from the pairs inside this one run. Fold the
    # prior sessions' persisted lane effects together with the one just
    # measured and re-aggregate over all of them.
    #
    # prior_session_records are the patch's already-committed validation
    # records; this run's own measurement is appended last, so the gate always
    # sees every valid session including the current one. Nothing is selected
    # or dropped -- aggregate_session_effects() consumes them all, and the
    # stopping rule decides whether that is yet enough.
    #
    # Only the gain field is re-aggregated. The control-regression budget is a
    # per-run property (this build must not have broken the control lane in
    # THIS run), not a claim being established across occasions.
    if contract.acceptance.effect_evidence_policy == "session_ci95_threshold_bound_v1":
        # The stub must carry gpu_architectures like a real record does, or
        # aggregate_session_effects' hardware filter would drop the very
        # session just measured.
        this_session = {
            "gpu_architectures": [amdgpu_targets],
            "lane_effects": collect_lane_effect_records(
                rd08_qualification=None,
                rd73_qualification={"mtp": mtp, "decode_control": decode_control},
            ),
        }
        gain_field = (
            "end_to_end_gain_pct" if contract.acceptance.end_to_end_gain_pct is not None
            else "target_kernel_gain_pct"
        )
        aggregated_effects = dict(aggregated_effects)
        aggregated_effects.update(experiment_contract.aggregate_session_effects(
            [*prior_session_records, this_session],
            field=gain_field, role="positive", metric="mtp_wall_tps",
            # Only sessions measured on THIS hardware may be pooled.
            architectures=[amdgpu_targets],
        ))
    resource_gate = experiment_contract.evaluate_resource_gate(
        contract, {"graph_cache_entries": resource["result"]},
    )
    trigger_proof = experiment_contract.evaluate_trigger_proof([
        experiment_contract.TriggerEvidence(
            role="positive", lane_id="rd73-mtp-subject",
            candidate_launches=1 if activation["subject_hit"] else 0,
        ),
    ])
    if activation["control_hit"]:
        trigger_proof = {
            "passed": False,
            "reasons": list(trigger_proof.get("reasons") or []) + [
                "control-role lane observed the target marker -- the negative "
                "control is invalid, so trigger proof cannot be trusted"
            ],
            "checked_lanes": trigger_proof.get("checked_lanes", 0),
            "untriggered_lanes": list(trigger_proof.get("untriggered_lanes") or []),
        }
    promotion = experiment_contract.evaluate_promotion_gate(
        contract, correctness_gate=correctness_gate, aggregated_effects=aggregated_effects,
        trigger_proof=trigger_proof, resource_gate=resource_gate,
    )
    qualification_doc = {
        "contract_id": contract.id, "contract_hash": contract.contract_hash,
        "activation_artifact": activation["artifact"], "mtp_lane_artifact": mtp["artifact"],
        "decode_control_artifact": decode_control["artifact"], "resource_artifact": resource["artifact"],
        "correctness_artifact": correctness["artifact"],
        "correctness_gate": correctness_gate, "aggregated_effects": aggregated_effects,
        "resource_gate": resource_gate, "trigger_proof": trigger_proof, "promotion": promotion,
    }
    artifact_ref = _write_bound_artifact(run_dir, "rd73-contract-qualification.json", qualification_doc)

    # VA23: emit the generic-adapter performance artifact.
    #
    # _builtin_benchmark() (patch/validation.py) requires an artifact with a
    # non-empty "metrics" dict; without one the declared performance/controls
    # checks ERROR with "benchmark artifact requires non-empty metrics", and
    # patch-verify-evidence then reports "no recorded benchmark execution"
    # even though a real, paired, bootstrapped benchmark demonstrably ran.
    # That is a false negative in the direction that HIDES real results.
    #
    # This is a faithful projection of already-measured values into the
    # schema the generic validator reads -- the same thing RD58's producer does via its controls_doc. Nothing here is
    # computed for the first time, and nothing is invented: every number
    # below is copied from the lane effects the contract gate itself just
    # consumed. The lane artifacts remain the authoritative record and stay
    # separately hash-bound; this document references them rather than
    # replacing them.
    performance_doc = {
        "campaign_id": contract.contract_hash,
        "passed": bool(promotion.get("passed")),
        "contract_id": contract.id,
        "target_metric": "mtp_wall_tps",
        "metrics": {
            # LaneEffect is a frozen dataclass (experiment/contract.py);
            # dataclasses.asdict() is its faithful serialisation, so the
            # recorded fields are exactly the measured
            # role/metric/geometric_effect_pct/decision the contract gate
            # consumed -- not a re-derivation.
            "mtp_verify": {
                "effect": dataclasses.asdict(mtp["effect"]),
                "artifact": mtp["artifact"],
            },
            "decode_control": {
                "effect": dataclasses.asdict(decode_control["effect"]),
                "artifact": decode_control["artifact"],
            },
            "aggregated_effects": aggregated_effects,
        },
        "promotion": promotion,
    }
    performance_artifact = _write_bound_artifact(
        run_dir, "rd73-performance.json", performance_doc,
    )
    return {
        "performance_artifact": performance_artifact,
        "activation": activation, "mtp": mtp, "decode_control": decode_control,
        "resource": resource, "correctness": correctness,
        "correctness_gate": correctness_gate, "aggregated_effects": aggregated_effects,
        # VA23: the NAMED correctness result, so the adapter's own
        # _contract_correctness_gate can be fed the same way RD08's and
        # RD58's are (see the compute_contract_correctness_gate() call in
        # run()). Without this RD73 passes None there and the gate reports
        # missing_checks -> BLOCKED, even though bit_identical was really
        # evaluated here. Returned as the CorrectnessResult itself, not a
        # bool, so a failure carries its detail through unchanged.
        "correctness_named_results": {"bit_identical": correctness_result},
        "resource_gate": resource_gate, "trigger_proof": trigger_proof, "promotion": promotion,
        "artifact": artifact_ref,
    }


def _run_framework_configuration(args: argparse.Namespace, descriptor, cfg) -> int:
    """Build the canonical native framework composition and persist schema-5 proof."""
    from bigcherry.build import generated_tree
    from bigcherry.patch import evidence as patch_validation_evidence
    from bigcherry.patch import source as psi
    from bigcherry.patch import validation_policy
    from bigcherry.patch import validation
    from bigcherry.core import paths as bc_paths

    if not validation_policy.is_framework_configuration_patch(descriptor):
        raise PatchCampaignError("--framework-configuration requires a local packaged framework patch without an RD/contract binding")
    if any(getattr(args, name, False) for name in (
        "run_rd08_lanes", "run_rd08_contract",
        "run_rd73_contract",
        "correctness_evidence",
    )):
        raise PatchCampaignError("framework configuration cannot be combined with runtime qualification modes")
    if args.amdgpu_targets is None:
        raise PatchCampaignError("framework configuration requires explicit AMDGPU compile targets")
    targets = tuple(target.strip() for target in re.split(r"[,;]", args.amdgpu_targets) if target.strip())
    if not targets or any(not re.fullmatch(r"gfx[0-9a-f]+", target) for target in targets):
        raise PatchCampaignError("framework configuration requires explicit AMDGPU compile targets")
    args.amdgpu_targets = ";".join(targets)
    from bigcherry.core.context import ProjectContext
    base_repo = ProjectContext.resolve(work_root=os.environ.get("BC_CACHE")).upstream_repo
    baseline_source = "bigcherry-qualification-tuning"
    base_revision, composition = psi.resolve_source_composition(
        baseline_source, focal=None, base_ref=cfg.pinned, base_repo=base_repo,
    )
    if (descriptor.patch_id, descriptor.implementation_digest) not in composition:
        raise PatchCampaignError(
            f"framework source {baseline_source!r} does not contain focal patch {descriptor.patch_id!r}"
        )
    source = psi.materialize_composition(
        base_repo=base_repo, worktree_root=args.worktree_root / "framework",
        resolved_revision=base_revision, composition=composition,
        overlay_root=psi.REPO_ROOT / "src", requested_revision=cfg.pinned,
    )
    idempotent = psi.verify_composition_idempotent(
        base_repo=base_repo, source=source, worktree_root=args.worktree_root / "framework",
        resolved_revision=base_revision, composition=composition,
        overlay_root=psi.REPO_ROOT / "src", requested_revision=cfg.pinned,
    )
    if not idempotent:
        raise PatchCampaignError("framework composition did not reapply idempotently")
    source_tree = psi.git_worktree_tree(source)
    source_manifest = psi._read_manifest(source)
    if not source_manifest or source_manifest.get("source_tree_oid") != source_tree:
        raise PatchCampaignError("framework source attestation is missing or stale")
    source_identity = psi._make_source_identity_v2(
        resolved_revision=base_revision, composition=composition, overlay_root=psi.REPO_ROOT / "src",
    )
    source_identity["materialization_plan_id"] = source_identity["source_key"]
    if any(source_manifest.get(key) != value for key, value in source_identity.items()):
        raise PatchCampaignError("framework materialization identity is stale")
    build_root = (args.build_root or args.workdir) / source.name
    # Qualification owns fresh directories, never retroactively attests a
    # historical build whose inputs were not observed during compilation.
    for role in ("production", "diagnostic"):
        if (build_root / f"framework-{role}").exists():
            raise PatchCampaignError("framework qualification requires a fresh build-root; preserve the previous run")
    generated_dir = build_root / "generated"
    generate_registry(source=source, amdgpu_targets=args.amdgpu_targets, generated_dir=generated_dir)
    # Same four compile inputs returned by catalog.emit().compile_input_paths;
    # JSON manifests contain timestamps and are not compiler inputs.
    compile_inputs = tuple(generated_dir / name for name in (
        "hip-autotune-registry.inc", "hip-autotune-build-hash.h",
        "hip-autotune-arch.h", "hip-autotune-mmvq-instances.inc",
    ))
    missing = [str(path) for path in compile_inputs if not path.is_file()]
    if missing:
        raise PatchCampaignError(f"generated compiler inputs missing: {missing}")
    generated_manifest = generated_tree.build_manifest(generated_dir, compile_inputs=compile_inputs)
    proof = {}

    def generated_proof(phase, build_dir):
        compiled_copy = build_dir / "generated-inputs"
        if psi.git_worktree_tree(source) != source_tree:
            raise PatchCampaignError(f"source changed at {phase}")
        generated_tree.verify_tree(generated_dir, generated_manifest)
        generated_tree.verify_tree(compiled_copy, generated_manifest)
        copied_manifest = generated_tree.build_manifest(
            compiled_copy,
            compile_inputs=tuple(compiled_copy / name for name in generated_manifest["compile_inputs"]),
        )
        if copied_manifest["compile_inputs_hash"] != generated_manifest["compile_inputs_hash"]:
            raise PatchCampaignError(f"{build_dir.name}: compiled-copy input hash disagrees with generated manifest")
        proof[build_dir.name] = copied_manifest

    import shutil
    for role in ("production", "diagnostic"):
        shutil.copytree(generated_dir, build_root / f"framework-{role}" / "generated-inputs")
    common = ["-DGGML_HIP_RCCL=ON", "-DGGML_HIP_DISPATCH_REPLAY=ON",
              "-DGGML_HIP_AUTOTUNE=OFF", "-DGGML_HIP_AUTOTUNE_RECORD=OFF",
              "-DGGML_HIP_REPLAY_DIAGNOSTICS=OFF"]
    production_args = common + ["-DGGML_HIP_DISPATCH_DIAGNOSTICS=OFF",
        f"-DGGML_HIP_AUTOTUNE_GENERATED_DIR={build_root / 'framework-production' / 'generated-inputs'}"]
    diagnostic_args = common + ["-DGGML_HIP_DISPATCH_DIAGNOSTICS=ON",
        f"-DGGML_HIP_AUTOTUNE_GENERATED_DIR={build_root / 'framework-diagnostic' / 'generated-inputs'}"]
    production_bin = build_tree(
        name="framework-production", hip_path=args.hip_path,
        amdgpu_targets=args.amdgpu_targets, workdir=build_root,
        targets=["llama-server"], source=source,
        extra_cmake_args=production_args, generated_proof_callback=generated_proof,
    )
    diagnostic_bin = build_tree(
        name="framework-diagnostic", hip_path=args.hip_path,
        amdgpu_targets=args.amdgpu_targets, workdir=build_root,
        targets=["llama-server"], source=source,
        extra_cmake_args=diagnostic_args, generated_proof_callback=generated_proof,
    )
    env = _hip_env(args.hip_path)
    exe = ".exe" if sys.platform == "win32" else ""
    production = capture_completed_build_evidence(
        build_root / "framework-production", source_root=source,
        architecture=args.amdgpu_targets, binary=production_bin / f"llama-server{exe}",
        requested_cmake_args=_full_requested_cmake_args(hip_path=args.hip_path, amdgpu_targets=args.amdgpu_targets, extra_cmake_args=production_args), build_env=env,
    )
    diagnostic = capture_completed_build_evidence(
        build_root / "framework-diagnostic", source_root=source,
        architecture=args.amdgpu_targets, binary=diagnostic_bin / f"llama-server{exe}",
        requested_cmake_args=_full_requested_cmake_args(hip_path=args.hip_path, amdgpu_targets=args.amdgpu_targets, extra_cmake_args=diagnostic_args), build_env=env,
    )
    from bigcherry.build.builds import inspect_dispatch_build
    compiler_observations = {}
    for role, diagnostic_on in (("production", False), ("diagnostic", True)):
        observed = inspect_dispatch_build(build_root / f"framework-{role}")
        counts = observed["compiled_definition_counts"]
        if observed["issues"] or bool(counts["GGML_HIP_DISPATCH_DIAGNOSTICS"]) != diagnostic_on:
            raise PatchCampaignError(f"{role} diagnostic compiler state disagrees with qualification role")
        compiler_observations[role] = {key: observed[key] for key in (
            "hip_compile_command_count", "compiled_definition_counts", "coverage_translation_unit", "issues",
        )}
    run_dir = args.workdir / "framework" / descriptor.patch_id
    run_dir.mkdir(parents=True, exist_ok=False)
    generated_artifact = _write_bound_artifact(run_dir, "generated-tree.json", generated_manifest)
    source_artifact = _write_bound_artifact(run_dir, "source-tree.json", source_manifest)
    builds = {"production": production.campaign_identity(), "diagnostic": diagnostic.campaign_identity()}
    for role in builds:
        compiler_observations[role]["build_identity"] = builds[role]
    build_artifacts = {role: _write_bound_artifact(run_dir, f"{role}-build.json", {
        **completed.to_dict(), "generated_inputs_verification": "compiled-copy-v1",
        "generated_inputs": proof[f"framework-{role}"],
        "source_slice_id": source_manifest["source_slice_id"], "source_tree": source_tree,
    }) for role, completed in (("production", production), ("diagnostic", diagnostic))}
    plan = validation_policy.require_execution_package(descriptor, root=bc_paths.PATCHES)
    ctx = validation.ValidationContext(
        descriptor=descriptor, base_revision=base_revision,
        control_source=None, subject_source=None,
        package_root=bc_paths.PATCHES / descriptor.package_root, run_dir=run_dir,
        register_artifact=validation.make_default_register_artifact(run_dir),
        configuration_evidence={
            "apply": {"single_composition": True, "verified": True, "idempotent": idempotent,
                      "artifact": source_artifact},
            "builds": {role: {"completed": True, "artifact": artifact}
                       for role, artifact in build_artifacts.items()},
        },
    )
    results = {spec.check_id: validation.evaluate_check(spec, ctx) for spec in plan.checks}
    verdict = validation.compute_verdict(plan, results)
    _print(f"adapter eligible: {verdict.eligible}")
    for check_id, result in results.items():
        _print(f"{check_id}: {result.status}: {result.summary}")
    checks = {name: asdict(result) for name, result in results.items()}
    artifacts = {artifact.path: artifact.sha256 for result in results.values() for artifact in result.artifacts}
    artifacts[generated_artifact["path"]] = generated_artifact["sha256"]
    record = patch_validation_evidence.make_framework_configuration_record(
        descriptor=descriptor, patch_path=bc_paths.PATCHES / descriptor.implementation_path,
        base_ref=cfg.pinned, base_revision=base_revision, source_name=baseline_source,
        source_composition=composition, source_tree=source_tree,
        source_slice_id=source_manifest["source_slice_id"], compiled_targets=tuple(
            target.strip() for target in re.split(r"[,;]", args.amdgpu_targets) if target.strip()
        ),
        builds=builds,
        source_identity=source_identity, compiler_observations=compiler_observations,
        generated_inputs={role: {"proof": "compiled-copy-v1",
            "compile_inputs_hash": proof[f"framework-{role}"]["compile_inputs_hash"],
            "tree_manifest": proof[f"framework-{role}"], "build_identity": builds[role],
        } for role in builds},
        check_results=checks, artifact_hashes=artifacts, campaign_workdir=run_dir,
    )
    path = patch_validation_evidence.write_record(record)
    _print(f"framework configuration evidence: {path}")
    return 0 if record["eligible_for_validated_state"] else 1


_STANDARD_BENCHMARK_MODEL_IDS: tuple[str, ...] = (
    "tierM-ministral14b-q4km", "tierB-qwen9b-q6k", "tierL-qwen27b-q8",
)


def _run_performance_benchmark(args: argparse.Namespace, descriptor, cfg) -> int:
    """PVPS02 step 4: the generic --run-performance-benchmark entry point.

    Deliberately does NOT go through the legacy --model/--manifest/one-
    architecture tune/replay/stock/control/subject flow run() otherwise
    builds -- that flow is keyed to exactly one amdgpu-targets value and
    a single --model, neither of which fits a cross-architecture,
    cross-model matrix. This materializes source ONCE and builds ONE
    control/subject llama-bench binary pair covering every applicable
    architecture (AMDGPU_TARGETS=";".join(architectures), a single fat
    multi-ISA build -- cmake/HIP natively support a semicolon-separated
    target list), reused across every architecture/model cell instead of
    rebuilding per architecture. User direction (2026-09-11, after the
    prior per-architecture-build version's real-hardware merge gate
    passed on all 3 architectures): device SELECTION is still per-cell
    via HIP_VISIBLE_DEVICES (see the real-hardware finding on that
    mechanism above this function), only the BUILD is now shared.

    execution_identity/device-visibility enforcement is turned on here
    for the FIRST time in this module (steps 1-3 deliberately left it
    off) -- every cell fails closed before launch on a missing/
    insufficient/malformed --device-map entry for its architecture, and
    every measured process is attested for real architecture + device
    count (not physical card identity -- see DeviceVisibility's own
    docstring for why that distinction is load-bearing, never
    overclaimed in evidence)."""
    from bigcherry.experiment import attestation
    from bigcherry.experiment.execution import require_device_visibility
    from bigcherry.patch import source as psi

    wiring = resolve_benchmark_wiring(descriptor)

    requested_arches = tuple(args.benchmark_architecture or ())
    if requested_arches:
        architectures = requested_arches
    else:
        platform_targets = tuple(cfg.platforms["linux-multi"].targets)
        architectures = tuple(
            arch for arch in platform_targets if arch in descriptor.validation_architectures
        )
    if not architectures:
        raise PatchCampaignError(
            f"{descriptor.patch_id}: no applicable architecture -- pass --benchmark-architecture "
            "explicitly, or declare validation-architectures overlapping "
            "config/recipes.toml's platform.linux-multi targets"
        )

    model_ids = tuple(args.benchmark_model or _STANDARD_BENCHMARK_MODEL_IDS)
    device_map = parse_device_map(list(args.device_map or ()))
    model_root: Path = args.model_root

    baseline_source = getattr(args, "baseline_source", "bigcherry")
    control_revision, control_composition = psi.resolve_source_composition(
        baseline_source, focal=None, base_ref=cfg.pinned, base_repo=LLAMA_CPP_SRC,
    )
    subject_revision, subject_composition = psi.resolve_source_composition(
        baseline_source, focal=args.patch, base_ref=cfg.pinned, base_repo=LLAMA_CPP_SRC,
    )
    if control_revision != subject_revision:
        raise RuntimeError("control and subject source plans resolved different base revisions")
    base_revision = subject_revision
    _print(f"performance benchmark: materializing control/subject source @ {base_revision[:12]} ...")
    control_src = psi.materialize_composition(
        base_repo=LLAMA_CPP_SRC, worktree_root=args.worktree_root / "control",
        resolved_revision=base_revision, composition=control_composition,
        overlay_root=psi.REPO_ROOT / "src", requested_revision=cfg.pinned,
    )
    subject_src = psi.materialize_composition(
        base_repo=LLAMA_CPP_SRC, worktree_root=args.worktree_root / "subject",
        resolved_revision=base_revision, composition=subject_composition,
        overlay_root=psi.REPO_ROOT / "src", requested_revision=cfg.pinned,
    )

    cells: list[dict[str, object]] = []
    build_root: Path = (args.build_root or args.workdir) / subject_src.name
    exe = ".exe" if sys.platform == "win32" else ""
    build_env = _hip_env(args.hip_path)

    # User direction (2026-09-11): ONE fat multi-ISA control/subject
    # build covering every applicable architecture, not a separate build
    # per architecture -- AMDGPU_TARGETS accepts a semicolon-separated
    # list natively. GPT review (req_e608313764834497, 2026-09-11): the
    # legacy validation path builds control and validation-subject with
    # IDENTICAL extra_cmake_args=[] and asserts that parity
    # (assert_validation_subject_parity) precisely because differing
    # build instrumentation would confound a measured patch effect with
    # build-configuration noise -- both binaries here remain plain,
    # symmetric, autotune-instrumentation-free builds, just built once
    # for the joined target list instead of once per architecture.
    joined_targets = ";".join(architectures)
    control_bin = build_tree(
        name="perf-control", hip_path=args.hip_path,
        amdgpu_targets=joined_targets, workdir=build_root, targets=["llama-bench"],
        source=control_src, extra_cmake_args=[],
    )
    subject_bin = build_tree(
        name="perf-subject", hip_path=args.hip_path,
        amdgpu_targets=joined_targets, workdir=build_root, targets=["llama-bench"],
        source=subject_src, extra_cmake_args=[],
    )
    control_binary = control_bin / f"llama-bench{exe}"
    subject_binary = subject_bin / f"llama-bench{exe}"
    control_cmake_args = _full_requested_cmake_args(
        hip_path=args.hip_path, amdgpu_targets=joined_targets, extra_cmake_args=[],
    )
    control_build_evidence = capture_completed_build_evidence(
        build_root / "perf-control", source_root=control_src,
        architecture=joined_targets, binary=control_binary,
        requested_cmake_args=control_cmake_args, build_env=build_env,
    )
    subject_build_evidence = capture_completed_build_evidence(
        build_root / "perf-subject", source_root=subject_src,
        architecture=joined_targets, binary=subject_binary,
        requested_cmake_args=control_cmake_args, build_env=build_env,
    )
    assert_validation_subject_parity(
        control_build_evidence, subject_build_evidence, patch_id=args.patch,
    )

    for architecture in architectures:
        for model_id in model_ids:
            cell: dict[str, object] = {"architecture": architecture, "model": model_id}
            try:
                resolved_model = resolve_benchmark_model(
                    model_id, model_root=model_root,
                )
            except PatchCampaignError as exc:
                cell.update(status="skipped", reason=f"model resolution failed: {exc}")
                cells.append(cell)
                continue
            try:
                device_ids = resolve_device_pool(
                    device_map, architecture, resolved_model.device_count,
                )
            except PatchCampaignError as exc:
                cell.update(status="skipped", reason=str(exc))
                cells.append(cell)
                continue

            # Real-hardware finding (2026-09-11, PNRO17): only
            # HIP_VISIBLE_DEVICES is set here -- also setting
            # ROCR_VISIBLE_DEVICES to the same value actively breaks
            # non-prefix-from-0 device selection (confirmed on real
            # gfx1100/gfx1201/gfx1030 hardware during this matrix's own
            # merge-gate run). See require_device_visibility()/
            # DeviceVisibility's docstrings for the full mechanism.
            env_overrides = {"HIP_VISIBLE_DEVICES": ",".join(device_ids)}
            visibility = require_device_visibility(
                context=f"performance-benchmark {architecture}/{model_id}",
                env=env_overrides, exact_count=resolved_model.device_count,
            )
            execution_identity = attestation.ExecutionIdentity(
                backend="ROCm", architectures=(architecture,) * resolved_model.device_count,
            )
            executor_func = BENCHMARK_EXECUTOR_FUNCS[wiring.executor]
            outcome = executor_func(
                control_binary=control_binary, subject_binary=subject_binary,
                model=resolved_model.path, hip_path=args.hip_path,
                patch_args=wiring.patch_args, runtime_args=resolved_model.runtime_args,
                pairs=args.bench_repetitions,
                log_context=f"performance-benchmark {architecture}/{model_id}",
                env_overrides=env_overrides, execution_identity=execution_identity,
            )
            cell.update(
                status="executed", device_visibility=visibility.document(),
                commands=outcome.commands, raw_logs=outcome.raw_logs,
                metrics={
                    workload: {"stats": run.stats, "runs": list(run.runs)}
                    for workload, run in outcome.runs.items()
                },
            )
            cells.append(cell)

    performance_doc = {
        "patch_id": descriptor.patch_id, "executor": wiring.executor,
        "architectures": list(architectures), "models": list(model_ids),
        "cells": cells,
    }
    performance_path = args.workdir / "performance-matrix.json"
    _atomic_write_json(performance_path, performance_doc)
    executed = [c for c in cells if c["status"] == "executed"]
    skipped = [c for c in cells if c["status"] == "skipped"]
    _print(
        f"performance benchmark matrix: {len(executed)} executed, {len(skipped)} skipped "
        f"(of {len(cells)} planned cells) -- {performance_path}"
    )
    for cell in skipped:
        _print(f"  skipped {cell['architecture']}/{cell['model']}: {cell['reason']}")
    return 0 if executed else 1


# ---------------------------------------------------------------------------
# Externally measured contract-evidence provenance
#
# RD39-43 can consume measurements produced outside this module, but those
# measurements are promotion-eligible only when their complete provenance is
# bound here.  This is deliberately campaign-local rather than part of
# experiment.contract: contracts define WHAT must be proven; this structure
# proves WHICH concrete executions supplied the facts. (GPT design,
# req_ea5a7ab7d6634e09.)
# ---------------------------------------------------------------------------

_QUALIFICATION_EVIDENCE_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_REQUIRED_BUILD_IDENTITY_KEYS = frozenset({
    "effective_build_id",
    "compile_verification_id",
    "compile_commands_digest",
    "hip_compile_commands_digest",
    "runtime_bundle_hash",
    "runtime_artifacts",
})

_REQUIRED_SOURCE_IDENTITY_KEYS = frozenset({
    "resolved_revision",
    "source_tree",
    "composition",
})

_RAW_EVIDENCE_KINDS = frozenset({
    "benchmark_raw",
    "correctness_raw",
    "logits_raw",
    "trigger_raw",
})


def _canonical_json(value: object, *, label: str) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise PatchCampaignError(f"{label} is not canonical JSON data: {exc}") from exc
    return encoded


def _canonical_json_mapping(value: Mapping[str, object], *, label: str) -> str:
    if not isinstance(value, Mapping) or not value:
        raise PatchCampaignError(f"{label} must be a non-empty mapping")
    encoded = _canonical_json(dict(value), label=label)
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise PatchCampaignError(f"{label} must encode a JSON object")
    return encoded


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise PatchCampaignError(f"cannot hash evidence file {path}: {exc}") from exc
    return digest.hexdigest()


@dataclass(frozen=True)
class QualificationEvidenceExecution:
    """One concrete process execution contributing qualification evidence."""

    name: str
    role: str
    command: tuple[str, ...]
    # Explicit selector state, including keys deliberately unset.
    selector_env: tuple[tuple[str, str | None], ...]
    expected_execution: ExecutionIdentity
    observed_execution: ExecutionAttestation

    def document(self) -> dict[str, object]:
        return {
            "name": self.name,
            "role": self.role,
            "command": list(self.command),
            "selector_env": {k: v for k, v in self.selector_env},
            "expected_execution": {
                "backend": self.expected_execution.backend,
                "architectures": list(self.expected_execution.architectures),
                "locators": (
                    list(self.expected_execution.locators)
                    if self.expected_execution.locators is not None
                    else {}
                ),
            },
            "observed_execution": self.observed_execution.document(),
        }


def make_qualification_execution(
    *,
    name: str,
    role: str,
    command: Iterable[str],
    selector_env: Mapping[str, str | None],
    expected_execution: ExecutionIdentity,
    observed_execution: ExecutionAttestation,
) -> QualificationEvidenceExecution:
    if not name:
        raise PatchCampaignError("qualification execution name must be non-empty")
    if not role:
        raise PatchCampaignError(f"qualification execution {name!r}: role must be non-empty")

    argv = tuple(str(arg) for arg in command)
    if not argv:
        raise PatchCampaignError(f"qualification execution {name!r}: command is empty")

    # HIP/ROCR selector state is identity-relevant even when one is explicitly
    # absent.  Recording only HIP_VISIBLE_DEVICES would lose the double-filter
    # condition that has caused real invalid runs in this project.
    missing_selector_keys = {
        "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"
    } - set(selector_env)
    if missing_selector_keys:
        raise PatchCampaignError(
            f"qualification execution {name!r}: selector_env missing "
            f"{sorted(missing_selector_keys)}"
        )

    if expected_execution.locators is None:
        raise PatchCampaignError(
            f"qualification execution {name!r}: expected device locators are "
            "required for promotion-grade provenance"
        )

    reasons = compare_execution_identity(expected_execution, observed_execution)
    if reasons:
        raise PatchCampaignError(
            f"qualification execution {name!r}: execution attestation mismatch: "
            f"{', '.join(reasons)}"
        )

    return QualificationEvidenceExecution(
        name=name,
        role=role,
        command=argv,
        selector_env=tuple(sorted(selector_env.items())),
        expected_execution=expected_execution,
        observed_execution=observed_execution,
    )


@dataclass(frozen=True)
class QualificationEvidenceArtifact:
    kind: str
    execution: str
    path: str
    sha256: str
    size_bytes: int

    def document(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "execution": self.execution,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


def bind_qualification_raw_artifact(
    *,
    run_dir: Path,
    kind: str,
    execution: str,
    source: Path,
) -> QualificationEvidenceArtifact:
    """Copy immutable raw evidence into run_dir and bind its SHA-256."""

    if kind not in _RAW_EVIDENCE_KINDS:
        raise PatchCampaignError(
            f"unknown qualification raw-artifact kind {kind!r}; "
            f"expected one of {sorted(_RAW_EVIDENCE_KINDS)}"
        )
    if not execution:
        raise PatchCampaignError("raw evidence execution name must be non-empty")
    if not source.is_file():
        raise PatchCampaignError(f"raw evidence file does not exist: {source}")

    digest = _sha256_file(source)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", source.name)
    target = (
        run_dir / "artifacts" / "raw" /
        f"{kind}-{digest[:16]}-{safe_name}"
    )
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        if _sha256_file(target) != digest:
            raise PatchCampaignError(
                f"existing bound evidence artifact has wrong digest: {target}"
            )
    else:
        shutil.copyfile(source, target)

    return QualificationEvidenceArtifact(
        kind=kind,
        execution=execution,
        path=target.relative_to(run_dir).as_posix(),
        sha256=digest,
        size_bytes=target.stat().st_size,
    )


@dataclass(frozen=True)
class QualificationEvidenceManifest:
    schema_version: int
    campaign_id: str

    contract_id: str
    contract_hash: str
    patch_ids: tuple[str, ...]
    base_revision: str

    target_metric: str
    positive_pct_deltas: tuple[float, ...]
    control_pct_deltas: tuple[float, ...]

    correctness_results: tuple[object, ...]
    trigger_evidence: tuple[object, ...]

    # Canonical JSON strings make these flexible identity documents deeply
    # immutable despite the outer dataclass being frozen.
    subject_source_identity_json: str
    control_source_identity_json: str
    subject_build_identity_json: str
    control_build_identity_json: str

    model_ref: str
    model_sha256: str
    model_size_bytes: int

    executions: tuple[QualificationEvidenceExecution, ...]
    artifacts: tuple[QualificationEvidenceArtifact, ...]

    manifest_sha256: str

    def document(self, *, include_manifest_sha256: bool = True) -> dict[str, object]:
        correctness_doc = [
            {
                "check": r.check,
                "passed": r.passed,
                "detail": r.detail,
            }
            for r in self.correctness_results
        ]
        trigger_doc = [
            {
                "role": e.role,
                "lane_id": e.lane_id,
                "candidate_launches": e.candidate_launches,
                "expected_route_selected": e.expected_route_selected,
            }
            for e in self.trigger_evidence
        ]

        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "contract_id": self.contract_id,
            "contract_hash": self.contract_hash,
            "patch_ids": list(self.patch_ids),
            "base_revision": self.base_revision,
            "target_metric": self.target_metric,
            "positive_pct_deltas": list(self.positive_pct_deltas),
            "control_pct_deltas": list(self.control_pct_deltas),
            "correctness_results": correctness_doc,
            "trigger_evidence": trigger_doc,
            "source_identity": {
                "subject": json.loads(self.subject_source_identity_json),
                "control": json.loads(self.control_source_identity_json),
            },
            "build_identity": {
                "subject": json.loads(self.subject_build_identity_json),
                "control": json.loads(self.control_build_identity_json),
            },
            "model": {
                "ref": self.model_ref,
                "sha256": self.model_sha256,
                "size_bytes": self.model_size_bytes,
            },
            "executions": [e.document() for e in self.executions],
            "artifacts": [a.document() for a in self.artifacts],
        }

        if include_manifest_sha256:
            result["manifest_sha256"] = self.manifest_sha256

        return result


def _qualification_manifest_digest(
    manifest: QualificationEvidenceManifest,
) -> str:
    payload = manifest.document(include_manifest_sha256=False)
    return hashlib.sha256(
        _canonical_json(payload, label="qualification evidence manifest").encode("utf-8")
    ).hexdigest()


def build_qualification_evidence_manifest(
    *,
    campaign_id: str,
    contract: object,
    patch_ids: Iterable[str],
    base_revision: str,
    target_metric: str,
    positive_pct_deltas: Iterable[float],
    control_pct_deltas: Iterable[float],
    correctness_results: Mapping[str, object],
    trigger_evidence: Iterable[object],
    subject_source_identity: Mapping[str, object],
    control_source_identity: Mapping[str, object],
    subject_build_identity: Mapping[str, object],
    control_build_identity: Mapping[str, object],
    model_ref: str,
    model_path: Path,
    executions: Iterable[QualificationEvidenceExecution],
    artifacts: Iterable[QualificationEvidenceArtifact],
) -> QualificationEvidenceManifest:
    """Construct one immutable, content-bound promotion evidence manifest."""

    patch_ids_t = tuple(patch_ids)
    positive_t = tuple(float(v) for v in positive_pct_deltas)
    control_t = tuple(float(v) for v in control_pct_deltas)
    correctness_t = tuple(correctness_results[k] for k in sorted(correctness_results))
    trigger_t = tuple(trigger_evidence)
    executions_t = tuple(executions)
    artifacts_t = tuple(artifacts)

    if not campaign_id:
        raise PatchCampaignError("qualification manifest campaign_id is required")
    if not patch_ids_t or len(set(patch_ids_t)) != len(patch_ids_t):
        raise PatchCampaignError(
            "qualification manifest patch_ids must be non-empty and unique"
        )
    if not re.fullmatch(r"[0-9a-fA-F]{40}", base_revision):
        raise PatchCampaignError(
            "qualification manifest base_revision must be a full 40-hex SHA"
        )
    if not target_metric:
        raise PatchCampaignError("qualification manifest target_metric is required")

    if not positive_t or not control_t:
        raise PatchCampaignError(
            "qualification manifest requires both positive and control measurements"
        )
    if len(positive_t) != len(control_t):
        raise PatchCampaignError(
            "qualification manifest positive/control measurement counts differ"
        )
    if not all(math.isfinite(v) for v in (*positive_t, *control_t)):
        raise PatchCampaignError(
            "qualification manifest percentage deltas must all be finite"
        )

    if not correctness_t:
        raise PatchCampaignError(
            "qualification manifest requires explicit correctness results"
        )
    if not trigger_t:
        raise PatchCampaignError(
            "qualification manifest requires explicit trigger evidence"
        )

    if not model_ref:
        raise PatchCampaignError("qualification manifest model_ref is required")
    if not model_path.is_file():
        raise PatchCampaignError(
            f"qualification manifest model does not exist: {model_path}"
        )

    execution_names = [e.name for e in executions_t]
    if not execution_names or len(set(execution_names)) != len(execution_names):
        raise PatchCampaignError(
            "qualification manifest executions must be non-empty and uniquely named"
        )

    # These four evidence classes must all have raw files bound.  A logit file
    # can satisfy the correctness obligation while correctness_raw can carry
    # the parsed checker output; require at least one of those two.
    artifact_kinds = {a.kind for a in artifacts_t}
    if "benchmark_raw" not in artifact_kinds:
        raise PatchCampaignError(
            "qualification manifest has no benchmark_raw artifact"
        )
    if not ({"correctness_raw", "logits_raw"} & artifact_kinds):
        raise PatchCampaignError(
            "qualification manifest has no correctness_raw/logits_raw artifact"
        )
    if "trigger_raw" not in artifact_kinds:
        raise PatchCampaignError(
            "qualification manifest has no trigger_raw artifact"
        )

    known_executions = set(execution_names)
    for artifact in artifacts_t:
        if artifact.execution not in known_executions:
            raise PatchCampaignError(
                f"artifact {artifact.path!r} references unknown execution "
                f"{artifact.execution!r}"
            )

    def _checked_identity(
        value: Mapping[str, object],
        *,
        label: str,
        required_keys: frozenset[str],
    ) -> str:
        missing = required_keys - set(value)
        if missing:
            raise PatchCampaignError(
                f"{label} missing identity fields: {sorted(missing)}"
            )
        return _canonical_json_mapping(value, label=label)

    subject_source_json = _checked_identity(
        subject_source_identity,
        label="subject_source_identity",
        required_keys=_REQUIRED_SOURCE_IDENTITY_KEYS,
    )
    control_source_json = _checked_identity(
        control_source_identity,
        label="control_source_identity",
        required_keys=_REQUIRED_SOURCE_IDENTITY_KEYS,
    )
    subject_build_json = _checked_identity(
        subject_build_identity,
        label="subject_build_identity",
        required_keys=_REQUIRED_BUILD_IDENTITY_KEYS,
    )
    control_build_json = _checked_identity(
        control_build_identity,
        label="control_build_identity",
        required_keys=_REQUIRED_BUILD_IDENTITY_KEYS,
    )

    manifest = QualificationEvidenceManifest(
        schema_version=_QUALIFICATION_EVIDENCE_SCHEMA_VERSION,
        campaign_id=campaign_id,
        contract_id=contract.id,
        contract_hash=contract.contract_hash,
        patch_ids=patch_ids_t,
        base_revision=base_revision.lower(),
        target_metric=target_metric,
        positive_pct_deltas=positive_t,
        control_pct_deltas=control_t,
        correctness_results=correctness_t,
        trigger_evidence=trigger_t,
        subject_source_identity_json=subject_source_json,
        control_source_identity_json=control_source_json,
        subject_build_identity_json=subject_build_json,
        control_build_identity_json=control_build_json,
        model_ref=model_ref,
        model_sha256=_sha256_file(model_path),
        model_size_bytes=model_path.stat().st_size,
        executions=executions_t,
        artifacts=artifacts_t,
        manifest_sha256="",
    )

    return dataclasses.replace(
        manifest,
        manifest_sha256=_qualification_manifest_digest(manifest),
    )


def validate_qualification_evidence_manifest(
    manifest: QualificationEvidenceManifest,
    *,
    contract: object,
    run_dir: Path,
) -> None:
    """Fail closed before manually gathered measurements reach contract gates."""

    if manifest.schema_version != _QUALIFICATION_EVIDENCE_SCHEMA_VERSION:
        raise PatchCampaignError(
            f"unsupported qualification evidence schema "
            f"{manifest.schema_version!r}"
        )
    if manifest.contract_id != contract.id:
        raise PatchCampaignError(
            f"qualification manifest contract mismatch: "
            f"{manifest.contract_id!r} != {contract.id!r}"
        )
    if manifest.contract_hash != contract.contract_hash:
        raise PatchCampaignError(
            "qualification manifest was produced for a different contract hash"
        )

    expected_digest = _qualification_manifest_digest(manifest)
    if (
        not _SHA256_RE.fullmatch(manifest.manifest_sha256)
        or manifest.manifest_sha256 != expected_digest
    ):
        raise PatchCampaignError(
            "qualification manifest content hash is missing or invalid"
        )

    if not _SHA256_RE.fullmatch(manifest.model_sha256):
        raise PatchCampaignError("qualification manifest model SHA-256 is invalid")

    for execution in manifest.executions:
        reasons = compare_execution_identity(
            execution.expected_execution,
            execution.observed_execution,
        )
        if reasons:
            raise PatchCampaignError(
                f"qualification execution {execution.name!r} no longer "
                f"validates: {', '.join(reasons)}"
            )

    for artifact in manifest.artifacts:
        path = run_dir / artifact.path
        if not path.is_file():
            raise PatchCampaignError(
                f"bound qualification artifact is missing: {path}"
            )
        if path.stat().st_size != artifact.size_bytes:
            raise PatchCampaignError(
                f"bound qualification artifact size changed: {path}"
            )
        actual = _sha256_file(path)
        if actual != artifact.sha256:
            raise PatchCampaignError(
                f"bound qualification artifact digest changed: {path}"
            )


def _lane_effect_from_pct_deltas(
    *, role: str, metric: str, pct_deltas: list[float],
) -> "experiment_contract.LaneEffect":
    """Binds real, already-measured per-round percentage deltas (the shape
    manually-run paired llama-bench campaigns report, e.g. patches
    1215/1216's README round-by-round evidence) into a LaneEffect, without
    re-running anything. ``pct_deltas`` is one signed percentage change per
    paired round (subject vs control, same round); converted to the
    ``pair_ratios`` sufficient statistic (1 + delta/100) that
    bootstrap_fixed_composite_mean()/aggregate_contract_effects() require
    for an interval, and to the geometric mean effect for the point
    estimate -- the same estimator block_bootstrap_effect() uses elsewhere
    in this module, just computed here directly since there is no raw
    per-pair timing to re-derive it from."""
    from bigcherry.experiment import contract as experiment_contract

    if not pct_deltas:
        raise PatchCampaignError("_lane_effect_from_pct_deltas: pct_deltas must be non-empty")
    ratios = tuple(1.0 + (delta / 100.0) for delta in pct_deltas)
    if any(ratio <= 0 for ratio in ratios):
        raise PatchCampaignError(
            f"_lane_effect_from_pct_deltas: non-positive ratio derived from deltas {pct_deltas!r}"
        )
    log_ratios = [math.log(ratio) for ratio in ratios]
    point_pct = 100.0 * (math.exp(statistics.mean(log_ratios)) - 1.0)
    lane = experiment_contract.LaneEffect(
        role=role, metric=metric, geometric_effect_pct=point_pct,
        paired_rounds=len(ratios), pair_ratios=ratios,
    )
    ci = experiment_contract.bootstrap_fixed_composite_mean([lane])
    if ci is not None:
        lane = dataclasses.replace(lane, ci95_low_pct=ci[0], ci95_high_pct=ci[1])
    return lane




def _build_standard_campaign_scaffold(
    *,
    patch_id: str,
    base_ref: str,
    baseline_source: str,
    hip_path: Path,
    amdgpu_targets: str,
    workdir: Path,
    worktree_root: Path,
    build_root: Path | None,
) -> StandardCampaignScaffold:
    """Materialize control/subject/stock sources and build the five
    standard campaign trees in the historical order, capturing per-build
    evidence and asserting validation-subject/control parity. Moved
    verbatim from run() (PA36 sub-slice 2, dev-gpt-agent
    req_2ecda033763949a9 T2) so the generic producer path and run() share
    one owner of the five-build contract."""
    from bigcherry.patch import source as psi # noqa: E402

    control_revision, control_composition = psi.resolve_source_composition(
        baseline_source, focal=None, base_ref=base_ref, base_repo=LLAMA_CPP_SRC,
    )
    subject_revision, subject_composition = psi.resolve_source_composition(
        baseline_source, focal=patch_id, base_ref=base_ref, base_repo=LLAMA_CPP_SRC,
    )
    if control_revision != subject_revision:
        raise RuntimeError("control and subject source plans resolved different base revisions")
    base_revision = subject_revision
    _print(f"materializing control and subject source plans @ {base_revision[:12]} ...")
    control_src = psi.materialize_composition(
        base_repo=LLAMA_CPP_SRC, worktree_root=worktree_root / "control",
        resolved_revision=base_revision, composition=control_composition,
        overlay_root=psi.REPO_ROOT / "src", requested_revision=base_ref,
    )
    patched_src = psi.materialize_composition(
        base_repo=LLAMA_CPP_SRC, worktree_root=worktree_root / "subject",
        resolved_revision=base_revision, composition=subject_composition,
        overlay_root=psi.REPO_ROOT / "src", requested_revision=base_ref,
    )
    _print(f"control source: {control_src}")
    _print(f"subject source: {patched_src}")
    control_idempotent = psi.verify_composition_idempotent(
        base_repo=LLAMA_CPP_SRC, source=control_src, worktree_root=worktree_root / "control",
        resolved_revision=base_revision, composition=control_composition,
        overlay_root=psi.REPO_ROOT / "src", requested_revision=base_ref,
    )
    subject_idempotent = psi.verify_composition_idempotent(
        base_repo=LLAMA_CPP_SRC, source=patched_src, worktree_root=worktree_root / "subject",
        resolved_revision=base_revision, composition=subject_composition,
        overlay_root=psi.REPO_ROOT / "src", requested_revision=base_ref,
    )
    stock_src = psi.materialize_stock_source(
        base_repo=LLAMA_CPP_SRC, worktree_root=worktree_root / "stock", base_revision=base_revision,
    )
    _print(f"stock source: {stock_src}")

    # Build trees are keyed by --build-root, not --workdir: build_tree()/
    # ensure_stock_baseline() always reconfigure (cheap/incremental) but
    # `cmake --build` itself only recompiles what actually changed, so a
    # build tree is still effectively reusable across runs on this
    # machine+arch as long as its SOURCE (an isolated, content-addressed
    # worktree, not the shared vendor/llama.cpp tree -- HI82) hasn't changed
    # identity. --workdir (record/tune/promote/replay/bench/report output)
    # is what needs to be fresh per patch+model.
    actual_build_root: Path = (build_root or workdir) / patched_src.name

    # One shared out-of-tree registry serves both the tune and replay builds
    # of this same patched source -- both need it (ggml-hip/CMakeLists.txt
    # gates on GGML_HIP_AUTOTUNE OR GGML_HIP_DISPATCH_REPLAY), and it is
    # pure generated-from-source content, not build-mode-specific.
    generated_dir = actual_build_root / "generated"
    generate_registry(
        source=patched_src, amdgpu_targets=amdgpu_targets, generated_dir=generated_dir,
    )

    exe = ".exe" if sys.platform == "win32" else ""
    build_env = _hip_env(hip_path)

    tune_extra_cmake_args = [
        "-DGGML_HIP_AUTOTUNE=ON", "-DGGML_HIP_AUTOTUNE_RECORD=ON",
        "-DGGML_HIP_ROUTING_TRANSFORM=ON",
        f"-DGGML_HIP_AUTOTUNE_GENERATED_DIR={generated_dir}",
    ]
    tune_cmake_args = _full_requested_cmake_args(
        hip_path=hip_path, amdgpu_targets=amdgpu_targets,
        extra_cmake_args=tune_extra_cmake_args,
    )
    tune_bin = build_tree(
        name="tune", hip_path=hip_path, amdgpu_targets=amdgpu_targets,
        workdir=actual_build_root, targets=["llama-server", "llama-bench"], source=patched_src,
        extra_cmake_args=tune_extra_cmake_args,
    )
    # HI82 item 7: refuse to hand a build to Campaign() until its actual
    # compiled command lines are proven to match configured intent -- a
    # build that silently lost a flag (the HI81 shape) must never reach
    # benchmarking. Raises BuildIdentityError uncaught, which is the
    # intended fail-closed behavior: no partial/best-effort campaign runs
    # against an unverified build. Reuses builds.py's existing identity/
    # reuse contract (effective_build_id/runtime_bundle_hash) rather than
    # a second, parallel identity authority -- see HI82 review history.
    tune_build_evidence = capture_completed_build_evidence(
        actual_build_root / "tune", source_root=patched_src, architecture=amdgpu_targets,
        binary=tune_bin / f"llama-server{exe}", extra_binaries=(tune_bin / f"llama-bench{exe}",),
        requested_cmake_args=tune_cmake_args, build_env=build_env,
    )
    _print(
        f"tune build: {tune_build_evidence.effective_build_id[:12]} / "
        f"{tune_build_evidence.runtime_bundle_hash[:12]} / "
        f"{tune_build_evidence.compile_verification_id[:12]}"
    )

    replay_extra_cmake_args = [
        "-DGGML_HIP_DISPATCH_REPLAY=ON",
        f"-DGGML_HIP_AUTOTUNE_GENERATED_DIR={generated_dir}",
    ]
    replay_cmake_args = _full_requested_cmake_args(
        hip_path=hip_path, amdgpu_targets=amdgpu_targets,
        extra_cmake_args=replay_extra_cmake_args,
    )
    replay_bin = build_tree(
        name="replay", hip_path=hip_path, amdgpu_targets=amdgpu_targets,
        workdir=actual_build_root, targets=["llama-server", "llama-bench"], source=patched_src,
        extra_cmake_args=replay_extra_cmake_args,
    )
    replay_build_evidence = capture_completed_build_evidence(
        actual_build_root / "replay", source_root=patched_src, architecture=amdgpu_targets,
        binary=replay_bin / f"llama-server{exe}",
        extra_binaries=(replay_bin / f"llama-bench{exe}",),
        requested_cmake_args=replay_cmake_args, build_env=build_env,
    )
    _print(
        f"replay build: {replay_build_evidence.effective_build_id[:12]} / "
        f"{replay_build_evidence.runtime_bundle_hash[:12]} / "
        f"{replay_build_evidence.compile_verification_id[:12]}"
    )

    stock_build_root = (build_root or workdir) / stock_src.name
    stock_bin = ensure_stock_baseline(
        hip_path=hip_path, amdgpu_targets=amdgpu_targets,
        workdir=stock_build_root, stock_src=stock_src,
    )
    stock_cmake_args = _full_requested_cmake_args(
        hip_path=hip_path, amdgpu_targets=amdgpu_targets, extra_cmake_args=[],
    )
    stock_build_evidence = capture_completed_build_evidence(
        stock_build_root / "stock", source_root=stock_src, architecture=amdgpu_targets,
        binary=stock_bin / f"llama-bench{exe}", requested_cmake_args=stock_cmake_args,
        build_env=build_env,
    )
    _print(
        f"stock build: {stock_build_evidence.effective_build_id[:12]} / "
        f"{stock_build_evidence.runtime_bundle_hash[:12]} / "
        f"{stock_build_evidence.compile_verification_id[:12]}"
    )

    # RS10: the authoritative control source is independently built as well;
    # it is not merely a recorded tree next to a subject-only campaign.
    control_build_root = (build_root or workdir) / control_src.name
    control_bin = build_tree(
        name="control", hip_path=hip_path, amdgpu_targets=amdgpu_targets,
        workdir=control_build_root, targets=["llama-server", "llama-bench"],
        source=control_src, extra_cmake_args=[],
    )
    control_build_evidence = capture_completed_build_evidence(
        control_build_root / "control", source_root=control_src,
        architecture=amdgpu_targets, binary=control_bin / f"llama-bench{exe}",
        requested_cmake_args=stock_cmake_args, build_env=build_env,
    )
    _print(
        f"control build: {control_build_evidence.effective_build_id[:12]} / "
        f"{control_build_evidence.runtime_bundle_hash[:12]} / "
        f"{control_build_evidence.compile_verification_id[:12]}"
    )

    # VA14-B: the validation-domain subject is a real, independently-built
    # parity binary from patched_src -- NOT the tune-mode binary (that build
    # carries GGML_HIP_AUTOTUNE/AUTOTUNE_RECORD/ROUTING_TRANSFORM
    # instrumentation the control build never had, which would confound a
    # measured RD08 lane effect with instrumentation overhead, not just the
    # patch). Built with exactly control's extra_cmake_args=[].
    validation_subject_bin = build_tree(
        name="validation-subject", hip_path=hip_path, amdgpu_targets=amdgpu_targets,
        workdir=actual_build_root, targets=["llama-server", "llama-bench"], source=patched_src,
        extra_cmake_args=[],
    )
    # GPT round 3 (req_e75c4936e2354351): capture symmetrically with
    # control_build_evidence below (binary=llama-bench only, no
    # extra_binaries) -- an asymmetric capture is not a like-for-like
    # comparison even when the underlying build tree is parity.
    validation_subject_build_evidence = capture_completed_build_evidence(
        actual_build_root / "validation-subject", source_root=patched_src,
        architecture=amdgpu_targets, binary=validation_subject_bin / f"llama-bench{exe}",
        requested_cmake_args=stock_cmake_args, build_env=build_env,
    )
    assert_validation_subject_parity(
        control_build_evidence, validation_subject_build_evidence, patch_id=patch_id,
    )
    _print(
        f"validation-subject build: {validation_subject_build_evidence.effective_build_id[:12]} / "
        f"{validation_subject_build_evidence.runtime_bundle_hash[:12]} / "
        f"{validation_subject_build_evidence.compile_verification_id[:12]}"
    )
    return StandardCampaignScaffold(
        base_revision=base_revision,
        control_composition=control_composition,
        subject_composition=subject_composition,
        control_source=control_src,
        subject_source=patched_src,
        stock_source=stock_src,
        control_idempotent=control_idempotent,
        subject_idempotent=subject_idempotent,
        build_root=actual_build_root,
        build_env=build_env,
        tune_bin=tune_bin,
        replay_bin=replay_bin,
        stock_bin=stock_bin,
        control_bin=control_bin,
        validation_subject_bin=validation_subject_bin,
        tune_build_evidence=tune_build_evidence,
        replay_build_evidence=replay_build_evidence,
        stock_build_evidence=stock_build_evidence,
        control_build_evidence=control_build_evidence,
        validation_subject_build_evidence=validation_subject_build_evidence,
    )



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
    from bigcherry.patch import source as psi # noqa: E402
    from bigcherry.patch import registry as patch_registry, validation as patch_validation
    from bigcherry.patch import validation_policy as patch_validation_policy # noqa: E402
    from bigcherry.core import paths as bc_paths # noqa: E402
    from bigcherry.core import config as campaign_config # noqa: E402

    registry = patch_registry.load_registry(bc_paths.PATCHES)
    descriptor = registry.get(args.patch)

    # GPT round 2 (req_71217bba406f4941, VA04 real-hardware finding): the
    # pinned ref MUST be resolved before any source materialization --
    # the hardcoded literal "HEAD" below used to silently build against whatever
    # the shared vendor/llama.cpp checkout's HEAD happened to be at run
    # time, while the evidence record was later labeled base_ref=cfg.pinned
    # regardless of whether HEAD actually matched the pin. A real RD04
    # hardware run on Brutus resolved and built against vendor HEAD while
    # its own evidence claimed pin b10705 -- VA08's stale-detection
    # correctly caught the mismatch and rejected the record. cfg is loaded
    # ONCE here and reused for evidence writing below (no duplicate load).
    cfg = campaign_config.load(bc_paths.RECIPES)

    # VA02 execution-side anti-grandfather guard (unconditional, per GPT
    # round-5 code review req_86cfd3a0bff04716: this command IS "start a
    # real validation run" -- there is no tracked-status branch here,
    # because otherwise build_plan_for_patch() legitimately returning None
    # for a patch with neither a contract nor an adapter would let this
    # command continue straight into source materialization/build without
    # ever producing real evidence tied to a check, regardless of any
    # lint-side structural-grandfather exemption).
    validation_plan = patch_validation_policy.require_execution_package(
        descriptor, root=bc_paths.PATCHES,
    )
    if validation_plan is not None:
        _print(f"validation plan: {len(validation_plan.checks)} checks; required={validation_plan.required_capabilities}")

    if getattr(args, "framework_configuration", False):
        return _run_framework_configuration(args, descriptor, cfg)

    if getattr(args, "run_performance_benchmark", False):
        return _run_performance_benchmark(args, descriptor, cfg)

    worktree_root: Path = args.worktree_root
    # RV80/B6: the baseline is the source's EXPLICIT named composition from
    # config/recipes.toml (never the retired implicit state=='validated'
    # scan), resolved through the exact-composition validator; the base ref
    # resolves to an immutable SHA that enters the v2 source identity.
    baseline_source = getattr(args, "baseline_source", "bigcherry")
    # PA36 sub-slice 2 (dev-gpt-agent req_2ecda033763949a9, T2): the five
    # standard-campaign builds (tune -> replay -> stock -> control ->
    # validation-subject) live in _build_standard_campaign_scaffold(),
    # shared with the generic standard_campaign="run" producer path. Local
    # aliases below keep the rest of run() structurally unchanged.
    scaffold = _build_standard_campaign_scaffold(
        patch_id=args.patch, base_ref=cfg.pinned, baseline_source=baseline_source,
        hip_path=args.hip_path, amdgpu_targets=args.amdgpu_targets,
        workdir=workdir, worktree_root=worktree_root, build_root=args.build_root,
    )
    base_revision = scaffold.base_revision
    control_composition = scaffold.control_composition
    subject_composition = scaffold.subject_composition
    control_src = scaffold.control_source
    patched_src = scaffold.subject_source
    stock_src = scaffold.stock_source
    control_idempotent = scaffold.control_idempotent
    subject_idempotent = scaffold.subject_idempotent
    build_root = scaffold.build_root
    build_env = scaffold.build_env
    tune_bin = scaffold.tune_bin
    replay_bin = scaffold.replay_bin
    stock_bin = scaffold.stock_bin
    control_bin = scaffold.control_bin
    validation_subject_bin = scaffold.validation_subject_bin
    tune_build_evidence = scaffold.tune_build_evidence
    replay_build_evidence = scaffold.replay_build_evidence
    stock_build_evidence = scaffold.stock_build_evidence
    control_build_evidence = scaffold.control_build_evidence
    validation_subject_build_evidence = scaffold.validation_subject_build_evidence
    exe = ".exe" if sys.platform == "win32" else ""



    from bigcherry.e2e_smoke_campaign import (  # noqa: E402
        Campaign, CampaignError, CampaignIdentityContext,
    )

    # Hoisted: HI83's evidence record (below) needs the same values.
    patch_digest = psi.patch_implementation_digest(args.patch)
    control_source_tree = psi.git_worktree_tree(control_src)
    patched_source_tree = psi.git_worktree_tree(patched_src)

    identity_context = CampaignIdentityContext(
        patch_name=args.patch,
        patch_digest=patch_digest,
        patched_source_tree=patched_source_tree,
        gpu_architecture=args.amdgpu_targets,
        build_identities={
            "tune": tune_build_evidence.campaign_identity(),
            "replay": replay_build_evidence.campaign_identity(),
            "stock": stock_build_evidence.campaign_identity(),
        },
    )

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
        identity_context=identity_context,
    )

    # Bind the workdir before writing any patch-specific activation
    # evidence. campaign.run() will check it again; this earlier call
    # prevents a trace probe from writing evidence into a stale/mismatched
    # campaign directory.
    campaign.ensure_campaign_identity()

    activation_evidence = None
    activation_verdict = None
    trace_marker_regex = args.trace_marker_regex
    trace_description = args.trace_description
    if validation_plan is not None:
        trace_specs = tuple(
            spec for spec in validation_plan.checks
            if spec.capability == "activation" and spec.validator == "trace-marker"
        )
        if len(trace_specs) > 1:
            raise PatchCampaignError(
                f"{args.patch}: validation plan declares multiple trace-marker activation checks"
            )
        if trace_specs:
            configured_marker = trace_specs[0].config.get("marker-regex")
            if not isinstance(configured_marker, str) or not configured_marker:
                raise PatchCampaignError(
                    f"{args.patch}: trace-marker activation check has no marker-regex"
                )
            if trace_marker_regex is not None and trace_marker_regex != configured_marker:
                raise PatchCampaignError(
                    f"{args.patch}: CLI trace marker conflicts with validation.toml"
                )
            trace_marker_regex = configured_marker
            trace_description = trace_description or f"{args.patch} activation"
        elif trace_marker_regex is not None or trace_description is not None:
            raise PatchCampaignError(
                f"{args.patch}: trace CLI options require a trace-marker validation check"
            )
    # GPT round 6 (req_bc329f6ae30c4e4c, VA15 real-hardware finding): the
    # generic tune-binary/fusion-disabled probe is redundant for
    # --run-rd08-contract -- it is replaced by RD08's own authoritative
    # validation-subject/control trigger probe below, which is a valid
    # negative control for RD08's specific marker (the generic probe's
    # negative control, GGML_CUDA_DISABLE_FUSION=1, is not). Skipping it
    # here also avoids wasted GPU time on a probe whose result gets
    # overwritten anyway.
    # VA06: --run-rd73-contract also skips the generic probe -- the
    # generic tune-binary/GGML_CUDA_DISABLE_FUSION negative control is
    # not valid for RD73 (graph-cache keying, not a fusion path), and
    # the generic probe's plain llama-bench invocation (no -sm tensor)
    # cannot even load RD73's real 27B contract model on Brutus's dual
    # gfx1100 GPUs. RD73's own authoritative activation evidence comes
    # from evaluate_rd73_activation_evidence() inside
    # run_rd73_contract_qualification().
    trace_result = None if (args.run_rd08_contract or args.run_rd73_contract) else run_trace_activation_probes(
        marker_regex=trace_marker_regex, description=trace_description,
        binary=tune_bin / f"llama-bench{exe}", model=args.model,
        hip_path=args.hip_path, workdir=workdir / "campaign",
        bench_prompt=args.bench_prompt, bench_gen=args.bench_gen,
    )
    # VA11A: real bound trace_evidence for ValidationContext (was {}
    # unconditionally, which made _builtin_trace_marker always BLOCKED --
    # GPT round-7 review, req_3d12aa6668b14bb1). Built from the real
    # positive/negative probe logs run_trace_activation_probes() already
    # wrote to workdir/"campaign"/"logs"/... -- the same directory
    # ValidationContext.run_dir (campaign_run_dir) resolves bound artifacts
    # against, so no copy is needed, only a path+sha256 reference.
    trace_evidence: dict[str, object] = {}
    if trace_result is not None:
        activation_evidence, trace_detail = trace_result
        # This stage establishes activation evidence only -- the existing
        # patch_activation verdict contract accepts correctness_passed=None;
        # a later patch-class-specific correctness check can strengthen
        # this without changing the trace-probe mechanism.
        activation_verdict = verdict(activation_evidence, correctness_passed=None)
        write_activation_json(
            workdir / "campaign" / "activation.json", activation_evidence, activation_verdict,
            extra={
                "campaign_identity_digest": campaign.campaign_identity_digest,
                "trace_probe": trace_detail,
            },
        )
        _print(f"activation: {activation_evidence.status} ({activation_evidence.mechanism})")

        def _bind_existing(relative_log_path: str) -> dict[str, str]:
            target = (workdir / "campaign" / relative_log_path).resolve()
            return {
                "path": relative_log_path,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }

        trace_evidence = {
            "positive": {
                "marker_regex": trace_detail["marker_regex"],
                "artifact": _bind_existing(trace_detail["positive"]["log"]),
            },
            "negative": {
                "marker_regex": trace_detail["marker_regex"],
                "artifact": _bind_existing(trace_detail["negative_control"]["log"]),
            },
        }

    # GPT round 6 (req_bc329f6ae30c4e4c, VA15 real-hardware finding): the
    # generic S1-S7 record/tune/promote/replay/bench/report campaign is
    # unrelated to RD08's own contract evidence -- lanes/correctness/
    # trigger/promotion never consume promoted.jsonl, dispatch.cache,
    # replay coverage, or S6/S7 results. Making that unrelated pipeline's
    # own promotion decision (which can legitimately promote zero
    # candidates on a real, honest run -- that is not a bug) a hard
    # prerequisite of --run-rd08-contract was itself the real bug,
    # discovered on real hardware (VA15). campaign.ensure_campaign_identity()
    # above still ran, so campaign.campaign_identity_digest remains valid
    # for RD08's evidence below.
    # VA06: --run-rd73-contract also skips the generic S1-S7 campaign
    # pipeline, for the same reason RD08/RD04/RD58 do -- RD73's real
    # evidence comes entirely from run_rd73_contract_qualification(),
    # which never reads promoted.jsonl/dispatch.cache/replay coverage/S6
    # or S7 results. This was actually the real, first root cause hit
    # during the Brutus qualification run (a manifest_hash mismatch
    # inside this unrelated pipeline) -- discovered before this
    # exclusion was added; kept for defense-in-depth even though a
    # correctly-generated manifest can also make the S1-S7 path succeed.
    if not (args.run_rd08_contract or args.run_rd73_contract):
        try:
            campaign.run()
        except CampaignError as exc:
            _print(f"CAMPAIGN FAILED: {exc}")
            return 1

        report_path = workdir / "campaign" / "report.md"
        _print(f"done -- report: {report_path}")
        print(report_path.read_text(encoding="utf-8"))

    # HI83: record what this campaign proved (or didn't), tracked so
    # STATE="validated" can eventually be checked against it. This is
    # purely additive evidence production -- it does not gate anything in
    # this campaign, and nothing in bigcherry apply/build consumes it yet
    # (see plan item HI83's notes for why hard enforcement is deliberately
    # deferred). A campaign with no correctness evidence and/or no
    # activation probe for this patch still writes a real record; it is
    # simply not eligible_for_validated_state.
    from bigcherry.patch import evidence as patch_validation_evidence # noqa: E402

    # cfg is loaded once, above, before source resolution -- reused here
    # (was previously loaded a second time in this exact spot, after
    # source materialization had already resolved against "HEAD").
    # RS04: the evidence record's patch file path resolves through the
    # registry descriptor (flat or packaged) -- no f"{patch_id}.py" guessing
    # in this caller.
    _descriptor = descriptor
    _patch_file = registry.root / _descriptor.implementation_path
    campaign_run_dir = workdir / "campaign"
    correctness_summary = None
    # VA11A: ctx.correctness_evidence must carry a BOUND artifact reference
    # ({"artifact": {"path", "sha256"}}) -- _builtin_backend_ops reads via
    # ctx.correctness_evidence.get("artifact"), and a raw decoded dict with
    # no "artifact" key made every backend-ops check unconditionally BLOCKED
    # (real bug, confirmed by reading validation.py::_builtin_backend_ops
    # before this fix -- GPT round-7 review, req_3d12aa6668b14bb1).
    correctness_evidence: dict[str, object] = {}
    performance_evidence: dict[str, object] = {}
    if args.correctness_evidence is not None and args.run_rd08_contract:
        raise PatchCampaignError(
            f"{args.patch}: --correctness-evidence and --run-rd08-contract are ambiguous "
            "together -- --run-rd08-contract already produces its own authoritative "
            "correctness.json"
        )
    if args.correctness_evidence is not None:
        correctness_summary = patch_validation_evidence.load_correctness_summary(
            args.correctness_evidence, patch_id=args.patch,
            subject_digest=patch_validation_evidence.patch_validation_subject_digest(
                _patch_file
            ),
            base_revision=base_revision, patched_source_tree=patched_source_tree,
            campaign_identity_digest=campaign.campaign_identity_digest,
            gpu_architectures=(args.amdgpu_targets,),
        )
        correctness_path = campaign_run_dir / "correctness.json"
        _atomic_write_json(correctness_path, correctness_summary)
        correctness_evidence = {
            "artifact": {
                "path": correctness_path.relative_to(campaign_run_dir).as_posix(),
                "sha256": hashlib.sha256(correctness_path.read_bytes()).hexdigest(),
            }
        }
    build_evidence = {
        "control": {
            "build_id": control_build_evidence.effective_build_id,
            "source_tree": control_source_tree,
            "architecture": args.amdgpu_targets,
            "options": control_build_evidence.effective_configure,
            "compile_commands": _write_bound_artifact(
                campaign_run_dir, "build/control-compile-commands.json",
                control_build_evidence.verification.to_dict(),
            ),
            "runtime_bundle": _write_bound_artifact(
                campaign_run_dir, "build/control-runtime-bundle.json",
                control_build_evidence.runtime_artifacts,
            ),
        },
        "subject": {
            "build_id": validation_subject_build_evidence.effective_build_id,
            "source_tree": patched_source_tree,
            "architecture": args.amdgpu_targets,
            "options": validation_subject_build_evidence.effective_configure,
            "compile_commands": _write_bound_artifact(
                campaign_run_dir, "build/subject-compile-commands.json",
                validation_subject_build_evidence.verification.to_dict(),
            ),
            "runtime_bundle": _write_bound_artifact(
                campaign_run_dir, "build/subject-runtime-bundle.json",
                validation_subject_build_evidence.runtime_artifacts,
            ),
        },
    }
    apply_evidence = {
        "control": {
            "verified": True, "idempotent": control_idempotent,
            "artifact": _write_bound_artifact(
                campaign_run_dir, "apply/control.json",
                {"source_tree": control_source_tree, "composition": list(control_composition)},
            ),
        },
        "subject": {
            "verified": True, "idempotent": subject_idempotent,
            "artifact": _write_bound_artifact(
                campaign_run_dir, "apply/subject.json",
                {"source_tree": patched_source_tree, "composition": list(subject_composition)},
            ),
        },
    }

    # VA14-B/VA14-final: RD08 execution, opt-in and scoped to RD08 only.
    # --run-rd08-lanes stays diagnostic-only (execution + evidence, never
    # feeds eligibility). --run-rd08-contract is the authoritative full-
    # qualification path (lanes + real named correctness + real trigger
    # proof, composed via evaluate_promotion_gate()) and is the ONLY thing
    # allowed to populate contract_promotions below. The two are mutually
    # exclusive to avoid a redundant duplicate lane run.
    if args.run_rd08_lanes and args.run_rd08_contract:
        raise PatchCampaignError(
            f"{args.patch}: --run-rd08-contract already runs the lanes -- "
            "do not also pass --run-rd08-lanes"
        )
    rd08_lane_evidence: dict[str, object] | None = None
    rd08_qualification: dict[str, object] | None = None
    contract_promotions: dict[str, dict[str, object]] = {}
    if (args.run_rd08_lanes or args.run_rd08_contract) and descriptor.experiment_contract != "RD08-Q6K-MMVQ-VDR2":
        raise PatchCampaignError(
            f"{args.patch}: --run-rd08-lanes/--run-rd08-contract are RD08-only today"
        )
    if args.run_rd08_contract:
        from bigcherry.patch import validation as _pv

        rd08_contract = _pv.load_contract_for_descriptor(descriptor)
        if rd08_contract is None:
            raise PatchCampaignError(
                f"{args.patch}: --run-rd08-contract requires a resolvable RD08 contract"
            )
        rd08_qualification = run_rd08_contract_qualification(
            contract=rd08_contract, descriptor=descriptor, base_revision=base_revision,
            control_binary=control_bin / f"llama-bench{exe}",
            subject_binary=validation_subject_bin / f"llama-bench{exe}", model=args.model,
            model_ref=rd08_contract.positive.models[0], marker_regex=trace_marker_regex,
            hip_path=args.hip_path, amdgpu_targets=args.amdgpu_targets,
            worktree_root=args.worktree_root, build_root=build_root, build_env=build_env,
            run_dir=campaign_run_dir,
            control_build_identity=control_build_evidence.campaign_identity(),
            subject_build_identity=validation_subject_build_evidence.campaign_identity(),
        )
        contract_promotions[rd08_contract.id] = rd08_qualification["promotion"]
        _print(f"rd08 contract qualification: {rd08_qualification['artifact']['path']}")
        _print(
            f"rd08 promotion: "
            f"{'PASS' if rd08_qualification['promotion'].get('passed') else rd08_qualification['promotion'].get('status', 'FAIL')}"
        )
        correctness_summary = {
            "schema_version": patch_validation_evidence.CORRECTNESS_SCHEMA_VERSION,
            "patch_id": args.patch,
            "patch_validation_subject_digest": patch_validation_evidence.patch_validation_subject_digest(
                _patch_file
            ),
            "base_revision": base_revision, "patched_source_tree": patched_source_tree,
            "campaign_identity_digest": campaign.campaign_identity_digest,
            "gpu_architectures": [args.amdgpu_targets],
            "disposition": "passed" if rd08_qualification["correctness_gate"].get("passed") else "failed",
            "mechanism": "rd08-bit-identical-5shape-3seed",
            "detail": rd08_qualification["correctness"]["results"]["bit_identical"].detail,
        }
        correctness_path = campaign_run_dir / "correctness.json"
        _atomic_write_json(correctness_path, correctness_summary)
        correctness_evidence = {
            "artifact": {
                "path": correctness_path.relative_to(campaign_run_dir).as_posix(),
                "sha256": hashlib.sha256(correctness_path.read_bytes()).hexdigest(),
            }
        }
        # GPT round 4 (req_4544a9240b6d45df): RD08's validation.toml declares
        # its correctness/performance/controls checks with
        # validator="autotune-campaign", which reads ctx.performance_evidence
        # (_builtin_autotune_campaign requires a bound artifact with a
        # truthy campaign_id) -- without this, those three checks always
        # error/fail and validation_verdict.eligible can never become True,
        # so the record's final eligibility can never become True either.
        # Bind the real RD08 qualification result here -- campaign_id is the
        # actual completed campaign's own identity digest, not fabricated.
        performance_doc = {
            "campaign_id": campaign.campaign_identity_digest,
            "passed": bool(rd08_qualification["promotion"].get("passed")),
            "target_kernel_gain_pct": rd08_qualification["aggregated_effects"].get(
                "target_kernel_gain_pct"
            ),
            "max_control_regression_pct": rd08_qualification["aggregated_effects"].get(
                "max_control_regression_pct"
            ),
        }
        performance_path = campaign_run_dir / "performance.json"
        _atomic_write_json(performance_path, performance_doc)
        performance_evidence = {
            "artifact": {
                "path": performance_path.relative_to(campaign_run_dir).as_posix(),
                "sha256": hashlib.sha256(performance_path.read_bytes()).hexdigest(),
            }
        }
        # GPT round 4: the adapter's trace-marker check AND the record's
        # top-level activation field must use RD08's OWN authoritative
        # subject-hit/control-miss trigger evidence, not the earlier generic
        # tune-binary/GGML_CUDA_DISABLE_FUSION=1 probe -- that probe is not a
        # valid negative control for RD08's specific MMVQ marker.
        def _bind_rd08_log(relative_log_path: str) -> dict[str, str]:
            target = (campaign_run_dir / relative_log_path).resolve()
            return {
                "path": relative_log_path,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }

        trace_evidence = {
            "positive": {
                "marker_regex": trace_marker_regex,
                "artifact": _bind_rd08_log(rd08_qualification["trigger"]["subject_log_path"]),
            },
            "negative": {
                "marker_regex": trace_marker_regex,
                "artifact": _bind_rd08_log(rd08_qualification["trigger"]["control_log_path"]),
            },
        }
        activation_evidence = ActivationEvidence(
            status=(
                "executed"
                if rd08_qualification["trigger"]["subject_hit"]
                and not rd08_qualification["trigger"]["control_hit"]
                else "not_executed"
            ),
            mechanism="rd08-trigger-marker", detail=f"marker={trace_marker_regex!r}",
        )
        activation_verdict = verdict(activation_evidence, correctness_passed=None)
        # GPT round 5 (req_12dd706a42e341bd): the RD08 override above only
        # changed the in-memory activation_evidence/activation_verdict --
        # campaign/activation.json on disk still held the earlier generic
        # tune/fusion-disabled probe's result while the record qualifies
        # using RD08-authoritative evidence. Rewrite it with the real
        # RD08 result so the on-disk artifact and the record agree.
        write_activation_json(
            campaign_run_dir / "activation.json", activation_evidence, activation_verdict,
            extra={
                "campaign_identity_digest": campaign.campaign_identity_digest,
                "rd08_trigger": {
                    "subject_hit": rd08_qualification["trigger"]["subject_hit"],
                    "control_hit": rd08_qualification["trigger"]["control_hit"],
                    "artifact": rd08_qualification["trigger"]["artifact"],
                },
            },
        )
    elif args.run_rd08_lanes:
        from bigcherry.patch import validation as _pv

        rd08_contract = _pv.load_contract_for_descriptor(descriptor)
        if rd08_contract is None:
            raise PatchCampaignError(
                f"{args.patch}: --run-rd08-lanes requires a resolvable RD08 contract"
            )
        rd08_lane_evidence = run_rd08_validation_lanes(
            contract=rd08_contract, control_binary=control_bin / f"llama-bench{exe}",
            subject_binary=validation_subject_bin / f"llama-bench{exe}", model=args.model,
            model_ref=rd08_contract.positive.models[0],
            hip_path=args.hip_path, run_dir=campaign_run_dir,
            control_build_identity=control_build_evidence.campaign_identity(),
            subject_build_identity=validation_subject_build_evidence.campaign_identity(),
        )
        _print(f"rd08 lanes: {rd08_lane_evidence['artifact']['path']}")

    if args.run_rd73_contract:
        if (
            args.run_rd08_lanes
            or args.run_rd08_contract
            
        ):
            raise PatchCampaignError(
                f"{args.patch}: --run-rd73-contract is mutually exclusive with the "
                "other specialized evidence-producer modes"
            )
        if descriptor.experiment_contract != "RD73-STABLE-GRAPH-CACHE-KEY":
            raise PatchCampaignError(
                f"{args.patch}: --run-rd73-contract is RD73-only today"
            )
        if args.rd73_corpus is None:
            raise PatchCampaignError(
                f"{args.patch}: --run-rd73-contract requires --rd73-corpus"
            )
        from bigcherry.patch import validation as _pv

        rd73_contract = _pv.load_contract_for_descriptor(descriptor)
        if rd73_contract is None:
            raise PatchCampaignError(
                f"{args.patch}: --run-rd73-contract requires a resolvable RD73 contract"
            )
        rd73_qualification = run_rd73_contract_qualification(
            contract=rd73_contract,
            control_server_binary=control_bin / f"llama-server{exe}",
            subject_server_binary=validation_subject_bin / f"llama-server{exe}",
            model=args.model, marker_regex=trace_marker_regex, corpus_path=args.rd73_corpus,
            run_dir=campaign_run_dir,
            # RV99: every measurement session already committed for this
            # patch. Under a session policy the gate aggregates these together
            # with the session about to be measured, so a run can establish a
            # bound that no single run could. Read from the tracked evidence
            # file -- which is exactly why lane_effects had to be persisted
            # there, and why RV96 had to make a build hold more than one
            # record before any of this could work.
            prior_session_records=patch_validation_evidence.load_records(args.patch),
            amdgpu_targets=args.amdgpu_targets,
        )
        contract_promotions[rd73_contract.id] = rd73_qualification["promotion"]

        # VA23: bind RD73's real contract-produced evidence into the generic
        # adapter, exactly as RD58 does. Before this, the RD73 branch produced
        # authoritative artifacts but bound none of them, so the declared
        # performance/controls checks ERRORed ("benchmark artifact requires
        # non-empty metrics") and correctness stayed BLOCKED -- making
        # patch-verify-evidence report that no benchmark ran when one had.
        #
        # correctness is bound ONLY when the bit_identical evaluation actually
        # produced an artifact. On Rd73CorrectnessError the artifact is None
        # and correctness must stay BLOCKED rather than silently pass: a
        # correctness check that could not be evaluated is not a correctness
        # check that succeeded.
        performance_evidence = {"artifact": rd73_qualification["performance_artifact"]}
        if rd73_qualification["correctness"].get("artifact") is not None:
            correctness_evidence = {"artifact": rd73_qualification["correctness"]["artifact"]}
        # VA23: the activation lane already ran a real positive/negative
        # marker probe; bind its bound log refs so _builtin_trace_marker()
        # can re-read and re-verify them. The validator does its own regex
        # check against both logs, so this supplies evidence for independent
        # verification rather than asserting the outcome.
        trace_evidence = {
            "positive": rd73_qualification["activation"]["positive"],
            "negative": rd73_qualification["activation"]["negative"],
        }
        # RV95: the three bindings above satisfy validation.toml's DECLARED
        # checks, but not the record's own top-level activation/correctness
        # fields -- make_record() reads those from activation_evidence and
        # correctness_summary, which the RD73 branch never set. They stayed
        # at disposition="unknown", so verify_validated_patch() rejected an
        # otherwise-passing record with "activation is not executed+
        # activation-verified; correctness did not pass" even while
        # check_results._contract_correctness_gate.passed was true. Bind them
        # from the SAME real evidence RD08/RD58 use, in the same shape.
        activation_evidence = ActivationEvidence(
            status=(
                "executed"
                if rd73_qualification["activation"]["subject_hit"]
                and not rd73_qualification["activation"]["control_hit"]
                else "not_executed"
            ),
            mechanism="rd73-trigger-marker", detail=f"marker={trace_marker_regex!r}",
        )
        activation_verdict = verdict(activation_evidence, correctness_passed=None)
        write_activation_json(
            campaign_run_dir / "activation.json", activation_evidence, activation_verdict,
            extra={
                "campaign_identity_digest": campaign.campaign_identity_digest,
                "rd73_trigger": {
                    "subject_hit": rd73_qualification["activation"]["subject_hit"],
                    "control_hit": rd73_qualification["activation"]["control_hit"],
                    "artifact": rd73_qualification["activation"]["artifact"],
                },
            },
        )
        # Disposition comes from the contract's own correctness gate, which
        # is already fail-closed: an Rd73CorrectnessError leaves the artifact
        # None and records passed=False, so a correctness check that could
        # not be evaluated reports "failed" here rather than silently passing.
        correctness_summary = {
            "schema_version": patch_validation_evidence.CORRECTNESS_SCHEMA_VERSION,
            "patch_id": args.patch,
            "patch_validation_subject_digest": patch_validation_evidence.patch_validation_subject_digest(
                _patch_file
            ),
            "base_revision": base_revision, "patched_source_tree": patched_source_tree,
            "campaign_identity_digest": campaign.campaign_identity_digest,
            "gpu_architectures": [args.amdgpu_targets],
            "disposition": (
                "passed" if rd73_qualification["correctness_gate"].get("passed") else "failed"
            ),
            "mechanism": "rd73-mtp-bit-identical",
            "detail": (
                "paired MTP control/subject completions compared byte-for-byte; "
                f"{len(rd73_qualification['correctness'].get('rows') or ())} row(s) compared"
            ),
        }
        correctness_path = campaign_run_dir / "correctness.json"
        _atomic_write_json(correctness_path, correctness_summary)

        _print(f"rd73 contract qualification: {rd73_qualification['artifact']['path']}")
        _print(
            f"rd73 promotion: "
            f"{'PASS' if rd73_qualification['promotion'].get('passed') else rd73_qualification['promotion'].get('status', 'FAIL')}"
        )


    validation_check_results: dict[str, object] = {}
    validation_verdict = None
    if validation_plan is not None:
        # VA11A: package_root lets a packaged patch's custom validator
        # actually resolve its check(ctx) file (was always None -- any
        # custom check would fail closed for every packaged RD patch).
        package_root = (
            (registry.root / descriptor.package_root)
            if descriptor.package_root is not None else {}
        )
        # VA15 real-hardware finding: validation_plan.contract is a
        # patch_validation.ContractBinding -- a lightweight PROJECTION
        # (contract_id/hash/expected_effect/etc) that deliberately does NOT
        # carry .correctness/.acceptance/etc. compute_contract_correctness_gate()
        # needs the real experiment_contract.ExperimentContract, which
        # run_rd08_contract already loaded as rd08_contract for the
        # --run-rd08-contract path; other contract-bound patches load it
        # fresh here the same way that block does. PA36-F step 2: also the
        # source of ValidationContext's plural contracts/contract_hashes
        # below -- loaded once, before the context is constructed.
        full_contract = (
            rd08_contract if rd08_qualification is not None
            else patch_validation.load_contract_for_descriptor(descriptor)
        )
        validation_ctx = patch_validation.ValidationContext(
            descriptor=descriptor, base_revision=base_revision,
            control_source=control_src, subject_source=patched_src, stock_source=stock_src,
            package_root=package_root,
            control_tree=control_source_tree, subject_tree=patched_source_tree,
            # GPT round 3: RD58's real build check must be evaluated
            # against ITS test-save-load-state builds, not the generic
            # llama-bench builds the final record no longer identifies it
            # with.
            build_identities={
                "control": control_build_evidence.effective_build_id,
                "subject": validation_subject_build_evidence.effective_build_id,
            },
            build_evidence=build_evidence,
            apply_evidence=apply_evidence,
            architecture=args.amdgpu_targets, model=str(args.model),
            contracts=(full_contract,) if full_contract is not None else (),
            contract_hashes=(
                {full_contract.id: full_contract.contract_hash}
                if full_contract is not None else {}
            ),
            run_dir=campaign_run_dir,
            register_artifact=patch_validation.make_default_register_artifact(campaign_run_dir),
            trace_evidence=trace_evidence, correctness_evidence=correctness_evidence,
            performance_evidence=performance_evidence,
        )
        evaluated = {
            spec.check_id: patch_validation.evaluate_check(spec, validation_ctx)
            for spec in validation_plan.checks
        }
        validation_verdict = patch_validation.compute_verdict(validation_plan, evaluated)
        # GPT round 2 (req_3616cc1d90dc4512, blocker #3): RD58's own real
        # test-save-load-state evidence produces a named
        # state_restore_integrity CorrectnessResult -- thread it through
        # the same way rd08_qualification's named results are threaded,
        # so the contract's own correctness gate actually reflects the
        # real evidence instead of reporting missing_checks.
        contract_correctness_gate = compute_contract_correctness_gate(
            full_contract,
            (
                rd08_qualification["correctness"]["results"] if rd08_qualification is not None
                else (
                    # VA23: RD73's bit_identical result is real and already
                    # evaluated inside run_rd73_contract_qualification(); thread
                    # it here exactly as RD08's, so the gate
                    # reflects the evidence instead of reporting missing_checks.
                    rd73_qualification["correctness_named_results"]
                    if rd73_qualification is not None else {}
                )
            ),
        )
        validation_check_results = {
            check_id: asdict(result) for check_id, result in evaluated.items()
        }
        if rd08_qualification is not None:
            validation_check_results["_contract_qualification"] = {
                "promotion": rd08_qualification["promotion"],
                "trigger_proof": rd08_qualification["trigger_proof"],
                "aggregated_effects": rd08_qualification["aggregated_effects"],
                "artifact": rd08_qualification["artifact"],
            }
        if contract_correctness_gate is not None:
            validation_check_results["_contract_correctness_gate"] = contract_correctness_gate
            _print(
                f"contract correctness gate: "
                f"{'passed' if contract_correctness_gate.get('passed') else contract_correctness_gate.get('status', 'not passed')}"
            )
        _print(
            f"validation verdict: {'eligible' if validation_verdict.eligible else 'ineligible'} "
            f"({len(validation_verdict.reasons)} blocking reasons)"
        )

    validation_contracts, validation_contract_verdicts = build_contract_evidence_for_persistence(
        validation_plan.contracts if validation_plan is not None else (), contract_promotions,
    )

    validation_record = patch_validation_evidence.make_record(
        patch_id=args.patch, patch_path=_patch_file,
        patch_implementation_digest=patch_digest, base_ref=cfg.pinned,
        base_revision=base_revision, framework_baseline_digest=psi.composition_digest(subject_composition),
        patched_source_tree=patched_source_tree, gpu_architectures=args.amdgpu_targets,
        activation_evidence=activation_evidence, activation_disposition=activation_verdict,
        correctness=correctness_summary, campaign_identity_digest=campaign.campaign_identity_digest,
        build_identities=identity_context.build_identities,
        # VA07: real validation-build domain, distinct from the campaign
        # {tune,replay,stock} domain above. subject is intentionally the
        # same physical build as campaign.tune today (the tune build IS
        # the patch under validation) -- the schema records both roles
        # explicitly rather than assuming that equality.
        # GPT round 2 (blocker #1): RD58's real validation build is
        # test-save-load-state, not the generic llama-bench control/
        # validation-subject builds -- record ITS identities when RD58 ran.
        validation_build_identities={
                "control": control_build_evidence.campaign_identity(),
                "subject": validation_subject_build_evidence.campaign_identity(),
            },
        campaign_workdir=workdir / "campaign",
        check_results=validation_check_results,
        # VA14 final slice: eligible_for_validated_state for a bound-contract
        # patch requires BOTH the adapter verdict AND every bound contract's
        # own evaluate_promotion_gate() PASS (contract_promotions). A bound
        # contract with no promotion result at all still forces False.
        # RV95: it additionally requires the record's own activation/
        # correctness dispositions -- the same values make_record() persists
        # just below and verify_validated_patch() later reads -- so this flag
        # can no longer report eligible for a record the evidence verifier
        # rejects. See compute_persisted_validation_eligible()'s docstring.
        validation_eligible=compute_persisted_validation_eligible(
            _descriptor, validation_verdict, contract_promotions,
            activation_disposition=activation_verdict, correctness=correctness_summary,
        ),
        # RV99: persist the measurements, not only the verdict derived from
        # them, so an interval can be re-derived and sessions aggregated from
        # committed evidence alone.
        lane_effects=collect_lane_effect_records(
            rd08_qualification=rd08_qualification, rd73_qualification=rd73_qualification,
        ),
        representation=_descriptor.representation,
        validation_implementation_digest=_descriptor.validation_digest,
        contracts=validation_contracts,
        contract_verdicts=validation_contract_verdicts,
        baseline_composition={"source": baseline_source, "base_revision": base_revision,
                              "patches": list(control_composition)},
        control_composition={"base_revision": base_revision, "patches": list(control_composition)},
        subject_composition={"base_revision": base_revision, "patches": list(subject_composition)},
        control_tree=control_source_tree,
        subject_tree=patched_source_tree,
        stock_tree=psi.git_worktree_tree(stock_src),
    )
    validation_record_path = patch_validation_evidence.write_record(validation_record)
    _print(f"validation evidence: {validation_record_path}")
    _print(
        "STATE='validated' eligible: "
        + ("yes" if validation_record["eligible_for_validated_state"] else "no")
    )

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bigcherry patch-validation-campaign")
    parser.add_argument("--patch", required=True,
                         help="patch module name under patches/")
    parser.add_argument(
        "--framework-configuration", action="store_true", default=False,
        help="run the explicit schema-5 local framework-configuration build path",
    )
    parser.add_argument("--baseline-source", default="bigcherry",
                         help="explicit named source composition for CONTROL; SUBJECT adds "
                              "only the focal patch. The focal must be absent from this "
                              "baseline; dependencies/conflicts remain enforced.")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--hip-path", required=True, type=Path)
    parser.add_argument(
        "--amdgpu-targets", default=None,
        help="e.g. gfx1100 or gfx1201 -- required for every mode EXCEPT "
             "--run-performance-benchmark, which resolves its own architecture list "
             "(--benchmark-architecture or the recipes/validation-architectures "
             "intersection) per cell.",
    )
    parser.add_argument("--manifest", type=Path)
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
    parser.add_argument(
        "--trace-marker-regex", default=None,
        help="optional generic activation marker regex; patch-specific probe configuration "
             "stays outside the campaign orchestrator",
    )
    parser.add_argument(
        "--trace-description", default=None,
        help="human-readable description paired with --trace-marker-regex",
    )
    parser.add_argument(
        "--correctness-evidence", type=Path, default=None,
        help="HI83: machine-readable patch-level correctness evidence bound to "
             "this patch/source/campaign identity; without it the campaign still "
             "runs and records evidence, but the record is not eligible for "
             "STATE='validated'",
    )
    parser.add_argument(
        "--run-rd08-lanes", action="store_true", default=False,
        help="VA14-B: execute RD08's real positive(decode)/control(prefill) lane pairs "
             "against the parity-verified control/validation-subject builds and persist "
             "the raw evidence. Diagnostic-only -- does not affect eligibility. RD08-only; "
             "an error for any other patch. Mutually exclusive with --run-rd08-contract.",
    )
    parser.add_argument(
        "--run-rd08-contract", action="store_true", default=False,
        help="VA14 final slice: the authoritative RD08 full-qualification path -- real "
             "lane execution + real bit-identical correctness "
             "(require_rd08_correctness_evidence()) + real trigger proof, composed via "
             "evaluate_promotion_gate(). The only path that can make an RD08-bound patch "
             "eligible_for_validated_state. RD08-only; an error for any other patch. "
             "Mutually exclusive with --run-rd08-lanes and --correctness-evidence.",
    )
    parser.add_argument(
        "--run-rd73-contract", action="store_true", default=False,
        help="VA06: the RD73 full-qualification path -- activation + graph-cache resource "
             "evidence + real paired MTP-verify performance (server harness) + decode "
             "control + bit-identical correctness, composed via evaluate_promotion_gate(). "
             "Populates contract_promotions for RD73 (a real PASS/FAIL/INVALID verdict is "
             "printed) and rebinds the generic adapter's performance/correctness/trace "
             "evidence plus the record's own activation/correctness dispositions, so a "
             "passing run satisfies verify_validated_patch() as well as the eligibility "
             "flag. RD73-only; an error for any other patch. Mutually exclusive with the "
             "RD08/RD58 execution modes. Requires --rd73-corpus.",
    )
    parser.add_argument(
        "--producer-corpus", type=Path, default=None,
        help="text corpus for --validation-producer's ProducerContext.corpus "
             "-- the generic (non-RD73-specific) analog of --rd73-corpus, "
             "threaded into any patch-local producer that "
             "declares a correctness check requiring a real backend-reference "
             "corpus (e.g. 1203's RD05/RD07 backend_reference checks).",
    )
    parser.add_argument(
        "--rd73-corpus", type=Path, default=None,
        help="VA06: prompt corpus JSONL for --run-rd73-contract's MTP server lane "
             "(bench/server_completion.py's load_corpus() format).",
    )
    parser.add_argument(
        "--run-performance-benchmark", action="store_true", default=False,
        help="PVPS02: the generic paired-benchmark entry point for ANY patch whose "
             "validation.toml wires a recognized benchmark-executor on its required "
             "performance check -- not RD04/RD08/etc-specific. Builds one control/subject "
             "llama-bench per applicable architecture and runs the standard (or "
             "--benchmark-model-selected) model matrix across them. Diagnostic-only for "
             "eligibility, same as the legacy per-patch modes -- never populates "
             "contract_promotions. Mutually exclusive with the RD04/RD08/RD58/RD73 modes; "
             "does NOT require --model/--manifest/--amdgpu-targets (those are for the "
             "legacy single-architecture flow).",
    )
    parser.add_argument(
        "--model-root", type=Path, default=None,
        help="PVPS02: host model root config/models.toml paths are relative to (e.g. "
             "/mnt/vault/llm-models on Brutus). Required with --run-performance-benchmark.",
    )
    parser.add_argument(
        "--benchmark-model", action="append", default=None,
        help="PVPS02: a config/models.toml id to benchmark (repeatable). Omit for the "
             "standard model set (tierM-ministral14b-q4km, tierB-qwen9b-q6k, "
             "tierL-qwen27b-q8).",
    )
    parser.add_argument(
        "--benchmark-architecture", action="append", default=None,
        help="PVPS02: an amdgpu target to benchmark (repeatable). Omit to default to the "
             "intersection of config/recipes.toml's platform.linux-multi targets and the "
             "patch's own validation-architectures.",
    )
    parser.add_argument(
        "--device-map", action="append", default=None,
        help="PVPS02: ARCH=ID[,ID...] (repeatable) -- the real, ordered device pool for "
             "one architecture. Required with --run-performance-benchmark; never inferred. "
             "Also consumed by --validation-producer (device indices there must be integers).",
    )
    parser.add_argument(
        "--validation-producer",
        dest="validation_producer",
        metavar="PATCH/PRODUCER_ID",
        default=None,
        help="PA36-F step 5: select one patch-local validation producer "
             "(patches/<patch>/validation/producer.toml's [producer.<PRODUCER_ID>]) "
             "and execute it through the generic execute_validation_producer() "
             "dispatcher. Mutually exclusive with every --run-rdXX-*/--run-patchXXXX-* "
             "legacy execution mode -- this is the non-legacy replacement path "
             "(PA36's atomic migration sequence retires the legacy flags one at a "
             "time). Repeatable --producer-input NAME=VALUE supplies its declared "
             "inputs.",
    )
    parser.add_argument(
        "--producer-input",
        dest="producer_inputs",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="PA36-F step 5: one producer-declared input (repeatable). Fails closed "
             "if the producer does not declare NAME, a required NAME is missing, or "
             "the same NAME is given twice.",
    )
    args = parser.parse_args(argv)
    if args.validation_producer is not None:
        # PA36-F step 5, GPT design section 5 (req_8ec9b90c05f84a30): generic
        # dispatch plugs in immediately after parse_args()/common patch
        # resolution, before the first other RD-only
        # guard. Reject generically by NAME PATTERN, never a hardcoded tuple
        # of known RD flags -- a new --run-rdNN-* flag added later is caught
        # automatically, with no edit required here.
        legacy_modes = tuple(
            name for name, value in vars(args).items()
            if value
            and (
                re.fullmatch(r"run_rd\d+.*", name)
                or re.fullmatch(r"run_patch\d+.*", name)
            )
        )
        if legacy_modes:
            parser.error(
                "--validation-producer is mutually exclusive with legacy execution "
                f"mode(s): {', '.join(sorted(legacy_modes))}"
            )
        selector_patch, producer_id = _parse_validation_producer_selector(args.validation_producer)
        # --patch stays required at the parser level (retiring that
        # requirement is the atomic migration sequence's job, not step 5's);
        # while it is, this just enforces it can never silently diverge from
        # the selector instead of asking the user to specify the patch twice.
        if args.patch != selector_patch:
            parser.error(
                f"--patch {args.patch!r} does not match --validation-producer's patch "
                f"component {selector_patch!r} -- do not specify a different patch twice"
            )
        provided_inputs = _parse_producer_inputs(args.producer_inputs)
        return _run_validation_producer(args, producer_id=producer_id, provided_inputs=provided_inputs)
    if (
        not args.framework_configuration
        and not args.run_performance_benchmark
        and (args.model is None or args.manifest is None or args.amdgpu_targets is None)
    ):
        parser.error("runtime qualification requires --model, --manifest, and --amdgpu-targets")
    if args.run_performance_benchmark:
        if args.model_root is None or not args.device_map:
            parser.error("--run-performance-benchmark requires --model-root and --device-map")
        if any(getattr(args, name, False) for name in (
            "run_rd08_lanes", "run_rd08_contract",
            "run_rd73_contract",
        )):
            parser.error("--run-performance-benchmark is mutually exclusive with the legacy RD modes")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

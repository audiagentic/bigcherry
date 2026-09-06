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
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from pathlib import Path

from bigcherry.build.builds import capture_completed_build_evidence
from bigcherry.campaign.bench_runner import (  # noqa: F401
    BENCH_RUNNER_ROOT, BenchRunnerError, run_bench_runner_server_bench,
)
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
    )
    positive_hit = pattern.search(positive_output) is not None

    _print(f"activation probe: {description} (fusion-disabled control)")
    negative_output = _run_one_trace_probe(
        name="fusion-disabled", binary=binary, model=model, hip_path=hip_path, workdir=workdir,
        bench_prompt=bench_prompt, bench_gen=bench_gen, disable_fusion=True,
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
    contract: object | None, named_results: "dict[str, object] | None" = None,
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
    except experiment_contract.ExperimentContractError:
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
    control_build_evidence: object, validation_subject_build_evidence: object, *, patch_id: str,
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
    evaluate_promotion_gate(), and the eligibility cutover are deferred)."""
    from bigcherry.experiment import contract as experiment_contract
    from bigcherry.experiment import execution as experiment_execution
    from bigcherry.campaign.benchmark import sanitize_environment

    positive_pattern = re.compile(r"tg128\s*\|\s*([0-9.]+)")
    control_pattern = re.compile(r"pp512\s*\|\s*([0-9.]+)")

    # GPT round 3 (req_e75c4936e2354351): _hip_env() alone does not strip
    # inherited GGML_HIP_DISPATCH_*/GGML_HIP_FORCE_*/GGML_HIP_TUNE_*
    # overrides from the ambient shell -- a validation lane must run in the
    # same "stock" contamination-free environment sanitize_environment()
    # already establishes for native campaign arms, plus clearing
    # BIGCHERRY_*/GGML_CUDA_DISABLE_FUSION which that helper does not touch.
    clean_env = sanitize_environment(_hip_env(hip_path), mode="stock")
    for key in list(clean_env):
        if key.startswith("BIGCHERRY_") or key == "GGML_CUDA_DISABLE_FUSION":
            clean_env.pop(key, None)

    # Raw per-arm stdout/stderr, keyed by (workload, mode, pair) -- VA14-B's
    # own docstring promises this is retained; run_paired_lane()'s
    # PairedLaneRun.runs only carries {pair, mode, metrics} (VA14's already-
    # reviewed contract), so it is captured here, at the runner boundary,
    # instead of changing that primitive's API.
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
                # A nonzero-returncode run already fails via its own real
                # signal (LaneExecutionError downstream); GPU-execution
                # evidence is only meaningful to demand of an apparently
                # successful run, which is exactly the case that silently
                # accepted a CPU-fallback result before this fix.
                _require_real_gpu_execution(
                    completed.stdout, completed.stderr,
                    context=f"rd08 {workload} lane ({Path(command[0]).name})",
                )
            return experiment_execution.RunnerOutput(
                returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr,
            )
        return _runner

    decode_control_cmd, decode_subject_cmd = rd08_validation_lane_commands(
        control_binary=control_binary, subject_binary=subject_binary, model=model, workload="decode",
    )
    decode_run = experiment_execution.run_paired_lane(
        metric="tg128", control_command=decode_control_cmd, subject_command=decode_subject_cmd,
        pattern=positive_pattern, pairs=pairs, runner=_make_runner("decode"),
    )
    prefill_control_cmd, prefill_subject_cmd = rd08_validation_lane_commands(
        control_binary=control_binary, subject_binary=subject_binary, model=model, workload="prefill",
    )
    prefill_run = experiment_execution.run_paired_lane(
        metric="pp512", control_command=prefill_control_cmd, subject_command=prefill_subject_cmd,
        pattern=control_pattern, pairs=pairs, runner=_make_runner("prefill"),
    )
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

    try:
        rows = rd08_correctness.require_rd08_correctness_evidence(
            subject_binary=subject_bin / f"test-backend-ops{exe}",
            control_binary=control_bin / f"test-backend-ops{exe}",
            runner=_correctness_runner,
        )
        result = experiment_contract.CorrectnessResult(
            check="bit_identical", passed=True,
            detail=f"{len(rows)} (shape,seed) pairs bit-identical",
        )
        rows_doc = [{"shape": r.shape_name, "seed": r.seed, "ok": r.ok} for r in rows]
    except rd08_correctness.Rd08CorrectnessError as exc:
        result = experiment_contract.CorrectnessResult(
            check="bit_identical", passed=False, detail=str(exc),
        )
        rows_doc = []

    correctness_doc = {
        "check": "bit_identical", "passed": result.passed, "detail": result.detail,
        "subject_source_tree": psi.git_worktree_tree(subject_src),
        "control_source_tree": psi.git_worktree_tree(control_src),
        "subject_build_identity": subject_build_evidence.campaign_identity(),
        "control_build_identity": control_build_evidence.campaign_identity(),
        "rows": rows_doc,
    }
    artifact_ref = _write_bound_artifact(run_dir, "rd08-correctness.json", correctness_doc)
    return {
        "results": {"bit_identical": result}, "artifact": artifact_ref,
        "subject_build_identity": subject_build_evidence.campaign_identity(),
        "control_build_identity": control_build_evidence.campaign_identity(),
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


def run_rd04_benchmark_evidence(
    *, control_binary: Path, subject_binary: Path, model: Path, hip_path: Path,
    run_dir: Path, campaign_id: str, amdgpu_targets: str,
    control_build_identity: dict[str, object], subject_build_identity: dict[str, object],
    pairs: int = 3,
) -> dict[str, object]:
    """VA04 hardware-free preflight slice (GPT session ses_5bbee8ce5c9a4265,
    req_da015a1366044ad1): an RD04-scoped validation-domain paired
    benchmark producer, analogous to RD08's real lanes (run_rd08_
    validation_lanes()) but deliberately without contract promotion or
    generalisation -- this slice only proves a real benchmark executed
    and binds it as evidence; qualification against RD04's real
    acceptance thresholds is separate, later, real-hardware work. Does
    NOT depend on the generic S1-S7 campaign succeeding -- that pipeline's
    own promotion decision is unrelated to RD04's own validation-domain
    evidence (the exact real bug VA15 found and fixed for RD08).

    ``passed`` means "benchmark evidence executed successfully" (both
    paired lanes completed with finite statistics), NOT "RD04 met its
    3.39% target" -- threshold qualification stays separate."""
    import math

    from bigcherry.experiment import execution as experiment_execution
    from bigcherry.campaign.benchmark import sanitize_environment

    rd04_flags = ["-fa", "on", "-ctk", "bf16", "-ctv", "bf16"]

    def _command(binary: Path, workload_flags: list[str]) -> list[str]:
        return [str(binary), "-m", str(model), *workload_flags, *rd04_flags, "-ngl", "99"]

    clean_env = sanitize_environment(_hip_env(hip_path), mode="stock")
    for key in list(clean_env):
        if key.startswith("BIGCHERRY_") or key == "GGML_CUDA_DISABLE_FUSION":
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
                    context=f"rd04 {workload} lane ({Path(command[0]).name})",
                )
            return experiment_execution.RunnerOutput(
                returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr,
            )
        return _runner

    decode_control_cmd = _command(control_binary, ["-p", "0", "-n", "128"])
    decode_subject_cmd = _command(subject_binary, ["-p", "0", "-n", "128"])
    decode_run = experiment_execution.run_paired_lane(
        metric="tg128", control_command=decode_control_cmd, subject_command=decode_subject_cmd,
        pattern=re.compile(r"tg128\s*\|\s*([0-9.]+)"), pairs=pairs, runner=_make_runner("decode"),
    )
    prefill_control_cmd = _command(control_binary, ["-p", "512", "-n", "0"])
    prefill_subject_cmd = _command(subject_binary, ["-p", "512", "-n", "0"])
    prefill_run = experiment_execution.run_paired_lane(
        metric="pp512", control_command=prefill_control_cmd, subject_command=prefill_subject_cmd,
        pattern=re.compile(r"pp512\s*\|\s*([0-9.]+)"), pairs=pairs, runner=_make_runner("prefill"),
    )

    def _finite(stats: dict[str, object]) -> bool:
        value = stats.get("geometric_effect_pct")
        return isinstance(value, (int, float)) and math.isfinite(value)

    passed = _finite(decode_run.stats) and _finite(prefill_run.stats)

    performance_doc = {
        "passed": passed, "campaign_id": campaign_id,
        "model": str(model), "architecture": amdgpu_targets,
        "validation_build_identities": {
            "control": control_build_identity, "subject": subject_build_identity,
        },
        "commands": {
            "decode": {"control": decode_control_cmd, "subject": decode_subject_cmd},
            "prefill": {"control": prefill_control_cmd, "subject": prefill_subject_cmd},
        },
        "raw_logs": raw_logs,
        "metrics": {
            "decode": {"metric": "tg128", "stats": decode_run.stats, "runs": list(decode_run.runs)},
            "prefill": {"metric": "pp512", "stats": prefill_run.stats, "runs": list(prefill_run.runs)},
        },
    }
    performance_path = run_dir / "performance.json"
    _atomic_write_json(performance_path, performance_doc)
    artifact_ref = {
        "path": "performance.json",
        "sha256": hashlib.sha256(performance_path.read_bytes()).hexdigest(),
    }
    return {"performance_doc": performance_doc, "artifact": artifact_ref, "passed": passed}


def run_rd58_state_restore_evidence(
    *, control_binary: Path, subject_binary: Path, model: Path, hip_path: Path,
    run_dir: Path, campaign_id: str,
    control_build_identity: dict[str, object], subject_build_identity: dict[str, object],
    observed_devices: dict[str, object] | None = None,
    repetitions: int = 3,
) -> dict[str, object]:
    """VA05: an RD58-scoped validation-domain state-restore evidence
    producer -- unlike RD04/RD08 (paired benchmark lanes), RD58 is a
    correctness/reliability contract with NO performance claim. Runs the
    real ``test-save-load-state`` binary (a real llama.cpp upstream test,
    not a new correctness engine) against parity control/validation-
    subject builds, under the SAME GGML_CUDA_REGISTER_HOST=1 env and
    ``-sm tensor`` topology (the ambient dual-GPU visibility is preserved
    -- RD58's contract requires 2+ real GPUs, this producer never
    restricts to one device the way RD04/RD08's single-GPU producers do).

    Three real, independent claims, all from the SAME repeated runs:
    - correctness: the subject binary's test-save-load-state exit code
      (0 iff every one of its 5 internal tests, including "Test 4: seq
      copy (host)", actually passed -- that internal baseline-vs-
      restored-continuation comparison IS the real correctness proof;
      this function does not re-implement or second-guess it).
    - activation: subject-hit/control-miss on the real
      "pinned state buffer (... bytes) for restore" registration-success
      marker RD58's own diagnostic emits -- NOT the generic tune-binary/
      fusion-disabled negative control, which is meaningless here.
    - controls: a real repeated control/subject execution record proving
      no crash/regression across ``repetitions`` real restore cycles --
      no invented latency threshold (this contract carries none)."""
    from bigcherry.campaign.benchmark import sanitize_environment

    marker_pattern = re.compile(r"pinned state buffer \(\d+ bytes\) for restore")

    # GPT direction (session ses_5bbee8ce5c9a4265): preserve ambient
    # dual-GPU visibility -- unlike RD04/RD08's single-GPU producers,
    # this must NOT restrict HIP_VISIBLE_DEVICES/ROCR_VISIBLE_DEVICES
    # (sanitize_environment() never touches those; it only strips stale
    # GGML_HIP_DISPATCH_*/FORCE_*/TUNE_* overrides, same as RD04/RD08).
    env = sanitize_environment(_hip_env(hip_path), mode="stock")
    for key in list(env):
        if key.startswith("BIGCHERRY_"):
            env.pop(key, None)
    env["GGML_CUDA_REGISTER_HOST"] = "1"

    def _command(binary: Path) -> list[str]:
        return [str(binary), "-m", str(model), "-sm", "tensor", "-ngl", "99"]

    def _run(binary: Path, role: str, index: int) -> dict[str, object]:
        command = _command(binary)
        completed = subprocess.run(command, capture_output=True, text=True, check=False, env=env)
        combined = completed.stdout + "\n" + completed.stderr
        return {
            "role": role, "index": index, "command": command,
            "returncode": completed.returncode,
            "marker_hit": marker_pattern.search(combined) is not None,
            "stdout": completed.stdout, "stderr": completed.stderr,
        }

    subject_runs = [_run(subject_binary, "subject", i) for i in range(repetitions)]
    control_runs = [_run(control_binary, "control", i) for i in range(repetitions)]

    correctness_passed = all(r["returncode"] == 0 for r in subject_runs)
    subject_hit = any(r["marker_hit"] for r in subject_runs)
    control_hit = any(r["marker_hit"] for r in control_runs)
    # GPT round 3: the controls artifact claims "no crash/regression
    # across repeated control/subject execution" -- it must fail if
    # EITHER arm fails, not just the control arm (a subject that only
    # crashed on repeated runs would otherwise be reported as a controls
    # pass).
    controls_passed = (
        all(r["returncode"] == 0 for r in control_runs)
        and all(r["returncode"] == 0 for r in subject_runs)
    )
    hardware_doc = dict(observed_devices) if observed_devices else {}

    def _strip_output(runs: list[dict[str, object]]) -> list[dict[str, object]]:
        # Persist returncode/marker_hit/command in full; truncate raw
        # stdout/stderr to a bounded tail per run so the artifact stays
        # a reasonable size across `repetitions` real process launches.
        return [
            {**r, "stdout": r["stdout"][-2000:], "stderr": r["stderr"][-2000:]}
            for r in runs
        ]

    correctness_doc = {
        "ops": ["STATE_RESTORE_SEQ_CP_HOST"], "passed": correctness_passed,
        "campaign_id": campaign_id,
        "validation_build_identities": {
            "control": control_build_identity, "subject": subject_build_identity,
        },
        "hardware": hardware_doc,
        "subject_runs": _strip_output(subject_runs),
    }
    correctness_path = run_dir / "rd58-correctness.json"
    _atomic_write_json(correctness_path, correctness_doc)
    correctness_artifact = {
        "path": "rd58-correctness.json",
        "sha256": hashlib.sha256(correctness_path.read_bytes()).hexdigest(),
    }

    trigger_doc = {
        "marker_regex": marker_pattern.pattern, "subject_hit": subject_hit, "control_hit": control_hit,
        "subject_runs": _strip_output(subject_runs), "control_runs": _strip_output(control_runs),
    }
    subject_log_path = run_dir / "rd58-trigger-subject.log"
    control_log_path = run_dir / "rd58-trigger-control.log"
    subject_log_path.write_text(
        "\n---\n".join(f"{r['role']}-{r['index']}:\n{r['stdout']}\n{r['stderr']}" for r in subject_runs),
        encoding="utf-8",
    )
    control_log_path.write_text(
        "\n---\n".join(f"{r['role']}-{r['index']}:\n{r['stdout']}\n{r['stderr']}" for r in control_runs),
        encoding="utf-8",
    )
    trigger_path = run_dir / "rd58-trigger.json"
    _atomic_write_json(trigger_path, trigger_doc)
    trigger_artifact = {
        "path": "rd58-trigger.json",
        "sha256": hashlib.sha256(trigger_path.read_bytes()).hexdigest(),
    }

    controls_doc = {
        "campaign_id": campaign_id, "passed": controls_passed, "repetitions": repetitions,
        "metrics": {
            "control_pass_count": sum(1 for r in control_runs if r["returncode"] == 0),
            "subject_pass_count": sum(1 for r in subject_runs if r["returncode"] == 0),
        },
        "validation_build_identities": {
            "control": control_build_identity, "subject": subject_build_identity,
        },
        "hardware": hardware_doc,
    }
    controls_path = run_dir / "performance.json"
    _atomic_write_json(controls_path, controls_doc)
    controls_artifact = {
        "path": "performance.json",
        "sha256": hashlib.sha256(controls_path.read_bytes()).hexdigest(),
    }

    return {
        "correctness_passed": correctness_passed, "correctness_artifact": correctness_artifact,
        "subject_hit": subject_hit, "control_hit": control_hit, "trigger_artifact": trigger_artifact,
        "subject_log_path": "rd58-trigger-subject.log", "control_log_path": "rd58-trigger-control.log",
        "controls_passed": controls_passed, "controls_artifact": controls_artifact,
    }


_LANE_EFFECT_FIELDS = (
    "geometric_effect_pct", "ci95_low_pct", "ci95_high_pct", "paired_rounds",
)


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
        promotions.get(contract_id, {}).get("passed") is True
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
    host: str = "127.0.0.1", control_port: int = 18080, subject_port: int = 18081,
    spec_draft_n_max: int = 4,
    n_predict: int = 128, warmup_pairs: int = 2, measured_pairs: int = 10,
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
    from bigcherry.tuning.server_runner import ServerRunner

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
    rd73_env = {"BIGCHERRY_PATCH_TRACE": "1", "BIGCHERRY_RD73_RESOURCE_TRACE": "1"}
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
        runner = ServerRunner(
            binary=binaries[arm], model=model, host=host, port=ports[arm],
            extra_args=server_args, log_path=log_path, env_overrides=rd73_env,
        )
        with runner:
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
    host: str = "127.0.0.1", control_port: int = 18082, subject_port: int = 18083,
    pairs: int = 3, extra_flags: tuple[str, ...] = ("-sm", "tensor", "--fit", "off"),
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
    from bigcherry.tuning.server_runner import ServerRunner

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
        runner = ServerRunner(
            binary=binaries[arm], model=model, host=host, port=ports[arm],
            extra_args=server_args, log_path=logs_dir / f"rd73-decode-{arm}-server-{index}.log",
        )
        with runner:
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
    host: str = "127.0.0.1", port: int = 18084, burst_requests: int = 20, n_predict: int = 32,
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
    from bigcherry.tuning.server_runner import ServerRunner

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
    runner = ServerRunner(
        binary=subject_binary, model=model, host=host, port=port,
        extra_args=server_args, log_path=log_path, env_overrides=rd73_env,
    )
    sampling = sc.SamplingConfig(temperature=1.0, top_p=0.95, top_k=20)
    config = sc.SessionConfig(
        session_id="rd73-resource-burst", corpus_id=corpus_path.stem, corpus_sha256="",
        bigcherry_revision="rd73-va06", llama_pin="", llama_revision="", model_id=str(model),
        server_argv=server_args, spec_type="draft-mtp", spec_n_max=4,
        spec_draft_k="default", spec_draft_v="default", sampling=sampling,
        n_predict=n_predict, order_seed=12345,
    )
    with runner:
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

    mtp = run_rd73_mtp_server_lane(
        control_binary=control_server_binary, subject_binary=subject_server_binary, model=model,
        corpus_path=corpus_path, run_dir=run_dir, warmup_pairs=warmup_pairs, measured_pairs=measured_pairs,
    )
    activation = evaluate_rd73_activation_evidence(
        marker_regex=marker_regex, control_log_path=mtp["control_log_path"],
        subject_log_path=mtp["subject_log_path"], run_dir=run_dir,
    )
    resource = run_rd73_resource_burst_session(
        subject_binary=subject_server_binary, model=model, corpus_path=corpus_path, run_dir=run_dir,
    )
    decode_control = run_rd73_decode_control_lane(
        control_binary=control_server_binary, subject_binary=subject_server_binary, model=model,
        run_dir=run_dir, pairs=decode_pairs,
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
    # schema the generic validator reads -- the same thing RD58 does via
    # run_rd58_state_restore_evidence()'s controls_doc. Nothing here is
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
        "run_rd08_lanes", "run_rd08_contract", "run_rd04_benchmark",
        "run_rd58_state_restore", "run_rd73_contract", "correctness_evidence",
    )):
        raise PatchCampaignError("framework configuration cannot be combined with runtime qualification modes")
    targets = tuple(target.strip() for target in re.split(r"[,;]", args.amdgpu_targets) if target.strip())
    if not targets or any(not re.fullmatch(r"gfx[0-9a-f]+", target) for target in targets):
        raise PatchCampaignError("framework configuration requires explicit AMDGPU compile targets")
    args.amdgpu_targets = ";".join(targets)
    from bigcherry.core.context import ProjectContext
    base_repo = ProjectContext.resolve(work_root=os.environ.get("BC_CACHE")).upstream_repo
    baseline_source = "bigcherry-native"
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
    for role, diagnostic_on in (("production", False), ("diagnostic", True)):
        observed = inspect_dispatch_build(build_root / f"framework-{role}")
        counts = observed["compiled_definition_counts"]
        if observed["issues"] or bool(counts["GGML_HIP_DISPATCH_DIAGNOSTICS"]) != diagnostic_on:
            raise PatchCampaignError(f"{role} diagnostic compiler state disagrees with qualification role")
    run_dir = args.workdir / "framework" / descriptor.patch_id
    run_dir.mkdir(parents=True, exist_ok=False)
    generated_artifact = _write_bound_artifact(run_dir, "generated-tree.json", generated_manifest)
    source_artifact = _write_bound_artifact(run_dir, "source-tree.json", source_manifest)
    builds = {"production": production.campaign_identity(), "diagnostic": diagnostic.campaign_identity()}
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
        generated_inputs={role: {"proof": "compiled-copy-v1",
            "compile_inputs_hash": proof[f"framework-{role}"]["compile_inputs_hash"],
            "tree_manifest": proof[f"framework-{role}"], "build_identity": builds[role],
        } for role in builds},
        check_results=checks, artifact_hashes=artifacts, campaign_workdir=run_dir,
    )
    path = patch_validation_evidence.write_record(record)
    _print(f"framework configuration evidence: {path}")
    return 0 if record["eligible_for_validated_state"] else 1


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

    worktree_root: Path = args.worktree_root
    # RV80/B6: the baseline is the source's EXPLICIT named composition from
    # config/recipes.toml (never the retired implicit state=='validated'
    # scan), resolved through the exact-composition validator; the base ref
    # resolves to an immutable SHA that enters the v2 source identity.
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
    _print(f"materializing control and subject source plans @ {base_revision[:12]} ...")
    control_src = psi.materialize_composition(
        base_repo=LLAMA_CPP_SRC, worktree_root=worktree_root / "control",
        resolved_revision=base_revision, composition=control_composition,
        overlay_root=psi.REPO_ROOT / "src", requested_revision=cfg.pinned,
    )
    patched_src = psi.materialize_composition(
        base_repo=LLAMA_CPP_SRC, worktree_root=worktree_root / "subject",
        resolved_revision=base_revision, composition=subject_composition,
        overlay_root=psi.REPO_ROOT / "src", requested_revision=cfg.pinned,
    )
    _print(f"control source: {control_src}")
    _print(f"subject source: {patched_src}")
    control_idempotent = psi.verify_composition_idempotent(
        base_repo=LLAMA_CPP_SRC, source=control_src, worktree_root=worktree_root / "control",
        resolved_revision=base_revision, composition=control_composition,
        overlay_root=psi.REPO_ROOT / "src", requested_revision=cfg.pinned,
    )
    subject_idempotent = psi.verify_composition_idempotent(
        base_repo=LLAMA_CPP_SRC, source=patched_src, worktree_root=worktree_root / "subject",
        resolved_revision=base_revision, composition=subject_composition,
        overlay_root=psi.REPO_ROOT / "src", requested_revision=cfg.pinned,
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

    tune_extra_cmake_args = [
        "-DGGML_HIP_AUTOTUNE=ON", "-DGGML_HIP_AUTOTUNE_RECORD=ON",
        "-DGGML_HIP_ROUTING_TRANSFORM=ON",
        f"-DGGML_HIP_AUTOTUNE_GENERATED_DIR={generated_dir}",
    ]
    tune_cmake_args = _full_requested_cmake_args(
        hip_path=args.hip_path, amdgpu_targets=args.amdgpu_targets,
        extra_cmake_args=tune_extra_cmake_args,
    )
    tune_bin = build_tree(
        name="tune", hip_path=args.hip_path, amdgpu_targets=args.amdgpu_targets,
        workdir=build_root, targets=["llama-server", "llama-bench"], source=patched_src,
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
        build_root / "tune", source_root=patched_src, architecture=args.amdgpu_targets,
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
        hip_path=args.hip_path, amdgpu_targets=args.amdgpu_targets,
        extra_cmake_args=replay_extra_cmake_args,
    )
    replay_bin = build_tree(
        name="replay", hip_path=args.hip_path, amdgpu_targets=args.amdgpu_targets,
        workdir=build_root, targets=["llama-server", "llama-bench"], source=patched_src,
        extra_cmake_args=replay_extra_cmake_args,
    )
    replay_build_evidence = capture_completed_build_evidence(
        build_root / "replay", source_root=patched_src, architecture=args.amdgpu_targets,
        binary=replay_bin / f"llama-server{exe}",
        extra_binaries=(replay_bin / f"llama-bench{exe}",),
        requested_cmake_args=replay_cmake_args, build_env=build_env,
    )
    _print(
        f"replay build: {replay_build_evidence.effective_build_id[:12]} / "
        f"{replay_build_evidence.runtime_bundle_hash[:12]} / "
        f"{replay_build_evidence.compile_verification_id[:12]}"
    )

    stock_build_root = (args.build_root or workdir) / stock_src.name
    stock_bin = ensure_stock_baseline(
        hip_path=args.hip_path, amdgpu_targets=args.amdgpu_targets,
        workdir=stock_build_root, stock_src=stock_src,
    )
    stock_cmake_args = _full_requested_cmake_args(
        hip_path=args.hip_path, amdgpu_targets=args.amdgpu_targets, extra_cmake_args=[],
    )
    stock_build_evidence = capture_completed_build_evidence(
        stock_build_root / "stock", source_root=stock_src, architecture=args.amdgpu_targets,
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
    control_build_root = (args.build_root or workdir) / control_src.name
    control_bin = build_tree(
        name="control", hip_path=args.hip_path, amdgpu_targets=args.amdgpu_targets,
        workdir=control_build_root, targets=["llama-server", "llama-bench"],
        source=control_src, extra_cmake_args=[],
    )
    control_build_evidence = capture_completed_build_evidence(
        control_build_root / "control", source_root=control_src,
        architecture=args.amdgpu_targets, binary=control_bin / f"llama-bench{exe}",
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
        name="validation-subject", hip_path=args.hip_path, amdgpu_targets=args.amdgpu_targets,
        workdir=build_root, targets=["llama-server", "llama-bench"], source=patched_src,
        extra_cmake_args=[],
    )
    # GPT round 3 (req_e75c4936e2354351): capture symmetrically with
    # control_build_evidence below (binary=llama-bench only, no
    # extra_binaries) -- an asymmetric capture is not a like-for-like
    # comparison even when the underlying build tree is parity.
    validation_subject_build_evidence = capture_completed_build_evidence(
        build_root / "validation-subject", source_root=patched_src,
        architecture=args.amdgpu_targets, binary=validation_subject_bin / f"llama-bench{exe}",
        requested_cmake_args=stock_cmake_args, build_env=build_env,
    )
    assert_validation_subject_parity(
        control_build_evidence, validation_subject_build_evidence, patch_id=args.patch,
    )
    _print(
        f"validation-subject build: {validation_subject_build_evidence.effective_build_id[:12]} / "
        f"{validation_subject_build_evidence.runtime_bundle_hash[:12]} / "
        f"{validation_subject_build_evidence.compile_verification_id[:12]}"
    )

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
    # VA04: --run-rd04-benchmark also skips the generic probe -- RD04 has
    # no real activation marker yet (see README.md's Known limitations),
    # and the generic negative control (GGML_CUDA_DISABLE_FUSION) is not
    # valid for a flash-attention patch. Activation stays explicitly
    # BLOCKED for this slice rather than fabricated.
    # VA06: --run-rd73-contract also skips the generic probe -- the
    # generic tune-binary/GGML_CUDA_DISABLE_FUSION negative control is
    # not valid for RD73 (graph-cache keying, not a fusion path), and
    # the generic probe's plain llama-bench invocation (no -sm tensor)
    # cannot even load RD73's real 27B contract model on Brutus's dual
    # gfx1100 GPUs. RD73's own authoritative activation evidence comes
    # from evaluate_rd73_activation_evidence() inside
    # run_rd73_contract_qualification().
    trace_result = None if (args.run_rd08_contract or args.run_rd04_benchmark or args.run_rd58_state_restore or args.run_rd73_contract) else run_trace_activation_probes(
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
    if not (args.run_rd08_contract or args.run_rd04_benchmark or args.run_rd58_state_restore or args.run_rd73_contract):
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

    # VA04: --run-rd04-benchmark, RD04-only, mutually exclusive with the
    # RD08 execution modes above (each patch's own real evidence producer
    # is used only for its own contract). Binds only performance_evidence
    # -- contract_promotions stays empty for RD04 in this slice, so
    # eligible_for_validated_state remains False even on a full PASS: this
    # command produces truthful ported-benched-level evidence, never a
    # pretend contract-promotion PASS.
    if args.run_rd04_benchmark:
        if args.run_rd08_lanes or args.run_rd08_contract:
            raise PatchCampaignError(
                f"{args.patch}: --run-rd04-benchmark is mutually exclusive with the "
                "RD08 execution modes"
            )
        if descriptor.experiment_contract != "RD04-BF16-FLASH-ATTN-TILE":
            raise PatchCampaignError(
                f"{args.patch}: --run-rd04-benchmark is RD04-only today"
            )
        rd04_result = run_rd04_benchmark_evidence(
            control_binary=control_bin / f"llama-bench{exe}",
            subject_binary=validation_subject_bin / f"llama-bench{exe}", model=args.model,
            hip_path=args.hip_path, run_dir=campaign_run_dir,
            campaign_id=campaign.campaign_identity_digest, amdgpu_targets=args.amdgpu_targets,
            control_build_identity=control_build_evidence.campaign_identity(),
            subject_build_identity=validation_subject_build_evidence.campaign_identity(),
        )
        performance_evidence = {"artifact": rd04_result["artifact"]}
        _print(
            f"rd04 benchmark evidence: {rd04_result['artifact']['path']} "
            f"({'executed' if rd04_result['passed'] else 'did not execute cleanly'})"
        )

    # VA05: --run-rd58-state-restore, RD58-only, mutually exclusive with
    # the RD04/RD08 execution modes. Builds its own parity control/
    # validation-subject test-save-load-state binaries (not
    # llama-server/llama-bench -- a different real llama.cpp test
    # target) and binds correctness_evidence/trace_evidence/
    # performance_evidence from real, repeated, dual-GPU execution.
    # contract_promotions stays empty -- eligible_for_validated_state
    # remains False even on a full PASS.
    if args.run_rd58_state_restore:
        if args.run_rd04_benchmark or args.run_rd08_lanes or args.run_rd08_contract:
            raise PatchCampaignError(
                f"{args.patch}: --run-rd58-state-restore is mutually exclusive with the "
                "RD04/RD08 execution modes"
            )
        if descriptor.experiment_contract != "RD58-PIN-STATE-BUFFER-MULTIGPU-RESTORE":
            raise PatchCampaignError(
                f"{args.patch}: --run-rd58-state-restore is RD58-only today"
            )
        # GPT round 2 (req_3616cc1d90dc4512): the contract's own
        # scope.gpu_count.minimum=2 was not execution-enforced anywhere --
        # preserving ambient device visibility is necessary but not
        # sufficient; a real one-GPU invocation could still execute (and
        # potentially qualify) against a contract that requires 2+.
        # Fail closed before any build: require HIP_VISIBLE_DEVICES and
        # ROCR_VISIBLE_DEVICES both explicitly set, consistent with each
        # other, and exposing at least the contract's declared minimum.
        from bigcherry.experiment import contract as _ec

        rd58_contract_check = _ec.load_contracts(
            REPO_ROOT / "config" / "experiment-contracts.toml"
        ).contracts["RD58-PIN-STATE-BUFFER-MULTIGPU-RESTORE"]
        required_gpu_count = (
            rd58_contract_check.scope.gpu_count.minimum
            if rd58_contract_check.scope.gpu_count is not None else None
        )
        hip_devices_raw = os.environ.get("HIP_VISIBLE_DEVICES")
        rocr_devices_raw = os.environ.get("ROCR_VISIBLE_DEVICES")
        if not hip_devices_raw or not rocr_devices_raw:
            raise PatchCampaignError(
                f"{args.patch}: --run-rd58-state-restore requires HIP_VISIBLE_DEVICES and "
                "ROCR_VISIBLE_DEVICES to be explicitly set (this contract requires "
                f"{required_gpu_count}+ real GPUs -- an unset/ambient-default device list "
                "cannot be trusted to expose that many)"
            )
        hip_device_ids = [d.strip() for d in hip_devices_raw.split(",") if d.strip()]
        rocr_device_ids = [d.strip() for d in rocr_devices_raw.split(",") if d.strip()]
        if hip_device_ids != rocr_device_ids:
            raise PatchCampaignError(
                f"{args.patch}: HIP_VISIBLE_DEVICES ({hip_devices_raw!r}) and "
                f"ROCR_VISIBLE_DEVICES ({rocr_devices_raw!r}) must match exactly"
            )
        # GPT round 3: a duplicated id (e.g. "0,0") must not count as 2
        # distinct real GPUs.
        if len(set(hip_device_ids)) != len(hip_device_ids):
            raise PatchCampaignError(
                f"{args.patch}: HIP_VISIBLE_DEVICES/ROCR_VISIBLE_DEVICES contains duplicate "
                f"device ids ({hip_devices_raw!r}) -- this does not expose distinct real GPUs"
            )
        if required_gpu_count is not None and len(hip_device_ids) < required_gpu_count:
            raise PatchCampaignError(
                f"{args.patch}: --run-rd58-state-restore requires {required_gpu_count}+ GPUs; "
                f"HIP_VISIBLE_DEVICES/ROCR_VISIBLE_DEVICES expose only {len(hip_device_ids)} "
                f"({hip_devices_raw!r})"
            )
        rd58_observed_devices = {
            "hip_visible_devices": hip_device_ids, "rocr_visible_devices": rocr_device_ids,
            "gpu_count": len(hip_device_ids),
        }

        rd58_build_root = build_root / "rd58-state-restore"
        rd58_control_bin = build_tree(
            name="rd58-control", hip_path=args.hip_path, amdgpu_targets=args.amdgpu_targets,
            workdir=rd58_build_root, targets=["test-save-load-state"], source=control_src,
            extra_cmake_args=[],
        )
        rd58_subject_bin = build_tree(
            name="rd58-subject", hip_path=args.hip_path, amdgpu_targets=args.amdgpu_targets,
            workdir=rd58_build_root, targets=["test-save-load-state"], source=patched_src,
            extra_cmake_args=[],
        )
        # GPT round 2: RD58's real evidence-producing binary is
        # test-save-load-state, not llama-server/llama-bench -- capture
        # ITS build identities (not the unrelated generic control_build_evidence/
        # validation_subject_build_evidence) and use them everywhere RD58
        # evidence/the final record references a validation build.
        rd58_cmake_args = _full_requested_cmake_args(
            hip_path=args.hip_path, amdgpu_targets=args.amdgpu_targets, extra_cmake_args=[],
        )
        rd58_control_build_evidence = capture_completed_build_evidence(
            rd58_build_root / "rd58-control", source_root=control_src,
            architecture=args.amdgpu_targets, binary=rd58_control_bin / f"test-save-load-state{exe}",
            requested_cmake_args=rd58_cmake_args, build_env=build_env,
        )
        rd58_subject_build_evidence = capture_completed_build_evidence(
            rd58_build_root / "rd58-subject", source_root=patched_src,
            architecture=args.amdgpu_targets, binary=rd58_subject_bin / f"test-save-load-state{exe}",
            requested_cmake_args=rd58_cmake_args, build_env=build_env,
        )
        assert_validation_subject_parity(
            rd58_control_build_evidence, rd58_subject_build_evidence, patch_id=args.patch,
        )
        # GPT round 3: RD58's evidence-producing binary is
        # test-save-load-state, not the generic llama-bench control/
        # subject builds -- ValidationContext must see THESE build
        # identities/evidence when RD58 ran, mirroring the shape the
        # generic path already builds for build_evidence above.
        rd58_build_evidence = {
            "control": {
                "build_id": rd58_control_build_evidence.effective_build_id,
                "source_tree": control_source_tree,
                "architecture": args.amdgpu_targets,
                "options": rd58_control_build_evidence.effective_configure,
                "compile_commands": _write_bound_artifact(
                    campaign_run_dir, "build/rd58-control-compile-commands.json",
                    rd58_control_build_evidence.verification.to_dict(),
                ),
                "runtime_bundle": _write_bound_artifact(
                    campaign_run_dir, "build/rd58-control-runtime-bundle.json",
                    rd58_control_build_evidence.runtime_artifacts,
                ),
            },
            "subject": {
                "build_id": rd58_subject_build_evidence.effective_build_id,
                "source_tree": patched_source_tree,
                "architecture": args.amdgpu_targets,
                "options": rd58_subject_build_evidence.effective_configure,
                "compile_commands": _write_bound_artifact(
                    campaign_run_dir, "build/rd58-subject-compile-commands.json",
                    rd58_subject_build_evidence.verification.to_dict(),
                ),
                "runtime_bundle": _write_bound_artifact(
                    campaign_run_dir, "build/rd58-subject-runtime-bundle.json",
                    rd58_subject_build_evidence.runtime_artifacts,
                ),
            },
        }
        rd58_result = run_rd58_state_restore_evidence(
            control_binary=rd58_control_bin / f"test-save-load-state{exe}",
            subject_binary=rd58_subject_bin / f"test-save-load-state{exe}", model=args.model,
            hip_path=args.hip_path, run_dir=campaign_run_dir,
            campaign_id=campaign.campaign_identity_digest,
            control_build_identity=rd58_control_build_evidence.campaign_identity(),
            subject_build_identity=rd58_subject_build_evidence.campaign_identity(),
            observed_devices=rd58_observed_devices,
        )
        correctness_evidence = {"artifact": rd58_result["correctness_artifact"]}
        performance_evidence = {"artifact": rd58_result["controls_artifact"]}

        # GPT round 2: RD58's contract requires the NAMED
        # state_restore_integrity check, not the adapter's generic
        # backend-ops capability -- compute_contract_correctness_gate()
        # only receives named results for RD08 today, so RD58's real
        # test-save-load-state evidence was never actually connected to
        # its own contract's correctness gate. contract_promotions stays
        # empty regardless -- this is diagnostic, not a promotion.
        rd58_contract_correctness_result = _ec.CorrectnessResult(
            check="state_restore_integrity", passed=rd58_result["correctness_passed"],
            detail=(
                "subject test-save-load-state (all 5 internal tests, including "
                "Test 4: seq copy (host)) "
                + ("passed" if rd58_result["correctness_passed"] else "failed")
            ),
        )
        rd58_contract_correctness_named_results = {
            "state_restore_integrity": rd58_contract_correctness_result,
        }

        def _bind_rd58_log(relative_log_path: str) -> dict[str, str]:
            target = (campaign_run_dir / relative_log_path).resolve()
            return {
                "path": relative_log_path,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }

        trace_evidence = {
            "positive": {
                "marker_regex": trace_marker_regex,
                "artifact": _bind_rd58_log(rd58_result["subject_log_path"]),
            },
            "negative": {
                "marker_regex": trace_marker_regex,
                "artifact": _bind_rd58_log(rd58_result["control_log_path"]),
            },
        }
        activation_evidence = ActivationEvidence(
            status=(
                "executed" if rd58_result["subject_hit"] and not rd58_result["control_hit"]
                else "not_executed"
            ),
            mechanism="rd58-trigger-marker", detail=f"marker={trace_marker_regex!r}",
        )
        activation_verdict = verdict(activation_evidence, correctness_passed=None)
        write_activation_json(
            campaign_run_dir / "activation.json", activation_evidence, activation_verdict,
            extra={
                "campaign_identity_digest": campaign.campaign_identity_digest,
                "rd58_trigger": {
                    "subject_hit": rd58_result["subject_hit"],
                    "control_hit": rd58_result["control_hit"],
                    "artifact": rd58_result["trigger_artifact"],
                },
            },
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
            "disposition": "passed" if rd58_result["correctness_passed"] else "failed",
            "mechanism": "rd58-test-save-load-state-real-execution",
            "detail": (
                "subject test-save-load-state (all 5 internal tests, including "
                "Test 4: seq copy (host)) "
                + ("passed" if rd58_result["correctness_passed"] else "failed")
            ),
        }
        correctness_path = campaign_run_dir / "correctness.json"
        _atomic_write_json(correctness_path, correctness_summary)
        _print(
            f"rd58 state-restore evidence: correctness="
            f"{'pass' if rd58_result['correctness_passed'] else 'fail'} "
            f"activation={'executed' if activation_evidence.status == 'executed' else 'not_executed'} "
            f"controls={'pass' if rd58_result['controls_passed'] else 'fail'}"
        )

    # VA06: RD73 execution, opt-in and scoped to RD73 only. Mirrors RD08's
    # --run-rd08-contract dispatch (mutual exclusion, descriptor check,
    # single orchestrator call, contract_promotions population, pass/fail
    # print).
    #
    # RV95: this block now ALSO rebinds the generic adapter evidence, the
    # way the RD08 block above does. It previously bound none of it, and
    # then bound only part of it (performance/correctness/trace, VA23) --
    # leaving the record's own top-level activation/correctness fields at
    # disposition="unknown". That produced a FAIL-OPEN disagreement: the
    # campaign printed "STATE='validated' eligible: yes" for records that
    # verify_validated_patch() rejected. Both halves are bound here now, so
    # a passing RD73 record satisfies the evidence verifier as well as the
    # eligibility flag. The real, auditable per-contract PASS/FAIL/INVALID
    # verdict is produced and printed here as before.
    rd73_qualification: dict[str, object] | None = None
    if args.run_rd73_contract:
        if args.run_rd08_lanes or args.run_rd08_contract or args.run_rd04_benchmark or args.run_rd58_state_restore:
            raise PatchCampaignError(
                f"{args.patch}: --run-rd73-contract is mutually exclusive with the "
                "RD04/RD08/RD58 execution modes"
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
            if descriptor.package_root is not None else None
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
            build_identities=(
                {
                    "control": rd58_control_build_evidence.effective_build_id,
                    "subject": rd58_subject_build_evidence.effective_build_id,
                } if args.run_rd58_state_restore else {
                    "control": control_build_evidence.effective_build_id,
                    "subject": validation_subject_build_evidence.effective_build_id,
                }
            ),
            build_evidence=(rd58_build_evidence if args.run_rd58_state_restore else build_evidence),
            apply_evidence=apply_evidence,
            architecture=args.amdgpu_targets, model=str(args.model),
            contract=validation_plan.contract,
            contract_hash=(validation_plan.contract.contract_hash if validation_plan.contract else None),
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

        # VA15 real-hardware finding: validation_plan.contract is a
        # patch_validation.ContractBinding -- a lightweight PROJECTION
        # (contract_id/hash/expected_effect/etc) that deliberately does NOT
        # carry .correctness/.acceptance/etc. compute_contract_correctness_gate()
        # needs the real experiment_contract.ExperimentContract, which
        # run_rd08_contract already loaded as rd08_contract for the
        # --run-rd08-contract path; other contract-bound patches load it
        # fresh here the same way that block does.
        full_contract = (
            rd08_contract if rd08_qualification is not None
            else patch_validation.load_contract_for_descriptor(descriptor)
        )
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
                else rd58_contract_correctness_named_results if args.run_rd58_state_restore
                # VA23: RD73's bit_identical result is real and already
                # evaluated inside run_rd73_contract_qualification(); thread
                # it here exactly as RD08's and RD58's are, so the gate
                # reflects the evidence instead of reporting missing_checks.
                else rd73_qualification["correctness_named_results"]
                if rd73_qualification is not None
                else None
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
        validation_build_identities=(
            {
                "control": rd58_control_build_evidence.campaign_identity(),
                "subject": rd58_subject_build_evidence.campaign_identity(),
            }
            if args.run_rd58_state_restore else {
                "control": control_build_evidence.campaign_identity(),
                "subject": validation_subject_build_evidence.campaign_identity(),
            }
        ),
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
    parser.add_argument("--amdgpu-targets", required=True, help="e.g. gfx1100 or gfx1201")
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
        "--run-rd04-benchmark", action="store_true", default=False,
        help="VA04: execute RD04's real paired decode/prefill benchmark lanes against the "
             "parity-verified control/validation-subject builds and bind the result into "
             "ctx.performance_evidence. Diagnostic-only for eligibility -- does not attempt "
             "correctness/activation proof or contract promotion, so "
             "eligible_for_validated_state stays False. RD04-only; an error for any other "
             "patch. Mutually exclusive with the RD08 execution modes.",
    )
    parser.add_argument(
        "--run-rd58-state-restore", action="store_true", default=False,
        help="VA05: build parity control/validation-subject test-save-load-state binaries "
             "and run RD58's real state-restore correctness/activation/controls evidence "
             "(GGML_CUDA_REGISTER_HOST=1, -sm tensor -- requires 2+ real GPUs, ambient "
             "device visibility is preserved, never restricted to one device). "
             "Diagnostic-only for eligibility -- does not attempt contract promotion, so "
             "eligible_for_validated_state stays False. RD58-only; an error for any other "
             "patch. Mutually exclusive with the RD04/RD08 execution modes.",
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
             "RD04/RD08/RD58 execution modes. Requires --rd73-corpus.",
    )
    parser.add_argument(
        "--rd73-corpus", type=Path, default=None,
        help="VA06: prompt corpus JSONL for --run-rd73-contract's MTP server lane "
             "(bench/server_completion.py's load_corpus() format).",
    )
    args = parser.parse_args(argv)
    if not args.framework_configuration and (args.model is None or args.manifest is None):
        parser.error("runtime qualification requires --model and --manifest")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

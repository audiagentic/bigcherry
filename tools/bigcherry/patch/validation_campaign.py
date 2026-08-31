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
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

from bigcherry.build.builds import capture_completed_build_evidence
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


def _run_one_trace_probe(
    *, name: str, binary: Path, model: Path, hip_path: Path, workdir: Path,
    bench_prompt: int, bench_gen: int, disable_fusion: bool,
) -> str:
    import subprocess

    binary = Path(binary)
    model = Path(model)
    if not binary.is_file():
        raise PatchCampaignError(f"activation probe binary does not exist: {binary}")
    if not model.is_file():
        raise PatchCampaignError(f"activation probe model does not exist: {model}")

    command = [
        str(binary.resolve()), "-m", str(model.resolve()),
        "-p", str(bench_prompt), "-n", str(bench_gen), "-r", "1",
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
) -> tuple[list[str], list[str]]:
    """VA14-B: the real, minimal llama-bench command pair for one RD08 lane
    -- control_command, subject_command -- differing only by binary path,
    consistent with metric_for_workload()'s decode->tg128/prefill->pp512
    mapping (decode: -p 0 -n 128; prefill: -p 512 -n 0)."""
    if workload == "decode":
        workload_flags = ["-p", "0", "-n", "128"]
    elif workload == "prefill":
        workload_flags = ["-p", "512", "-n", "0"]
    else:
        raise PatchCampaignError(f"rd08 lane: no llama-bench flag mapping for workload {workload!r}")
    control_command = [str(control_binary), "-m", str(model), *workload_flags]
    subject_command = [str(subject_binary), "-m", str(model), *workload_flags]
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

    try:
        rows = rd08_correctness.require_rd08_correctness_evidence(
            subject_binary=subject_bin / f"test-backend-ops{exe}",
            control_binary=control_bin / f"test-backend-ops{exe}",
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


def compute_persisted_validation_eligible(
    descriptor: object, validation_verdict: object | None,
    contract_promotions: "dict[str, dict[str, object]] | None" = None,
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
    such a patch ever claims, exactly as before."""
    if not descriptor.experiment_contracts:
        if validation_verdict is None:
            return None
        return validation_verdict.eligible
    if validation_verdict is None or not validation_verdict.eligible:
        return False
    promotions = contract_promotions or {}
    return all(
        promotions.get(contract_id, {}).get("passed") is True
        for contract_id in descriptor.experiment_contracts
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

    registry = patch_registry.load_registry(bc_paths.PATCHES)
    descriptor = registry.get(args.patch)

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

    worktree_root: Path = args.worktree_root
    # RV80/B6: the baseline is the source's EXPLICIT named composition from
    # config/recipes.toml (never the retired implicit state=='validated'
    # scan), resolved through the exact-composition validator; the base ref
    # resolves to an immutable SHA that enters the v2 source identity.
    control_revision, control_composition = psi.resolve_source_composition(
        "bigcherry", focal=None, base_ref="HEAD", base_repo=LLAMA_CPP_SRC,
    )
    subject_revision, subject_composition = psi.resolve_source_composition(
        "bigcherry", focal=args.patch, base_ref="HEAD", base_repo=LLAMA_CPP_SRC,
    )
    if control_revision != subject_revision:
        raise RuntimeError("control and subject source plans resolved different base revisions")
    base_revision = subject_revision
    _print(f"materializing control and subject source plans @ {base_revision[:12]} ...")
    control_src = psi.materialize_composition(
        base_repo=LLAMA_CPP_SRC, worktree_root=worktree_root / "control",
        resolved_revision=base_revision, composition=control_composition,
        overlay_root=psi.REPO_ROOT / "src", requested_revision="HEAD",
    )
    patched_src = psi.materialize_composition(
        base_repo=LLAMA_CPP_SRC, worktree_root=worktree_root / "subject",
        resolved_revision=base_revision, composition=subject_composition,
        overlay_root=psi.REPO_ROOT / "src", requested_revision="HEAD",
    )
    _print(f"control source: {control_src}")
    _print(f"subject source: {patched_src}")
    control_idempotent = psi.verify_composition_idempotent(
        base_repo=LLAMA_CPP_SRC, source=control_src, worktree_root=worktree_root / "control",
        resolved_revision=base_revision, composition=control_composition,
        overlay_root=psi.REPO_ROOT / "src", requested_revision="HEAD",
    )
    subject_idempotent = psi.verify_composition_idempotent(
        base_repo=LLAMA_CPP_SRC, source=patched_src, worktree_root=worktree_root / "subject",
        resolved_revision=base_revision, composition=subject_composition,
        overlay_root=psi.REPO_ROOT / "src", requested_revision="HEAD",
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
    trace_result = None if args.run_rd08_contract else run_trace_activation_probes(
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
    if not args.run_rd08_contract:
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
    from bigcherry.core import config as campaign_config # noqa: E402
    from bigcherry.patch import evidence as patch_validation_evidence # noqa: E402

    cfg = campaign_config.load(bc_paths.RECIPES)
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
            build_identities={
                "control": control_build_evidence.effective_build_id,
                "subject": validation_subject_build_evidence.effective_build_id,
            },
            build_evidence=build_evidence, apply_evidence=apply_evidence,
            architecture=args.amdgpu_targets, model=str(args.model),
            contract=validation_plan.contract,
            contract_hash=(validation_plan.contract.contract_hash if validation_plan.contract else None),
            run_dir=campaign_run_dir,
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
        contract_correctness_gate = compute_contract_correctness_gate(
            full_contract,
            (rd08_qualification["correctness"]["results"] if rd08_qualification is not None else None),
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
        validation_build_identities={
            "control": control_build_evidence.campaign_identity(),
            "subject": validation_subject_build_evidence.campaign_identity(),
        },
        campaign_workdir=workdir / "campaign",
        check_results=validation_check_results,
        # VA14 final slice: eligible_for_validated_state for a bound-contract
        # patch now requires BOTH the adapter verdict AND every bound
        # contract's own evaluate_promotion_gate() PASS (contract_promotions,
        # populated only by --run-rd08-contract today). A bound contract with
        # no promotion result at all still forces False -- see
        # compute_persisted_validation_eligible()'s docstring.
        validation_eligible=compute_persisted_validation_eligible(
            _descriptor, validation_verdict, contract_promotions
        ),
        representation=_descriptor.representation,
        validation_implementation_digest=_descriptor.validation_digest,
        contract_id=_descriptor.experiment_contract,
        contract_hash=(validation_plan.contract.contract_hash if validation_plan and validation_plan.contract else None),
        baseline_composition={"base_revision": base_revision},
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
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

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
    descriptor: object, correctness_summary: dict[str, object] | None,
) -> dict[str, object] | None:
    """VA11A: named-correctness independence, computed as a real,
    independent diagnostic -- NEVER wired into this slice's own eligibility
    (compute_verdict() stays the only thing that gates VA11A; composing
    adapter-verdict AND contract-qualification into final eligibility is
    VA11B's job). Returns None when the patch has no bound contract at all,
    or the contract declares no required correctness checks.

    ANY non-empty required_checks -- one name or several -- is reported
    BLOCKED here, never a real pass. GPT round 8 (req_84fca34f83064678)
    corrected an earlier version of this function that treated a
    single-required-check contract as safely provable from the one generic
    --correctness-evidence summary: that summary has no machine-readable
    field naming WHICH check it actually proves (load_correctness_summary()
    validates identity/disposition only), so constructing
    CorrectnessResult(check=required_checks[0], passed=disposition=='passed')
    silently asserted the summary proved that specific named check when it
    never claimed to. A real per-named-check evidence producer (token-bound
    to the check it proves) is VA14's job, not this plumbing slice's."""
    if descriptor.experiment_contract is None:
        return None
    from bigcherry.patch import validation as patch_validation

    full_contract = patch_validation.load_contract_for_descriptor(descriptor)
    if full_contract is None:
        return None
    required_checks = full_contract.correctness.required_checks
    if not required_checks:
        return None
    return {
        "passed": False,
        "status": "blocked",
        "detail": (
            f"contract requires correctness check(s) {list(required_checks)!r}; no "
            "per-named-check evidence producer exists yet (VA14) -- refusing to infer "
            "a named check's result from one generic, untagged correctness summary"
        ),
    }


def compute_persisted_validation_eligible(
    descriptor: object, validation_verdict: object | None,
) -> bool | None:
    """VA13/VA11A fail-closed invariant (GPT round 8, req_84fca34f83064678):
    this slice explicitly cannot grant full Experiment-Contract
    qualification. Persisting the raw adapter verdict for a bound-contract
    patch would let patch-verify-evidence read it as fully 'validated' off
    adapter-only evidence (compute_verdict() checks only that the
    validation.toml adapter's own checks pass -- it says nothing about
    contract-level named-correctness/performance/promotion qualification,
    which is VA14's job). Force False whenever a contract is bound; the
    adapter verdict remains a real diagnostic in check_results/
    validation_verdict, it just cannot flow into
    eligible_for_validated_state until VA14 composes it with real contract
    qualification. A patch with NO bound contract is unaffected -- the
    adapter verdict is the only qualification such a patch ever claims."""
    if descriptor.experiment_contract is not None:
        return False
    if validation_verdict is None:
        return None
    return validation_verdict.eligible


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
    trace_result = run_trace_activation_probes(
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
            "build_id": tune_build_evidence.effective_build_id,
            "source_tree": patched_source_tree,
            "architecture": args.amdgpu_targets,
            "options": tune_build_evidence.effective_configure,
            "compile_commands": _write_bound_artifact(
                campaign_run_dir, "build/subject-compile-commands.json",
                tune_build_evidence.verification.to_dict(),
            ),
            "runtime_bundle": _write_bound_artifact(
                campaign_run_dir, "build/subject-runtime-bundle.json",
                tune_build_evidence.runtime_artifacts,
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
                "subject": tune_build_evidence.effective_build_id,
            },
            build_evidence=build_evidence, apply_evidence=apply_evidence,
            architecture=args.amdgpu_targets, model=str(args.model),
            contract=validation_plan.contract,
            contract_hash=(validation_plan.contract.contract_hash if validation_plan.contract else None),
            run_dir=campaign_run_dir,
            trace_evidence=trace_evidence, correctness_evidence=correctness_evidence,
        )
        evaluated = {
            spec.check_id: patch_validation.evaluate_check(spec, validation_ctx)
            for spec in validation_plan.checks
        }
        validation_verdict = patch_validation.compute_verdict(validation_plan, evaluated)

        contract_correctness_gate = compute_contract_correctness_gate(
            descriptor, correctness_summary
        )
        validation_check_results = {
            check_id: asdict(result) for check_id, result in evaluated.items()
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
            "subject": tune_build_evidence.campaign_identity(),
        },
        campaign_workdir=workdir / "campaign",
        check_results=validation_check_results,
        # VA13/VA11A fail-closed invariant (GPT round 8, req_84fca34f83064678):
        # this slice explicitly cannot grant full Experiment-Contract
        # qualification -- persisting the raw adapter verdict here would let
        # patch-verify-evidence read a bound-contract patch as "validated"
        # off adapter-only evidence. Force False whenever a contract is
        # bound; the adapter verdict remains a real diagnostic in
        # check_results/validation_verdict, it just cannot flow into
        # eligible_for_validated_state until VA14 composes it with real
        # contract qualification.
        validation_eligible=compute_persisted_validation_eligible(_descriptor, validation_verdict),
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
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

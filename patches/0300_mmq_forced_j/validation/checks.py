"""Hardware-free host proof for the emitted MMQ forced-J transform.

This check compiles and runs the two verbatim emitted blocks that 0300 splices
into upstream mmq.cuh / mmq.cu -- the lifted J-switch helpers (``_HELPERS``)
and the forced-J variant entry points (``_MMQ_SOURCE``) -- against a minimal
host stub of the CUDA/ROCm surface they touch.  It proves the emitted C++ is
behaviourally correct on the host:

* the 16-case ``mul_mat_q_launch_forced_J`` switch routes to exactly the J it
  is given, and aborts on any J outside ``8..128 step 8``;
* the lifted native scan (``mul_mat_q_compute_J_best``) still answers J_best
  and, per PRBE107, optimises tile size against ``ncols_opt`` (not
  ``ncols_max``) -- the MoE case where the two diverge must pick the
  ncols_opt answer;
* ``mul_mat_q_switch_J`` uses the forced value when one is supplied and the
  scan's answer otherwise, through one shared launcher, so forcing
  ``J == J_best`` is native rather than merely equivalent to it;
* the HI71 dense-shape-aware tail envelope
  (``ggml_cuda_mmq_variant_is_eligible``) rejects a forced J outside the
  envelope that upstream's own J_max search over the real batch width would
  have produced, and requires a real ``ncols_max`` (asserts on a sentinel).

It is not evidence of HIP, GPU, or architecture qualification.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from bigcherry.patch.validation import (
    BLOCKED,
    ERROR,
    FAIL,
    PASS,
    ArtifactRef,
    ValidationResult,
    _artifact_is_bound,
)


CHECK_ID = "mmq-forced-j-transform"
SCOPE = "host compiled emitted-transform proof; not HIP artifact or GPU architecture/throughput qualification"

# The two emitted blocks, named by the FilePatch path and the Edit that carries
# each one verbatim.  The fixture asserts the compiled text equals edit.text, so
# a drift between the emitted bytes and the block under test fails closed.
_HELPERS_EDIT = ("ggml/src/ggml-cuda/mmq.cuh", "mmq-helpers-and-switch")
_SOURCE_EDIT = ("ggml/src/ggml-cuda/mmq.cu", "mmq-eligibility-and-entry")

# Host stub for the CUDA/ROCm surface the emitted blocks reference.  Only the
# shapes the transform depends on are real; the kernel launcher records the J it
# was asked to launch so the switch's routing can be asserted.
PREAMBLE = r'''
#include <climits>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

enum ggml_type {
    GGML_TYPE_Q4_0 = 0,
    GGML_TYPE_Q8_0 = 1,
    GGML_TYPE_COUNT = 64,
};

struct ggml_tensor {};
struct ggml_backend_cuda_context {};
typedef void * cudaStream_t;

struct ggml_cuda_mmq_config {
    ggml_type type;
    int J;
    int block;
    int threads;
    int smem;
};

static const ggml_cuda_mmq_config kMmqUndefined = { GGML_TYPE_COUNT, 0, 0, 0, 0 };

ggml_cuda_mmq_config ggml_cuda_mmq_get_config(ggml_type type, int J, bool fallback, int cc) {
    (void)cc;
    if (type == GGML_TYPE_Q8_0 && !fallback) {
        if (J == 8)  return { GGML_TYPE_Q8_0, 8,  1, 256, 1024 };
        if (J == 16) return { GGML_TYPE_Q8_0, 16, 2, 256, 2048 };
    }
    return kMmqUndefined;
}

size_t mmq_get_nbytes_shared(const ggml_cuda_mmq_config & config, int cc) {
    (void)cc;
    return (size_t) config.smem;
}

int ggml_cuda_mmq_get_J_max(ggml_type type, bool fallback, int cc, int64_t ncols) {
    (void)cc;
    if (type != GGML_TYPE_Q8_0 || fallback) return 0;
    if (ncols >= 16) return 16;
    if (ncols >= 8) return 8;
    return 0;
}

struct ggml_cuda_device_info { int cc; size_t smpbo; };
struct ggml_cuda_info_t { ggml_cuda_device_info devices[1]; };
int ggml_cuda_get_device() { return 0; }
ggml_cuda_info_t ggml_cuda_info() {
    ggml_cuda_info_t info = {};
    info.devices[0] = { 90900, 65536 };
    return info;
}

#define GGML_ABORT(msg) throw std::runtime_error(msg)
#define GGML_ASSERT(cond) do { if (!(cond)) throw std::runtime_error("assert: " #cond); } while (0)
#define GGML_UNUSED(x) (void) (x)

struct mmq_args {
    ggml_type type;
    int64_t ncols_opt;
    int64_t ncols_max;
};

static std::vector<int> kLaunchJs;
template <ggml_type type, int J, bool fallback>
static void launch_mul_mat_q(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    (void)ctx; (void)args; (void)stream; (void)type; (void)fallback;
    kLaunchJs.push_back(J);
}

void ggml_cuda_mul_mat_q(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                         const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
                         int forced_J);
static int kMmqCalls = 0;
static int kMmqForcedJ = -1;
void ggml_cuda_mul_mat_q(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                         const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
                         int forced_J) {
    (void)ctx; (void)src0; (void)src1; (void)ids; (void)dst;
    ++kMmqCalls;
    kMmqForcedJ = forced_J;
}
'''

MAIN = r'''
static int kFailures = 0;
static void expect(bool cond, const char * label) {
    std::printf("%s %s\n", cond ? "ok  " : "FAIL", label);
    if (!cond) ++kFailures;
}

int main() {
    ggml_backend_cuda_context ctx;
    cudaStream_t stream = nullptr;
    ggml_tensor t;
    const mmq_args any = { GGML_TYPE_Q8_0, 8, 8 };

    for (int J = 8; J <= 128; J += 8) {
        kLaunchJs.clear();
        mul_mat_q_launch_forced_J<GGML_TYPE_Q8_0, false>(ctx, any, stream, J);
        expect(kLaunchJs.size() == 1 && kLaunchJs[0] == J, "switch-routes-J");
    }
    for (int J : { 4, 136, 0, -8 }) {
        kLaunchJs.clear();
        bool aborted = false;
        try {
            mul_mat_q_launch_forced_J<GGML_TYPE_Q8_0, false>(ctx, any, stream, J);
        } catch (const std::runtime_error &) { aborted = true; }
        expect(aborted && kLaunchJs.empty(), "switch-default-aborts");
    }

    const mmq_args native = { GGML_TYPE_Q8_0, 8, 24 };
    kLaunchJs.clear();
    mul_mat_q_switch_J<GGML_TYPE_Q8_0, false>(ctx, native, stream);
    expect(kLaunchJs.size() == 1 && kLaunchJs[0] == 8, "native-scan-answer-launches");

    // PRBE107: ncols_opt=4 with ncols_max=24.  A ncols_opt scan picks J=8; the
    // pre-fix ncols_max scan would have picked J=128 (the width-filling tile).
    const mmq_args moe = { GGML_TYPE_Q8_0, 4, 24 };
    kLaunchJs.clear();
    mul_mat_q_switch_J<GGML_TYPE_Q8_0, false>(ctx, moe, stream);
    expect(kLaunchJs.size() == 1 && kLaunchJs[0] == 8, "scan-optimizes-ncols_opt");
    expect(mul_mat_q_compute_J_best<GGML_TYPE_Q8_0, false>(moe) == 8, "compute-J-best-delegates-ncols_opt");

    kLaunchJs.clear();
    mul_mat_q_switch_J<GGML_TYPE_Q8_0, false>(ctx, moe, stream, 16);
    expect(kLaunchJs.size() == 1 && kLaunchJs[0] == 16, "forced-replaces-scan");

    const int native_j = mul_mat_q_compute_J_best<GGML_TYPE_Q8_0, false>(moe);
    kLaunchJs.clear();
    mul_mat_q_switch_J<GGML_TYPE_Q8_0, false>(ctx, moe, stream, native_j);
    expect(kLaunchJs.size() == 1 && kLaunchJs[0] == native_j, "forced-equals-best-is-native");

    expect( ggml_cuda_mmq_config_is_eligible(GGML_TYPE_Q8_0, 8,  false, 90900, 65536), "eligible-defined-row");
    expect( ggml_cuda_mmq_config_is_eligible(GGML_TYPE_Q8_0, 16, false, 90900, 65536), "eligible-defined-row-16");
    expect(!ggml_cuda_mmq_config_is_eligible(GGML_TYPE_Q8_0, 24, false, 90900, 65536), "rejected-undefined-row");
    expect(!ggml_cuda_mmq_config_is_eligible(GGML_TYPE_Q8_0, 8,  true,  90900, 65536), "rejected-undefined-fallback");
    expect(!ggml_cuda_mmq_config_is_eligible(GGML_TYPE_Q8_0, 4,  false, 90900, 65536), "rejected-below-8");
    expect(!ggml_cuda_mmq_config_is_eligible(GGML_TYPE_Q8_0, 136,false, 90900, 65536), "rejected-above-128");
    expect(!ggml_cuda_mmq_config_is_eligible(GGML_TYPE_Q8_0, 12, false, 90900, 65536), "rejected-not-multiple-of-8");
    expect(!ggml_cuda_mmq_config_is_eligible(GGML_TYPE_Q8_0, 8,  false, 90900, 1023), "rejected-smem-limit");
    expect( ggml_cuda_mmq_config_is_eligible(GGML_TYPE_Q8_0, 8,  false, 90900, 1024), "eligible-smem-boundary");

    expect( ggml_cuda_mmq_variant_is_eligible(GGML_TYPE_Q8_0, 8,  false, 0, 0, 16), "envelope-small-j-real-width");
    expect(!ggml_cuda_mmq_variant_is_eligible(GGML_TYPE_Q8_0, 16, false, 0, 0, 3),  "envelope-tail-exceeds-jmax");
    bool width_asserted = false;
    try {
        ggml_cuda_mmq_variant_is_eligible(GGML_TYPE_Q8_0, 8, false, 0, 0, 0);
    } catch (const std::runtime_error &) { width_asserted = true; }
    expect(width_asserted, "envelope-requires-real-width");

    expect( ggml_cuda_mmq_type_is_supported(GGML_TYPE_Q8_0, 0, 0), "type-supported-by-table");
    expect(!ggml_cuda_mmq_type_is_supported(GGML_TYPE_Q4_0, 0, 0), "type-unsupported-not-in-table");

    kMmqCalls = 0;
    kMmqForcedJ = -1;
    ggml_cuda_mul_mat_q_variant(ctx, &t, &t, &t, &t, 16, 1);
    expect(kMmqCalls == 1 && kMmqForcedJ == 16, "variant-forwards-forced-j");
    ggml_cuda_mul_mat_q_variant(ctx, &t, &t, &t, &t, 0, 1);
    expect(kMmqCalls == 2 && kMmqForcedJ == 0, "variant-native-zero-passthrough");

    std::printf("failures: %d\n", kFailures);
    return kFailures == 0 ? 0 : 1;
}
'''


def check_apply(ctx: Any) -> ValidationResult:
    evidence = getattr(ctx, "configuration_evidence", None)
    if not isinstance(evidence, dict):
        return ValidationResult("apply", "apply", BLOCKED, "single-composition apply evidence is required")
    apply = evidence.get("apply")
    if not isinstance(apply, dict):
        return ValidationResult("apply", "apply", BLOCKED, "single-composition apply evidence is missing")
    if apply.get("single_composition") is not True:
        return ValidationResult("apply", "apply", FAIL, "apply evidence is not for one framework composition")
    if apply.get("verified") is not True or apply.get("idempotent") is not True:
        return ValidationResult("apply", "apply", FAIL, "framework source apply/idempotence proof failed")
    artifact = apply.get("artifact")
    if not isinstance(artifact, dict) or not _artifact_is_bound(artifact, getattr(ctx, "run_dir", None)):
        return ValidationResult("apply", "apply", BLOCKED, "bound single-composition apply artifact is required")
    return ValidationResult("apply", "apply", PASS, "single framework composition apply and idempotence are verified",
                            artifacts=(ArtifactRef("apply", str(artifact["path"]), str(artifact["sha256"])),))


def check_build(ctx: Any) -> ValidationResult:
    evidence = getattr(ctx, "configuration_evidence", None)
    if not isinstance(evidence, dict):
        return ValidationResult("build", "build", BLOCKED, "framework production/diagnostic build evidence is required")
    builds = evidence.get("builds")
    if not isinstance(builds, dict):
        return ValidationResult("build", "build", BLOCKED, "production and diagnostic build evidence is missing")
    artifacts: list[ArtifactRef] = []
    for role in ("production", "diagnostic"):
        item = builds.get(role)
        if not isinstance(item, dict) or item.get("completed") is not True:
            return ValidationResult("build", "build", FAIL, f"{role} framework build is not completed")
        artifact = item.get("artifact")
        if not isinstance(artifact, dict) or not _artifact_is_bound(artifact, getattr(ctx, "run_dir", None)):
            return ValidationResult("build", "build", BLOCKED, f"bound {role} framework build artifact is required")
        artifacts.append(ArtifactRef(role, str(artifact["path"]), str(artifact["sha256"])))
    return ValidationResult("build", "build", PASS, "completed production and diagnostic framework builds are verified",
                            artifacts=tuple(artifacts))


def _result(status: str, summary: str, *, details: list[str] | None = None,
            artifacts: tuple[ArtifactRef, ...] = ()) -> ValidationResult:
    return ValidationResult(
        check_id=CHECK_ID,
        capability="configuration",
        status=status,
        summary=summary,
        details=tuple(details or ()),
        artifacts=artifacts,
    )


def _load_patch(package_root: Path) -> Any:
    path = package_root / "patch.py"
    spec = importlib.util.spec_from_file_location("mmq_forced_j_patch", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load patch module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _emitted_text(module: Any, file_path: str, edit_id: str) -> str:
    for file_patch in module.PATCHES:
        if file_patch.path != file_path:
            continue
        for edit in file_patch.edits:
            if edit.id == edit_id:
                return edit.text
        raise RuntimeError(f"edit {edit_id} not found in {file_path}")
    raise RuntimeError(f"file patch {file_path} not found")


def _fixture(module: Any, work: Path) -> tuple[Path | None, dict[str, Any]]:
    # Tie the compiled text to the exact bytes the patch edits emit.  If the
    # block a human edited no longer matches the edit carrying it, the proof is
    # no longer about this patch and must fail.
    helpers_from_edit = _emitted_text(module, *_HELPERS_EDIT)
    source_from_edit = _emitted_text(module, *_SOURCE_EDIT)
    identity = {
        "helpers_edit_matches_block": helpers_from_edit == module._HELPERS,
        "source_edit_matches_block": source_from_edit == module._MMQ_SOURCE,
    }
    report: dict[str, Any] = {"emitted_identity": identity}
    if not (identity["helpers_edit_matches_block"] and identity["source_edit_matches_block"]):
        return None, report
    source = work / "mmq-forced-j.cpp"
    source.write_text(PREAMBLE + module._HELPERS + "\n" + module._MMQ_SOURCE + "\n" + MAIN,
                      encoding="utf-8", newline="")
    return source, report


def check(ctx: Any) -> ValidationResult:
    """Compile and run the emitted forced-J transform with a real host compiler."""
    try:
        raw_run_dir = getattr(ctx, "run_dir", None)
        raw_package_root = getattr(ctx, "package_root", None)
        run_dir = Path(raw_run_dir) if raw_run_dir is not None else None
        package_root = Path(raw_package_root) if raw_package_root is not None else None
        register = getattr(ctx, "register_artifact", None)
        if run_dir is None or package_root is None or register is None:
            return _result(BLOCKED, "run_dir, package_root, and register_artifact are required")
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            return _result(BLOCKED, "no clang++ or g++ host compiler is available")

        with tempfile.TemporaryDirectory(dir=run_dir, prefix="mmq-forced-j-") as temp:
            work = Path(temp)
            module = _load_patch(package_root)
            source, report = _fixture(module, work)
            report["scope"] = SCOPE
            if source is None:
                report["outcome"] = FAIL
                report["compiler"] = compiler
                report_path = work / "mmq-forced-j-transform.json"
                report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
                report_ref = register("mmq-forced-j-transform.json", report_path)
                if not isinstance(report_ref, ArtifactRef) or not _artifact_is_bound(asdict(report_ref), run_dir):
                    return _result(ERROR, "register_artifact returned an invalid ArtifactRef")
                return _result(FAIL, "emitted block no longer matches the patch edit under proof",
                               artifacts=(report_ref,))

            version_command = [compiler, "--version"]
            version = subprocess.run(version_command, capture_output=True, text=True, timeout=60)
            if version.returncode != 0:
                return _result(ERROR, "host compiler identity command failed",
                               details=[version.stdout, version.stderr])
            exe = work / "mmq-forced-j.exe"
            command = [compiler, "-std=c++17", "-O0", str(source), "-o", str(exe)]
            run_command = [str(exe)]
            built = subprocess.run(command, capture_output=True, text=True, timeout=120)
            ran = (subprocess.run(run_command, capture_output=True, text=True, timeout=60)
                   if built.returncode == 0 else None)
            outcome = {
                "command": command,
                "run_command": run_command,
                "compile_returncode": built.returncode,
                "compile_stdout": built.stdout,
                "compile_stderr": built.stderr,
                "run_returncode": None if ran is None else ran.returncode,
                "run_stdout": None if ran is None else ran.stdout,
                "run_stderr": None if ran is None else ran.stderr,
            }
            report["outcome"] = PASS if (outcome["compile_returncode"] == 0 and
                                         outcome["run_returncode"] == 0) else FAIL
            report.update({"compiler": compiler, "version_command": version_command,
                           "compiler_version": version.stdout + version.stderr,
                           "compiler_version_returncode": version.returncode,
                           "outcomes": [outcome]})
            report_path = work / "mmq-forced-j-transform.json"
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
            artifact_list: list[ArtifactRef] = [register("mmq-forced-j.cpp", source)]
            compiler_file = work / "compiler.txt"
            compiler_file.write_text(report["compiler_version"], encoding="utf-8")
            artifact_list.append(register("compiler.txt", compiler_file))
            command_file = work / "command.txt"
            command_file.write_text("\n".join(outcome["command"]), encoding="utf-8")
            artifact_list.append(register("command.txt", command_file))
            for stream in ("compile_stdout", "compile_stderr", "run_stdout", "run_stderr"):
                stream_file = work / f"{stream}.txt"
                stream_file.write_text(outcome[stream] or "", encoding="utf-8")
                artifact_list.append(register(f"{stream}.txt", stream_file))
            artifact_list.append(register("mmq-forced-j-transform.json", report_path))
            artifacts = tuple(artifact_list)
            if not all(isinstance(ref, ArtifactRef) and
                       _artifact_is_bound(asdict(ref), run_dir) for ref in artifacts):
                return _result(ERROR, "register_artifact returned an invalid ArtifactRef")
            if report["outcome"] != PASS:
                return _result(FAIL, "host compile or run failed", artifacts=artifacts)
            return _result(PASS, "emitted forced-J transform compiled and ran clean on the host",
                           details=["host C++ proof only; no HIP/GPU inference"], artifacts=artifacts)
    except Exception as exc:
        return _result(ERROR, f"mmq-forced-j-transform raised: {exc}")

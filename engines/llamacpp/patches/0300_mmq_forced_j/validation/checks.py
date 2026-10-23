"""Host proof for 0300's b11552 MMQ forced-J migration.

The proof compiles the exact helper/source blocks emitted by patch.py against a
small host stub. It validates configuration eligibility, b11552's ncols_opt J
scan, explicit precision plumbing, and the required forced-J variant forward.
It also fails closed if the pre-allocation force-selection edit drifts from the
block in patch.py. This is not HIP/GPU qualification.
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
    BLOCKED, ERROR, FAIL, PASS, ArtifactRef, ValidationResult, _artifact_is_bound,
)

CHECK_ID = "mmq-forced-j-transform"
SCOPE = "host C++ emitted-transform proof; not HIP artifact or hardware qualification"

_HELPERS_EDIT = ("ggml/src/ggml-cuda/bc-mmq-forced-j.cuh", "mmq-helpers-owned")
_SOURCE_EDIT = ("ggml/src/ggml-cuda/bc-mmq-forced-j-source.cuh", "mmq-source-owned")
_FORCE_EDIT = ("ggml/src/ggml-cuda/mmq.cu", "mmq-force-before-padding")

PREAMBLE = r'''
#include <climits>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

enum ggml_prec { GGML_PREC_Q8 = 30, GGML_PREC_Q4 = 40 };
enum ggml_type {
    GGML_TYPE_Q4_0 = 0,
    GGML_TYPE_Q8_0 = 1,
    GGML_TYPE_Q6_K = 2,
    GGML_TYPE_COUNT = 64,
};

struct ggml_tensor { ggml_type type = GGML_TYPE_Q8_0; };
struct ggml_backend_cuda_context {};

struct ggml_cuda_mmq_config {
    ggml_type type;
    int J;
    int nthreads;
    int smem;
};
static const ggml_cuda_mmq_config kUndefined = { GGML_TYPE_COUNT, 0, 0, 0 };

ggml_cuda_mmq_config ggml_cuda_mmq_get_config(
        ggml_type type, int J, bool fallback, int cc, ggml_prec prec_src1) {
    if (type == GGML_TYPE_Q8_0 && !fallback && prec_src1 == GGML_PREC_Q8) {
        if (J == 8)  return { GGML_TYPE_Q8_0, 8, 256, 1024 };
        if (J == 16) return { GGML_TYPE_Q8_0, 16, 256, 2048 };
    }
    if (type == GGML_TYPE_Q6_K && !fallback && prec_src1 == GGML_PREC_Q4 && J == 16) {
        return { GGML_TYPE_Q6_K, 16, 256, 2048 };
    }
    if (type == GGML_TYPE_Q4_0 && fallback && cc >= 90000 &&
            prec_src1 == GGML_PREC_Q8 && J == 8) {
        return { GGML_TYPE_Q4_0, 8, 128, 512 };
    }
    return kUndefined;
}

size_t mmq_get_nbytes_shared(const ggml_cuda_mmq_config & config, int) {
    return (size_t) config.smem;
}

struct ggml_cuda_device_info { int cc; size_t smpbo; };
struct ggml_cuda_info_t { ggml_cuda_device_info devices[1]; };
int ggml_cuda_get_device() { return 0; }
ggml_cuda_info_t ggml_cuda_info() {
    ggml_cuda_info_t info = {};
    info.devices[0] = { 90900, 65536 };
    return info;
}

#define GGML_ASSERT(cond) do { if (!(cond)) throw std::runtime_error("assert: " #cond); } while (0)
#define GGML_UNUSED(x) (void) (x)

static int kMmqCalls = 0;
static int kForcedJ = -1;
void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *,
        const ggml_tensor *, ggml_tensor *, int forced_J) {
    ++kMmqCalls;
    kForcedJ = forced_J;
}
'''

MAIN = r'''
static int failures = 0;
static void expect(bool condition, const char * name) {
    std::printf("%s %s\n", condition ? "ok  " : "FAIL", name);
    if (!condition) ++failures;
}

int main() {
    expect(ggml_cuda_mmq_config_is_eligible(
        GGML_TYPE_Q8_0, 8, false, 90900, 65536, GGML_PREC_Q8), "eligible-q8-j8");
    expect(!ggml_cuda_mmq_config_is_eligible(
        GGML_TYPE_Q8_0, 12, false, 90900, 65536, GGML_PREC_Q8), "reject-j12");
    expect(ggml_cuda_mmq_config_is_eligible(
        GGML_TYPE_Q4_0, 8, true, 90900, 65536, GGML_PREC_Q8), "fallback-cc-plumbing");

    expect(ggml_cuda_mmq_native_j_best(
        GGML_TYPE_Q8_0, false, 4, GGML_PREC_Q8) == 8, "native-small-ncols-opt");
    expect(ggml_cuda_mmq_native_j_best(
        GGML_TYPE_Q8_0, false, 24, GGML_PREC_Q8) == 16, "native-wide-ncols-opt");
    expect(ggml_cuda_mmq_native_j_best(
        GGML_TYPE_Q6_K, false, 16, GGML_PREC_Q4) == 16, "explicit-q4-precision");
    expect(ggml_cuda_mmq_native_j_best(
        GGML_TYPE_Q6_K, false, 16, GGML_PREC_Q8) == 0, "q8-cannot-see-q4-row");

    expect(ggml_cuda_mmq_type_is_supported(GGML_TYPE_Q8_0, 0, 0), "type-supported");
    expect(!ggml_cuda_mmq_type_is_supported(GGML_TYPE_Q6_K, 0, 0), "q6-q8-unsupported");

    ggml_backend_cuda_context ctx;
    ggml_tensor t;
    ggml_cuda_mul_mat_q_variant(ctx, &t, &t, &t, &t, 16, 1);
    expect(kMmqCalls == 1 && kForcedJ == 16, "variant-forwards-forced-j");
    ggml_cuda_mul_mat_q_variant(ctx, &t, &t, &t, &t, 0, 0);
    expect(kMmqCalls == 2 && kForcedJ == 0, "variant-zero-means-native");

    bool asserted = false;
    try {
        (void) ggml_cuda_mmq_variant_is_eligible(
            GGML_TYPE_Q8_0, 8, false, 0, 0, 0);
    } catch (const std::runtime_error &) {
        asserted = true;
    }
    expect(asserted, "variant-requires-real-width");

    std::printf("failures: %d\n", failures);
    return failures == 0 ? 0 : 1;
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
    return ValidationResult(
        "apply", "apply", PASS, "single framework composition apply and idempotence are verified",
        artifacts=(ArtifactRef("apply", str(artifact["path"]), str(artifact["sha256"])),),
    )


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
    return ValidationResult(
        "build", "build", PASS, "completed production and diagnostic framework builds are verified",
        artifacts=tuple(artifacts),
    )


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
    helpers = _emitted_text(module, *_HELPERS_EDIT)
    source_block = _emitted_text(module, *_SOURCE_EDIT)
    force_block = _emitted_text(module, *_FORCE_EDIT)
    identity = {
        "helpers_edit_matches_block": helpers == module._HELPERS,
        "source_edit_matches_block": source_block == module._MMQ_SOURCE,
        "force_edit_matches_block": force_block == module._FORCE_SELECTION,
        "force_precedes_padding_anchor": any(
            edit.id == _FORCE_EDIT[1] and edit.anchor == r"^    GGML_ASSERT\(J_best > 0\);$"
            for patch in module.PATCHES if patch.path == _FORCE_EDIT[0]
            for edit in patch.edits
        ),
    }
    report: dict[str, Any] = {"emitted_identity": identity}
    if not all(identity.values()):
        return None, report
    source = work / "mmq-forced-j.cpp"
    source.write_text(PREAMBLE + module._HELPERS + "\n" + module._MMQ_SOURCE + "\n" + MAIN,
                      encoding="utf-8", newline="")
    return source, report


def check(ctx: Any) -> ValidationResult:
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
            report["outcome"] = PASS if (
                outcome["compile_returncode"] == 0 and outcome["run_returncode"] == 0
            ) else FAIL
            report.update({
                "compiler": compiler,
                "version_command": version_command,
                "compiler_version": version.stdout + version.stderr,
                "compiler_version_returncode": version.returncode,
                "outcomes": [outcome],
            })
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
            return _result(
                PASS,
                "emitted b11552 MMQ helpers compiled and ran clean on host C++",
                details=["host C++ proof only; hardware qualification remains external"],
                artifacts=artifacts,
            )
    except Exception as exc:
        return _result(ERROR, f"mmq-forced-j-transform raised: {exc}")

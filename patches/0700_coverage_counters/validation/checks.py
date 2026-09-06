"""Hardware-free host proof for the emitted family coverage hooks.

This check deliberately proves only source transformation and host C++
compile/run behaviour.  It is not evidence of HIP, GPU, or architecture
qualification.
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

from bigcherry.patch.apply import apply_patch
from bigcherry.patch.validation import (
    BLOCKED,
    ERROR,
    FAIL,
    PASS,
    ArtifactRef,
    ValidationResult,
    _artifact_is_bound,
)


FAMILIES = ["MMQ", "MMVQ", "MMVF", "MMF", "BLAS"]
SCOPE = "host compiled emitted-hook proof; not HIP artifact or GPU architecture/throughput qualification"


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
        check_id="family-hook-isolation",
        capability="configuration",
        status=status,
        summary=summary,
        details=tuple(details or ()),
        artifacts=artifacts,
    )


def _load_patch(package_root: Path) -> Any:
    path = package_root / "patch.py"
    spec = importlib.util.spec_from_file_location("coverage_counters_patch", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load patch module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(module: Any, directory: Path) -> tuple[Path, dict[str, Any]]:
    functions: list[str] = []
    transforms: list[dict[str, Any]] = []
    for index, family in enumerate(FAMILIES):
        original = module._count(
            "GGML_HIP_FAMILY_" + family,
            "nullptr" if family != "BLAS" else None,
        )
        source = directory / f"hook-{index}.cpp"
        source.write_text(original, encoding="utf-8", newline="")
        patch = module.PATCHES[index]
        # The final edit is the emitted-hook upgrade under proof.  Its anchor
        # intentionally recognizes the historical insertion from this patch.
        edit_patch = type(patch)(source.name, (patch.edits[-1],), patch.description)
        first = apply_patch(edit_patch, directory)
        updated = source.read_text(encoding="utf-8")
        second = apply_patch(edit_patch, directory)
        transforms.append({
            "family": family,
            "first_ok": first.ok,
            "first_changed": first.changed,
            "second_ok": second.ok,
            "second_changed": second.changed,
            "dispatch_calls_before": original.count("ggml_hip_dispatch_family("),
            "dispatch_calls_after": updated.count("ggml_hip_dispatch_family("),
        })
        functions.append(f"void entry{index}() {{\n{updated}\n}}")

    source = directory / "coverage-hooks.cpp"
    definitions = "\n".join(
        f"const int GGML_HIP_FAMILY_{family} = {index};"
        for index, family in enumerate(FAMILIES)
    )
    source.write_text(
        """
int ctx, src0, src1, ids, dst;
int dispatch_calls, executed_calls, probe_calls;
bool ggml_hip_dispatch_family(...) { ++dispatch_calls; return false; }
bool ggml_hip_dispatch_is_reentrant();
void ggml_hip_coverage_count_executed(int);
#ifdef GGML_HIP_DISPATCH_DIAGNOSTICS
bool ggml_hip_dispatch_is_reentrant() { ++probe_calls; return false; }
void ggml_hip_coverage_count_executed(int) { ++executed_calls; }
#endif
""" + definitions + "\n" + "\n".join(functions) + """
int main() {
    entry0(); entry1(); entry2(); entry3(); entry4();
#ifdef GGML_HIP_DISPATCH_DIAGNOSTICS
    return !(dispatch_calls == 4 && executed_calls == 5 && probe_calls == 5);
#else
    return !(dispatch_calls == 4 && executed_calls == 0 && probe_calls == 0);
#endif
}
""",
        encoding="utf-8",
        newline="",
    )
    return source, {"transforms": transforms}


def check(ctx: Any) -> ValidationResult:
    """Compile and run the emitted-hook proof with a real host compiler."""
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

        with tempfile.TemporaryDirectory(dir=run_dir, prefix="family-hook-") as temp:
            work = Path(temp)
            module = _load_patch(package_root)
            source, report = _fixture(module, work)
            report["scope"] = SCOPE
            if any(not item["first_ok"] or not item["first_changed"] or
                   not item["second_ok"] or item["second_changed"] or
                   item["dispatch_calls_before"] != item["dispatch_calls_after"]
                   for item in report["transforms"]):
                report["outcome"] = FAIL
                report["compiler"] = compiler
                report_path = work / "family-hook-isolation.json"
                report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
                source_ref = register("coverage-hooks.cpp", source)
                if not isinstance(source_ref, ArtifactRef) or not _artifact_is_bound(asdict(source_ref), run_dir):
                    return _result(ERROR, "register_artifact returned an invalid ArtifactRef")
                report_ref = register("family-hook-isolation.json", report_path)
                if not isinstance(report_ref, ArtifactRef) or not _artifact_is_bound(asdict(report_ref), run_dir):
                    return _result(ERROR, "register_artifact returned an invalid ArtifactRef")
                return _result(FAIL, "one or more emitted-hook transforms were wrong or non-idempotent",
                               artifacts=(source_ref, report_ref))

            version_command = [compiler, "--version"]
            version = subprocess.run(version_command, capture_output=True, text=True, timeout=60)
            if version.returncode != 0:
                return _result(ERROR, "host compiler identity command failed",
                               details=[version.stdout, version.stderr])
            outcomes: list[dict[str, Any]] = []
            for diagnostics in (False, True):
                exe = work / ("diagnostics.exe" if diagnostics else "production.exe")
                command = [compiler, "-std=c++17", "-O0", "-DGGML_HIP_DISPATCH", str(source), "-o", str(exe)]
                if diagnostics:
                    command.append("-DGGML_HIP_DISPATCH_DIAGNOSTICS")
                run_command = [str(exe)]
                built = subprocess.run(command, capture_output=True, text=True, timeout=60)
                ran = (subprocess.run(run_command, capture_output=True, text=True, timeout=60)
                       if built.returncode == 0 else None)
                outcomes.append({
                    "diagnostics": diagnostics,
                    "command": command,
                    "run_command": run_command,
                    "compile_returncode": built.returncode,
                    "compile_stdout": built.stdout,
                    "compile_stderr": built.stderr,
                    "run_returncode": None if ran is None else ran.returncode,
                    "run_stdout": None if ran is None else ran.stdout,
                    "run_stderr": None if ran is None else ran.stderr,
                })
            report.update({"compiler": compiler, "version_command": version_command,
                           "compiler_version": version.stdout + version.stderr,
                           "compiler_version_returncode": version.returncode, "outcomes": outcomes})
            report["outcome"] = PASS if all(
                item["compile_returncode"] == 0 and item["run_returncode"] == 0 for item in outcomes
            ) else FAIL
            report_path = work / "family-hook-isolation.json"
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
            artifact_list: list[ArtifactRef] = [register("coverage-hooks.cpp", source)]
            compiler_file = work / "compiler.txt"
            compiler_file.write_text(report["compiler_version"], encoding="utf-8")
            artifact_list.append(register("compiler.txt", compiler_file))
            for item in outcomes:
                label = "diagnostics-on" if item["diagnostics"] else "diagnostics-off"
                command_file = work / f"{label}-command.txt"
                command_file.write_text("\n".join(item["command"]), encoding="utf-8")
                artifact_list.append(register(f"{label}-command.txt", command_file))
                for stream in ("compile_stdout", "compile_stderr", "run_stdout", "run_stderr"):
                    stream_file = work / f"{label}-{stream}.txt"
                    stream_file.write_text(item[stream] or "", encoding="utf-8")
                    artifact_list.append(register(f"{label}-{stream}.txt", stream_file))
            artifact_list.append(register("family-hook-isolation.json", report_path))
            artifacts = tuple(artifact_list)
            if not all(isinstance(ref, ArtifactRef) and
                       _artifact_is_bound(asdict(ref), run_dir) for ref in artifacts):
                return _result(ERROR, "register_artifact returned an invalid ArtifactRef")
            if report["outcome"] != PASS:
                return _result(FAIL, "host compile or run failed", artifacts=artifacts)
            return _result(PASS, "all five emitted hooks compiled and ran in diagnostics OFF and ON modes",
                           details=["host C++ proof only; no HIP/GPU inference"], artifacts=artifacts)
    except Exception as exc:
        return _result(ERROR, f"family-hook-isolation raised: {exc}")

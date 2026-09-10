"""Patch-local, hardware-free CMake source-selection validation."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from bigcherry.patch.apply import FilePatch, apply_patch
from bigcherry.patch.validation import (
    BLOCKED,
    ERROR,
    FAIL,
    PASS,
    ArtifactRef,
    ValidationResult,
    _artifact_is_bound,
)


_CHECK_ID = "coverage-source-selection"
_CAPABILITY = "configuration"


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


def _result(status: str, summary: str, *, details: tuple[str, ...] = (),
            artifacts: tuple[ArtifactRef, ...] = ()) -> ValidationResult:
    return ValidationResult(
        check_id=_CHECK_ID,
        capability=_CAPABILITY,
        status=status,
        summary=summary,
        details=details,
        artifacts=artifacts,
    )


def _load_patch(package_root: Path) -> Any:
    patch_path = package_root / "patch.py"
    if not patch_path.is_file():
        raise FileNotFoundError(f"package patch.py is missing: {patch_path}")
    spec = importlib.util.spec_from_file_location("bigcherry_0100_validation_patch", patch_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load package patch module: {patch_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_list(module: Any, root: Path) -> tuple[str, bool, bool]:
    """Materialize the real package edit and return (list, first, second)."""
    source = root / "CMakeLists.txt"
    source.write_text(module._HIP_DEFINITIONS, encoding="utf-8", newline="")
    patch = FilePatch("CMakeLists.txt", module.HIP_BACKEND_PATCH.edits[1:])
    first = apply_patch(patch, root)
    if not first.ok or not first.changed:
        raise RuntimeError(f"coverage source patch did not apply: {first.results!r}")
    second = apply_patch(patch, root)
    if not second.ok or second.changed:
        raise RuntimeError(f"coverage source patch is not idempotent: {second.results!r}")
    text = source.read_text(encoding="utf-8")
    start = text.index("    set(_BC_DISPATCH_SOURCES")
    end = text.index("    list(APPEND GGML_SOURCES_ROCM ${_BC_DISPATCH_SOURCES})", start)
    return text[start:end], first.changed, second.changed


def check(ctx: Any) -> ValidationResult:
    """Prove host CMake source selection; this is not HIP build evidence."""
    package_root = getattr(ctx, "package_root", None)
    run_dir = getattr(ctx, "run_dir", None)
    register = getattr(ctx, "register_artifact", None)
    if package_root is None or run_dir is None or not callable(register):
        return _result(BLOCKED, "package_root, run_dir, and register_artifact are required")
    cmake = shutil.which("cmake")
    if not cmake:
        return _result(BLOCKED, "CMake is unavailable")

    try:
        package_root = Path(package_root).resolve()
        run_dir = Path(run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        module = _load_patch(package_root)
        with tempfile.TemporaryDirectory(prefix="coverage-selection-", dir=run_dir) as temp:
            work = Path(temp)
            source_list, first_changed, second_changed = _source_list(module, work)
            script = "cmake_minimum_required(VERSION 3.18)\n" + source_list + """
if ("../ggml-cuda/hip-autotune-coverage.cpp" IN_LIST _BC_DISPATCH_SOURCES)
    set(actual ON)
else()
    set(actual OFF)
endif()
if (NOT actual STREQUAL expected)
    message(FATAL_ERROR "Coverage selection ${actual}; expected ${expected}")
endif()
"""
            script_path = work / "coverage-selection.cmake"
            script_path.write_text(script, encoding="utf-8", newline="")
            version = subprocess.run([cmake, "--version"], capture_output=True, text=True,
                                     check=False, timeout=30)
            cases = (
                ("production-diagnostics-OFF", (), "OFF"),
                ("diagnosticON", ("GGML_HIP_DISPATCH_DIAGNOSTICS",), "ON"),
                ("AUTOTUNE_RECORDON", ("GGML_HIP_AUTOTUNE_RECORD",), "ON"),
                ("AUTOTUNEON", ("GGML_HIP_AUTOTUNE",), "ON"),
            )
            observations: list[dict[str, Any]] = []
            for name, enabled, expected in cases:
                command = [cmake, "-DGGML_HIP_DISPATCH_REPLAY=ON", f"-Dexpected={expected}"]
                command.extend(f"-D{option}=ON" for option in enabled)
                command.extend(["-P", str(script_path)])
                result = subprocess.run(command, capture_output=True, text=True,
                                        check=False, timeout=30)
                observations.append({
                    "name": name,
                    "command": command,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                })
            report = {
                "scope": "host CMake source selection; not HIP binary, architecture, or performance proof",
                "cmake": cmake,
                "cmake_version_command": [cmake, "--version"],
                "cmake_version_returncode": version.returncode,
                "cmake_version_stdout": version.stdout,
                "cmake_version_stderr": version.stderr,
                "source_script": script,
                "patch_first_changed": first_changed,
                "patch_second_changed": second_changed,
                "observations": observations,
            }
            report_path = work / "coverage-selection.json"
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
            script_ref = register("coverage-selection.cmake", script_path)
            report_ref = register("coverage-selection.json", report_path)
            if (not isinstance(script_ref, ArtifactRef)
                    or not isinstance(report_ref, ArtifactRef)
                    or not _artifact_is_bound(asdict(script_ref), run_dir)
                    or not _artifact_is_bound(asdict(report_ref), run_dir)):
                raise TypeError("register_artifact returned an invalid or unbound ArtifactRef")
            artifacts = (script_ref, report_ref)
            if version.returncode != 0:
                return _result(ERROR, "CMake version command failed", artifacts=artifacts)
        failures = tuple(item["name"] for item in observations if item["returncode"] != 0)
        if failures:
            return _result(FAIL, "CMake source-selection matrix failed", details=failures, artifacts=artifacts)
        return _result(PASS, "CMake source-selection matrix passed", artifacts=artifacts)
    except Exception as exc:
        return _result(ERROR, f"CMake source-selection check errored: {exc}")

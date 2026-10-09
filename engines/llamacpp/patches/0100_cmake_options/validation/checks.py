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


_CHECK_ID = "serving-source-selection"
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


_ANCHOR_TEXT = (
    'file(GLOB   SRCS "../ggml-cuda/template-instances/mmf*.cu")\n'
    'list(APPEND GGML_SOURCES_ROCM ${SRCS})\n'
)


def _source_list(module: Any, root: Path) -> tuple[str, bool, bool]:
    """Materialize the real package edit against its actual anchor text
    (not the already-applied _HIP_DEFINITIONS -- writing that as the
    starting file would make the edit's own guard match immediately,
    reporting "already-applied" rather than exercising the real anchored
    insert) and return (list, first, second)."""
    source = root / "CMakeLists.txt"
    source.write_text(_ANCHOR_TEXT, encoding="utf-8", newline="")
    patch = FilePatch("CMakeLists.txt", module.HIP_BACKEND_PATCH.edits)
    first = apply_patch(patch, root)
    if not first.ok or not first.changed:
        raise RuntimeError(f"serving source patch did not apply: {first.results!r}")
    second = apply_patch(patch, root)
    if not second.ok or second.changed:
        raise RuntimeError(f"serving source patch is not idempotent: {second.results!r}")
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
        with tempfile.TemporaryDirectory(prefix="serving-selection-", dir=run_dir) as temp:
            work = Path(temp)
            source_list, first_changed, second_changed = _source_list(module, work)
            # Prove the serving/shared package's own source selection: the
            # common dispatch/transform/telemetry/signature/blake2b sources
            # are always present when this package's outer guard fires, the
            # replay-only source is present iff GGML_HIP_DISPATCH_REPLAY, and
            # none of 0110_campaign_tune_record_build's tuner/record/coverage
            # sources ever appear here -- this package does not declare or
            # reference GGML_HIP_AUTOTUNE_RECORD/DIAGNOSTICS/AUTOTUNE at all
            # in its own source list (moved entirely to 0110 by PA27).
            script = "cmake_minimum_required(VERSION 3.18)\n" + source_list + """
foreach(_forbidden IN ITEMS
        "../ggml-cuda/hip-autotune-coverage.cpp"
        "../ggml-cuda/hip-autotune-record.cpp"
        "../ggml-cuda/hip-autotune-tuner.cu")
    if (_forbidden IN_LIST _BC_DISPATCH_SOURCES)
        message(FATAL_ERROR "serving source list must never include ${_forbidden}")
    endif()
endforeach()
foreach(_common IN ITEMS
        "../ggml-cuda/hip-autotune-dispatch.cu"
        "../ggml-cuda/hip-autotune-transform.cu"
        "../ggml-cuda/hip-autotune-reduce-telemetry.cpp"
        "../ggml-cuda/hip-autotune-signature.cpp"
        "../ggml-cuda/hip-autotune-blake2b.cpp")
    if (NOT _common IN_LIST _BC_DISPATCH_SOURCES)
        message(FATAL_ERROR "serving source list is missing common source ${_common}")
    endif()
endforeach()
if ("../ggml-cuda/hip-autotune-replay.cpp" IN_LIST _BC_DISPATCH_SOURCES)
    set(actual ON)
else()
    set(actual OFF)
endif()
if (NOT actual STREQUAL expected)
    message(FATAL_ERROR "replay.cpp selection ${actual}; expected ${expected}")
endif()
"""
            script_path = work / "serving-selection.cmake"
            script_path.write_text(script, encoding="utf-8", newline="")
            version = subprocess.run([cmake, "--version"], capture_output=True, text=True,
                                     check=False, timeout=30)
            cases = (
                ("autotune-only-no-replay", (), "OFF"),
                ("replay-only", ("GGML_HIP_DISPATCH_REPLAY",), "ON"),
            )
            observations: list[dict[str, Any]] = []
            for name, enabled, expected in cases:
                # "autotune-only-no-replay" exercises this package's outer
                # `if (GGML_HIP_AUTOTUNE OR GGML_HIP_DISPATCH_REPLAY)` guard
                # via the foreign, 0110-owned GGML_HIP_AUTOTUNE variable
                # (simulating 0110 applied alongside this package) without
                # GGML_HIP_DISPATCH_REPLAY -- this package's own source list
                # must still be the common set, with no replay.cpp.
                command = [cmake, f"-Dexpected={expected}"]
                command.extend(f"-D{option}=ON" for option in enabled)
                if not enabled:
                    command.append("-DGGML_HIP_AUTOTUNE=ON")
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
            report_path = work / "serving-selection.json"
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
            script_ref = register("serving-selection.cmake", script_path)
            report_ref = register("serving-selection.json", report_path)
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

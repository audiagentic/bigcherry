"""Staged lane builds (BU06): the header-only engine prefix and the stage marker."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from bigcherry.campaign import runtime_bundle


def _configure(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    (src / "abi").mkdir(parents=True)
    (src / "abi" / "rad_abi.h").write_text("#define RAD_ABI_VERSION 45\n")
    (src / "components" / "x").mkdir(parents=True)
    (src / "components" / "x" / "CMakeLists.txt").write_text("install(FILES ../../abi/rad_abi.h DESTINATION include/x)\n")
    (src / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\nproject(t NONE)\n"
        "install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/abi/ DESTINATION include/radiance FILES_MATCHING PATTERN \"*.h\")\n"
        "file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/radiance-config.cmake \"\")\n"
        "install(FILES ${CMAKE_CURRENT_BINARY_DIR}/radiance-config.cmake DESTINATION lib/cmake/radiance)\n"
        "install(FILES abi/rad_abi.h DESTINATION share/radiance)\n"
        "add_subdirectory(components/x)\n"
    )
    build = tmp_path / "build"
    subprocess.run(["cmake", "-S", str(src), "-B", str(build)], check=True, capture_output=True)
    return build


def _run(argv, cwd):
    subprocess.run(argv, cwd=cwd, check=True, capture_output=True)


def test_development_files_are_the_headers_and_package_files_only(tmp_path):
    build = _configure(tmp_path)
    prefix = tmp_path / "prefix"
    runtime_bundle.install_development_files(build, prefix, _run)
    files = sorted(str(p.relative_to(prefix)) for p in prefix.rglob("*") if p.is_file())
    assert files == ["include/radiance/rad_abi.h", "lib/cmake/radiance/radiance-config.cmake"]


def test_unconfigured_build_is_refused(tmp_path):
    with pytest.raises(runtime_bundle.RuntimeBundleError):
        runtime_bundle.install_development_files(tmp_path, tmp_path / "prefix", _run)


def test_stage_marker_is_a_safe_file_name(tmp_path):
    marker = runtime_bundle.stage_marker(tmp_path, "component:moe-remote,serving")
    assert marker.parent == tmp_path / "lane-stage"
    assert marker.name == "component_moe-remote_serving.done"

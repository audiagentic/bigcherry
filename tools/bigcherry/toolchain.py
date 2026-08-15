"""Relevant build environment and toolchain identity capture."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path


BUILD_ENVIRONMENT_KEYS = (
    "CC", "CXX", "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS",
    "CMAKE_PREFIX_PATH", "HIP_PATH", "ROCM_PATH", "HSA_OVERRIDE_GFX_VERSION",
)


def capture_environment(environ: dict[str, str] | None = None) -> tuple[tuple[str, str], ...]:
    values = environ if environ is not None else os.environ
    return tuple(sorted((key, values[key]) for key in BUILD_ENVIRONMENT_KEYS if key in values))


def _version(executable: str | os.PathLike[str] | None) -> dict[str, str | None]:
    if not executable:
        return {"path": None, "version": None}
    resolved = shutil.which(str(executable)) or str(Path(executable).expanduser())
    try:
        result = subprocess.run(
            [resolved, "--version"], check=True, capture_output=True, text=True,
            timeout=15,
        )
        output = (result.stdout or result.stderr).strip().splitlines()
        version = output[0] if output else ""
    except (OSError, subprocess.SubprocessError):
        version = None
    return {"path": str(Path(resolved).resolve()), "version": version}


def capture_toolchain(
    *,
    cmake: str = "cmake",
    ninja: str = "ninja",
    c_compiler: str | None = None,
    cxx_compiler: str | None = None,
    rocm_version: str | None = None,
) -> dict[str, object]:
    return {
        "cmake": _version(cmake),
        "ninja": _version(ninja),
        "c_compiler": _version(c_compiler or os.environ.get("CC")),
        "cxx_compiler": _version(cxx_compiler or os.environ.get("CXX")),
        "rocm_version": rocm_version,
        "host_os": platform.platform(),
        "architecture": platform.machine(),
    }

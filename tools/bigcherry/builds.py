"""Content-addressed build plans, effective build IDs, and reuse checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

from .context import ProjectContext


class BuildIdentityError(ValueError):
    pass


def _canonical(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_canonical(item) for item in value]
    return value


def _digest(domain: str, value: object) -> str:
    encoded = json.dumps(_canonical(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.blake2b(domain.encode() + b"\0" + encoded, digest_size=16).hexdigest()


@dataclass(frozen=True)
class BuildPlan:
    source_slice_id: str
    phase: str
    platform: str
    targets: tuple[str, ...]
    cmake_options: tuple[tuple[str, str], ...] = ()
    variant_set: str | None = None
    inventory_hash: str | None = None
    winners_hash: str | None = None
    resource_report_hashes: tuple[str, ...] = ()
    toolchain_request: tuple[tuple[str, str], ...] = ()
    environment: tuple[tuple[str, str], ...] = ()

    def canonical(self) -> dict[str, object]:
        return _canonical(asdict(self))  # type: ignore[return-value]

    @property
    def build_plan_id(self) -> str:
        return _digest("bigcherry/build-plan/v1", self.canonical())


def effective_build_id(configure_record: dict[str, object]) -> str:
    """Hash the normalized post-configure record, not the requested label."""
    if not isinstance(configure_record, dict) or not configure_record:
        raise BuildIdentityError("effective configure record must be a non-empty object")
    return _digest("bigcherry/effective-build/v1", configure_record)


def binary_hash(binary: Path) -> str:
    try:
        digest = hashlib.sha256()
        with binary.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError as exc:
        raise BuildIdentityError(f"cannot hash binary {binary}: {exc}") from exc


def build_directory(context: ProjectContext, source_slice_id: str, plan: BuildPlan) -> Path:
    return context.work_root / "builds" / source_slice_id / plan.build_plan_id


def resolve_runtime_artifacts(binary: Path) -> tuple[Path, ...]:
    """Every regular (non-symlink) file this build's own compile step
    produced alongside ``binary`` -- llama.cpp's CMake build places every
    project shared library directly next to its executables (libggml*.so*,
    libllama*.so*), never installed elsewhere. This is a directory-
    membership proxy for "this build's runtime output closure", not a
    linker-dependency walk (ldd/readelf) -- deliberately: system libraries
    (libc, the ROCm runtime itself) live outside this directory and are NOT
    part of what this specific build produced; a change there is a
    toolchain/environment identity question, not a build-output one.

    gpt-auto-agent review finding: a reuse check that only hashes the
    requested launcher (e.g. llama-bench) can accept a cache hit even if
    the actual HIP dispatch implementation (libggml-hip.so) changed --
    RE09's own investigation established that the meaningful logic lives
    there, not in the small launcher stub.
    """
    directory = binary.parent
    artifacts = [binary]
    for candidate in sorted(directory.glob("*.so*")):
        if candidate.is_file() and not candidate.is_symlink():
            artifacts.append(candidate)
    return tuple(artifacts)


def runtime_bundle_hash(artifacts: dict[str, str]) -> str:
    """Canonical hash of a ``{filename: sha256}`` runtime artifact map --
    what validate_reuse() now trusts instead of a single binary_hash alone.
    """
    return _digest("bigcherry/runtime-bundle/v1", artifacts)


#: CMakeCache.txt keys that actually describe what got built: resolved
#: compiler paths, build type, GPU targets, and every autotune/HIP option.
#: Deliberately not the whole cache -- that also carries unrelated
#: find_package probe results and generator-internal bookkeeping (line
#: numbers, help strings) that say nothing about build identity and would
#: make effective_build_id() sensitive to changes that do not matter.
_EFFECTIVE_CONFIGURE_PREFIXES = (
    "CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER", "CMAKE_BUILD_TYPE", "AMDGPU_TARGETS", "GGML_HIP",
)


def parse_effective_configure(cmake_cache: Path) -> dict[str, str]:
    """The post-configure record ``validate_reuse``/``effective_build_id``
    need: read back from CMakeCache.txt, not from the requested BuildPlan.
    ``BuildPlan`` records what was asked for; this records what CMake
    actually resolved, which is what a reuse decision needs to trust.
    """
    if not cmake_cache.is_file():
        raise BuildIdentityError(f"no CMakeCache.txt at {cmake_cache} -- configure did not run")
    record: dict[str, str] = {}
    for line in cmake_cache.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        key_type, sep, value = line.partition("=")
        if not sep:
            continue
        key = key_type.partition(":")[0]
        if key.startswith(_EFFECTIVE_CONFIGURE_PREFIXES):
            record[key] = value
    if not record:
        raise BuildIdentityError(f"no relevant configure keys found in {cmake_cache}")
    return record


def validate_reuse(
    metadata: dict[str, object],
    plan: BuildPlan,
    *,
    binary: Path,
    expected_toolchain: object | None = None,
    runtime_bundle_hash: str | None = None,
) -> None:
    if metadata.get("source_slice_id") != plan.source_slice_id:
        raise BuildIdentityError("source_slice_id does not match build plan")
    if metadata.get("build_plan_id") != plan.build_plan_id:
        raise BuildIdentityError("build_plan_id does not match requested plan")
    record = metadata.get("effective_configure")
    if not isinstance(record, dict):
        raise BuildIdentityError("effective configure metadata is missing")
    recorded_id = metadata.get("build_id")
    if recorded_id != effective_build_id(record):
        raise BuildIdentityError("recorded build_id does not recompute")
    if expected_toolchain is not None and metadata.get("toolchain") != expected_toolchain:
        raise BuildIdentityError("toolchain identity does not match")
    if not binary.is_file():
        raise BuildIdentityError("requested binary is missing")
    if metadata.get("binary_hash") != binary_hash(binary):
        raise BuildIdentityError("binary hash does not match recorded identity")
    # binary_hash alone only proves the requested launcher is unchanged --
    # RE09 established the meaningful HIP dispatch logic lives in
    # libggml-hip.so, not the launcher. When the caller supplies a
    # freshly-computed runtime_bundle_hash (see resolve_runtime_artifacts),
    # require it to match too, so a changed dependent library invalidates
    # reuse even when the launcher itself is byte-identical.
    if runtime_bundle_hash is not None and metadata.get("runtime_bundle_hash") != runtime_bundle_hash:
        raise BuildIdentityError("runtime bundle hash does not match recorded identity")

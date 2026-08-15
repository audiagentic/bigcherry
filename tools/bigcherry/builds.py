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


def validate_reuse(
    metadata: dict[str, object],
    plan: BuildPlan,
    *,
    binary: Path,
    expected_toolchain: object | None = None,
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

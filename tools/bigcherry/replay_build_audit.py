"""Offline audit of the production replay CMake source partition.

The replay build is deliberately audited from the patch description rather
than from a configured build tree.  This keeps the check usable on machines
without HIP and catches accidental broad-glob/link-graph regressions before a
GPU build is started.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


class ReplayBuildAuditError(ValueError):
    """The replay source partition is missing or contains forbidden inputs."""


REPLAY_REQUIRED = frozenset({
    "hip-autotune-dispatch.cu",
    "hip-autotune-transform.cu",
    "hip-autotune-reduce-telemetry.cpp",
    "hip-autotune-signature.cpp",
    "hip-autotune-blake2b.cpp",
    "hip-autotune-coverage.cpp",
    "hip-autotune-replay.cpp",
})
REPLAY_FORBIDDEN = frozenset({
    "hip-autotune-record.cpp",
    "hip-autotune-tuner.cu",
    "hip-autotune-io.cpp",
    "hip-autotune-journal.cpp",
    "hip-autotune-smi.cpp",
})


@dataclass(frozen=True)
class ReplaySourceAudit:
    unconditional: frozenset[str]
    replay_only: frozenset[str]
    tuning_only: frozenset[str]
    sqlite_mentions: tuple[str, ...]

    @property
    def replay_sources(self) -> frozenset[str]:
        return self.unconditional | self.replay_only


def _source_names(block: str) -> frozenset[str]:
    return frozenset(re.findall(r"hip-autotune-[A-Za-z0-9_-]+\.(?:cpp|cu)", block))


def audit_replay_source_partition(patch_source: str) -> ReplaySourceAudit:
    """Validate the replay/tune source partition in ``0100_cmake_options``.

    This intentionally checks the source graph and option guards, not CMake's
    generated implementation.  It therefore remains deterministic and does
    not mutate a checkout or require a compiler.
    """
    marker = "set(_BC_DISPATCH_SOURCES"
    start = patch_source.find(marker)
    if start < 0:
        raise ReplayBuildAuditError("explicit BigCherry source partition is missing")
    end = patch_source.find("list(APPEND GGML_SOURCES_ROCM ${_BC_DISPATCH_SOURCES})", start)
    if end < 0:
        raise ReplayBuildAuditError("BigCherry source partition has no append boundary")
    base = patch_source[start:end]
    replay_start = base.find("if (GGML_HIP_DISPATCH_REPLAY)")
    tune_start = base.find("if (GGML_HIP_AUTOTUNE)")
    first_guard = min(v for v in (replay_start, tune_start) if v >= 0)
    unconditional = _source_names(base[:first_guard])

    replay_match = re.search(
        r"if \(GGML_HIP_DISPATCH_REPLAY\)(.*?)endif\(\)", base, re.S)
    tune_match = re.search(
        r"if \(GGML_HIP_AUTOTUNE\)(.*?)endif\(\)", base, re.S)
    if not replay_match or not tune_match:
        raise ReplayBuildAuditError("replay/tune source guards are incomplete")
    replay_only = _source_names(replay_match.group(1))
    tuning_only = _source_names(tune_match.group(1))

    if not REPLAY_REQUIRED <= unconditional | replay_only:
        missing = sorted(REPLAY_REQUIRED - unconditional - replay_only)
        raise ReplayBuildAuditError(f"replay source graph is missing: {', '.join(missing)}")
    forbidden = (unconditional | replay_only) & REPLAY_FORBIDDEN
    if forbidden:
        raise ReplayBuildAuditError(
            f"production replay graph carries forbidden sources: {', '.join(sorted(forbidden))}"
        )
    if "hip-autotune-replay.cpp" not in replay_only:
        raise ReplayBuildAuditError("replay loader is not guarded by GGML_HIP_DISPATCH_REPLAY")
    if not REPLAY_FORBIDDEN & tuning_only:
        raise ReplayBuildAuditError("tuning-only source guard no longer contains tuner machinery")

    sqlite_mentions = tuple(
        line.strip() for line in patch_source.splitlines()
        if re.search(r"sqlite|SQLite", line)
    )
    if any(re.search(r"target_link_(?:libraries|options).*sqlite|[-/]lsqlite", line, re.I)
           for line in sqlite_mentions):
        raise ReplayBuildAuditError("replay patch contains an SQLite link directive")

    return ReplaySourceAudit(unconditional, replay_only, tuning_only, sqlite_mentions)

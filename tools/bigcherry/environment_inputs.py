"""Typed, hashed, immutable non-patch environment inputs for campaigns (RRBC03).

Represents driver/ICD, OS/kernel, ROCm/Vulkan/toolchain, env overrides, ASPM
policy, and offline-tuning-file (hipBLASLt/CK/rocblas-gemm-tune) inputs as
typed EnvironmentInput records, composed into a campaign/build/experiment's
identity WITHOUT mixing system state into patch_set_id.

Reuses (does not duplicate): tuning.journal.canonical/atomic_write for
canonical serialization and durable writes, experiment.bundle.safe_environment
for the existing allow-listed/secret-rejecting environment capture, and
build.toolchain.capture_environment for the existing build-environment key
set. This module adds the typed EnvironmentInput/EnvironmentInputSet
container and the digest composition rule on top of those.

Per the external design review (2026-09-10, folded into RRBC03's plan-item
notes): one raw digest cannot blindly represent both build-time and run-time
identity, so every EnvironmentInput carries an explicit `scope`. Offline
tuning files are keyed by logical role + content digest, never by host path
(a path is provenance, not identity). Mutating actions record the requested
action plus pre/post observation and a restoration outcome, never a silent
host mutation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .experiment.bundle import SECRET, file_hash
from .tuning.journal import canonical, checksum as _blake2b_checksum

Kind = Literal["observation", "action", "artifact"]
Scope = Literal["build", "run", "both", "evidence_only"]

_NAME_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")


class EnvironmentInputError(RuntimeError):
    pass


@dataclass(frozen=True)
class EnvironmentInput:
    """One typed, immutable environment fact.

    kind:
      observation -- a read-only fact about the current environment
                      (driver version, OS/kernel, toolchain version).
      action      -- a requested host mutation (ASPM policy write, env
                      override); `values` holds the requested action,
                      `pre`/`post` (also EnvironmentInput observations)
                      record state before/after, `restored` records whether
                      the action was reverted.
      artifact    -- an offline-tuning-file input (hipBLASLt/CK solution
                      cache, rocblas-gemm-tune CSV). Identified by
                      `logical_role` + content digest, never by host path.

    scope: which identity this input is composed into -- "build" (affects
    reusable build identity), "run" (affects run/execution identity only),
    "both", or "evidence_only" (recorded for provenance, composed into
    neither).
    """

    kind: Kind
    name: str
    scope: Scope
    values: tuple[tuple[str, str], ...] = ()
    files: tuple[tuple[str, str], ...] = ()  # (logical_role, content sha/blake digest)
    reversible: bool = False
    pre: "EnvironmentInput | None" = None
    post: "EnvironmentInput | None" = None
    restored: bool | None = None

    def __post_init__(self) -> None:
        if not _NAME_RE.match(self.name):
            raise EnvironmentInputError(f"invalid EnvironmentInput name: {self.name!r}")
        if self.kind not in ("observation", "action", "artifact"):
            raise EnvironmentInputError(f"invalid EnvironmentInput kind: {self.kind!r}")
        if self.scope not in ("build", "run", "both", "evidence_only"):
            raise EnvironmentInputError(f"invalid EnvironmentInput scope: {self.scope!r}")
        if self.kind == "action" and self.reversible and self.restored is None:
            raise EnvironmentInputError(
                f"reversible action {self.name!r} must record a restored outcome"
            )
        for key, value in self.values:
            if SECRET.search(key) or SECRET.search(value):
                raise EnvironmentInputError(f"secret-pattern value refused: {self.name}.{key}")
        for logical_role, digest in self.files:
            if not logical_role or "/" in logical_role or "\\" in logical_role:
                raise EnvironmentInputError(
                    f"file entry must use a logical role, not a path: {logical_role!r}"
                )
            if not re.fullmatch(r"[0-9a-f]{32,64}", digest.lower()):
                raise EnvironmentInputError(f"file entry digest is not a valid hex digest: {digest!r}")

    def canonical_document(self) -> dict:
        doc: dict = {
            "kind": self.kind,
            "name": self.name,
            "scope": self.scope,
            "values": sorted(self.values),
            "files": sorted(self.files),
            "reversible": self.reversible,
        }
        if self.kind == "action":
            doc["pre"] = self.pre.canonical_document() if self.pre else None
            doc["post"] = self.post.canonical_document() if self.post else None
            doc["restored"] = self.restored
        return doc

    def digest(self) -> str:
        return _blake2b_checksum(self.canonical_document())


@dataclass(frozen=True)
class EnvironmentInputSet:
    """An ordered, deduplicated collection of EnvironmentInputs plus the
    composed digests for each scope, per the build/run/both/evidence_only
    split (never one blind digest over everything)."""

    inputs: tuple[EnvironmentInput, ...]

    def __post_init__(self) -> None:
        names = [i.name for i in self.inputs]
        if len(names) != len(set(names)):
            raise EnvironmentInputError("duplicate EnvironmentInput name in set")

    def _scoped(self, scope: Scope) -> list[EnvironmentInput]:
        return sorted(
            (i for i in self.inputs if scope in (i.scope, "both") or i.scope == scope),
            key=lambda i: i.name,
        )

    def digest_for(self, scope: Literal["build", "run"]) -> str:
        """Digest composed of inputs scoped to `scope` or "both". Never
        includes "evidence_only" inputs -- those are provenance, not identity."""
        included = [i for i in self.inputs if i.scope in (scope, "both")]
        included.sort(key=lambda i: i.name)
        return _blake2b_checksum({"scope": scope, "inputs": [i.canonical_document() for i in included]})

    def build_digest(self) -> str:
        return self.digest_for("build")

    def run_digest(self) -> str:
        return self.digest_for("run")

    def to_document(self) -> dict:
        return {
            "schema": "bigcherry.environment-inputs.v1",
            "build_digest": self.build_digest(),
            "run_digest": self.run_digest(),
            "inputs": [i.canonical_document() for i in sorted(self.inputs, key=lambda i: i.name)],
        }


def artifact_input(
    *, name: str, logical_role: str, path: Path, scope: Scope = "run",
) -> EnvironmentInput:
    """Build an artifact EnvironmentInput for an offline-tuning file (e.g. a
    hipBLASLt/CK solution cache or a rocblas-gemm-tune override CSV).
    Identified by logical_role + content digest -- never by host path."""
    digest = file_hash(path)
    return EnvironmentInput(
        kind="artifact", name=name, scope=scope, files=((logical_role, digest),),
    )


def observation_input(
    *, name: str, scope: Scope, values: dict[str, str],
) -> EnvironmentInput:
    return EnvironmentInput(
        kind="observation", name=name, scope=scope,
        values=tuple(sorted(values.items())),
    )


def action_input(
    *, name: str, scope: Scope, values: dict[str, str],
    pre: EnvironmentInput, post: EnvironmentInput | None, restored: bool,
) -> EnvironmentInput:
    """A requested host-mutating action. `post`/`restored` are None/False
    respectively only while the action is still in flight; a published,
    evidence-accepted action must have both set."""
    return EnvironmentInput(
        kind="action", name=name, scope=scope,
        values=tuple(sorted(values.items())), reversible=True,
        pre=pre, post=post, restored=restored,
    )

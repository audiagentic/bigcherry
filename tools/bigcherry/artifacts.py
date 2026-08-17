"""Immutable, hash-verified campaign artifact publication."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .provenance import ProvenanceError, ProvenanceV2, require_promotable, validate_for_kind


class ArtifactError(ValueError):
    pass


@dataclass(frozen=True)
class ArtifactRef:
    """GPT-auto-agent implementation plan (2026-08-17), RE25.1: the same
    shape pipeline.ArtifactRef already has (pipeline.py keeps importing
    this one for compatibility -- see its own module docstring). Lives
    here, beside descriptor persistence, rather than in pipeline.py.
    """
    kind: str
    path: Path
    content_hash: str
    provenance: dict[str, object]
    #: Empty ("") for an ArtifactRef built the old way (publish_bytes/
    #: publish_file/publish_json + a hand-built provenance dict) --
    #: only publish_*_ref()/rehydrate() populate a real descriptor-backed
    #: artifact_id. Never used for equality/hashing decisions elsewhere;
    #: purely a rehydration handle.
    artifact_id: str = ""


DESCRIPTOR_SCHEMA_VERSION = 1


def _artifact_id(
    *, kind: str, relative_path: str, content_hash: str, provenance: ProvenanceV2,
) -> str:
    body = {
        "schema_version": DESCRIPTOR_SCHEMA_VERSION,
        "kind": kind,
        "relative_path": relative_path,
        "content_hash": content_hash,
        "provenance": provenance.document(),
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2b(b"bigcherry/artifact-descriptor/v1\0" + encoded, digest_size=16).hexdigest()


@dataclass(frozen=True)
class ArtifactDescriptor:
    """A persistable, independently rehydratable identity for one
    published artifact (RE25.1) -- ArtifactStore persists BYTES; this is
    what lets a SEPARATE process recover a byte-verified, provenance-
    complete ArtifactRef given only (store root, artifact_id), which is
    exactly what RE11's cross-process campaign resume needs and today
    cannot do (CampaignRun.records is process memory only).
    """
    schema_version: int
    artifact_id: str
    kind: str
    relative_path: str
    content_hash: str
    provenance: ProvenanceV2

    @classmethod
    def create(
        cls, *, kind: str, relative_path: str, content_hash: str, provenance: ProvenanceV2,
    ) -> "ArtifactDescriptor":
        artifact_id = _artifact_id(
            kind=kind, relative_path=relative_path, content_hash=content_hash, provenance=provenance)
        return cls(
            schema_version=DESCRIPTOR_SCHEMA_VERSION, artifact_id=artifact_id, kind=kind,
            relative_path=relative_path, content_hash=content_hash, provenance=provenance,
        )

    def document(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "relative_path": self.relative_path,
            "content_hash": self.content_hash,
            "provenance": self.provenance.document(),
        }

    @classmethod
    def from_document(cls, document: object) -> "ArtifactDescriptor":
        if not isinstance(document, dict):
            raise ArtifactError("artifact descriptor must be an object")
        if document.get("schema_version") != DESCRIPTOR_SCHEMA_VERSION:
            raise ArtifactError("artifact descriptor has an unsupported schema_version")
        kind = document.get("kind")
        relative_path = document.get("relative_path")
        content_hash = document.get("content_hash")
        artifact_id = document.get("artifact_id")
        # Explicit per-field isinstance (not all(isinstance(...))): behaviour
        # is identical, but pyright cannot narrow the four locals through the
        # generator form and would reject the _artifact_id/constructor calls.
        if (
            not isinstance(kind, str) or not kind
            or not isinstance(relative_path, str) or not relative_path
            or not isinstance(content_hash, str) or not content_hash
            or not isinstance(artifact_id, str) or not artifact_id
        ):
            raise ArtifactError(
                "artifact descriptor requires non-empty kind/relative_path/content_hash/artifact_id")
        try:
            provenance = ProvenanceV2.from_document(document.get("provenance"))
        except ProvenanceError as exc:
            raise ArtifactError(f"artifact descriptor has invalid provenance: {exc}") from exc
        # Re-derive artifact_id from the body EXCLUDING the claimed
        # artifact_id itself, and require it to match -- a descriptor
        # whose JSON was hand-edited (provenance/content_hash/kind/path
        # changed) while its filename/claimed artifact_id was left alone
        # must not be trusted just because the file parses.
        expected = _artifact_id(
            kind=kind, relative_path=relative_path, content_hash=content_hash, provenance=provenance)
        if expected != artifact_id:
            raise ArtifactError(
                f"artifact descriptor artifact_id {artifact_id!r} disagrees with its own "
                f"recomputed identity {expected!r} -- refusing to trust a tampered descriptor"
            )
        return cls(
            schema_version=DESCRIPTOR_SCHEMA_VERSION, artifact_id=artifact_id, kind=kind,
            relative_path=relative_path, content_hash=content_hash, provenance=provenance,
        )


def _descriptor_relative(artifact_id: str) -> str:
    return f".descriptors/v1/{artifact_id[:2]}/{artifact_id}.json"


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def _path(self, relative: str | Path) -> Path:
        candidate = (self.root / Path(relative)).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ArtifactError("artifact path escapes the store") from exc
        return candidate

    @staticmethod
    def digest(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def publish_bytes(self, relative: str | Path, data: bytes) -> str:
        target = self._path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = self.digest(data)
        if target.exists():
            if target.read_bytes() != data:
                raise ArtifactError(f"immutable artifact already exists with different bytes: {relative}")
            return digest
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return digest

    def publish_file(self, relative: str | Path, source: Path) -> str:
        try:
            data = source.read_bytes()
        except OSError as exc:
            raise ArtifactError(f"cannot read artifact source {source}: {exc}") from exc
        return self.publish_bytes(relative, data)

    def publish_json(self, relative: str | Path, value: object) -> str:
        return self.publish_bytes(
            relative,
            (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )

    def verify(self, relative: str | Path, expected_hash: str) -> bool:
        target = self._path(relative)
        return target.is_file() and self.digest(target.read_bytes()) == expected_hash

    def resolve(self, relative: str | Path) -> Path:
        target = self._path(relative)
        if not target.is_file():
            raise ArtifactError(f"artifact is missing: {relative}")
        return target

    # ------------------------------------------------------------ RE25.1 descriptors
    #
    # GPT-auto-agent implementation plan (2026-08-17): descriptors live in
    # their OWN central namespace (.descriptors/v1/<id[:2]>/<id>.json), not
    # one sidecar per byte path -- one immutable build byte object can
    # legitimately participate in several later campaign runs, each with
    # its own distinct provenance/descriptor (RE07's runtime-bundle
    # content-hash-by-full-manifest design already established this same
    # principle for a different artifact kind).

    def persist_descriptor(self, descriptor: ArtifactDescriptor) -> Path:
        """Verify the descriptor's own referenced bytes exist and match
        BEFORE publishing it -- a descriptor must never claim an identity
        for bytes that were never actually published under it."""
        if not self.verify(descriptor.relative_path, descriptor.content_hash):
            raise ArtifactError(
                f"refusing to persist descriptor {descriptor.artifact_id!r}: referenced "
                f"artifact {descriptor.relative_path!r} does not exist or does not match "
                f"content_hash {descriptor.content_hash!r}"
            )
        descriptor_relative = _descriptor_relative(descriptor.artifact_id)
        encoded = (json.dumps(descriptor.document(), indent=2, sort_keys=True) + "\n").encode("utf-8")
        self.publish_bytes(descriptor_relative, encoded)
        return self._path(descriptor_relative)

    def load_descriptor(self, artifact_id: str) -> ArtifactDescriptor:
        descriptor_relative = _descriptor_relative(artifact_id)
        target = self._path(descriptor_relative)
        if not target.is_file():
            raise ArtifactError(f"no artifact descriptor for artifact_id={artifact_id!r}")
        try:
            document = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"artifact descriptor {artifact_id!r} is not valid JSON: {exc}") from exc
        descriptor = ArtifactDescriptor.from_document(document)
        if descriptor.artifact_id != artifact_id:
            raise ArtifactError(
                f"artifact descriptor stored at {descriptor_relative!r} has artifact_id "
                f"{descriptor.artifact_id!r}, not the requested {artifact_id!r}"
            )
        return descriptor

    def rehydrate(
        self, artifact_id: str, *, expected_kind: str | None = None, require_promotable: bool = False,
    ) -> ArtifactRef:
        """The real cross-process recovery primitive RE11 needs: given
        only (store root, artifact_id), reconstruct a byte-verified,
        provenance-validated ArtifactRef -- treating BOTH the descriptor
        and the bytes it points at as claims to re-prove, never as an
        authority to trust blindly (matches RE04's cache-hit
        re-verification philosophy applied to a new kind of cached fact).
        """
        descriptor = self.load_descriptor(artifact_id)
        if expected_kind is not None and descriptor.kind != expected_kind:
            raise ArtifactError(
                f"artifact {artifact_id!r} has kind {descriptor.kind!r}, expected {expected_kind!r}")
        try:
            (validate_for_kind if not require_promotable else _require_promotable_wrapper)(
                descriptor.provenance.document(), kind=descriptor.kind)
        except ProvenanceError as exc:
            raise ArtifactError(f"artifact {artifact_id!r} failed provenance validation: {exc}") from exc
        target = self._path(descriptor.relative_path)
        if not target.is_file():
            raise ArtifactError(f"artifact {artifact_id!r} references missing bytes: {descriptor.relative_path!r}")
        actual_hash = self.digest(target.read_bytes())
        if actual_hash != descriptor.content_hash:
            raise ArtifactError(
                f"artifact {artifact_id!r} bytes have changed since publication "
                f"(recorded {descriptor.content_hash!r}, actual {actual_hash!r}) -- refusing to trust them"
            )
        return ArtifactRef(
            kind=descriptor.kind, path=target, content_hash=descriptor.content_hash,
            provenance=descriptor.provenance.document(), artifact_id=descriptor.artifact_id,
        )

    def _publish_ref(
        self, relative: str | Path, *, publish: "Callable[[], str]", kind: str, provenance: ProvenanceV2,
    ) -> ArtifactRef:
        content_hash = publish()
        if not self.verify(relative, content_hash):
            raise ArtifactError(f"published {relative} failed verification")
        descriptor = ArtifactDescriptor.create(
            kind=kind, relative_path=str(Path(relative).as_posix()),
            content_hash=content_hash, provenance=provenance,
        )
        self.persist_descriptor(descriptor)
        return ArtifactRef(
            kind=kind, path=self._path(relative), content_hash=content_hash,
            provenance=provenance.document(), artifact_id=descriptor.artifact_id,
        )

    def publish_bytes_ref(
        self, relative: str | Path, data: bytes, *, kind: str, provenance: ProvenanceV2,
    ) -> ArtifactRef:
        return self._publish_ref(relative, publish=lambda: self.publish_bytes(relative, data),
                                 kind=kind, provenance=provenance)

    def publish_file_ref(
        self, relative: str | Path, source: Path, *, kind: str, provenance: ProvenanceV2,
    ) -> ArtifactRef:
        return self._publish_ref(relative, publish=lambda: self.publish_file(relative, source),
                                 kind=kind, provenance=provenance)

    def publish_json_ref(
        self, relative: str | Path, value: object, *, kind: str, provenance: ProvenanceV2,
    ) -> ArtifactRef:
        return self._publish_ref(relative, publish=lambda: self.publish_json(relative, value),
                                 kind=kind, provenance=provenance)


def _require_promotable_wrapper(document: object, *, kind: str) -> ProvenanceV2:
    return require_promotable(document, kind=kind)

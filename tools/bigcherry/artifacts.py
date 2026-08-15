"""Immutable, hash-verified campaign artifact publication."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


class ArtifactError(ValueError):
    pass


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

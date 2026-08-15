"""Per-source lifecycle service with provenance checks before child work."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import provenance


class PipelineError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArtifactRef:
    kind: str
    path: Path
    content_hash: str
    provenance: dict[str, object]


class PipelineService:
    """Domain lifecycle boundary; scheduling remains outside this service."""

    def __init__(self, executor: Callable[[str, tuple[ArtifactRef, ...]], tuple[ArtifactRef, ...]]):
        self._executor = executor

    @staticmethod
    def _check_inputs(
        inputs: tuple[ArtifactRef, ...], expected: dict[str, str]
    ) -> None:
        for artifact in inputs:
            try:
                provenance.require_compatible(artifact.provenance, **expected)
            except provenance.ProvenanceError as exc:
                raise PipelineError(
                    f"{artifact.kind} cannot be consumed by this lifecycle stage: {exc}"
                ) from exc

    def run(
        self,
        stage: str,
        *,
        inputs: tuple[ArtifactRef, ...] = (),
        expected: dict[str, str],
    ) -> tuple[ArtifactRef, ...]:
        self._check_inputs(inputs, expected)
        try:
            outputs = self._executor(stage, inputs)
        except Exception as exc:
            raise PipelineError(f"stage {stage} failed before publishing outputs: {exc}") from exc
        if not isinstance(outputs, tuple):
            raise PipelineError(f"stage {stage} executor returned a non-tuple output set")
        for artifact in outputs:
            try:
                provenance.require_compatible(artifact.provenance, **expected)
            except provenance.ProvenanceError as exc:
                raise PipelineError(f"stage {stage} returned incompatible output: {exc}") from exc
        return outputs

    def record(self, *, expected: dict[str, str], inputs: tuple[ArtifactRef, ...] = ()):
        return self.run("record", expected=expected, inputs=inputs)

    def build_inventory(self, *, expected: dict[str, str], inputs: tuple[ArtifactRef, ...]):
        return self.run("inventory", expected=expected, inputs=inputs)

    def tune(self, *, expected: dict[str, str], inputs: tuple[ArtifactRef, ...]):
        return self.run("tune", expected=expected, inputs=inputs)

    def promote(self, *, expected: dict[str, str], inputs: tuple[ArtifactRef, ...]):
        return self.run("promote", expected=expected, inputs=inputs)

    def build_replay_full(self, *, expected: dict[str, str], inputs: tuple[ArtifactRef, ...]):
        return self.run("replay-full", expected=expected, inputs=inputs)

    def export_replay_cache(self, *, expected: dict[str, str], inputs: tuple[ArtifactRef, ...]):
        return self.run("replay-export", expected=expected, inputs=inputs)

    def validate_replay(self, *, expected: dict[str, str], inputs: tuple[ArtifactRef, ...]):
        return self.run("replay-validate", expected=expected, inputs=inputs)

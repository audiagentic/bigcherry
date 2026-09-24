"""Shared building blocks for patch-local validation producers.

Producers stay patch-local (they own their contract's measurement); these
helpers only remove repeated plumbing: picking the one mapped device for the
run's architecture, starting an attested llama-server on it, content-binding
a registered model, and turning a paired llama-bench lane into a LaneEffect.
"""

from __future__ import annotations

import hashlib
import tomllib
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from bigcherry.core import paths as bc_paths
from bigcherry.experiment import execution as experiment_execution
from bigcherry.experiment.attestation import ExecutionIdentity
from bigcherry.experiment.server_execution import AttestedServerSession
from bigcherry.patch import validation_producer as vp


def single_architecture(ctx: vp.ProducerContext, allowed: tuple[str, ...], *, label: str) -> str:
    targets = ctx.fat_targets.targets
    if len(targets) != 1 or targets[0] not in allowed:
        raise vp.ValidationProducerError(
            f"{label}: needs exactly one of {allowed!r} per run; got {targets!r}"
        )
    return targets[0]


def select_device(ctx: vp.ProducerContext, architecture: str, *, label: str) -> vp.ProducerDeviceContext:
    matches = [d for d in ctx.runtime.device_contexts(device_map=ctx.device_map) if d.architecture == architecture]
    if len(matches) != 1:
        raise vp.ValidationProducerError(
            f"{label}: --device-map must select exactly one {architecture} device; got {len(matches)}"
        )
    return matches[0]


def model_identity(path: Path, *, model_id: str, label: str) -> dict[str, object]:
    """Verify ``path`` is the registered file for ``model_id`` and hash it."""
    try:
        registry = tomllib.loads(bc_paths.MODELS.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise vp.ValidationProducerError(f"{label}: cannot read {bc_paths.MODELS}: {exc}") from exc
    entry = next(
        (m for m in registry.get("models", []) if isinstance(m, dict) and m.get("id") == model_id),
        None,
    )
    if not isinstance(entry, dict):
        raise vp.ValidationProducerError(f"{label}: model registry has no {model_id!r} entry")
    if not path.is_file():
        raise vp.ValidationProducerError(f"{label}: model file does not exist: {path}")
    if path.name != Path(str(entry.get("path", ""))).name:
        raise vp.ValidationProducerError(f"{label}: {path.name!r} is not the registered file for {model_id}")
    size = entry.get("size-bytes")
    if not isinstance(size, int) or isinstance(size, bool) or path.stat().st_size != size:
        raise vp.ValidationProducerError(f"{label}: {path} size does not match registry size {size!r}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"model_id": model_id, "path": str(path), "size_bytes": size, "sha256": digest.hexdigest()}


def server_session_factory(
    ctx: vp.ProducerContext,
    *,
    device: vp.ProducerDeviceContext,
    architecture: str,
    binary: Path,
    model: Path,
    log_path: Path,
    env: Mapping[str, str] | None = None,
    server_args: tuple[str, ...] = ("-ngl", "99", "-c", "1024", "--parallel", "1"),
) -> Callable[[], AbstractContextManager[Any]]:
    """An attested llama-server on ``device``; fusion is never disabled."""
    if device.locator is not None:
        expected = ExecutionIdentity(backend="ROCm", architectures=(architecture,), locators=(device.locator,))
        by_locator: dict[str, str] | None = {device.locator: architecture}
    else:
        expected = device.execution_identity
        by_locator = None
    server_env = dict(ctx.build_env)
    server_env.update(dict(device.env_overrides))
    for key in device.env_unset:
        server_env.pop(key, None)
    server_env.pop("GGML_CUDA_DISABLE_FUSION", None)
    server_env.update(env or {})
    env_unset = (*device.env_unset, "GGML_CUDA_DISABLE_FUSION")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _open() -> AbstractContextManager[Any]:
        return AttestedServerSession(
            binary=binary,
            model=model,
            expected=expected,
            extra_args=server_args,
            log_path=log_path,
            env_overrides=server_env,
            env_unset=env_unset,
            architecture_by_locator=by_locator,
        )

    return _open


def lane_effect(outcome: vp.ProducerPairedBenchmarkOutcome, *, workload: str, metric: str, role: str,
                rounds: int, label: str):
    """(LaneEffect, run) for a single-workload paired outcome with exactly ``rounds`` rounds."""
    if set(outcome.runs) != {workload}:
        raise vp.ValidationProducerError(
            f"{label}: {role} lane must produce exactly one {workload} lane; got {sorted(outcome.runs)!r}"
        )
    run = outcome.runs[workload]
    if dict(run.stats).get("paired_rounds") != rounds:
        raise vp.ValidationProducerError(
            f"{label}: {role} lane has {run.stats.get('paired_rounds')!r} paired rounds; expected {rounds}"
        )
    return experiment_execution.lane_effect_from_run(role, metric, run), run

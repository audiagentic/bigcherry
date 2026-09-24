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


MTP_SERVER_ARGS = (
    "--parallel", "1", "--metrics", "-sm", "tensor", "--fit", "off",
    "--spec-type", "draft-mtp", "--spec-draft-n-max", "4",
)


def mtp_server_lane(
    ctx: vp.ProducerContext,
    *,
    control_binary: Path,
    subject_binary: Path,
    expected: ExecutionIdentity,
    env: Mapping[str, str],
    label: str,
    role: str = "positive",
    server_args: tuple[str, ...] = MTP_SERVER_ARGS,
    warmup_pairs: int = 2,
    measured_pairs: int = 10,
    n_predict: int = 128,
) -> tuple[Any, dict[str, list[dict[str, Any]]], dict[str, Path]]:
    """Paired MTP speculative-decode lane on llama-server (metric mtp_wall_tps).

    Every request gets a fresh server so control and subject never share a
    GPU (the large-tier model needs ~13GB/GPU under -sm tensor). Prompts come
    from ``ctx.corpus``; the client-measured wall-clock tokens/s is the
    sample. Returns (LaneEffect, per-arm request records, per-arm combined log).
    """
    import re

    from bigcherry.bench import server_completion as sc

    if ctx.model is None or ctx.corpus is None:
        raise vp.ValidationProducerError(f"{label}: the MTP lane needs --model and --producer-corpus")
    prompts, corpus_sha256 = sc.load_corpus(ctx.corpus)
    logs_dir = ctx.workdir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    binaries = {"control": control_binary, "subject": subject_binary}
    per_request_logs: dict[str, list[Path]] = {"control": [], "subject": []}
    records: dict[str, list[dict[str, Any]]] = {"control": [], "subject": []}
    counters = {"control": 0, "subject": 0}
    session_kwargs = dict(
        corpus_id=ctx.corpus.stem,
        corpus_sha256=corpus_sha256,
        bigcherry_revision=label,
        llama_pin="",
        llama_revision="",
        model_id=str(ctx.model),
        server_argv=server_args,
        spec_type="draft-mtp",
        spec_n_max=4,
        spec_draft_k="default",
        spec_draft_v="default",
        sampling=sc.SamplingConfig(temperature=1.0, top_p=0.95, top_k=20),
        n_predict=n_predict,
        order_seed=12345,
    )
    configs = {arm: sc.SessionConfig(session_id=f"{label}-mtp-{arm}", **session_kwargs) for arm in binaries}
    server_env = dict(env)
    server_env.pop("ROCR_VISIBLE_DEVICES", None)

    def _runner(command: list[str]) -> experiment_execution.RunnerOutput:
        arm = command[-1]
        index = counters[arm]
        counters[arm] += 1
        log_path = logs_dir / f"{label}-mtp-{arm}-server-{index}.log"
        per_request_logs[arm].append(log_path)
        session = AttestedServerSession(
            binary=binaries[arm],
            model=ctx.model,
            expected=expected,
            extra_args=server_args,
            log_path=log_path,
            env_overrides=server_env,
            env_unset=("ROCR_VISIBLE_DEVICES",),
        )
        with session:
            transport = sc.HttpTransport(session.base_url)
            sc.validate_server(transport)
            record = sc.run_request(
                transport, prompts[index % len(prompts)], configs[arm], pass_number=1, order_index=index
            )
        records[arm].append(record)
        if not isinstance(record.get("wall_tps"), (int, float)):
            raise vp.ValidationProducerError(f"{label} MTP lane ({arm}, request {index}): no usable wall_tps")
        return experiment_execution.RunnerOutput(
            returncode=0, stdout=f"BIGCHERRY_MTP_LANE wall_tps={record['wall_tps']}\n", stderr=""
        )

    for _ in range(warmup_pairs):
        _runner(["mtp-lane", "control"])
        _runner(["mtp-lane", "subject"])
    paired = experiment_execution.run_paired_lane(
        metric="mtp_wall_tps",
        control_command=["mtp-lane", "control"],
        subject_command=["mtp-lane", "subject"],
        pattern=re.compile(r"BIGCHERRY_MTP_LANE wall_tps=([0-9.]+)"),
        pairs=measured_pairs,
        runner=_runner,
    )
    combined: dict[str, Path] = {}
    for arm, paths in per_request_logs.items():
        combined[arm] = ctx.workdir / f"{label}-mtp-{arm}.log"
        combined[arm].write_text(
            "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in paths if p.exists()),
            encoding="utf-8",
        )
    effect = experiment_execution.lane_effect_from_run(role, "mtp_wall_tps", paired)
    return effect, records, combined

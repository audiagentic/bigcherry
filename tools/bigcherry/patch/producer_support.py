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


_PREFLIGHT_CTX = 4096


def contract_measurement(contract_id: str) -> Any:
    """The contract's declared measurement procedure (defaults when absent)."""
    from bigcherry.core import paths as core_paths
    from bigcherry.experiment import contract as experiment_contract

    contract = _contract_registry(core_paths.EXPERIMENT_CONTRACTS).contracts.get(contract_id)
    if contract is None:
        raise vp.ValidationProducerError(f"unknown experiment contract {contract_id!r}")
    return contract.measurement or experiment_contract.Measurement()


def contract_paired_rounds(contract_id: str) -> int:
    """Paired rounds per lane, from the contract's acceptance.min_paired_rounds.

    Producers take the round count from the contract they are bound to rather
    than a local constant, so a pre-declared contract amendment (e.g. a new
    series with more rounds) is the single place the design changes."""
    from bigcherry.core import paths as core_paths

    registry = _contract_registry(core_paths.EXPERIMENT_CONTRACTS)
    contract = registry.contracts.get(contract_id)
    if contract is None:
        raise vp.ValidationProducerError(f"unknown experiment contract {contract_id!r}")
    rounds = contract.acceptance.min_paired_rounds
    if not isinstance(rounds, int) or rounds < 1:
        raise vp.ValidationProducerError(f"{contract_id}: acceptance.min_paired_rounds is not declared")
    return rounds


_REGISTRY_CACHE: dict[Path, Any] = {}


def _contract_registry(path: Path) -> Any:
    from bigcherry.experiment import contract as experiment_contract

    if path not in _REGISTRY_CACHE:
        _REGISTRY_CACHE[path] = experiment_contract.load_contracts(path)
    return _REGISTRY_CACHE[path]


def _is_tensor_split(server_args: tuple[str, ...]) -> bool:
    return any(a in ("-sm", "--split-mode") and b == "tensor" for a, b in zip(server_args, server_args[1:]))


def tensor_split_preflights(
    binaries: Mapping[str, Path], *, model: Path, server_args: tuple[str, ...],
    env: Mapping[str, str], workdir: Path, label: str,
) -> dict[str, Any]:
    """Attest each arm's tensor-split server once with the untimed RCCL
    preflight (its only diagnostic delta is NCCL_DEBUG + --verbosity 5).
    Timed processes of the same binary/model/args/devices are then bound to
    this attestation (AttestedServerSession tensor_split_preflight)."""
    import os

    from bigcherry.campaign import benchmark as campaign_benchmark
    from bigcherry.core import environment as bc_environment
    from bigcherry.experiment.attestation import ExecutionAttestation, ObservedDevice

    visible = env.get("HIP_VISIBLE_DEVICES", "")
    ids = [int(d) for d in visible.split(",") if d.strip()]
    if len(ids) < 2:
        raise vp.ValidationProducerError(f"{label}: tensor split needs HIP_VISIBLE_DEVICES with >= 2 devices")
    inventory = {d.index: d for d in bc_environment.load_default().host().devices}
    missing = [i for i in ids if i not in inventory or inventory[i].locator is None]
    if missing:
        raise vp.ValidationProducerError(f"{label}: devices {missing} lack a configured locator")
    expected = {
        "backend": "rocm",
        "architectures": [inventory[i].arch for i in ids],
        "locators": [inventory[i].locator for i in ids],
    }
    full_env = {k: v for k, v in os.environ.items() if k != "ROCR_VISIBLE_DEVICES"}
    full_env.update(env)
    # The internal HIP AllReduce (not RCCL) carries tensor split at this pin,
    # so no RCCL binding record exists. The preflight instead runs the same
    # binary/model/devices under -sm layer, whose "using device ROCmN (...)
    # (<pci>)" lines name each physical card; that split-mode swap is the
    # only argument delta and is recorded in the attestation telemetry.
    preflight_args = tuple("layer" if (prev in ("-sm", "--split-mode") and arg == "tensor") else arg
                           for prev, arg in zip(("",) + server_args, server_args))
    # Layer split puts a whole layer's KV (plus the MTP draft context) on one
    # card, which the model's default context does not fit (1241: OOM on
    # device 1). Device identity does not depend on context size, so the
    # preflight caps it; the cap is the second recorded argument delta.
    if not any(a in ("-c", "--ctx-size") for a in preflight_args):
        preflight_args = (*preflight_args, "-c", str(_PREFLIGHT_CTX))
    else:
        preflight_args = tuple(str(_PREFLIGHT_CTX) if prev in ("-c", "--ctx-size") else arg
                               for prev, arg in zip(("",) + preflight_args, preflight_args))
    out: dict[str, Any] = {}
    for arm, binary in binaries.items():
        document, _binding = campaign_benchmark._run_server_attestation_preflight(
            binary=binary, model=model, extra_args=preflight_args,
            output=workdir / f"{label}-{arm}-attestation", env=full_env, expected_execution=expected,
        )
        out[arm] = ExecutionAttestation(
            backend=document["backend"],
            devices=tuple(ObservedDevice(d["architecture"], d["locator"]) for d in document["devices"]),
            telemetry={"attested_by": "layer-split-preflight", "preflight_split_mode": "layer",
                       "preflight_ctx_size": _PREFLIGHT_CTX,
                       **document.get("telemetry", {})},
        )
    return out


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
    requests_per_start: int = 1,
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
    starts = {"control": 0, "subject": 0}
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
    preflights = (
        tensor_split_preflights(binaries, model=ctx.model, server_args=server_args, env=server_env,
                                workdir=logs_dir, label=label)
        if _is_tensor_split(server_args) else {arm: None for arm in binaries}
    )

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
            tensor_split_preflight=preflights[arm],
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

    def _serve_block(arm: str, count: int) -> list[str]:
        """One server start: a warm-up request, then `count` measured requests."""
        start = starts[arm]
        starts[arm] += 1
        log_path = logs_dir / f"{label}-mtp-{arm}-server-block-{start}.log"
        per_request_logs[arm].append(log_path)
        session = AttestedServerSession(
            binary=binaries[arm], model=ctx.model, expected=expected, extra_args=server_args,
            log_path=log_path, env_overrides=server_env, env_unset=("ROCR_VISIBLE_DEVICES",),
            tensor_split_preflight=preflights[arm],
        )
        outputs: list[str] = []
        with session:
            transport = sc.HttpTransport(session.base_url)
            sc.validate_server(transport)
            warm = counters[arm]
            sc.run_request(transport, prompts[warm % len(prompts)], configs[arm], pass_number=0, order_index=warm)
            for _ in range(count):
                index = counters[arm]
                counters[arm] += 1
                record = sc.run_request(
                    transport, prompts[index % len(prompts)], configs[arm], pass_number=1, order_index=index
                )
                records[arm].append(record)
                if not isinstance(record.get("wall_tps"), (int, float)):
                    raise vp.ValidationProducerError(f"{label} MTP lane ({arm}, request {index}): no usable wall_tps")
                outputs.append(f"BIGCHERRY_MTP_LANE wall_tps={record['wall_tps']}\n")
        return outputs

    pattern = re.compile(r"BIGCHERRY_MTP_LANE wall_tps=([0-9.]+)")
    if requests_per_start == 1:
        for _ in range(warmup_pairs):
            _runner(["mtp-lane", "control"])
            _runner(["mtp-lane", "subject"])
        paired = experiment_execution.run_paired_lane(
            metric="mtp_wall_tps",
            control_command=["mtp-lane", "control"],
            subject_command=["mtp-lane", "subject"],
            pattern=pattern,
            pairs=measured_pairs,
            runner=_runner,
        )
    else:
        # Batched: one server start per arm per block serves one unmeasured
        # warm-up request plus up to `requests_per_start` measured requests;
        # every measured request stays its own sample (same pair count, far
        # fewer model loads). Blocks alternate which arm runs first.
        samples: dict[str, list[str]] = {"control": [], "subject": []}
        block = 0
        while len(samples["control"]) < measured_pairs:
            count = min(requests_per_start, measured_pairs - len(samples["control"]))
            order = ("control", "subject") if block % 2 == 0 else ("subject", "control")
            for arm in order:
                samples[arm].extend(_serve_block(arm, count))
            block += 1
        replay = {arm: iter(values) for arm, values in samples.items()}

        def _replay(command: list[str]) -> experiment_execution.RunnerOutput:
            return experiment_execution.RunnerOutput(returncode=0, stdout=next(replay[command[-1]]), stderr="")

        paired = experiment_execution.run_paired_lane(
            metric="mtp_wall_tps",
            control_command=["mtp-lane", "control"],
            subject_command=["mtp-lane", "subject"],
            pattern=pattern,
            pairs=measured_pairs,
            runner=_replay,
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


def performance_metrics(*effects: Any) -> dict[str, dict[str, Any]]:
    """The benchmark validator's required ``metrics`` block: one entry per
    measured lane, keyed ``<role>_<metric>``."""
    import dataclasses

    metrics = {f"{effect.role}_{effect.metric}": dataclasses.asdict(effect) for effect in effects}
    if not metrics:
        raise vp.ValidationProducerError("performance artifact needs at least one measured lane")
    return metrics


_TBO_PASSED = __import__("re").compile(r"(\d+)/(\d+) tests passed")


def run_backend_ops(
    binary: Path, args: tuple[str, ...], env: Mapping[str, str], *, label: str, timeout_s: int = 3600
) -> tuple[str, int, int, int]:
    """Run test-backend-ops (test mode) and return (output, returncode, passed, total).

    ``total`` is the backend's case count from its "N/M tests passed" summary;
    a run that prints no summary fails closed.
    """
    import subprocess

    completed = subprocess.run(
        [str(binary), *args], capture_output=True, text=True, env=dict(env), check=False, timeout=timeout_s
    )
    text = (completed.stdout or "") + "\n" + (completed.stderr or "")
    counts = _TBO_PASSED.findall(text)
    if not counts:
        raise vp.ValidationProducerError(
            f"{label}: test-backend-ops printed no pass summary (exit {completed.returncode})"
        )
    passed, total = (int(v) for v in counts[-1])
    return text, completed.returncode, passed, total


def device_env(ctx: vp.ProducerContext, device: vp.ProducerDeviceContext, extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build env + the device's selector, device env_unset removed, plus ``extra``."""
    env = dict(ctx.build_env)
    env.update(dict(device.env_overrides))
    for key in device.env_unset:
        env.pop(key, None)
    env.update(extra or {})
    return env


def compact_log(text: str, *, head_lines: int = 400) -> str:
    """A bounded trace artifact: the log's head (startup, device attestation)
    plus every BIGCHERRY_ line. Attested servers run at --verbosity 5 and log
    each full-vocabulary response, so whole logs run to gigabytes."""
    lines = text.splitlines()
    markers = [line for line in lines[head_lines:] if "BIGCHERRY_" in line]
    omitted = len(lines) - head_lines - len(markers)
    tail = [f"... [{omitted} lines omitted; BIGCHERRY_ lines kept] ..."] if omitted > 0 else []
    return "\n".join(lines[:head_lines] + tail + markers) + "\n"

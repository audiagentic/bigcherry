"""PRBE35 (RD43/1216): patch-local validation producer.

RD43-CONCURRENT-JOIN-FUSION-GUARD is a correctness contract: 1216 keeps
op-fusion from absorbing the join node of 1215's concurrent shared-expert
region. Both arms therefore carry 1215 (its hard prerequisite) and differ
only by 1216:

  control = bigcherry + 1215
  subject = bigcherry + 1215 + 1216

Everything runs with GGML_CUDA_GRAPH_OPT=1, the only mode in which 1215
launches concurrent regions and the guard can engage.

Three claims, measured here:
- correctness (``backend_reference``): a fixed temperature-0 request on
  llama-server returns identical generated tokens and full-vocabulary
  logprobs within tolerance on both arms. Completing at all under graph
  capture is the primary hazard RD43 closes.
- activation: the subject server log carries the 1216 marker (the fusion
  horizon was actually capped at a join node); the control log cannot.
- controls: a paired decode llama-bench, 10 rounds, both arms under
  GGML_CUDA_GRAPH_OPT=1 (the contract allows at most a 1% regression).

The standard scaffold is skipped: its control is the plain baseline, which
cannot build the subject (1216 requires 1215).
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import importlib
import json
import math
import re
import sys
import tomllib
import urllib.error
import urllib.request
from array import array
from collections.abc import Iterable, Mapping
from pathlib import Path

from bigcherry.patch import validation_producer as vp  # type: ignore[import-not-found]

_CONTRACT_ARCHITECTURES: tuple[str, ...] = ("gfx1100", "gfx1201", "gfx1030")
_CONTRACT_ID = "RD43-CONCURRENT-JOIN-FUSION-GUARD"
_PREREQUISITE = "1215_rd394041_amd_stream_moe_overlap"
_MODEL_REF = "tierM-qwen35b-a3b-moe-mtp"
_GRAPH_OPT_ENV = {"GGML_CUDA_GRAPH_OPT": "1"}
_MARKER_REGEX = r"BIGCHERRY_PATCH_HIT patch=1216_rd43 path=join_fusion_cap"
_TOLERANCE = 0.0005
_N_PREDICT = 64
_PROMPT = (
    "Explain in one concise sentence why a shared expert can run beside the "
    "routed experts without changing the layer's output."
)
_TIMEOUT_S = 1800
_MIN_PAIRED_ROUNDS = 10

_ARTIFACT_NAME = "rd43-backend-reference.json"
_PERFORMANCE_ARTIFACT_NAME = "rd43-controls.json"
_SUBJECT_TRACE_ARTIFACT_NAME = "rd43-subject-server.log"
_CONTROL_TRACE_ARTIFACT_NAME = "rd43-control-server.log"


def _fail(message: str) -> vp.ValidationProducerError:
    return vp.ValidationProducerError(f"rd43: {message}")


def _content_identity(path: Path, *, model_id: str) -> dict[str, object]:
    """Verify the model file against the registry entry and content-bind it."""
    from bigcherry.core import paths as bc_paths  # type: ignore[import-not-found]

    try:
        registry = tomllib.loads(bc_paths.MODELS.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise _fail(f"cannot read {bc_paths.MODELS}: {exc}") from exc
    entry = next(
        (m for m in registry.get("models", []) if isinstance(m, dict) and m.get("id") == model_id),
        None,
    )
    if not isinstance(entry, dict):
        raise _fail(f"model registry has no {model_id!r} entry")
    if not path.is_file():
        raise _fail(f"model file does not exist: {path}")
    if path.name != Path(str(entry.get("path", ""))).name:
        raise _fail(f"{path.name!r} is not the registered file for {model_id}")
    size = entry.get("size-bytes")
    if not isinstance(size, int) or isinstance(size, bool) or path.stat().st_size != size:
        raise _fail(f"{path} size does not match the registry size {size!r}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "model_id": model_id,
        "path": str(path),
        "size_bytes": size,
        "sha256": digest.hexdigest(),
    }


def _request_payload(vocab_size: int) -> dict[str, object]:
    return {
        "prompt": _PROMPT,
        "n_predict": _N_PREDICT,
        "n_probs": vocab_size,
        "post_sampling_probs": False,
        "temperature": 0.0,
        "seed": 42,
        "cache_prompt": False,
        "ignore_eos": True,
        "return_tokens": True,
        "stream": True,
    }


def _stream_completion_rows(
    base_url: str, payload: Mapping[str, object], *, timeout_s: int
) -> Iterable[Mapping[str, object]]:
    """Yield one probability row per generated token from a streaming /completion."""
    request = urllib.request.Request(
        f"{base_url}/completion",
        data=json.dumps(dict(payload)).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    saw_stop = False
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="strict").strip()
                if not line or line.startswith((":", "event:")):
                    continue
                if not line.startswith("data:"):
                    raise _fail(f"malformed SSE line: {line[:120]!r}")
                encoded = line.removeprefix("data:").strip()
                if encoded == "[DONE]":
                    saw_stop = True
                    break
                try:
                    event = json.loads(encoded)
                except json.JSONDecodeError as exc:
                    raise _fail("malformed JSON in completion stream") from exc
                if not isinstance(event, Mapping) or not isinstance(event.get("stop"), bool):
                    raise _fail("completion stream event lacks a boolean stop")
                if event["stop"]:
                    saw_stop = True
                    continue
                rows = event.get("completion_probabilities")
                if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], Mapping):
                    raise _fail("expected exactly one completion_probabilities row per event")
                yield rows[0]
    except (urllib.error.URLError, OSError, TimeoutError, UnicodeError) as exc:
        raise _fail(f"streaming /completion failed: {exc}") from exc
    if not saw_stop:
        raise _fail("completion stream ended without a stop event")


def _dense_logprobs(
    row: Mapping[str, object], *, vocab_size: int, arm: str, step: int
) -> tuple[int, array]:
    """Validate one full-vocabulary row; return (generated id, id-indexed logprobs)."""
    generated_id = row.get("id")
    if not isinstance(generated_id, int) or isinstance(generated_id, bool) or not 0 <= generated_id < vocab_size:
        raise _fail(f"{arm} step {step} has an invalid generated token id")
    top = row.get("top_logprobs")
    if not isinstance(top, list) or len(top) != vocab_size:
        actual = len(top) if isinstance(top, list) else None
        raise _fail(f"{arm} step {step} is not full-vocabulary: expected {vocab_size}, got {actual!r}")
    values = array("d", [math.nan]) * vocab_size
    seen = bytearray(vocab_size)
    for entry in top:
        token_id = entry.get("id") if isinstance(entry, Mapping) else None
        logprob = entry.get("logprob") if isinstance(entry, Mapping) else None
        if not isinstance(token_id, int) or isinstance(token_id, bool) or not 0 <= token_id < vocab_size:
            raise _fail(f"{arm} step {step} has an invalid vocabulary token id {token_id!r}")
        if seen[token_id]:
            raise _fail(f"{arm} step {step} duplicates token id {token_id}")
        if not isinstance(logprob, (int, float)) or isinstance(logprob, bool) or not math.isfinite(float(logprob)):
            raise _fail(f"{arm} step {step} token {token_id} has an invalid logprob")
        seen[token_id] = 1
        values[token_id] = float(logprob)
    return generated_id, values


def _canonical_bytes(values: array) -> bytes:
    if sys.byteorder == "little":
        return values.tobytes()
    copied = array("d", values)
    copied.byteswap()
    return copied.tobytes()


def _vocab_size(base_url: str) -> int:
    """The served model's vocabulary size (llama-server /v1/models data[0].meta.n_vocab)."""
    try:
        with urllib.request.urlopen(f"{base_url}/v1/models", timeout=60) as response:
            models = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise _fail(f"cannot read /v1/models: {exc}") from exc
    data = models.get("data") if isinstance(models, Mapping) else None
    meta = data[0].get("meta") if isinstance(data, list) and data and isinstance(data[0], Mapping) else None
    value = meta.get("n_vocab") if isinstance(meta, Mapping) else None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise _fail("llama-server /v1/models does not report meta.n_vocab")
    return value


def run(ctx: vp.ProducerContext) -> vp.ProducerResult:
    experiment_contract = importlib.import_module("bigcherry.experiment.contract")
    experiment_execution = importlib.import_module("bigcherry.experiment.execution")
    ExecutionIdentity = importlib.import_module("bigcherry.experiment.attestation").ExecutionIdentity
    AttestedServerSession = importlib.import_module("bigcherry.experiment.server_execution").AttestedServerSession
    psi = importlib.import_module("bigcherry.patch.source")
    ActivationEvidence = importlib.import_module("bigcherry.patch.activation").ActivationEvidence

    targets = ctx.fat_targets.targets
    if len(targets) != 1 or targets[0] not in _CONTRACT_ARCHITECTURES:
        raise _fail(f"{_CONTRACT_ID} needs exactly one contract architecture per run; got {targets!r}")
    architecture = targets[0]
    if ctx.model is None:
        raise _fail("a real model (--model) is required")
    model = ctx.model
    model_identity = _content_identity(model, model_id=_MODEL_REF)

    server_pair = ctx.runtime.build_pair(
        targets=_CONTRACT_ARCHITECTURES,
        primary_target="llama-server",
        common_extra_patches=(_PREREQUISITE,),
        baseline_source="bigcherry",
        require_parity=True,
    )
    bench_pair = ctx.runtime.build_pair(
        targets=_CONTRACT_ARCHITECTURES,
        primary_target="llama-bench",
        common_extra_patches=(_PREREQUISITE,),
        baseline_source="bigcherry",
        require_parity=True,
    )

    devices = [d for d in ctx.runtime.device_contexts(device_map=ctx.device_map) if d.architecture == architecture]
    if len(devices) != 1:
        raise _fail(f"--device-map must select exactly one {architecture} device; got {len(devices)}")
    device = devices[0]

    if device.locator is not None:
        expected = ExecutionIdentity(backend="ROCm", architectures=(architecture,), locators=(device.locator,))
        architecture_by_locator = {device.locator: architecture}
    else:
        expected = device.execution_identity
        architecture_by_locator = None

    server_env = dict(ctx.build_env)
    server_env.update(dict(device.env_overrides))
    for key in device.env_unset:
        server_env.pop(key, None)
    server_env.pop("GGML_CUDA_DISABLE_FUSION", None)
    server_env.update(_GRAPH_OPT_ENV)
    server_env["BIGCHERRY_PATCH_TRACE"] = "1"
    env_unset = (*device.env_unset, "GGML_CUDA_DISABLE_FUSION")
    server_args = ("-ngl", "99", "-c", "1024", "--parallel", "1")

    logs_dir = ctx.workdir / "logs"
    scratch_dir = ctx.workdir / "scratch"
    logs_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    spool_path = scratch_dir / "rd43-control-full-vocab-logprobs.f64"
    control_log = logs_dir / "rd43-backend-reference-control-server.log"
    subject_log = logs_dir / "rd43-backend-reference-subject-server.log"

    control_tokens: list[int] = []
    subject_tokens: list[int] = []
    control_digest = hashlib.sha256()
    subject_digest = hashlib.sha256()
    attestations: dict[str, object] = {}
    max_abs_diff = 0.0
    worst: dict[str, object] | None = None
    first_mismatch: dict[str, int] | None = None
    comparable_steps = 0
    vocab_size = 0
    payload: dict[str, object] = {}

    def _session(binary: Path, log_path: Path):
        return AttestedServerSession(
            binary=binary,
            model=model,
            expected=expected,
            extra_args=server_args,
            log_path=log_path,
            env_overrides=server_env,
            env_unset=env_unset,
            architecture_by_locator=architecture_by_locator,
        )

    try:
        with _session(server_pair.control_bin, control_log) as control_session, spool_path.open("wb") as spool:
            if control_session.attestation is None:
                raise _fail("control server has no attestation")
            attestations["control"] = control_session.attestation.document()
            vocab_size = _vocab_size(control_session.base_url)
            payload = _request_payload(vocab_size)
            for step, row in enumerate(_stream_completion_rows(control_session.base_url, payload, timeout_s=_TIMEOUT_S)):
                if step >= _N_PREDICT:
                    raise _fail("control emitted more decode steps than requested")
                generated_id, values = _dense_logprobs(row, vocab_size=vocab_size, arm="control", step=step)
                encoded = _canonical_bytes(values)
                spool.write(encoded)
                control_digest.update(encoded)
                control_tokens.append(generated_id)
            if len(control_tokens) != _N_PREDICT:
                raise _fail(f"control emitted {len(control_tokens)} decode steps; expected {_N_PREDICT}")

        with _session(server_pair.subject_bin, subject_log) as subject_session, spool_path.open("rb") as spool:
            if subject_session.attestation is None:
                raise _fail("subject server has no attestation")
            attestations["subject"] = subject_session.attestation.document()
            if _vocab_size(subject_session.base_url) != vocab_size:
                raise _fail("control and subject report different vocabulary sizes")
            for step, row in enumerate(_stream_completion_rows(subject_session.base_url, payload, timeout_s=_TIMEOUT_S)):
                if step >= _N_PREDICT:
                    raise _fail("subject emitted more decode steps than requested")
                generated_id, subject_values = _dense_logprobs(row, vocab_size=vocab_size, arm="subject", step=step)
                subject_digest.update(_canonical_bytes(subject_values))
                subject_tokens.append(generated_id)
                control_values = array("d")
                try:
                    control_values.fromfile(spool, vocab_size)
                except EOFError as exc:
                    raise _fail("control logprob spool ended early") from exc
                if sys.byteorder != "little":
                    control_values.byteswap()
                if first_mismatch is None and control_tokens[step] != generated_id:
                    first_mismatch = {
                        "step": step,
                        "control_token_id": control_tokens[step],
                        "subject_token_id": generated_id,
                    }
                # Step k is still comparable when token k is the first mismatch
                # (same input context); later steps are diagnostic only.
                if first_mismatch is None or first_mismatch["step"] == step:
                    comparable_steps += 1
                    for token_id, (c, s) in enumerate(zip(control_values, subject_values, strict=True)):
                        delta = abs(s - c)
                        if delta > max_abs_diff:
                            max_abs_diff = delta
                            worst = {"step": step, "token_id": token_id, "control": c, "subject": s, "abs_diff": delta}
            if len(subject_tokens) != _N_PREDICT:
                raise _fail(f"subject emitted {len(subject_tokens)} decode steps; expected {_N_PREDICT}")
            if spool.read(1):
                raise _fail("control logprob spool has trailing data")
    finally:
        spool_path.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            scratch_dir.rmdir()

    tokens_match = first_mismatch is None
    passed = tokens_match and max_abs_diff <= _TOLERANCE
    compared = comparable_steps * vocab_size
    if first_mismatch is not None:
        detail = f"generated tokens diverged at step {first_mismatch['step']} after comparing {compared} logprobs"
    else:
        relation = "<=" if passed else ">"
        detail = (
            f"{_N_PREDICT} decode steps under GGML_CUDA_GRAPH_OPT=1, {compared} full-vocabulary logprobs; "
            f"max_abs_logprob_diff={max_abs_diff:.9g} {relation} tolerance={_TOLERANCE:.9g}"
        )
    backend_reference = experiment_contract.CorrectnessResult(check="backend_reference", passed=passed, detail=detail)

    marker = re.compile(_MARKER_REGEX)
    subject_text = subject_log.read_text(encoding="utf-8", errors="replace")
    control_text = control_log.read_text(encoding="utf-8", errors="replace")
    subject_hit = marker.search(subject_text) is not None
    control_hit = marker.search(control_text) is not None
    trigger_hit = subject_hit and not control_hit
    activation = ActivationEvidence(
        status="executed" if trigger_hit else ("unobservable" if subject_hit else "not_executed"),
        mechanism="trace_marker",
        detail=f"marker {_MARKER_REGEX!r} subject_hit={subject_hit} control_hit={control_hit}",
    )
    subject_trace_ref = ctx.runtime.write_text_artifact(name=_SUBJECT_TRACE_ARTIFACT_NAME, text=subject_text)
    control_trace_ref = ctx.runtime.write_text_artifact(name=_CONTROL_TRACE_ARTIFACT_NAME, text=control_text)

    controls_outcome = ctx.runtime.run_paired_llama_benchmark(
        control_binary=bench_pair.control_bin,
        subject_binary=bench_pair.subject_bin,
        model=model,
        workloads=("decode",),
        pairs=_MIN_PAIRED_ROUNDS,
        log_context="rd43-controls",
        device=device,
        env_overrides=_GRAPH_OPT_ENV,
    )
    if set(controls_outcome.runs) != {"decode"}:
        raise _fail(f"controls must produce exactly one decode lane; got {sorted(controls_outcome.runs)!r}")
    decode_run = controls_outcome.runs["decode"]
    if dict(decode_run.stats).get("paired_rounds") != _MIN_PAIRED_ROUNDS:
        raise _fail(f"controls decode lane has {decode_run.stats.get('paired_rounds')!r} paired rounds")
    control_effect = experiment_execution.lane_effect_from_run("control", "tg128", decode_run)

    controls_ref = ctx.runtime.write_artifact(
        name=_PERFORMANCE_ARTIFACT_NAME,
        payload={
            "passed": True,
            "schema_version": 1,
            "contract_id": _CONTRACT_ID,
            "architecture": architecture,
            "model_identity": model_identity,
            "env": _GRAPH_OPT_ENV,
            "build_identities": {r: dict(i) for r, i in bench_pair.validation_build_identities.items()},
            "control": {
                "metric": "tg128",
                "effect": dataclasses.asdict(control_effect),
                "runs": list(decode_run.runs),
                "stats": dict(decode_run.stats),
            },
        },
    )
    ctx.runtime.write_artifact(
        name=_ARTIFACT_NAME,
        payload={
            "schema_version": 1,
            "check": "backend_reference",
            "passed": passed,
            "detail": detail,
            "model_identity": model_identity,
            "env": _GRAPH_OPT_ENV,
            "request": payload,
            "comparison": {
                "method": "llama-server-streaming-full-vocab-logprob",
                "vocab_size": vocab_size,
                "decode_steps_compared": comparable_steps,
                "logprobs_compared": compared,
                "tolerance": _TOLERANCE,
                "generated_tokens_match": tokens_match,
                "first_generated_token_mismatch": first_mismatch,
                "max_abs_logprob_diff": max_abs_diff,
                "worst": worst,
                "control_logprobs_sha256": control_digest.hexdigest(),
                "subject_logprobs_sha256": subject_digest.hexdigest(),
            },
            "execution_attestation": attestations,
            "subject_source_tree": psi.git_worktree_tree(server_pair.subject_source),
            "control_source_tree": psi.git_worktree_tree(server_pair.control_source),
            "subject_build_identity": server_pair.validation_build_identities["subject"],
            "control_build_identity": server_pair.validation_build_identities["control"],
        },
    )
    trigger_evidence = experiment_execution.trigger_evidence_from_marker_probe(
        lane_id="rd43-server-subject", role="positive", positive_hit=trigger_hit
    )

    return vp.ProducerResult(
        correctness={
            "disposition": "passed" if passed else "failed",
            "mechanism": "rd43-full-vocab-backend-reference",
            "detail": detail,
        },
        validation_build_identities=bench_pair.validation_build_identities,
        activation_evidence=activation,
        performance_evidence={"artifact": {"path": controls_ref.path, "sha256": controls_ref.sha256}},
        trace_evidence={
            "positive": {"artifact": {"path": subject_trace_ref.path, "sha256": subject_trace_ref.sha256}},
            "negative": {"artifact": {"path": control_trace_ref.path, "sha256": control_trace_ref.sha256}},
        },
        check_results=(),
        lane_effects=(),
        contract_correctness_results=(backend_reference,),
        promotion_lane_effects={_CONTRACT_ID: (control_effect,)},
        promotion_target_metric={_CONTRACT_ID: "tg128"},
        promotion_trigger_evidence={_CONTRACT_ID: (trigger_evidence,)},
        emitted_artifacts=frozenset(
            {_ARTIFACT_NAME, _PERFORMANCE_ARTIFACT_NAME, _SUBJECT_TRACE_ARTIFACT_NAME, _CONTROL_TRACE_ARTIFACT_NAME}
        ),
    )

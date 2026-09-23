"""PA36 migration #3 (dev-gpt-agent req_110d0729beb44d8b): patch-local
validation producer for 1206 (RD13-MUL-MAT-ADD-VIEW-FUSION).

Mechanically migrated off validation_campaign.py's
run_rd13_backend_reference_check() -- the contract-grade
backend_reference producer (a fixed-prompt, temperature-0,
full-vocabulary /completion logprob comparison between the control
server (RD13 absent) and the subject server (1206 applied)). The
legacy helper trio (_rd13_stream_completion_rows /
_rd13_dense_logprobs / _rd13_canonical_bytes), the
_RD13_BACKEND_REFERENCE_* constants, and the --run-rd13-contract CLI
path are DELETED from shared code in the same change (no
compatibility layer, per the project's migrate-up doctrine). The dead
legacy run_rd13_ppl_check() PPL wrapper and this patch's
rd13_correctness.py module (its only home) are deleted too -- zero
live call sites.

Design rulings applied (req_110d0729beb44d8b):

- ONE named contract-correctness result: ``backend_reference`` only
  (the RD13 contract requires exactly that one check -- no
  ppl_equality, unlike RD04's dual derivation from one PPL run).
- The isolated llama-server pair is built ONCE as the fat multi-arch
  set via ``ctx.runtime.build_pair()`` (the PA36 build-once authority)
  and run per-device; ``require_parity=True`` demands
  assert_validation_subject_parity() on the pair. Build identities
  legitimately differ from the legacy per-arch builds -- the
  historical records are the semantic equivalence target, not the
  build-identity bytes.
- The producer owns the backend-reference correctness measurement and
  the contract performance/controls lanes. Correctness uses its isolated
  llama-server pair; performance and trigger evidence use the standard
  scaffold llama-bench pair. The producer returns real activation and
  trace evidence from its own marker probes, with the server environment
  still sanitizing GGML_CUDA_DISABLE_FUSION.
- Scaffold control/subject build identities are canonical because they
  produce the promotion measurements. Server-pair identities remain local
  to the backend-reference artifact. The required tierM control model is
  content-bound to the registry before any benchmark subprocess starts;
  threshold aggregation remains dispatcher-owned.
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
from typing import Protocol

from bigcherry.patch import validation_producer as vp  # type: ignore[import-not-found]

# One contract architecture per run (the historical RD13 rule): the
# operator names it via --amdgpu-targets; the binary itself is built
# ONCE as the production-matching fat multi-arch set and the real
# device is selected at run time.
_CONTRACT_ARCHITECTURES: tuple[str, ...] = ("gfx1100", "gfx1201", "gfx1030")

_CONTRACT_ID = "RD13-MUL-MAT-ADD-VIEW-FUSION"
_SUBJECT_PATCH = "1206_rd13_mul_mat_add_view_fusion"

# The contract's positive model reference (diagnostic metadata in the
# artifact; the REAL model file is ctx.model and must be the registered
# tierA-qwen4b-q6k for contract-grade evidence).
_MODEL_REF = "tierA-qwen4b-q6k"
_VOCAB_SIZE = 248_320
_TOLERANCE = 0.0005
_N_PREDICT = 64
_PROMPT = (
    "Explain in one concise sentence why a recurrent state update can be combined "
    "with a residual connection while preserving the model's output."
)
_TIMEOUT_S = 900

_ARTIFACT_NAME = "rd13-backend-reference.json"
_PERFORMANCE_ARTIFACT_NAME = "rd13-performance.json"
_SUBJECT_TRACE_ARTIFACT_NAME = "rd13-subject-trace.log"
_CONTROL_TRACE_ARTIFACT_NAME = "rd13-control-trace.log"
_MARKER_REGEX = (
    "BIGCHERRY_PATCH_HIT patch=1206_rd13 path=mul_mat_add_view_fusion_(?:f|q)"
)
_CONTROL_MODEL_REF = "tierM-gptoss20b-q6k"
_MIN_PAIRED_ROUNDS = 10


def _content_identity(
    path: Path, *, model_id: str, label: str = "control model"
) -> dict[str, object]:
    """Verify and content-bind a fixed contract model registry entry."""
    from bigcherry.core import paths as bc_paths  # type: ignore[import-not-found]

    try:
        registry = tomllib.loads(bc_paths.MODELS.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise vp.ValidationProducerError(
            f"rd13 {label}: cannot read {bc_paths.MODELS}: {exc}"
        ) from exc
    entry = next(
        (
            item
            for item in registry.get("models", [])
            if isinstance(item, dict) and item.get("id") == model_id
        ),
        None,
    )
    if not isinstance(entry, dict):
        raise vp.ValidationProducerError(
            f"rd13 {label}: registry has no {model_id!r} entry"
        )
    expected_name = Path(str(entry.get("path", ""))).name
    expected_size = entry.get("size-bytes")
    if not path.is_file():
        raise vp.ValidationProducerError(f"rd13 {label}: file does not exist: {path}")
    if path.name != expected_name:
        raise vp.ValidationProducerError(
            f"rd13 {label}: {path.name!r} does not match registry basename "
            f"{expected_name!r} for {model_id}"
        )
    if (
        not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or path.stat().st_size != expected_size
    ):
        raise vp.ValidationProducerError(
            f"rd13 {label}: {path} size {path.stat().st_size} does not match "
            f"registry size {expected_size!r} for {model_id}"
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "model_id": model_id,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "registry_path": str(entry.get("path", "")),
    }


def _scaffold_binary(
    ctx: vp.ProducerContext, *, role: str, target: str = "llama-bench"
) -> Path:
    binaries = ctx.validation_binaries.get(role)
    binary = binaries.get(target) if isinstance(binaries, Mapping) else None
    if not isinstance(binary, Path) or not binary.is_file():
        raise vp.ValidationProducerError(
            f"rd13 performance: standard scaffold {role} {target} binary is missing"
        )
    return binary


def _request_payload() -> dict[str, object]:
    """The fixed, pre-sampling-disabled full-vocabulary request both
    arms receive (legacy constants, verbatim)."""
    return {
        "prompt": _PROMPT,
        "n_predict": _N_PREDICT,
        "n_probs": _VOCAB_SIZE,
        "post_sampling_probs": False,
        "temperature": 0.0,
        "seed": 42,
        "cache_prompt": False,
        "ignore_eos": True,
        "return_tokens": True,
        "stream": True,
    }


class _ProbabilitySession(Protocol):
    """Structural view of ``AttestedServerSession``: the SSE completion
    consumer needs only the base URL (the real session type is imported
    at call time so tests can patch the source module)."""

    base_url: str


def _stream_completion_rows(
    session: _ProbabilitySession,
    payload: Mapping[str, object],
    *,
    timeout_s: int,
) -> Iterable[Mapping[str, object]]:
    """Yield one native llama-server probability row per generated token.

    The pinned server emits one ``completion_probabilities`` entry per SSE
    event when ``stream=true``.  Streaming is intentional: a 64-token,
    full-vocabulary Qwen response contains ~16M probability entries, so the
    non-streaming ``post_json()`` path would retain the whole JSON document
    in memory.
    """
    body = json.dumps(dict(payload)).encode("utf-8")
    request = urllib.request.Request(
        f"{session.base_url}/completion",
        data=body,
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
                    raise vp.ValidationProducerError(
                        f"rd13 backend_reference: malformed SSE line: {line[:120]!r}"
                    )
                encoded = line.removeprefix("data:").strip()
                if encoded == "[DONE]":
                    saw_stop = True
                    break
                try:
                    event = json.loads(encoded)
                except json.JSONDecodeError as exc:
                    raise vp.ValidationProducerError(
                        "rd13 backend_reference: malformed JSON in completion stream"
                    ) from exc
                if not isinstance(event, Mapping):
                    raise vp.ValidationProducerError(
                        "rd13 backend_reference: completion stream event is not an object"
                    )
                stop = event.get("stop")
                if not isinstance(stop, bool):
                    raise vp.ValidationProducerError(
                        "rd13 backend_reference: completion stream event lacks boolean stop"
                    )
                if stop:
                    saw_stop = True
                    continue
                rows = event.get("completion_probabilities")
                if (
                    not isinstance(rows, list)
                    or len(rows) != 1
                    or not isinstance(rows[0], Mapping)
                ):
                    raise vp.ValidationProducerError(
                        "rd13 backend_reference: expected exactly one completion_probabilities row per stream event"
                    )
                yield rows[0]
    except (urllib.error.URLError, OSError, TimeoutError, UnicodeError) as exc:
        raise vp.ValidationProducerError(
            f"rd13 backend_reference: streaming /completion failed: {exc}"
        ) from exc
    if not saw_stop:
        raise vp.ValidationProducerError(
            "rd13 backend_reference: completion stream ended without a stop event"
        )


def _dense_logprobs(
    row: Mapping[str, object],
    *,
    vocab_size: int,
    arm: str,
    step: int,
) -> tuple[int, array]:
    """Validate one full-vocabulary row and return token-id-indexed logprobs."""
    generated_id = row.get("id")
    if not isinstance(generated_id, int) or isinstance(generated_id, bool):
        raise vp.ValidationProducerError(
            f"rd13 backend_reference: {arm} step {step} has invalid generated token id"
        )
    if generated_id < 0 or generated_id >= vocab_size:
        raise vp.ValidationProducerError(
            f"rd13 backend_reference: {arm} step {step} generated token id {generated_id} "
            f"outside vocabulary [0,{vocab_size})"
        )

    top = row.get("top_logprobs")
    if not isinstance(top, list) or len(top) != vocab_size:
        actual = len(top) if isinstance(top, list) else None
        raise vp.ValidationProducerError(
            f"rd13 backend_reference: {arm} step {step} is not full-vocabulary -- "
            f"expected {vocab_size} top_logprobs, got {actual!r}"
        )

    values = array("d", [math.nan]) * vocab_size
    seen = bytearray(vocab_size)
    for entry in top:
        if not isinstance(entry, Mapping):
            raise vp.ValidationProducerError(
                f"rd13 backend_reference: {arm} step {step} has a non-object top_logprobs entry"
            )
        token_id = entry.get("id")
        logprob = entry.get("logprob")
        if (
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or token_id < 0
            or token_id >= vocab_size
        ):
            raise vp.ValidationProducerError(
                f"rd13 backend_reference: {arm} step {step} has invalid vocabulary token id {token_id!r}"
            )
        if seen[token_id]:
            raise vp.ValidationProducerError(
                f"rd13 backend_reference: {arm} step {step} duplicates vocabulary token id {token_id}"
            )
        if not isinstance(logprob, (int, float)) or isinstance(logprob, bool):
            raise vp.ValidationProducerError(
                f"rd13 backend_reference: {arm} step {step} token {token_id} has invalid logprob"
            )
        value = float(logprob)
        if not math.isfinite(value):
            raise vp.ValidationProducerError(
                f"rd13 backend_reference: {arm} step {step} token {token_id} has non-finite logprob"
            )
        seen[token_id] = 1
        values[token_id] = value

    # len(top)==vocab_size + unique in-range ids proves complete coverage.
    return generated_id, values


def _canonical_bytes(values: array) -> bytes:
    """Canonical little-endian f64 encoding for evidence digests/spooling."""
    if values.typecode != "d" or values.itemsize != 8:
        raise vp.ValidationProducerError(
            "rd13 backend_reference: platform does not expose 64-bit array('d')"
        )
    if sys.byteorder == "little":
        return values.tobytes()
    copied = array("d", values)
    copied.byteswap()
    return copied.tobytes()


def _lane_effect(
    outcome: vp.ProducerPairedBenchmarkOutcome,
    *,
    role: str,
    label: str,
):
    experiment_execution = importlib.import_module("bigcherry.experiment.execution")

    if set(outcome.runs) != {"decode"}:
        raise vp.ValidationProducerError(
            f"rd13 performance: {label} must produce exactly one decode lane; "
            f"got {sorted(outcome.runs)!r}"
        )
    run = outcome.runs["decode"]
    stats = getattr(run, "stats", None)
    if not isinstance(stats, Mapping):
        raise vp.ValidationProducerError(
            f"rd13 performance: {label} decode lane has no statistics"
        )
    if stats.get("paired_rounds") != _MIN_PAIRED_ROUNDS:
        raise vp.ValidationProducerError(
            f"rd13 performance: {label} decode lane has "
            f"{stats.get('paired_rounds')!r} usable paired rounds; "
            f"expected {_MIN_PAIRED_ROUNDS}"
        )
    try:
        return experiment_execution.lane_effect_from_run(role, "tg128", run)
    except (KeyError, TypeError, ValueError) as exc:
        raise vp.ValidationProducerError(
            f"rd13 performance: {label} decode lane statistics are incomplete"
        ) from exc


def run(ctx: vp.ProducerContext) -> vp.ProducerResult:
    experiment_contract = importlib.import_module("bigcherry.experiment.contract")
    experiment_execution = importlib.import_module("bigcherry.experiment.execution")
    ExecutionIdentity = importlib.import_module(
        "bigcherry.experiment.attestation"
    ).ExecutionIdentity
    AttestedServerSession = importlib.import_module(
        "bigcherry.experiment.server_execution"
    ).AttestedServerSession
    psi = importlib.import_module("bigcherry.patch.source")
    ActivationEvidence = importlib.import_module(
        "bigcherry.patch.activation"
    ).ActivationEvidence

    if (
        len(ctx.fat_targets.targets) != 1
        or ctx.fat_targets.targets[0] not in _CONTRACT_ARCHITECTURES
    ):
        raise vp.ValidationProducerError(
            f"rd13 backend_reference: {_CONTRACT_ID} requires exactly one "
            "contract architecture per run (gfx1100, gfx1201, or gfx1030); "
            f"got targets={ctx.fat_targets.targets!r}"
        )
    architecture = ctx.fat_targets.targets[0]

    # The full-vocabulary backend reference is a whole-MODEL comparison;
    # it needs a real model file and no PPL corpus. Fail closed before
    # any build or subprocess.
    if ctx.model is None:
        raise vp.ValidationProducerError(
            "rd13 backend_reference requires a real whole model (--model); "
            "refusing to run without one"
        )
    model = ctx.model
    positive_model_identity = _content_identity(
        model, model_id=_MODEL_REF, label="positive model"
    )
    control_model_raw = ctx.inputs.get("control_model")
    if not isinstance(control_model_raw, str) or not control_model_raw:
        raise vp.ValidationProducerError(
            "rd13 performance requires --producer-input control_model=<path>"
        )
    control_model = Path(control_model_raw)
    control_model_identity = _content_identity(
        control_model, model_id=_CONTROL_MODEL_REF
    )

    # The performance and controls lanes are deliberately run with the
    # standard scaffold pair. The server pair below remains correctness-only.
    bench_control = _scaffold_binary(ctx, role="control")
    bench_subject = _scaffold_binary(ctx, role="subject")
    scaffold_identities = ctx.validation_build_identities
    if set(scaffold_identities) != {"control", "subject"}:
        raise vp.ValidationProducerError(
            "rd13 performance requires scaffold control/subject build identities"
        )

    # The one sanctioned pair authority: control = baseline composition;
    # subject = baseline + focal (1206). Built ONCE as the fat multi-arch
    # set (the PA36 build-once rule), parity asserted, run per-device.
    pair = ctx.runtime.build_pair(
        targets=_CONTRACT_ARCHITECTURES,
        primary_target="llama-server",
        baseline_source="bigcherry",
        require_parity=True,
    )
    control_binary = pair.control_bin
    subject_binary = pair.subject_bin

    devices = ctx.runtime.device_contexts(device_map=ctx.device_map)
    matches = [d for d in devices if d.architecture == architecture]
    if len(matches) != 1:
        raise vp.ValidationProducerError(
            f"rd13 backend_reference: expected exactly one device mapped for "
            f"run architecture {architecture!r}, got {len(matches)}; "
            f"device_map={ {k: tuple(v) for k, v in ctx.device_map.items()}!r} "
            "-- --device-map must select one real device for it"
        )
    device = matches[0]

    positive_outcome = ctx.runtime.run_paired_llama_benchmark(
        control_binary=bench_control,
        subject_binary=bench_subject,
        model=model,
        workloads=("decode",),
        pairs=_MIN_PAIRED_ROUNDS,
        log_context="rd13-performance-positive",
        device=device,
    )
    control_outcome = ctx.runtime.run_paired_llama_benchmark(
        control_binary=bench_control,
        subject_binary=bench_subject,
        model=control_model,
        workloads=("decode",),
        pairs=_MIN_PAIRED_ROUNDS,
        log_context="rd13-performance-control",
        device=device,
    )
    positive_effect = _lane_effect(positive_outcome, role="positive", label="positive")
    control_effect = _lane_effect(control_outcome, role="control", label="control")

    subject_trace = ctx.runtime.run_trace_probe(
        binary=bench_subject,
        model=model,
        device=device,
        bench_prompt=512,
        bench_gen=128,
        log_context="rd13-trigger-subject",
        disable_fusion=False,
    )
    control_trace = ctx.runtime.run_trace_probe(
        binary=bench_control,
        model=model,
        device=device,
        bench_prompt=512,
        bench_gen=128,
        log_context="rd13-trigger-control",
        disable_fusion=False,
    )
    marker = re.compile(_MARKER_REGEX)
    subject_hit = marker.search(subject_trace) is not None
    control_hit = marker.search(control_trace) is not None
    trigger_hit = subject_hit and not control_hit
    activation = ActivationEvidence(
        status="executed"
        if trigger_hit
        else ("unobservable" if subject_hit else "not_executed"),
        mechanism="trace_marker",
        detail=(
            f"marker {_MARKER_REGEX!r} subject_hit={subject_hit} "
            f"control_hit={control_hit}"
        ),
    )
    subject_trace_ref = ctx.runtime.write_text_artifact(
        name=_SUBJECT_TRACE_ARTIFACT_NAME, text=subject_trace
    )
    control_trace_ref = ctx.runtime.write_text_artifact(
        name=_CONTROL_TRACE_ARTIFACT_NAME, text=control_trace
    )

    # PRBE111 (preserved) + PA36 RD13/1206 migration (GPT req_760c0fe82d7b4609
    # BLOCKER): the llama-server attestation channel derives the server
    # architecture ONLY via a verified locator->arch mapping (the
    # llama-bench/perplexity banner carries no PCI locator, so
    # device_contexts() omits locators from execution_identity). RD13
    # runs llama-server, so construct a locator-bearing ExecutionIdentity
    # from the verified physical locator and the {locator: architecture}
    # mapping. When the device carries no verified locator (test/mocked),
    # leave the mapping None and let attestation fail closed exactly as
    # before (never guess a mapping).
    if device.locator is not None:
        expected = ExecutionIdentity(
            backend="ROCm",
            architectures=(architecture,),
            locators=(device.locator,),
        )
        architecture_by_locator = {device.locator: architecture}
    else:
        expected = device.execution_identity
        architecture_by_locator = None

    # Sanctioned HIP-only device selector env (design ruling Q5): merge
    # the build env with the device overrides, pop env_unset LAST, and
    # strip GGML_CUDA_DISABLE_FUSION -- a stale ambient fusion-disable
    # would make the RD13 subject inert and cannot masquerade as
    # correctness evidence.
    server_env = dict(ctx.build_env)
    server_env.update(dict(device.env_overrides))
    for key in device.env_unset:
        server_env.pop(key, None)
    server_env.pop("GGML_CUDA_DISABLE_FUSION", None)
    env_unset = (*device.env_unset, "GGML_CUDA_DISABLE_FUSION")

    server_args = ("-ngl", "99", "-c", "1024", "--parallel", "1")
    payload = _request_payload()

    logs_dir = ctx.workdir / "logs"
    scratch_dir = ctx.workdir / "scratch"
    logs_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    spool_path = scratch_dir / "rd13-control-full-vocab-logprobs.f64"

    control_tokens: list[int] = []
    subject_tokens: list[int] = []
    control_digest = hashlib.sha256()
    subject_digest = hashlib.sha256()
    attestations: dict[str, object] = {}
    max_abs_diff = 0.0
    worst: dict[str, object] | None = None
    first_token_mismatch: dict[str, int] | None = None
    comparable_steps = 0

    try:
        with (
            AttestedServerSession(
                binary=control_binary,
                model=model,
                expected=expected,
                extra_args=server_args,
                log_path=logs_dir / "rd13-backend-reference-control-server.log",
                env_overrides=server_env,
                env_unset=env_unset,
                architecture_by_locator=architecture_by_locator,
            ) as control_session,
            spool_path.open("wb") as spool,
        ):
            if control_session.attestation is None:
                raise vp.ValidationProducerError(
                    "rd13 backend_reference: control server has no attestation"
                )
            attestations["control"] = control_session.attestation.document()
            for step, row in enumerate(
                _stream_completion_rows(control_session, payload, timeout_s=_TIMEOUT_S)
            ):
                if step >= _N_PREDICT:
                    raise vp.ValidationProducerError(
                        "rd13 backend_reference: control emitted more decode steps than requested"
                    )
                generated_id, values = _dense_logprobs(
                    row,
                    vocab_size=_VOCAB_SIZE,
                    arm="control",
                    step=step,
                )
                encoded = _canonical_bytes(values)
                spool.write(encoded)
                control_digest.update(encoded)
                control_tokens.append(generated_id)
            if len(control_tokens) != _N_PREDICT:
                raise vp.ValidationProducerError(
                    f"rd13 backend_reference: control emitted {len(control_tokens)} decode steps; "
                    f"expected {_N_PREDICT}"
                )

        with (
            AttestedServerSession(
                binary=subject_binary,
                model=model,
                expected=expected,
                extra_args=server_args,
                log_path=logs_dir / "rd13-backend-reference-subject-server.log",
                env_overrides=server_env,
                env_unset=env_unset,
                architecture_by_locator=architecture_by_locator,
            ) as subject_session,
            spool_path.open("rb") as spool,
        ):
            if subject_session.attestation is None:
                raise vp.ValidationProducerError(
                    "rd13 backend_reference: subject server has no attestation"
                )
            attestations["subject"] = subject_session.attestation.document()
            for step, row in enumerate(
                _stream_completion_rows(subject_session, payload, timeout_s=_TIMEOUT_S)
            ):
                if step >= _N_PREDICT:
                    raise vp.ValidationProducerError(
                        "rd13 backend_reference: subject emitted more decode steps than requested"
                    )
                generated_id, subject_values = _dense_logprobs(
                    row,
                    vocab_size=_VOCAB_SIZE,
                    arm="subject",
                    step=step,
                )
                encoded = _canonical_bytes(subject_values)
                subject_digest.update(encoded)
                subject_tokens.append(generated_id)

                control_values = array("d")
                try:
                    control_values.fromfile(spool, _VOCAB_SIZE)
                except EOFError as exc:
                    raise vp.ValidationProducerError(
                        "rd13 backend_reference: compact control spool ended early"
                    ) from exc
                if sys.byteorder != "little":
                    control_values.byteswap()

                control_generated = control_tokens[step]
                if first_token_mismatch is None and control_generated != generated_id:
                    first_token_mismatch = {
                        "step": step,
                        "control_token_id": control_generated,
                        "subject_token_id": generated_id,
                    }

                # Step k's distribution is still comparable when token k is
                # the first mismatch (its input context was identical).
                # Once a generated token diverges, later contexts are not
                # the same experiment and their numeric deltas are
                # diagnostic only.
                if first_token_mismatch is None or first_token_mismatch["step"] == step:
                    comparable_steps += 1
                    for token_id, (control_value, subject_value) in enumerate(
                        zip(control_values, subject_values, strict=True)
                    ):
                        delta = abs(subject_value - control_value)
                        if delta > max_abs_diff:
                            max_abs_diff = delta
                            worst = {
                                "step": step,
                                "token_id": token_id,
                                "control_logprob": control_value,
                                "subject_logprob": subject_value,
                                "abs_diff": delta,
                            }
            if len(subject_tokens) != _N_PREDICT:
                raise vp.ValidationProducerError(
                    f"rd13 backend_reference: subject emitted {len(subject_tokens)} decode steps; "
                    f"expected {_N_PREDICT}"
                )
            if spool.read(1):
                raise vp.ValidationProducerError(
                    "rd13 backend_reference: compact control spool has trailing data"
                )
    finally:
        spool_path.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            scratch_dir.rmdir()

    generated_tokens_match = first_token_mismatch is None
    passed = generated_tokens_match and max_abs_diff <= _TOLERANCE
    compared = comparable_steps * _VOCAB_SIZE
    if first_token_mismatch is not None:
        detail = (
            "generated token sequence diverged at step "
            f"{first_token_mismatch['step']} after comparing {compared} full-vocabulary logprobs"
        )
    else:
        relation = "<=" if passed else ">"
        detail = (
            f"{_N_PREDICT} decode steps, {compared} full-vocabulary logprobs; "
            f"max_abs_logprob_diff={max_abs_diff:.9g} {relation} tolerance={_TOLERANCE:.9g}"
        )
    backend_reference_result = experiment_contract.CorrectnessResult(
        check="backend_reference",
        passed=passed,
        detail=detail,
    )

    comparison = {
        "method": "llama-server-streaming-full-vocab-logprob",
        "contract_model_ref": _MODEL_REF,
        "vocab_size": _VOCAB_SIZE,
        "decode_steps_requested": _N_PREDICT,
        "decode_steps_compared": comparable_steps,
        "logprobs_compared": compared,
        "tolerance": _TOLERANCE,
        "generated_tokens_match": generated_tokens_match,
        "first_generated_token_mismatch": first_token_mismatch,
        "max_abs_logprob_diff": max_abs_diff,
        "worst": worst,
        "control_logprobs_sha256": control_digest.hexdigest(),
        "subject_logprobs_sha256": subject_digest.hexdigest(),
    }
    performance_doc = {
        # This is evidence completeness, not the promotion verdict. The
        # dispatcher evaluates the typed lane effects and owns thresholds.
        "passed": True,
        "schema_version": 1,
        "contract_id": _CONTRACT_ID,
        "architecture": architecture,
        "positive_model": str(model),
        "positive_model_identity": positive_model_identity,
        "control_model": str(control_model),
        "control_model_identity": control_model_identity,
        "benchmark_build_identities": {
            role: dict(identity) for role, identity in scaffold_identities.items()
        },
        "positive": {
            "metric": "tg128",
            "effect": dataclasses.asdict(positive_effect),
            "commands": {
                key: {inner: list(argv) for inner, argv in values.items()}
                for key, values in positive_outcome.commands.items()
            },
            "runs": list(positive_outcome.runs["decode"].runs),
            "stats": dict(positive_outcome.runs["decode"].stats),
        },
        "control": {
            "metric": "tg128",
            "effect": dataclasses.asdict(control_effect),
            "commands": {
                key: {inner: list(argv) for inner, argv in values.items()}
                for key, values in control_outcome.commands.items()
            },
            "runs": list(control_outcome.runs["decode"].runs),
            "stats": dict(control_outcome.runs["decode"].stats),
        },
        "metrics": {
            "positive_decode": dataclasses.asdict(positive_effect),
            "control_decode": dataclasses.asdict(control_effect),
        },
        "trigger": {"subject_hit": subject_hit, "control_hit": control_hit},
    }
    performance_ref = ctx.runtime.write_artifact(
        name=_PERFORMANCE_ARTIFACT_NAME, payload=performance_doc
    )

    doc = {
        "schema_version": 1,
        "check": backend_reference_result.check,
        "passed": backend_reference_result.passed,
        "detail": backend_reference_result.detail,
        "model": str(model),
        "request": payload,
        "comparison": comparison,
        "execution_attestation": attestations,
        "subject_source_tree": psi.git_worktree_tree(pair.subject_source),
        "control_source_tree": psi.git_worktree_tree(pair.control_source),
        "subject_build_identity": pair.validation_build_identities["subject"],
        "control_build_identity": pair.validation_build_identities["control"],
    }
    ctx.runtime.write_artifact(name=_ARTIFACT_NAME, payload=doc)
    trigger_evidence = experiment_execution.trigger_evidence_from_marker_probe(
        lane_id="rd13-decode-subject",
        role="positive",
        positive_hit=trigger_hit,
    )

    # Semantic correctness for the shared binder: exactly the legacy
    # correctness.json mechanism (the historical records' mechanism
    # string is preserved verbatim). The server pair identities remain
    # local to that artifact; the canonical ProducerResult identity is
    # the standard scaffold llama-bench pair used for promotion lanes.
    return vp.ProducerResult(
        correctness={
            "disposition": "passed" if passed else "failed",
            "mechanism": "rd13-full-vocab-backend-reference",
            "detail": detail,
        },
        validation_build_identities=ctx.validation_build_identities,
        activation_evidence=activation,
        performance_evidence={
            "artifact": {
                "path": performance_ref.path,
                "sha256": performance_ref.sha256,
            }
        },
        trace_evidence={
            "positive": {
                "artifact": {
                    "path": subject_trace_ref.path,
                    "sha256": subject_trace_ref.sha256,
                }
            },
            "negative": {
                "artifact": {
                    "path": control_trace_ref.path,
                    "sha256": control_trace_ref.sha256,
                }
            },
        },
        check_results=(),
        lane_effects=(),
        contract_correctness_results=(backend_reference_result,),
        promotion_lane_effects={_CONTRACT_ID: (positive_effect, control_effect)},
        promotion_target_metric={_CONTRACT_ID: "tg128"},
        promotion_trigger_evidence={_CONTRACT_ID: (trigger_evidence,)},
        emitted_artifacts=frozenset(
            {
                _ARTIFACT_NAME,
                _PERFORMANCE_ARTIFACT_NAME,
                _SUBJECT_TRACE_ARTIFACT_NAME,
                _CONTROL_TRACE_ARTIFACT_NAME,
            }
        ),
    )

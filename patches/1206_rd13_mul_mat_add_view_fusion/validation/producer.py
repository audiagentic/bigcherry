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
- The producer owns ONLY the measurement. Activation is the
  SCAFFOLD's generic two-probe trace-marker probe (trace_probe='run'
  in producer.toml): RD13's marker (validation.toml's marker-regex)
  is a real graph-fusion marker and its GGML_CUDA_DISABLE_FUSION
  negative control is valid -- the producer returns no activation or
  trace evidence. The producer's own server env sanitizes
  GGML_CUDA_DISABLE_FUSION out so a stale ambient fusion-disable
  cannot make the subject inert (legacy behavior, preserved).
- performance_evidence stays None: RD13's historical performance/
  controls dispositions are ERROR and remain so (this migration does
  not add benchmark semantics). Every canonical identity field is
  owned by the shared binder, never here.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import sys
import urllib.error
import urllib.request
from array import array
from collections.abc import Iterable, Mapping
from typing import Protocol

from bigcherry.patch import validation_producer as vp

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
                if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], Mapping):
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
    row: Mapping[str, object], *, vocab_size: int, arm: str, step: int,
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


def run(ctx: vp.ProducerContext) -> vp.ProducerResult:
    from bigcherry.experiment import contract as experiment_contract
    from bigcherry.experiment.server_execution import AttestedServerSession
    from bigcherry.patch import source as psi

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

    # PRBE111 (preserved): only build the locator->arch mapping when the
    # device identity actually carries real locators; otherwise leave it
    # None and let attestation fail closed exactly as before (never
    # guess a mapping).
    expected = device.execution_identity
    architecture_by_locator = (
        dict(zip(expected.locators, expected.architectures, strict=True))
        if expected.locators is not None else None
    )

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
        with AttestedServerSession(
            binary=control_binary,
            model=model,
            expected=expected,
            extra_args=server_args,
            log_path=logs_dir / "rd13-backend-reference-control-server.log",
            env_overrides=server_env,
            env_unset=env_unset,
            architecture_by_locator=architecture_by_locator,
        ) as control_session, spool_path.open("wb") as spool:
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
                    row, vocab_size=_VOCAB_SIZE, arm="control", step=step,
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

        with AttestedServerSession(
            binary=subject_binary,
            model=model,
            expected=expected,
            extra_args=server_args,
            log_path=logs_dir / "rd13-backend-reference-subject-server.log",
            env_overrides=server_env,
            env_unset=env_unset,
            architecture_by_locator=architecture_by_locator,
        ) as subject_session, spool_path.open("rb") as spool:
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
                    row, vocab_size=_VOCAB_SIZE, arm="subject", step=step,
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
        check="backend_reference", passed=passed, detail=detail,
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

    # Semantic correctness for the shared binder: exactly the legacy
    # correctness.json mechanism (the historical records' mechanism
    # string is preserved verbatim). Activation/performance stay
    # untouched: the scaffold probe owns activation; performance/
    # controls remain unsatisfied exactly as the historical records
    # show them.
    return vp.ProducerResult(
        correctness={
            "disposition": "passed" if passed else "failed",
            "mechanism": "rd13-full-vocab-backend-reference",
            "detail": detail,
        },
        validation_build_identities=pair.validation_build_identities,
        activation_evidence=None,
        performance_evidence=None,
        trace_evidence=None,
        check_results=(),
        lane_effects=(),
        contract_correctness_results=(backend_reference_result,),
        emitted_artifacts=frozenset({_ARTIFACT_NAME}),
    )

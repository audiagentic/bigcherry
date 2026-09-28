"""PROMPT-CACHE-CHECKPOINT-SELECTION (1005): patch-local validation producer.

Correctness-gated validation for the shorter-prefix checkpoint-selection
regression. A warm server primes prompt A, then runs prompt C, which shares a
shorter prefix and diverges. Server-side checkpoint logs must prove that C
actually restored an A-era checkpoint whose position/token count is shorter
than the newest checkpoint primed by A. C's greedy completion must exactly
match a fresh --cache-ram 0 reference.

If greedy parity fails, the producer records top-2 token IDs/logprobs at the
first divergent prediction across the original pair plus three fresh
warm/cold repetitions. Greedy parity remains exact; diagnostics do not create
a mismatch tolerance.

The contract also carries auxiliary paired tg128 decode no-regression evidence.
That lane is not the correctness gate and does not exercise checkpoint reuse.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

from bigcherry.experiment import contract as experiment_contract
from bigcherry.experiment import execution as experiment_execution
from bigcherry.patch import producer_support as support
from bigcherry.patch import validation_producer as vp
from bigcherry.patch.activation import ActivationEvidence

_LABEL = "pccs"
_ARCHITECTURE = "gfx1100"
_CONTRACT_ID = "PROMPT-CACHE-CHECKPOINT-SELECTION"
_MODEL_REF = "tierA-qwen4b-q6k"
_ROUNDS = support.contract_paired_rounds(_CONTRACT_ID)

_WARM_ARGS = ("-ngl", "99", "-c", "4096", "--parallel", "1", "--cache-ram", "512", "--fit", "off", "-v")
_COLD_ARGS = ("-ngl", "99", "-c", "4096", "--parallel", "1", "--cache-ram", "0", "--fit", "off")
_N_PREDICT = 64
_N_PROBS = 2
_DIAGNOSTIC_REPEATS = 3

_CHECKPOINT_RE = re.compile(
    r"(?P<action>created|restored) context checkpoint.*?"
    r"pos_min\s*=\s*(?P<pos_min>-?\d+).*?"
    r"pos_max\s*=\s*(?P<pos_max>-?\d+).*?"
    r"n_tokens\s*=\s*(?P<n_tokens>\d+)",
    re.IGNORECASE,
)

_PROMPT_A = (
    "Describe, in careful technical detail, how a paged key-value cache "
    "allows a language model server to reuse computation across requests "
    "that share a common prefix, and explain why hybrid recurrent "
    "architectures require this reuse to be exact rather than approximate."
)
_PROMPT_C = (
    "Describe, in careful technical detail, why prime numbers greater than "
    "two are always odd, and give a short proof."
)


def _fail(message: str) -> vp.ValidationProducerError:
    return vp.ValidationProducerError(f"{_LABEL}: {message}")


def _completion(session, prompt: str) -> dict:
    reply = session.post_json("/completion", {
        "prompt": prompt,
        "n_predict": _N_PREDICT,
        "temperature": 0.0,
        "top_k": 1,
        "seed": 42,
        "cache_prompt": True,
        "ignore_eos": True,
        "return_tokens": True,
        "n_probs": _N_PROBS,
    })
    tokens = reply.get("tokens")
    if not isinstance(tokens, list) or len(tokens) != _N_PREDICT:
        raise _fail("invalid greedy token vector from /completion")
    probabilities = reply.get("completion_probabilities")
    if not isinstance(probabilities, list) or len(probabilities) != _N_PREDICT:
        raise _fail("/completion did not return one probability record per generated token")
    return {"tokens": tokens, "probabilities": probabilities}


def _checkpoint_records(text: str, *, action: str) -> list[dict[str, int]]:
    records: list[dict[str, int]] = []
    for match in _CHECKPOINT_RE.finditer(text):
        if match.group("action").lower() != action:
            continue
        records.append({
            "pos_min": int(match.group("pos_min")),
            "pos_max": int(match.group("pos_max")),
            "n_tokens": int(match.group("n_tokens")),
        })
    return records


def _read_log_since(path: Path, offset: int) -> tuple[str, int]:
    data = path.read_text(encoding="utf-8", errors="replace")
    return data[offset:], len(data)


def _top2(probability_record: object) -> list[dict[str, object]]:
    if not isinstance(probability_record, dict):
        raise _fail("invalid completion probability record")
    candidates = probability_record.get("top_logprobs")
    if not isinstance(candidates, list):
        # Compatibility with the immediately preceding llama-server schema.
        candidates = probability_record.get("probs")
    if not isinstance(candidates, list):
        raise _fail("completion probability record has no top-logprob candidates")
    result: list[dict[str, object]] = []
    for candidate in candidates[:_N_PROBS]:
        if not isinstance(candidate, dict):
            raise _fail("invalid top-logprob candidate")
        token_id = candidate.get("id")
        logprob = candidate.get("logprob")
        # Older schema exposed probability rather than logprob and no token id.
        if logprob is None:
            logprob = candidate.get("prob")
        result.append({
            "id": token_id,
            "logprob": logprob,
            "token": candidate.get("token", candidate.get("tok_str")),
        })
    if len(result) < 2:
        raise _fail("completion probability record did not expose top-2 candidates")
    return result


def _prediction_record(run: dict, step: int) -> dict[str, object]:
    return {
        "selected_token_id": run["tokens"][step],
        "top2": _top2(run["probabilities"][step]),
    }


def _first_diff(cold: dict, warm: dict) -> int | None:
    return next((i for i, (a, b) in enumerate(zip(cold["tokens"], warm["tokens"])) if a != b), None)


def run(ctx: vp.ProducerContext) -> vp.ProducerResult:
    if ctx.fat_targets.targets != (_ARCHITECTURE,):
        raise _fail(f"{_CONTRACT_ID} is scoped to gfx1100; got {ctx.fat_targets.targets!r}")
    if ctx.model is None:
        raise _fail("the contract model (--model) is required")
    identity = support.model_identity(ctx.model, model_id=_MODEL_REF, label=_LABEL)
    device = support.select_device(ctx, _ARCHITECTURE, label=_LABEL)

    servers = {role: ctx.validation_binaries.get(role, {}).get("llama-server") for role in ("control", "subject")}
    benches = {role: ctx.validation_binaries.get(role, {}).get("llama-bench") for role in ("control", "subject")}
    if not all(isinstance(b, Path) and b.is_file() for b in (*servers.values(), *benches.values())):
        raise _fail("standard scaffold llama-server/llama-bench pair is missing")

    logs = ctx.workdir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    subject_server = servers["subject"]
    assert isinstance(subject_server, Path)

    warm_log = logs / "pccs-subject-warm.log"
    warm_factory = support.server_session_factory(
        ctx, device=device, architecture=_ARCHITECTURE, binary=subject_server, model=ctx.model,
        log_path=warm_log, env={}, server_args=_WARM_ARGS,
    )
    with warm_factory() as session:
        offset = len(warm_log.read_text(encoding="utf-8", errors="replace"))
        _completion(session, _PROMPT_A)
        a_log, offset = _read_log_since(warm_log, offset)
        primed = _checkpoint_records(a_log, action="created")
        if not primed:
            raise _fail("turn A produced no observable context checkpoint creation")
        primed_checkpoint = max(primed, key=lambda record: (record["n_tokens"], record["pos_max"]))

        warm = _completion(session, _PROMPT_C)
        c_log, _ = _read_log_since(warm_log, offset)
        restored = _checkpoint_records(c_log, action="restored")
        shorter_restores = [
            record for record in restored
            if record["n_tokens"] < primed_checkpoint["n_tokens"]
            and record["pos_max"] < primed_checkpoint["pos_max"]
        ]
        if not shorter_restores:
            raise _fail(
                "turn C did not prove shorter-prefix checkpoint reuse: no restored checkpoint "
                "was shorter than turn A's newest primed checkpoint"
            )
        restored_checkpoint = max(shorter_restores, key=lambda record: (record["n_tokens"], record["pos_max"]))

    activation_ref = ctx.runtime.write_artifact(
        name="pccs-activation.json",
        payload={
            "schema_version": 1, "contract_id": _CONTRACT_ID,
            "mechanism": "server-log-restored-context-checkpoint",
            "positive_hit": True,
            "turn_a_primed_checkpoint": primed_checkpoint,
            "turn_c_restored_checkpoint": restored_checkpoint,
            "proof": "turn C restored a checkpoint with both n_tokens and pos_max below turn A's newest primed checkpoint",
        },
    )

    cold_log = logs / "pccs-subject-cold.log"
    cold_factory = support.server_session_factory(
        ctx, device=device, architecture=_ARCHITECTURE, binary=subject_server, model=ctx.model,
        log_path=cold_log, env={}, server_args=_COLD_ARGS,
    )
    with cold_factory() as session:
        cold = _completion(session, _PROMPT_C)

    first_diff = _first_diff(cold, warm)
    passed = first_diff is None
    detail = (
        f"turn C ({_N_PREDICT} tokens) warm-vs-cold identical with shorter-prefix checkpoint restore proven"
        if passed
        else f"turn C warm-vs-cold diverge at step {first_diff}; shorter-prefix checkpoint restore proven"
    )
    correctness = experiment_contract.CorrectnessResult(check="greedy_parity", passed=passed, detail=detail)
    correctness_ref = ctx.runtime.write_artifact(
        name="pccs-correctness.json",
        payload={
            "schema_version": 1, "contract_id": _CONTRACT_ID, "check": "greedy_parity",
            "passed": passed, "detail": detail, "model_identity": identity,
            "first_divergence": first_diff, "cold_tokens": cold["tokens"], "warm_tokens": warm["tokens"],
            "prompt_a": _PROMPT_A, "prompt_c": _PROMPT_C,
            "activation_artifact": {"path": activation_ref.path, "sha256": activation_ref.sha256},
        },
    )

    emitted = {"pccs-activation.json", "pccs-correctness.json", "pccs-performance.json"}
    if first_diff is not None:
        diagnostic_runs: list[dict[str, object]] = [{
            "replicate": 0,
            "first_divergence": first_diff,
            "cold_at_first_divergence": _prediction_record(cold, first_diff),
            "warm_at_first_divergence": _prediction_record(warm, first_diff),
        }]
        for replicate in range(1, _DIAGNOSTIC_REPEATS + 1):
            repeat_warm_log = logs / f"pccs-diagnostic-warm-{replicate}.log"
            repeat_warm_factory = support.server_session_factory(
                ctx, device=device, architecture=_ARCHITECTURE, binary=subject_server, model=ctx.model,
                log_path=repeat_warm_log, env={}, server_args=_WARM_ARGS,
            )
            with repeat_warm_factory() as session:
                _completion(session, _PROMPT_A)
                repeat_warm = _completion(session, _PROMPT_C)

            repeat_cold_log = logs / f"pccs-diagnostic-cold-{replicate}.log"
            repeat_cold_factory = support.server_session_factory(
                ctx, device=device, architecture=_ARCHITECTURE, binary=subject_server, model=ctx.model,
                log_path=repeat_cold_log, env={}, server_args=_COLD_ARGS,
            )
            with repeat_cold_factory() as session:
                repeat_cold = _completion(session, _PROMPT_C)

            repeat_diff = _first_diff(repeat_cold, repeat_warm)
            record: dict[str, object] = {"replicate": replicate, "first_divergence": repeat_diff}
            if repeat_diff is not None:
                record["cold_at_first_divergence"] = _prediction_record(repeat_cold, repeat_diff)
                record["warm_at_first_divergence"] = _prediction_record(repeat_warm, repeat_diff)
            record["cold_at_original_divergence"] = _prediction_record(repeat_cold, first_diff)
            record["warm_at_original_divergence"] = _prediction_record(repeat_warm, first_diff)
            diagnostic_runs.append(record)

        ctx.runtime.write_artifact(
            name="pccs-divergence-diagnostic.json",
            payload={
                "schema_version": 1, "contract_id": _CONTRACT_ID,
                "original_first_divergence": first_diff,
                "n_probs": _N_PROBS,
                "fresh_repeat_pairs": _DIAGNOSTIC_REPEATS,
                "runs": diagnostic_runs,
                "note": "top-2 token IDs/logprobs are diagnostic only; greedy_parity remains exact",
            },
        )
        emitted.add("pccs-divergence-diagnostic.json")

    # Auxiliary no-regression evidence only; correctness disposition is greedy_parity above.
    control_bench = benches["control"]
    subject_bench = benches["subject"]
    assert isinstance(control_bench, Path) and isinstance(subject_bench, Path)
    control_outcome = ctx.runtime.run_paired_llama_benchmark(
        control_binary=control_bench, subject_binary=subject_bench, model=ctx.model,
        workloads=("decode",), pairs=_ROUNDS, log_context="pccs-control", device=device,
    )
    control_effect, control_run = support.lane_effect(
        control_outcome, workload="decode", metric="tg128", role="control", rounds=_ROUNDS, label=_LABEL
    )
    performance_ref = ctx.runtime.write_artifact(
        name="pccs-performance.json",
        payload={
            "schema_version": 1, "contract_id": _CONTRACT_ID, "model_identity": identity,
            "build_identities": {r: dict(i) for r, i in ctx.validation_build_identities.items()},
            "metrics": support.performance_metrics(control_effect),
            "control": {"metric": "tg128", "effect": dataclasses.asdict(control_effect),
                        "runs": list(control_run.runs), "stats": dict(control_run.stats)},
        },
    )

    return vp.ProducerResult(
        correctness={"disposition": "passed" if passed else "failed",
                     "mechanism": "pccs-checkpoint-shorter-prefix-warm-vs-cold", "detail": detail,
                     "artifact": {"path": correctness_ref.path, "sha256": correctness_ref.sha256}},
        validation_build_identities=ctx.validation_build_identities,
        activation_evidence=ActivationEvidence(
            status="executed",
            mechanism="server-log-restored-context-checkpoint",
            detail=(
                "turn C restored a checkpoint with both n_tokens and pos_max below turn A's "
                f"newest primed checkpoint (artifact: {activation_ref.path})"
            ),
        ),
        performance_evidence={"artifact": {"path": performance_ref.path, "sha256": performance_ref.sha256}},
        trace_evidence=None,
        check_results=(),
        lane_effects=(),
        contract_correctness_results=(correctness,),
        promotion_lane_effects={_CONTRACT_ID: (control_effect,)},
        promotion_target_metric={_CONTRACT_ID: "tg128"},
        promotion_trigger_evidence={_CONTRACT_ID: (experiment_execution.trigger_evidence_from_marker_probe(
            lane_id="pccs-server-shorter-prefix-restore", role="positive", positive_hit=True),)},
        emitted_artifacts=frozenset(emitted),
    )

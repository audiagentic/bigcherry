"""PROMPT-CACHE-CHECKPOINT-SELECTION (1005): patch-local validation producer.

Formalizes the exact regression sequence already manually validated on real
hardware (README.md, 2026-09-12/13): hybrid/recurrent memory (LLM_ARCH_QWEN35
and siblings, per llm_arch_supports_rs_rollback in llama-arch.cpp) is only
valid at its exact final position. Pre-fix, every checkpoint was treated as
range-valid, so a later request sharing only a SHORTER common prefix than a
primed checkpoint could incorrectly select/restore it.

Protocol (single server session, --cache-ram 512):
  turn A: a long prompt -- primes a checkpoint at that full-prompt position.
  turn C: a DIFFERENT prompt sharing only PROMPT_C's own short common prefix
          with turn A's prompt, then diverging -- the exact "shorter common
          prefix" scenario the fix must handle.
The warm (cache-assisted) completion of turn C is compared against a cold
reference: a fresh server process, --cache-ram 0 (cannot engage the
checkpoint-selection code path at all), given ONLY turn C's prompt from
scratch. For temp=0/seed=42 determinism these must match exactly if cache
reuse is correct.

Uses tierA-qwen4b-q6k (Qwen3.5-4B, LLM_ARCH_QWEN35, already registered) in
place of the original manual test's lfm2.5-8B (never registered in
config/models.toml) -- both are real hybrid/recurrent architectures under
the same llm_arch_supports_rs_rollback gate, so this is the same class of
test subject, not a methodology change.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from bigcherry.experiment import contract as experiment_contract
from bigcherry.patch import producer_support as support
from bigcherry.patch import validation_producer as vp

_LABEL = "pccs"
_ARCHITECTURE = "gfx1100"
_CONTRACT_ID = "PROMPT-CACHE-CHECKPOINT-SELECTION"
_MODEL_REF = "tierA-qwen4b-q6k"
_ROUNDS = support.contract_paired_rounds(_CONTRACT_ID)

_WARM_ARGS = ("-ngl", "99", "-c", "4096", "--parallel", "1", "--cache-ram", "512", "--fit", "off")
_COLD_ARGS = ("-ngl", "99", "-c", "4096", "--parallel", "1", "--cache-ram", "0", "--fit", "off")
_N_PREDICT = 64

# Turn A: a long prompt that primes a checkpoint at its full-prompt position.
_PROMPT_A = (
    "Describe, in careful technical detail, how a paged key-value cache "
    "allows a language model server to reuse computation across requests "
    "that share a common prefix, and explain why hybrid recurrent "
    "architectures require this reuse to be exact rather than approximate."
)
# Turn C: shares only its own short lead-in with PROMPT_A, then diverges --
# the "shorter common prefix" scenario the fix must handle correctly.
_PROMPT_C = (
    "Describe, in careful technical detail, why prime numbers greater than "
    "two are always odd, and give a short proof."
)


def _fail(message: str) -> vp.ValidationProducerError:
    return vp.ValidationProducerError(f"{_LABEL}: {message}")


def _completion_tokens(session, prompt: str) -> list[int]:
    reply = session.post_json("/completion", {
        "prompt": prompt,
        "n_predict": _N_PREDICT,
        "temperature": 0.0,
        "top_k": 1,
        "seed": 42,
        "cache_prompt": True,
        "ignore_eos": True,
        "return_tokens": True,
    })
    tokens = reply.get("tokens")
    if not isinstance(tokens, list) or len(tokens) != _N_PREDICT:
        raise _fail("invalid greedy token vector from /completion")
    return tokens


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

    # Warm session: turn A primes the checkpoint, turn C reuses (shorter common prefix).
    warm_factory = support.server_session_factory(
        ctx, device=device, architecture=_ARCHITECTURE, binary=subject_server, model=ctx.model,
        log_path=logs / "pccs-subject-warm.log", env={}, server_args=_WARM_ARGS,
    )
    with warm_factory() as session:
        _completion_tokens(session, _PROMPT_A)  # turn A: prime the checkpoint (result unused)
        warm_tokens = _completion_tokens(session, _PROMPT_C)  # turn C: shorter-prefix reuse

    # Cold reference: fresh process, --cache-ram 0 (cannot engage checkpoint-selection at all).
    cold_factory = support.server_session_factory(
        ctx, device=device, architecture=_ARCHITECTURE, binary=subject_server, model=ctx.model,
        log_path=logs / "pccs-subject-cold.log", env={}, server_args=_COLD_ARGS,
    )
    with cold_factory() as session:
        cold_tokens = _completion_tokens(session, _PROMPT_C)

    first_diff = next((i for i, (a, b) in enumerate(zip(cold_tokens, warm_tokens)) if a != b), None)
    passed = first_diff is None
    detail = (
        f"turn C ({_N_PREDICT} tokens) warm-vs-cold identical"
        if passed
        else f"turn C warm-vs-cold diverge at step {first_diff}"
    )
    correctness = experiment_contract.CorrectnessResult(check="greedy_parity", passed=passed, detail=detail)
    ctx.runtime.write_artifact(
        name="pccs-correctness.json",
        payload={
            "schema_version": 1, "contract_id": _CONTRACT_ID, "check": "greedy_parity",
            "passed": passed, "detail": detail, "model_identity": identity,
            "first_divergence": first_diff, "cold_tokens": cold_tokens, "warm_tokens": warm_tokens,
            "prompt_a": _PROMPT_A, "prompt_c": _PROMPT_C,
        },
    )

    # Performance/controls: standard paired tg128 decode, no cache-checkpoint
    # involvement -- proves the fix carries no ordinary-decode regression.
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
                     "mechanism": "pccs-checkpoint-shorter-prefix-warm-vs-cold", "detail": detail},
        validation_build_identities=ctx.validation_build_identities,
        activation_evidence=None,
        performance_evidence={"artifact": {"path": performance_ref.path, "sha256": performance_ref.sha256}},
        trace_evidence=None,
        check_results=(),
        lane_effects=(),
        contract_correctness_results=(correctness,),
        promotion_lane_effects={_CONTRACT_ID: (control_effect,)},
        promotion_target_metric={_CONTRACT_ID: "tg128"},
        emitted_artifacts=frozenset({"pccs-correctness.json", "pccs-performance.json"}),
    )

"""PA36 migration #2 (dev-gpt-agent req_69d011a2acf44acc): patch-local
validation producer for 1202 (RD04-BF16-FLASH-ATTN-TILE).

Mechanically migrated off validation_campaign.py's
run_rd04_contract_correctness() and run_rd04_benchmark_evidence() -- both
functions (and the --run-rd04-contract / --run-rd04-benchmark CLI paths)
are DELETED from shared code in the same change (no compatibility layer,
per the project's migrate-up doctrine).

One combined producer per the design ruling: RD04 has one bound contract,
so the PPL correctness pair and the paired benchmark are one evidence
workflow, run in one invocation.

The producer owns ONLY the measurements:

  1. The PPL correctness pair -- built ONCE by ``ctx.runtime.build_pair()``
     with ``primary_target="llama-perplexity"`` (that target is NOT one of
     the standard scaffold's five builds, so it cannot reuse them). One
     whole-model perplexity comparison (normal BigCherry control versus
     the same composition with 1202 applied) legitimately derives BOTH
     contract-internal results, backend_reference and ppl_equality --
     this project has only one real whole-model correctness signal
     available (real PPL), mirroring RD13's reasoning for the same
     situation. Forced ``-fa on -ctk bf16 -ctv bf16`` so the comparison
     actually exercises 1202's native-BF16 flash-attn path.

  2. The paired benchmark -- REUSES the standard scaffold's parity
     llama-bench binaries (``ctx.validation_binaries``); it never builds
     a second pair. ``passed`` means "benchmark executed with finite
     statistics on both lanes" -- NOT "RD04 met its 3.39% target"
     (threshold qualification remains separate, later, real-hardware
     work, exactly as in the legacy function).

It returns semantic evidence (``correctness`` = {disposition, mechanism,
detail}) plus the ``performance_evidence`` artifact ref for the fallback
benchmark validators -- every canonical identity field (patch digest,
source trees, campaign identity, root correctness.json / activation.json)
is owned by the shared binder in the generic dispatcher, never here.
``check_results`` is empty: all six declared checks are evaluated by
their fallback validators against the bound evidence.

Activation is deliberately left BLOCKED: 1202 contains no valid RD04
trace marker (its GGML_CUDA_DISABLE_FUSION negative control is invalid
for a flash-attention patch), so there is no subject-hit/control-miss
probe to run -- the historical records show the same blocked activation.
Adding a marker would modify the focal patch itself and expand the
migration beyond semantic equivalence.
"""

from __future__ import annotations

import math
import os
import subprocess
from collections.abc import Mapping
from typing import Protocol, cast

from bigcherry.patch import validation_producer as vp

# One contract architecture per run (the historical RD04 rule): the
# operator names it via --amdgpu-targets; the binary itself is built ONCE
# as the production-matching fat multi-arch set and the real device is
# selected at run time.
_CONTRACT_ARCHITECTURES: tuple[str, ...] = ("gfx1100", "gfx1201", "gfx1030")

_CONTRACT_ID = "RD04-BF16-FLASH-ATTN-TILE"
_SUBJECT_PATCH = "1202_rd04_bf16_flash_attn_tile"

# Forces the comparison onto 1202's native-BF16 flash-attn path
# (otherwise it could pass without ever touching the focal code).
_PPL_EXTRA_ARGS: tuple[str, ...] = ("-fa", "on", "-ctk", "bf16", "-ctv", "bf16")


class _PairedLaneRun(Protocol):
    """Structural view of ``experiment_execution.PairedLaneRun`` -- the
    producer-facing benchmark outcome exposes its runs opaquely, so the
    two fields this producer persists are named here instead of
    importing the campaign-internal type."""

    runs: tuple[Mapping[str, object], ...]
    stats: Mapping[str, object]


def run(ctx: vp.ProducerContext) -> vp.ProducerResult:
    from bigcherry.experiment import contract as experiment_contract
    from bigcherry.experiment import perplexity
    from bigcherry.patch import source as psi

    if (
        len(ctx.fat_targets.targets) != 1
        or ctx.fat_targets.targets[0] not in _CONTRACT_ARCHITECTURES
    ):
        raise vp.ValidationProducerError(
            f"rd04 correctness: {_CONTRACT_ID} requires exactly one "
            "contract architecture per run (gfx1100, gfx1201, or gfx1030); "
            f"got targets={ctx.fat_targets.targets!r}"
        )
    architecture = ctx.fat_targets.targets[0]
    # Real production builds compile ONE fat multi-arch binary and select
    # the real device to run it against at runtime -- match that instead
    # of rebuilding per architecture (the build-once-fat-multiarch rule).
    fat_targets = ";".join(_CONTRACT_ARCHITECTURES)

    # One canonical invocation requires both a real model and a real PPL
    # corpus; fail closed before any build or subprocess.
    if ctx.model is None or ctx.corpus is None:
        raise vp.ValidationProducerError(
            "rd04 requires BOTH a real whole model (--model) and a real PPL "
            "corpus (--producer-corpus); refusing to run with either missing"
        )
    model = ctx.model
    corpus = ctx.corpus

    # The one sanctioned pair authority: control = baseline composition;
    # subject = baseline + focal (1202). llama-perplexity is NOT one of
    # the scaffold's five builds, so this pair is producer-built.
    pair = ctx.runtime.build_pair(
        targets=_CONTRACT_ARCHITECTURES,
        primary_target="llama-perplexity",
        baseline_source="bigcherry",
        require_parity=True,
    )
    control_binary = pair.control_bin
    subject_binary = pair.subject_bin

    devices = ctx.runtime.device_contexts(device_map=ctx.device_map)
    device = next((d for d in devices if d.architecture == architecture), None)
    if device is None:
        raise vp.ValidationProducerError(
            f"rd04: no device mapped for run architecture {architecture!r}; "
            f"device_map={ {k: tuple(v) for k, v in ctx.device_map.items()}!r} "
            "-- --device-map must select a real device for it"
        )

    def _ppl_runner(argv, **kwargs):
        # perplexity.run_perplexity() invokes the runner as
        # runner(argv, ..., env=run_env) -- MERGE that env onto the
        # sanctioned HIP-only device selector env instead of forwarding
        # it as a second env keyword (subprocess.run would raise
        # "multiple values for keyword argument 'env'"). The device
        # selector is applied AFTER the supplied env and env_unset is
        # popped LAST, so HIP_VISIBLE_DEVICES always wins (GPT review
        # req_7a72896b609a48b5 MAJOR #4) and a stale ROCR_VISIBLE_DEVICES
        # can never survive into the child (never ROCR double-filtering).
        supplied_env = dict(kwargs.pop("env", None) or {})
        env = {**os.environ, **supplied_env}
        env.update(dict(device.env_overrides))
        for key in device.env_unset:
            env.pop(key, None)
        return subprocess.run(argv, env=env, **kwargs)

    try:
        subject_run = perplexity.run_perplexity(
            subject_binary,
            model=model,
            corpus=corpus,
            runner=_ppl_runner,
            extra_args=_PPL_EXTRA_ARGS,
        )
        control_run = perplexity.run_perplexity(
            control_binary,
            model=model,
            corpus=corpus,
            runner=_ppl_runner,
            extra_args=_PPL_EXTRA_ARGS,
        )
    except perplexity.PerplexityError as exc:
        comparison = None
        passed = False
        detail = (
            f"could not produce a real BF16 flash-attn perplexity comparison: {exc}"
        )
    else:
        comparison = perplexity.PerplexityComparison(
            subject=subject_run, control=control_run
        )
        passed = comparison.ok
        detail = (
            f"real BF16 flash-attn perplexity comparison: sigma={comparison.sigma:.4f} "
            f"vs threshold max_sigma={comparison.max_sigma} "
            f"(subject={comparison.subject.ppl:.4f}, control={comparison.control.ppl:.4f}, "
            f"delta={comparison.delta:.5f})"
        )

    # Both contract-internal results derive from the ONE real comparison
    # (the project's only real whole-model correctness signal).
    backend_reference_result = experiment_contract.CorrectnessResult(
        check="backend_reference",
        passed=passed,
        detail=detail,
    )
    ppl_equality_result = experiment_contract.CorrectnessResult(
        check="ppl_equality",
        passed=passed,
        detail=detail,
    )

    artifact_doc = {
        "schema_version": 1,
        "contract_id": _CONTRACT_ID,
        "base_revision": ctx.base_revision,
        "architecture": architecture,
        "compiled_targets": fat_targets,
        "model": str(model),
        "corpus": str(corpus),
        "subject_patch": _SUBJECT_PATCH,
        "perplexity_extra_args": list(_PPL_EXTRA_ARGS),
        "results": {
            "backend_reference": {
                "check": backend_reference_result.check,
                "passed": backend_reference_result.passed,
                "detail": backend_reference_result.detail,
            },
            "ppl_equality": {
                "check": ppl_equality_result.check,
                "passed": ppl_equality_result.passed,
                "detail": ppl_equality_result.detail,
            },
        },
        "subject_source_tree": psi.git_worktree_tree(pair.subject_source),
        "control_source_tree": psi.git_worktree_tree(pair.control_source),
        "subject_build_identity": pair.validation_build_identities["subject"],
        "control_build_identity": pair.validation_build_identities["control"],
        "comparison": (
            perplexity.comparison_to_dict(comparison)
            if comparison is not None
            else None
        ),
    }
    correctness_artifact_ref = ctx.runtime.write_artifact(
        name=f"rd04-correctness-{architecture}.json",
        payload=artifact_doc,
    )

    # The paired benchmark REUSES the scaffold parity llama-bench pair
    # (design ruling: one standard scaffold, no second build pair).
    bench_control = ctx.validation_binaries.get("control", {}).get("llama-bench")
    bench_subject = ctx.validation_binaries.get("subject", {}).get("llama-bench")
    if bench_control is None or bench_subject is None:
        raise vp.ValidationProducerError(
            "rd04 benchmark requires the standard scaffold's llama-bench "
            "binaries (standard_campaign='run' populates "
            "ProducerContext.validation_binaries); got "
            f"{ {k: sorted(v) for k, v in ctx.validation_binaries.items()}!r}"
        )

    outcome = ctx.runtime.run_paired_llama_benchmark(
        control_binary=bench_control,
        subject_binary=bench_subject,
        model=model,
        patch_args=_PPL_EXTRA_ARGS,
        pairs=3,
        log_context="rd04",
        device=device,
    )
    decode_run = cast("_PairedLaneRun", outcome.runs["decode"])
    prefill_run = cast("_PairedLaneRun", outcome.runs["prefill"])

    def _finite(stats: Mapping[str, object]) -> bool:
        value = stats.get("geometric_effect_pct")
        return isinstance(value, (int, float)) and math.isfinite(value)

    benchmark_passed = _finite(decode_run.stats) and _finite(prefill_run.stats)

    # The raw performance artifact embeds the SCAFFOLD benchmark
    # identities (the llama-bench pair was scaffold-built), BEFORE the
    # record-level validation identities are replaced by this producer's
    # own PPL pair in the return below.
    performance_doc = {
        "passed": benchmark_passed,
        "campaign_id": ctx.campaign_id,
        "model": str(model),
        "architecture": architecture,
        "validation_build_identities": {
            role: dict(ids) for role, ids in ctx.validation_build_identities.items()
        },
        "commands": {
            key: {inner_key: list(args) for inner_key, args in inner.items()}
            for key, inner in outcome.commands.items()
        },
        "raw_logs": list(outcome.raw_logs),
        "metrics": {
            "decode": {
                "metric": "tg128",
                "stats": dict(decode_run.stats),
                "runs": list(decode_run.runs),
            },
            "prefill": {
                "metric": "pp512",
                "stats": dict(prefill_run.stats),
                "runs": list(prefill_run.runs),
            },
        },
    }
    performance_artifact_ref = ctx.runtime.write_artifact(
        name=f"rd04-performance-{architecture}.json",
        payload=performance_doc,
    )

    # Semantic correctness for the shared binder: the conjunction of the
    # two contract-internal results (matches the legacy binding, which
    # derived the canonical correctness.json disposition the same way).
    return vp.ProducerResult(
        correctness={
            "disposition": "passed" if passed else "failed",
            "mechanism": "rd04-bf16-flash-attn-ppl-comparison",
            "detail": (
                f"backend_reference={'PASS' if backend_reference_result.passed else 'FAIL'} "
                f"({backend_reference_result.detail}); "
                f"ppl_equality={'PASS' if ppl_equality_result.passed else 'FAIL'} "
                f"({ppl_equality_result.detail})"
            ),
        },
        validation_build_identities=pair.validation_build_identities,
        activation_evidence=None,
        performance_evidence={
            "artifact": {
                "path": performance_artifact_ref.path,
                "sha256": performance_artifact_ref.sha256,
            }
        },
        trace_evidence=None,
        check_results=(),
        lane_effects=(),
        contract_correctness_results=(
            backend_reference_result,
            ppl_equality_result,
        ),
        emitted_artifacts=frozenset(
            {
                correctness_artifact_ref.name,
                performance_artifact_ref.name,
            }
        ),
    )

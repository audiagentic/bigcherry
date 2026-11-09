---
id: XDP01
order: 0
plan: run-xtx27-prefill
state: active
created-at: '2026-11-09T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Radiance dense 27B dual-XTX prefill performance

## Description

Owner direction (2026-11-09): establish good **Radiance** performance on two
RX 7900 XTX cards before integrating the R9700 into a dense layer pipeline.
Use the locally available Qwen3.8-27B MXFP4 model as the dense proxy for a
future Qwen4-27B; neither the new model's architecture nor its availability is
asserted here.

**Do not equate Strata-rdna3's quoted >6,000 token/s Flash-Next MoE prefill
with the dense 27B target.** Geometry, quantization, experts and prompt path
differ. Measure speedups only with the same dense model, tokenizer, prompt,
context, precision and transport. Historical 27B v1.9 TP2 records quoted
~2,251-2,256 prefill t/s at ~7.2K, but that is not current-pin evidence.
The existing TP2 MXFP4 layout is `qwen27b-2xtx`, WHT6, FP8 KV, MTP3.

## Coequal resident-KV target (XDP02)

Plan 1M classical attention KV as a hypothetical future-model capacity target. Current 27B's 64-layer geometry needs 16 GiB of FP8 attention KV on each TP2 XTX at 1M, leaving inadequate headroom beside 7.51 GiB weights. XDP02 compares full-layer R9700 suffix cuts, with explicit per-stage KV/GDN ownership. Keep the token embedding in host RAM to protect capacity; the current model natively declares 262,144 tokens, so 1M remains only a projection.

## Steps

1. **Reproducible current-bin baseline:** under an exclusive GPU lease on
   cards selected by inventory (gfx1100 x2), run exactly the same Radiance
   binary and model with exact and WHT6 host collective, drafting disabled,
   context 8192, 1 session, batch 2048, FP8 KV and no P2P. Measure 512,
   2,048 and 6,144 prompt-depth steps with at least two reps. Capture per-request
   prompt token counts, TTFT-derived prefill, decode, server log, VRAM,
   binary/build provenance, and shutdown. Use isolated scratch outputs.
2. **Chunk sensitivity:** compare WHT6 batch sizes 1,024/2,048/4,096 at
   constant 8,192 context, exact same model and prompt. Rotate order for
   qualification to avoid thermal/order confounding. Do not tune from a
   single unpaired run.
3. **Long prefill:** repeat at a genuinely supported 32,768 context
   (2K, 8K, 16K, 28K prompt depths), only after memory fit verification.
   Separate cache/graph warmup from steady-state. Report a TTFT metric
   distinct from GPU kernel-only rates. A single-client request is not
   a concurrency benchmark.
4. **Attribute bottlenecks:** `--profile-ops` as a separate diagnostic
   (profiling overhead excludes score). Compare GEMMs, GDN, attention,
   TP collectives, CPU dispatch gaps and H2D/D2H. Use rad-kbench/rocprofv3
   when per-op data is insufficient. Record current XTX upstream PCIe
   x8 constraint and host-staged transfers; do not assume endpoint x16
   reflects end-to-end host throughput.
5. **Small bounded optimisation:** qualify RAD48's native gfx11 INT8
   prefill tile against current pin, or optimise the dominant MXFP4
   kernel if profiling proves INT8 is not in this model's active path.
   Use isolated kernel fixture, reference parity, correctness and KLD;
   integrate via managed libr3 component/patch only, not vendor edits.
6. **Go/no-go:** paired old/new runs, exact and WHT6 correctness gates,
   memory/VRAM, p95 decode impact, model text/KL agreement and
   cold/warm prefill. Promote only a demonstrated improvement. Then
   resume the R9700 hybrid dense-stage investigation (HD2701).

## Detailed Solution & Technical Design

The card-free + GPU runner at
`tools/lab/radiance/xtx27_prefill.py` invokes the already-maintained
MEN03 `engine_bench.run` lifecycle, preserving its random-prefix cold
request behaviour and per-prompt streamed timing definition. It explicitly
requires `--run` to touch hardware; default mode validates inputs and
prints provenance/plan. Benchmark arms choose TP wire and 512..8192
prefill chunk; no source rebuild is required to compare transport/batching.

Reference invocation (from this worktree, **with the lease already held**):

```sh
PYTHONPATH=tools BIGCHERRY_ENVIRONMENT=config/environment.local.toml \
python3 tools/lab/radiance/xtx27_prefill.py \
  --build /path/to/completed-radiance-build \
  --model /path/to/qwen3.8-27b-mxfp4.rad \
  --corpus /path/to/kld-docs.txt \
  --output ${BC_WORK_ROOT}/xtx27-prefill/first-run \
  --devices 0,1 --arm exact:2048 --arm wht6:2048 --reps 2 \
  --run
```

The default (without `--run`) is a dry plan; each hardware arm writes a
separate MEN03 JSON/server log. `summary.json` is created even after a
failed measurement (which blocks subsequent arms). Experiment records are
outside the repository. Treat TTFT-derived tokens/s as an endpoint rate, NOT
the sum of kernel occupancy and NOT equivalent to Strata's benchmark until
their exact definitions/workloads are matched.

## Validation

- Card-free unit tests for input guards, exact TP flags, genuine per-depth
  repetition, missing/failed rows, and safe default dry-run.
- Explicit engine binary, complete runtime, model and corpus presence check.
- Record actual prompt token counts and require a clean engine shutdown;
  any missing row or failed arm fails the experiment.
- GPU lease, idle GPUs, model memory plan and normal server health required.
- Quality check on paired prompts and expert KLD before changing defaults.

## Effort & Risk

High. Active MXFP4 kernels may differ from the previously benchmarked INT8
RAD48 tile. XTX upstream PCIe x8 host communication and CPU scheduling may
dominate shorter requests; changing batch size may cost VRAM or latency.
WHT6 has known large-row lossiness.

## Acceptance Criteria

Current-pin MXFP4 dual-XTX baseline and profile captured; dominant
bottleneck identified; at least one bounded optimisation tested against an
unmodified control on identical 27B workload with no quality regression.
Parallel card-free three-card capacity design (XDP02) and BF16-derived heterogeneous quantization (XDP03) are active; runtime changes remain gated on TP2 performance, stage memory, quality and transfer proof.

## Notes

- Worktree: isolated experiment checkout, branch
  `exp/xtx27-prefill-opt`. Do not edit primary checkout.
- RAD48 native int8 tile was ~3.1x on one **isolated** 2K GEMM fixture,
  not a verified MXFP4 end-to-end speedup.
- RAD49's MoE TP2 profile (51% decode, 44% prefill in `moe_gemm_q`)
  describes **Flash-Next MoE**, not this dense 27B run.
- No edits to `vendor/radiance`, no push or PR without review.

## Change Log

- 2026-11-09: XDP03 added as a parallel candidate for faster XTX-compatible weight execution and KV-preserving mixed precision; no format presumed faster without measured KL and end-to-end bench.
- 2026-11-09: Started independent dual-XTX Radiance throughput harness and
  gated staged optimisation plan.

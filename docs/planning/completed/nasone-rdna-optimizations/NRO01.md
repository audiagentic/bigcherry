---
id: NRO01
order: 1
plan: nasone-rdna-optimizations
state: superseded
created-at: '2026-09-08T09:50:40+10:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: L
---

# Q8_0 wire format for the internal HIP AllReduce

## Description

Evaluate and port the Q8_0 wire-format portion of nasone32/llama.cpp-RDNA3-7900xtx-opt commit `e06dcf6300718227cb8cfda9e61fb12ccb693418` (`ggml: add Q8 wire, residual fusion, and AllReduce tracing`) as an atomic BigCherry experiment on top of the already-validated `1001_hip_internal_allreduce` provider. The source commit is compound; this item owns only FP32->Q8_0 wire quantization, transfer-size accounting, Q8_0 dequantized reduction, threshold selection, and the fallback to the existing exact/BF16 wire paths. NRO02 owns residual fusion and must not be used to claim NRO01's effect.

The reason this is P0 is empirical rather than source-reported optimism. BigCherry has already established that internal exact-FP32 AllReduce is valuable for latency-sensitive dual-XTX decode but performs poorly for large prefill reductions. Q8_0 wire compression attacks the transfer-volume term directly and may move the provider crossover without changing the higher-level Meta/Tensor-Parallel graph. It also changes numerical behavior, so a throughput win is insufficient unless the quality/correctness envelope is established separately.

The initial materialized patch is `1250_nro01_allreduce_q8_wire`, group `nasone-rdna`, state `untested`, requiring `1001_hip_internal_allreduce`. It must remain outside production patch sets until current-pin evidence qualifies it.

## Steps

1. Freeze source identity at nasone commit `e06dcf6300718227cb8cfda9e61fb12ccb693418` and BigCherry pin `b10705`; re-audit ancestry after every pin bump.
2. Extract only the Q8 wire mechanism: block-Q8_0 quantizer, wire-size calculation, Q8 finish/dequant-add path, runtime threshold/policy, and instrumentation required to prove activation. Do not import residual ADD fusion, P2P transport, ROCTx policy, or unrelated Meta graph changes in this arm.
3. Make Q8 opt-in initially. The draft must preserve exact-FP32 behavior when the selector is unset/disabled and must expose a deterministic threshold override for boundary sweeps.
4. Keep the existing copy-engine and chunked paths structurally intact. First implementation may target copy-engine reductions only if the small-reduction path cannot carry Q8 without additional synchronization; document any uncovered size range rather than silently changing it.
5. Add explicit wire-byte and activation accounting. Evidence must identify logical bytes, Q8 wire bytes, number of quantized blocks, tail handling, and actual provider path.
6. Build a correctness matrix with adversarial values: zeros, alternating signs, large dynamic range, denorm-like small values, non-multiple-of-32 tails, and independent rank inputs. Compare both ranks against a CPU FP32 reference and against the exact internal provider.
7. Run model-level quality gates on the positive workload. Because Q8 is lossy, bit-identical output is not an appropriate acceptance criterion; use bounded backend/model numerical checks and greedy divergence characterization before any throughput decision.
8. Sweep tensor size around the transfer crossover and Q8 threshold. Include small decode reductions, medium reductions, and large prefill reductions; record conversion time separately from transfer and finish time.
9. Compare four arms where supported: RCCL, exact internal, BF16 internal, Q8 internal. The causal treatment/control for PNRO01 is exact internal versus Q8 internal using the same patch composition and binary where possible.
10. Promote only an explicit size/topology envelope. A global default is forbidden unless the evidence proves it across decode and prefill controls.

## Detailed Solution & Technical Design

The source implementation quantizes each FP32 rank contribution into `block_q8_0` units, transfers those blocks, then dequantizes both rank contributions in the finish kernel. The design must distinguish logical element count from wire units: `ceil(ne / QK8_0)` blocks are transferred and the final block may contain padded elements. Tail values must not leak uninitialized memory into reduction or diagnostics.

The selector belongs to the AllReduce provider, not the generic matmul tuning candidate key. Contract/provenance identity must not become runtime dispatch identity. The minimal runtime state is a Q8 threshold/policy plus counters; no model name, contract hash, or plan ID may be compiled into selection logic.

Numerical policy is the primary risk. Each rank chooses its Q8 scale independently, so the result is not equivalent to quantizing the already-reduced FP32 sum. The experiment therefore measures the actual algorithm, not a simulated compression ratio. Correctness should include elementwise max/mean error, relative error with zero-safe handling, model PPL/KL or the project's available quality metric, and greedy-output divergence. Any tolerance must be pre-registered before the performance result is inspected.

PNRO02 may later reuse PNRO01's finish abstraction, but NRO01 itself must remain runnable without residual fusion. PNRO03 P2P transport is orthogonal: Q8 must first work correctly over the existing host-staged transport so transfer encoding and transport mechanism are not confounded.

## Code Samples & Guidance

Expected draft surfaces:

```text
ggml/src/ggml-cuda/allreduce.cu
  ggml_cuda_ar_quantize_q8_0_kernel
  ggml_cuda_ar_q8_0_add_kernel
  q8 threshold/policy parsing
  exact wire-byte accounting
```

The draft patch should include an opt-in marker such as `GGML_CUDA_AR_Q8_THRESHOLD`, with `0`/unset meaning disabled until validation establishes a safe default. Avoid reusing `GGML_CUDA_AR_BF16_THRESHOLD` semantics implicitly; the two representations have different numerical error.

## Files

- `docs/planning/active/nasone-rdna-optimizations/PNRO01.md`
- `patches/1250_nro01_allreduce_q8_wire/{patch.toml,patch.py,SUMMARY.md,README.md,TESTING.md}`
- `tools/tests/patch/test_nro_patch_packages.py` (shared package/static tests)
- Future: experiment contract and validation adapter/evidence once the draft applies/builds cleanly.

## Validation

Static: package discovery/lint, dependency closure, idempotent anchored apply, current-pin anchor check, second-apply no-op, malformed/tail fixture rejection, production-recipe non-inclusion.

Hardware: dual gfx1100 primary. Require per-element reference validation for synthetic reductions before any model run. Measure all tensor-size classes and both devices. Store actual wire bytes and activation evidence, not just an environment variable.

Performance: paired/interleaved exact-internal vs Q8-internal sessions; no trimming/winsorizing; follow current session-level CI policy if/when an Experiment Contract is registered. Prefill is a required hostile/control lane because internal exact already has a known large-reduction regression.

## Effort & Risk

High. The code surface is moderate, but lossy communication can produce plausible-looking throughput while silently altering model output. Threshold tuning can also overfit one model's reduction sizes. Treat numerical policy, tail handling, and provider crossover as separate failure modes.

## Standards

Follow `PATCH_SYSTEM.md`, `PATCH_AUTHORING.md`, `PATCH_VALIDATION.md`, Experiment Contract identity separation, fail-closed applicability, immutable source SHA provenance, and no production recipe inclusion before qualification. Preserve existing exact/BF16 fallback behavior.

## Acceptance Criteria

- Draft patch applies cleanly on `b10705`, builds HIP, and is inert when disabled.
- Synthetic Q8 reductions match pre-registered numerical tolerances on both ranks, including non-block-aligned tails.
- Activation evidence proves Q8 wire use and correct byte accounting.
- No correctness/quality gate failure on the positive model/workload.
- A statistically supported winning size envelope exists versus exact internal with <=1% unexplained control regression; large-prefill behavior is explicitly characterized.
- Promotion, if any, is selector-scoped by proven conditions rather than a blanket default.

## Notes

Source commit also contains residual fusion and tracing. PNRO02 intentionally owns residual fusion so Q8 compression can be evaluated causally. BigCherry already has separate split-reduce telemetry; do not duplicate generic tracing infrastructure merely because the fork commit contains it.

Superseded by: PNRO01
Migration: capability-rebaseline-v3-2026-09

## Change Log

- 2026-09-08T09:50:40+10:00 (created-by): Created from current nasone/BigCherry overlap audit; source commit frozen; P0.

## Ledger-events



- Pending: ag-ledger MCP was not available in the authoring session.
- 2026-09-09T11:24:25.965505+00:00 (updated-by): Updated: section:notes
- 2026-09-09T11:42:33.060472+00:00 (state-transition): State: pending → superseded
- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:57:59.826240+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:36.904401+00:00 (updated-by): Updated: section:ledger-events

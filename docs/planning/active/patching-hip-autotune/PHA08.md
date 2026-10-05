---
id: PHA08
order: 8
plan: patching-hip-autotune
state: pending
created-at: '2026-09-11T22:57:38.584940+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: L
---

# HIP Flash-Attention D=72 VLM aperture violation: root-cause before fallback (#28608/#28664)

## Description

Resolve the gfx1100 HIP tiled Flash-Attention D=72 aperture violation for VLM workloads. Prefer a kernel/root-cause fix; retain the CLIP-only matmul fallback as a narrow safety valve. Upstream PR #28664 closed without merge while issue #28608 retains evidence of the fault, so the workaround must not be assumed present merely because the PR existed.

Current BigCherry baseline is b11402. Reproduce against that pin before authoring a patch: the previous `050439614` pin named below is historical evidence only.

## Steps

1. Reproduce on current BigCherry pin b11402 on one and two 7900 XTX with representative Qwen/Gemma mmproj lanes at 1024/1600/2048/2560px; record Q sequence length and exact `flash_attn_tile<D,D,cols_per_block,...>` instantiation.
2. Use `cols_per_block=32/64` only as a diagnostic. Upstream evidence says 32 can move the crash earlier, so do not encode tile width as a workaround.
3. Instrument `ggml/src/ggml-cuda/fattn-tile.cuh` and helpers for Q/K/V/O byte ranges, shared-memory offsets, padded sequence extents, vectorized D=72 tail loads/stores, launch dimensions and workspace allocation ranges. Compare D=64/72/80 and generated gfx1100 ISA/resources; use gfx1201 as a comparison where the same kernel is selected.
4. Test the D=72 non-power-of-two/vector-tail hypothesis with a temporary guarded tail-safe load/store prototype. The prototype is diagnostic and must not become a D=72 clone kernel.
5. If the fault survives tail guarding, next isolate padded-sequence/workspace addressing. Stop pursuing tail geometry once byte-range telemetry proves every vector access is inside its allocation.
6. If no kernel fix is proven, retain only the #28664-style `tools/mtmd/clip.cpp::clip_graph::build_attn` fallback: HIP + CLIP + `d_head == 72`. Never disable main-model FA or all HIP FA.
7. Feed any architecture/shape policy through existing HIP-autotune ownership; do not add a second dispatch registry. Retire a local fallback when upstream lands an equivalent targeted fix.
8. Rebase interpretation on upstream commit `d89651a7` / #29435 when BigCherry next bumps past b11402. It changes FA scheduling only for NVIDIA DGX Spark two-stage async-KV kernels (`GGML_CUDA_CC_IS_NVIDIA && cc == DGX_SPARK`), so it is not an AMD D=72 fix and must not be cargo-culted. Its useful mechanism is diagnostic: mask-scan-induced work imbalance can justify whole-tile scheduling, but only after correctness is established and only if AMD profiling shows the same imbalance.

## Detailed Solution & Technical Design

Issue #28608 resolves the failure in `flash_attn_tile<72,72,64,1,false>`, scales with image/sequence size, and forcing a narrower tile does not cure it. That makes D=72 tail geometry, storage sizing, vector-width assumptions, or another allocation/indexing invariant higher-value hypotheses than the 64-column selection itself.

The optimisation opportunity is to recover tiled FA for D=72 rather than permanently paying matmul-attention cost. Audit vectorized `D/pack` loops, final-pack masking, padded key/query lengths, and any storage sized by floor division while wider loads/stores are issued. The image-size threshold may expose an address error that exists at shorter lengths but remains in-bounds.

### Diagnostic prototype

Do not start with a production patch. Add a temporary compile-time guarded path at the existing vector load/store helper boundary:

```cpp
// pseudocode: preserve the normal vector path for full packs
constexpr int pack = sizeof(vec_t) / sizeof(scalar_t);
const int lane0 = pack_index * pack;
if constexpr (D % pack != 0) {
    if (lane0 + pack > D) {
        scalar_t tmp[pack] = {};
        #pragma unroll
        for (int i = 0; i < pack && lane0 + i < D; ++i) {
            tmp[i] = src[lane0 + i];
        }
        consume(tmp);
    } else {
        consume(load_vec(src + lane0));
    }
} else {
    consume(load_vec(src + lane0));
}
```

Instrument the last byte touched by every D-tail vector access in debug builds. If the guarded prototype removes the aperture fault for 20 repeated 2048/2560px encodes and parity holds, replace the prototype with the smallest helper-level predicate that preserves vectorized full packs. If it does not, remove it and move to workspace/padded-sequence isolation; do not leave speculative code for downstream agents.

### Scheduling boundary

Fresh upstream #29435 (`d89651a7`, merged 2026-10-05) adds `async_kv_preload` to `launch_fattn` and disables Stream-K only for DGX Spark when mask scanning is active and whole-tile efficiency is already >=75%. This is explicitly NVIDIA-gated. PHA08 therefore owns no Stream-K/whole-tile optimisation. After D=72 correctness, an AMD scheduling experiment is permitted only if profiling shows mask-scan imbalance and must be added through generic HIP-autotune policy rather than this item.

PHA08 owns only the D=72 correctness gate and narrow candidate fix. Generic FA architecture/shape tuning remains with `patching-hip-autotune`; generic kernel work remains in the normal patch/upstream path.

## Files

- `ggml/src/ggml-cuda/fattn-tile.cuh` and the exact helper reached by the D=72 vector load/store
- `ggml/src/ggml-cuda/fattn-common.cuh` only for telemetry/scheduling comparison; do not port #29435's NVIDIA gate as a fix
- `tools/mtmd/clip.cpp` for the fallback only
- `patches/<new-id>/**` only after the diagnostic identifies a reproducible predicate
- `tools/bigcherry/tuning/correctness_evidence.py`, `tools/tests/tuning/**`, `docs/evidence/<run-id>/`

## Validation

Matrix: b11402 baseline; 1x/2x XTX; gfx1201 comparison where the same kernel is available; 1024/1600/2048/2560px; D=64/72/80 controls; cols-per-block 32/64 diagnostic; normal FA vs guarded-tail diagnostic vs CLIP matmul fallback.

Capture HSA fault absence, logits/embedding parity, image encode latency, peak VRAM/workspace, exact allocation/touched byte ranges, kernel time, VGPR/SGPR/spills and text-only throughput. Run at least 20 repeated 2048/2560px encodes before accepting a memory-safety fix.

Correctness gate: zero aperture violations and zero reported byte-range violations; parity to matmul fallback within existing VLM correctness tolerance. Performance gate for a production FA fix: >=5% VLM encode improvement versus fallback with no >2% healthy-shape regression. A diagnostic guard that fixes correctness but costs >2% on D=64/80 must be specialized/compiled away for divisible D before promotion.

Stop gates: (a) current b11402 does not reproduce across the full stress matrix -> record evidence and close without patch; (b) tail guard does not affect fault and byte telemetry is clean -> remove prototype and investigate workspace/padding only; (c) no root cause after bounded instrumentation -> ship/retain only narrow HIP+CLIP+D72 fallback, not unfinished speculative kernel work.

## Effort & Risk

Medium-high. A narrow fallback is low risk; a kernel fix is higher value but must prove the exact OOB/codegen predicate. Avoid speculative compiler workarounds without ISA/bounds evidence.

## Standards

Package-only changes remain fail-closed, idempotent, ancestry-pinned, benchmarked, and removable. No broad backend disablement.

## Acceptance Criteria

Preferred: a D=72 tiled-FA fix completes 20x 2048/2560px runs on gfx1100 with no aperture violation, numerical parity to matmul fallback, and >=5% VLM encode improvement versus fallback with no >2% healthy-shape regression. Otherwise, a narrow HIP+CLIP+D=72 fallback passes correctness/fault/non-regression gates. Current-pin non-reproduction requires the same stress matrix before closing the item.

## Notes

Provenance: shared ChatGPT conversation, 11 Sep 2026, '#28664 — direct 2×7900 XTX HIP Flash-Attention crash'; source https://github.com/ggml-org/llama.cpp/pull/28664.

Fresh upstream comparison: https://github.com/ggml-org/llama.cpp/commit/d89651a7b205c03c4a0b13cd0646d400dc929f79 (#29435). It is NVIDIA DGX-Spark-specific scheduling evidence, not an AMD correctness fix.

## Change Log

- 2026-09-11T22:57:38.584940+00:00: Created by agent.
- 2026-10-05: BCOP21 audit backfill added root-cause gate.
- 2026-10-05: Transplanted structured D=72 root-cause/tail-geometry plan from `automation-qfp-indexer-20261004`.
- 2026-10-05: Updated baseline to b11402; bounded the tail-safe diagnostic prototype; classified fresh upstream #29435 as NVIDIA-only scheduling evidence and added explicit stop gates.

## Ledger-events

- chg_20260911_225756_added-six-provenance-rich-buil_2622
- 2026-09-11T22:57:56.719160+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-05T04:55:02.911053+00:00 (updated-by): Updated: section:notes

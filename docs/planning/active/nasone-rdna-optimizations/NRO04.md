---
id: NRO04
order: 4
plan: nasone-rdna-optimizations
state: pending
created-at: '2026-09-08T09:50:40+10:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: L
---

# gfx1100 BF16/WMMA fused chunked GatedDeltaNet

## Description

Port the gfx11-specific BF16/WMMA chunked GatedDeltaNet evolution from nasone commit `4169fbbf50d24beb6d269a2350e7f780b85369e6`, building on BigCherry's existing `1221_rd50_gdn_chunked_recurrence`. RD50 carries AMD's earlier RDNA3.5 FP32 chunked kernel; it does not provide the current fork's first-generation gfx1100 WMMA implementation. This item therefore adds a real RX 7900 XTX execution path rather than duplicating RD50.

The source implementation uses BF16 WMMA operands with FP32 accumulation and FP32 recurrent state, fixed `S_v=128` geometry, and a separately validated gfx11 fragment layout. It is near-lossless, not bit-identical. The new kernel must remain segregated from RDNA4 WMMA code and must fall back to the existing FP32/sequential implementation outside the proven predicate.

## Steps

1. Require `1221_rd50_gdn_chunked_recurrence`; preserve RD50's existing code and fallback.
2. Extract only the gfx1100/gfx11 BF16 WMMA implementation and required helpers from nasone's `gated_delta_net_chunked_bf16_gfx11.cu`; fold it into the existing translation unit because BigCherry's patcher cannot create upstream files.
3. Gate host dispatch by runtime `GGML_CUDA_CC_IS_RDNA3(cc)` and exact supported geometry. Do not gate host code on device-pass-only `RDNA3` macros.
4. Compile WMMA intrinsics only under the gfx11 device pass. Non-gfx11 builds must compile cleanly and never select the path.
5. Add an opt-out/force-control lever; initial draft should not silently replace all GDN paths until numerical validation is complete.
6. Validate WMMA fragment layout with deterministic matrix probes before full GDN. Include identity, structured, random, and adversarial matrices.
7. Validate one-chunk and multi-chunk recurrence against the existing FP32 path and CPU/reference GDN, including recurrent state, not output alone.
8. Run long-sequence quality checks; quantify BF16-induced error, PPL/KL/greedy divergence as available.
9. Profile KKT solve and chunk-scan kernels separately: waves/workgroup, VGPR, LDS, occupancy, memory traffic, and end-to-end GDN time.
10. Promote only gfx1100 + exact supported shape; keep RDNA3.5/RDNA4 paths independent.

## Detailed Solution & Technical Design

The gfx1100 kernel is not a trivial datatype switch. It depends on the first-generation wave32 WMMA fragment layout: BF16 A/B fragments hold 16 elements/lane and FP32 accumulator rows are interleaved between lane halves. The source notes this layout was probed with 256/256 reference matmuls. BigCherry must reproduce that probe rather than trust comments as evidence.

The algorithm retains the chunked GDN transform: construct/invert the chunk-local KKT system, compute chunk-local attention/update terms, and scan chunks while recurrent state remains resident. BF16 is confined to WMMA operands/staging; accumulators, gate prefix sums, and persistent state remain FP32. This limits but does not eliminate numerical drift.

The dispatch must never cross-select gfx12. The source explicitly segregated gfx11 and gfx12 after an earlier combined refactor caused NaN/Inf on RDNA4. That failure mode is a design requirement: separate code path, separate compile guards, separate tests.

## Code Samples & Guidance

Target shape for first implementation:

```text
backend HIP
arch gfx1100/RDNA3
S_v = 128
scalar gate / supported non-KDA path
prefill n_tokens > chunk threshold
BF16 WMMA operands, FP32 accumulate/state
```

Do not broaden to `S_v=16/32/64` in this item unless the source WMMA implementation actually supports and validates them.

## Files

- `docs/planning/active/nasone-rdna-optimizations/NRO04.md`
- `patches/1253_nro04_gfx1100_bf16_chunked_gdn/{patch.toml,patch.py,SUMMARY.md,README.md,TESTING.md}`
- shared NRO package tests; future gfx1100 WMMA/GDN validation fixtures.

## Validation

Static: exact RD50 dependency, architecture guards, fallback marker, compile exclusion on non-gfx11 device pass.

Hardware: gfx1100 WMMA matrix probe, GDN op reference across sequence lengths/chunk boundaries, recurrent state comparison, multi-layer model PPL/greedy checks, graph/non-graph execution.

Performance: kernel-level KKT/scan timings and model prefill throughput over ubatch/context sweep. Record numerical quality alongside speed.

## Effort & Risk

High. Large architecture-specific kernel, non-bit-exact arithmetic, subtle fragment layout, high register/LDS pressure. Wrong fragment mapping can produce plausible but incorrect numbers.

## Standards

Correctness before performance; architecture-specific promotion; source SHA fixed; no conflation with RD50's RDNA3.5 evidence; no synthetic claim that gfx1100 inherits gfx1151 results.

## Acceptance Criteria

- gfx1100 WMMA primitive passes deterministic reference matrices.
- GDN output and state satisfy pre-registered tolerance across chunk boundaries and long sequences.
- Non-gfx1100 devices compile and cannot select the new path.
- Real gfx1100 prefill establishes positive kernel/E2E effect without quality regression.
- Fallback remains available and verified.

## Notes

NRO05 owns the MTP prefix/tail composition from the same source commit. This item is ordinary K==1/prefill chunked gfx1100 execution only.

## Change Log

- 2026-09-08T09:50:40+10:00 (created-by): Created as gfx1100 successor to RD50 from nasone block 02; P0.

## Ledger-events

- Pending: ag-ledger MCP unavailable in authoring session.

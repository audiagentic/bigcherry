---
id: RUM15
order: 15
plan: patching-radiance-upstream
state: pending
created-at: '2026-11-22T09:17:06.700561+00:00'
breadth: ''
skill: advanced
created-by: gpt-6
work: L
priority: P1
---

# Preserve IQ8R two-level weight-scale precision across native GPU GEMMs

## Description

Fix a measured precision loss in native IQ8R scale materialization, not just a CPU-versus-GPU mismatch. The native model stores U8 per-(row,256 K) scale and BF16 per-row scale. libref performs their product in FP32; gfx1100 libhalo and gfx1201 libr4d collapse this product into BF16 before GEMM. The frozen LTX02 source-backed 198-value IQ8R QKV oracle reproduces all CPU/GPU differences with that sole rounding choice. The existing GPU/GPU results are bitwise identical. Keep owner generic Radiance kernels/layouts, not LTX-specific architecture or a replacement model loader. Depends on LTX02 captures and RUM14's native loader seam; no production TP3 admission change.

## Steps

1. Freeze a failing native reference for actual checkpoint-backed 198 QKV elements and the captured input activation codes/scales: expect GPU to match canonical FP32 two-level decoded scales (currently 126/198), retain GPU/GPU exactness, and add synthetic worst-case U8*BF16 cases including ties, zero and large gains. Host libref is the higher-precision numeric oracle; NEVER downgrade it to BF16 for matching.
2. Inventory all existing IQ8R scale readers, native layout/unrelayout hooks, chained and unchained decode/prefill GEMMs in libr4d and libhalo, GPU layout tags/fingerprint, and Engine placement/refill. Measure the full model row/group counts and per-rank residency, baseline first-token, decode and M16/M64 throughput, source identity and launch overhead; capture existing component tests.
3. Evaluate two precise representations **without new kernel families**: (a) a tile-major FP32 folded scale for IQ8R only, reused by existing narrow/tiled kernels; (b) original U8 group scale with BF16 row factor multiplied in FP32 at fold (may save VRAM but adds operand reads/instructions). Select by numerical correctness, memory/admission and measured speed—not by equality with a degraded reference. For (a), the 1,306 IQ8R matrices total 127,309,824 group scales; F32 doubles the resident scale plane from 242.82 to 485.65 MiB (adds 242.82 MiB if all resident), distributed by actual rank allocation. No guessed hardware constants or silent RAM growth.
4. Implement the smallest shared, format-specific scale layout/accessor inside existing library paths. Keep all other I8/E4M3 layouts unchanged, preserve fragment order and 256-K group, exact source plane validation, source-managed Radiance patch lifecycle and ABI/layout fingerprint. Do not copy the engine loader, add transport, introduce LTX-only kernels or increase per-step synchronization.
5. Test real file relayout including bf16 ties, invalid plane lengths, wrong scale tag/layout, kernel selection, host/device stage, reload/refill, interrupted transfers, 1/16/64 row GEMM and 8/8/16 or 12/12/8 head partitions. Re-run native LTX C06 QKV/attention in both masked XTX-XTX-R9700 and R9700-XTX-XTX orders under exclusive GPU leases. Test the selected new path against higher-precision reference and actual TP1 activation captures. Do not use a guessed precision threshold.
6. Measure full-configuration memory admission (especially smaller card), VRAM versus host/RAM weights, GEMM timing 1/16/64 row, 4-GPU and 3-GPU mixed-stack canary if shared binaries changed, cold/warm weight load/refill and release ABI fingerprint. Compare added/deleted LOC, avoid adding a second scale math path. Only mark ready when the correctness proof and memory/throughput regression budget are explicitly reviewed.

## Detailed Solution & Technical Design

Architecture owner: pinned Radiance 2.2 ABI44 kernel code, not BigCherry LTX adapter. Canonical value is FP32(U8_scale * BF16_row_scale). Current libref: libref/ref_gemm.cpp:461-478 computes two_level[] in F32. Current device: libr4d/r4d_fp8_layout.cpp:236-243 declares an IQ8R BF16 scale plane; :303-318 calls rad_store_f32(...,RAD_BF16, product). libhalo/halo_fp8_layout.cpp:286-294 and :343-352 mirror that contract. Existing narrow/tiled kernels consume weight scales through r4d_gemm_fp8a8.hip and r4d_gemm_8a8_prefill.h; libhalo has matching equivalents. A format-tagged precision-preserving representation must be readable by ALL IQ8R consumers without altering FP8/I8 formats, projected GEMM shapes or source checkpoint. Prefer a shared scale-load trait/templated kernel specialization instead of duplicating decode and prefill kernels. Any new RadLayout bytes/tag changes ABI and layout fingerprint and must be deployed by a managed source patch, not an in-place edit to installed SDK or a test-only shim. RUM14 owns generic loader lifetime, LTX02 owns video/audio TP1 parity.

## Code Samples & Guidance



## Files

Frozen upstream: libref/ref_gemm.cpp, libr4d/r4d_fp8_layout.cpp, libr4d/r4d_gemm_fp8a8.hip, libr4d/r4d_gemm_8a8_prefill.h, libhalo/halo_fp8_layout.cpp and libhalo/halo_gemm_fp8a8.hip; managed engines/radiance/patches/, engines/radiance/recipes.toml; tests tools/lab/radiance/ltx-tp3/native-c01-gate and existing tools/tests/lab/test_ltx_iq8r_numeric.py. Evidence docs/evidence/2026-11-22-ltx25-ltx02-iq8r-scale-root-cause/README.md.

## Validation

Red/green checkpoint-based IQ8R scale and QKV numeric tests, host reference unchanged; native ref/halo/r4d component tests and exact layout/ABI fingerprint; source recipe apply/idempotence and installed stock same-SDK test; leased gfx1100/gfx1201 mixed rank reversed-order and native C06 full issue plus 1/16/64 row source parity; measured memory and performance.

## Effort & Risk

Large, P1. Shared GPU scale layout may add 242.82 MiB worst-case full-model resident memory if folded to F32; scale loads may affect 1-row latency, 16/64-row GEMM throughput, refills and runtime ABI. Do not presume that a numerically closer CPU comparison guarantees better generated-video quality without real TP1 activation tests.

## Standards



## Acceptance Criteria

Full precision of IQ8R decoded weight scale (no avoidable BF16 intermediate rounding) is preserved on both XTX and R9700 via the same declared generic Radiance layout. Actual checkpoint-backed QKV matches an approved canonical F32/TP1 oracle within a documented precision budget based on observed data; host reference remains unchanged. No new kernels/loader/ABI shims or duplicate paths; source/weight integrity, host-memory/offload/refill, layout fingerprints, admission and bounded failures still pass. Report measured time/memory tradeoff and review before enabling production. LTX world_size >1 stays refused until LTX04.

## Notes

2026-11-22 evidence: two AQL card permutations show 86,400/86,400 BF16 intermediate matches; all 36,864 INT8 activation codes and 144 F32 activation scales match CPU and GPU. Source-backed 198-sample QKV oracle isolates additional device BF16 weight-scale fold. This RUM item owns generic precision preservation; LTX02 only consumes output for model quality and does not duplicate this work.

2026-11-22 RUM15 diagnostic implementation (rad0061, not selected in serving): existing r4d/libr3 IQ8R group-256 F32 scale relayout and shared decode/prefill GEMM paths; libref remains F32. Four files, 14 source-anchored edits pass exact replay and idempotence; 23/23 focused Python tests including actual IQ8R source scales; full source-built gfx1201 R9700 plugin passed 35/35 native IQ8R numeric selftests with exact-F32 reference and unchanged 0.003 threshold (one previous BF16-reference case failed at 0.003963, now passes at 0.00165). gfx1100 native rewrite compile and mixed XTX/R9700 issued C06 parity, loader/refill and admission/perf gates remain open. Evidence docs/evidence/2026-11-22-rum15-f32-native-candidate/README.md.

2026-11-22 native runnable C06 result: diagnostic rad0061 F32 IQ8R scale source patch has 14/14 exact/idempotent anchors and full gfx1201 and gfx1100 native plugin builds. R9700 standalone IQ8R native selftest 35/35 (same exact-F32 oracle threshold). Existing model-backed C06 native AQL issued all four video/audio×head-split graphs in both masked device orders 0,1,2 and 2,0,1, with 108 retired instances/order, all exit 0. All 36,864 IQ8R QKV BF16 outputs agree exactly with unchanged CPU libref, and 12,287/12,288 final BF16 values agree; all 12,288 final values agree exactly between both mixed-GPU orders. Post-norm has 18/36,864 BF16 differences, max abs 0.00390625, and final video 12/12/8 has one difference max abs 0.015380859375 (relative L2 3.92e-5). 23/23 focused Python checks, real model, no fallback, no invented tolerances. Only M=1 seeded input qualifies; production remains unchanged because multirow, real TP1 activation quality, timing, VRAM/host/refill admission and layout-fingerprint release gates still open. See docs/evidence/2026-11-22-rum15-f32-native-candidate/README.md.

2026-11-22 RUM15 admission diagnostics: read-only RAD2 metadata check across all 1,306 IQ8R 2D matrices / 127,309,824 group-256 scales. F32 folded scales add exactly 254,619,648 bytes (242.8242 MiB) vs old BF16 across a fully resident full model; 138 MiB comes from DiT, 104.8242 MiB from other weights. For the 48-block C06 attn1 video/audio QKV and output projections only, 8/8/16 per-rank added scale bytes correspond to 7.5/7.5/15 MiB and 12/12/8 to 11.25/11.25/7.5 MiB. These are NOT live VRAM admission. Native stock-F32-stock kbench A/B/A under exclusive GPU leases on R9700 and XTX used identical shipped Qwen27B IQ8R fixture (not LTX shape), 54 matching timed IQ8R cases each; F32 median time change -0.59% R9700, -0.61% XTX. Important R9700 M33 N1024 K5120 group256 narrow GEMM slowdown +25.35% first pass and +24.89% independent second pass with <0.6% stock control drift; XTX M18/M32 N12288 K5120 slowdowns +13.71%/+13.47% once each. Do NOT select diagnostic FP32 scale patch for production until these per-shape regressions, actual LTX sizes, memory tier reload/refill, layout/cache invalidation and model TP1 capture parity are validated. See docs/evidence/2026-11-22-rum15-iq8r-scale-capacity-kbench/README.md.

2026-11-22 RUM15 narrow GEMM performance follow-up on merged diagnostic #1351 and admission evidence #1362: R9700 M33 N1024 K5120 IQ8R warm-cache FP32 scale regression +24.89% stock-vs-F32 A/B/A, repeatable. Existing native wave-count pin MW=4 (original MW8) was confirmed through kernel logs, stock 5.73/5.75 us vs F32 7.24 us (+26.13%); no benefit, so no dispatch workaround. Disassembled pinned gfx1201 native MW8 NT3 SB256 kernel stock vs F32: VGPR 167/167, SGPR 56/55, zero spills, LDS 13124/13380 B, static ds_load_b128 16/32, global_load_b128 28/28; not a spilling defect. Isolated direct global F32 scale load without shared-memory staging compiled and passed 35/35 existing native R9700 IQ8R tests with unchanged exact-F32 oracle, but same M33 case staged-F32 7.16/7.19 us vs direct-F32 7.20us (+0.35%); do not merge that experiment. Both R9700 and XTX ran exclusive-lease same-fixture stock-F32-stock rotating 256-MiB weight benchmarks, 54 IQ8R rows each; F32 median +0.63% R9700 and +0.27% XTX. R9700 M1 N1024 K5120 10.99 vs 11.95us (+8.74%) and M33 11.27 vs 13.24us (+17.48%), both steady controls. XTX M32 N5120 K6144 168.24/168.28us stock vs181.34us F32 (+7.77%). All are Qwen IQ8R fixture workloads, not LTX DiT shapes. Production remains unchanged. Next: actual LTX TP1 M1/16/64, per-rank residency, refill, old-layout cache invalidation; if cost survives then evaluate compact U8+BF16 original factors multiplied in F32 at fold within the same native kernel. Evidence docs/evidence/2026-11-22-rum15-narrow-iq8r-perf/README.md.

2026-11-23 RUM15 on the LTX GEMM shapes (docs/evidence/2026-11-23-rum15-ltx-iq8r-f32-scale-ab): rad-kbench timing of every gemm_nt_q IQ8R shape the LTX graph declares, stock against the F32-scale diagnostic build, one card per run under its own lease. The large-M tiled row, which every DiT block runs at 2048 tokens and above, is unchanged within noise: median -0.47% on gfx1201 and +0.26% on gfx1100 at M >= 2048 (48 shapes each). The M <= 64 row's median moves +0.90% and +0.45%, with single shapes up to +17-18%. The earlier narrow-shape regressions came from Qwen fixtures and do not represent LTX's prefill-like path. The 242.8 MiB resident-memory cost of the folded F32 scales still stands. Remaining: per-rank residency and refill on the real LTX layout, M=1/16/64 in the issued LTX lane, and the old-layout cache invalidation.

2026-11-25 second integration review: docs/design/radiance-integration-review-20261125.md is the exact-source review and controls conflicting earlier wording. Shipping pin remains b6b8751 (2.2.0), upstream observed main 1c104de4 is a development candidate, 119 commits and 327 changed files; no automatic repin or backport. No new compatibility layer, launch manifest, bump driver or identity machinery. Native int8 row group128 BF16 group scales and scalar IQR factored F32 scale are not proof of local IQ8R U8 x BF16 group256 precision. Compare actual source layout/tag/rounding and checkpoint-backed QKV against canonical F32 oracle; retain existing gfx1100/gfx1201 mixed/admission gates. Candidate layout changes may affect rad0061 reconciliation; no duplicate kernel or native backport.

## Change Log

- 2026-11-22T09:17:06.700561+00:00 (created-by): Created by gpt-6
- 2026-11-22T10:04:33.924676+00:00 (updated-by): Updated: section:notes
- 2026-11-22T10:10:14.030623+00:00 (updated-by): Updated: section:notes
- 2026-11-22T10:50:02.982673+00:00 (updated-by): Updated: section:notes
- 2026-11-22T11:27:10.667963+00:00 (updated-by): Updated: section:notes
- 2026-11-23T01:44:04.069729+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4649
- 2026-11-25T02:49:08.398079+00:00 (updated-by): Updated: section:notes

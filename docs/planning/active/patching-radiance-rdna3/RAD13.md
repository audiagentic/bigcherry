---
id: RAD13
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-10-19T12:23:26.909976+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# FP8 block-scaled expert path on gfx1100 for the moe worker (Qwen3.6-35B-A3B)

## Description

The moe worker is shape-generic (RCS09, PRs #317-#328): the gate is now the backend weight-format probe. gfx1100 lists only W4NL64, so Qwen3.6-35B-A3B FP8 (256 experts, hidden 2048, expert width 512, E4M3 weights with 128x128 bf16 block scales) is refused by name. This item adds the FP8 path. Design from GPT review 2026-10-19: FP8 W8A16 with BF16 activations first, preserving the checkpoint representation (codes uint8[N][K] E4M3FN, scales bf16[ceil(N/128)][ceil(K/128)]), FP16 WMMA for prefill and a separate decode GEMV; do not resurrect the shelved r4d_gemm_fp8a8 rewrite first (radiance 1.8 changed operand ownership, 128x128 prefill tiling, narrow path and epilogue, so its eight rewrite blocks no longer match).

## Steps

1. Backend entry and layout (S): gfx1100 lists MOE_FMT_FP8_E4M3_BLOCK with MOE_ACT_BF16 and bf16 128x128 scales; plan_weight/upload_weight for the FP8 layout (no W4 nibble packing, LUT or per-64 scales); both parity classes the same width; probe, byte sizing and upload round-trip tests without a model load.
2. GPU numerical harness (M): extend --route-selftest with synthetic experts from moe_fp8_ref.h run through the worker lane on gfx1100 against the host reference (GU 1024x2048 and DN 2048x512, adjacent-tile scales, zero/subnormal/negative/max E4M3, GEMM/GEMV threshold); BF16-rounded reference inputs; provisional tolerances atol/rtol 0.02 per GEMM and 0.05 per expert forward, calibrated on the synthetic distribution.
3. FP8 prefill GEMM (L): the existing moe_ plugin call is W4-specific (RAD_F8E4M3 activations, F32 group scales, W4 tables), so a new FP8 plugin dispatch variant is needed: first predecode FP8 to scaled FP16 at upload (2x memory) to validate numerics with a conventional FP16 WMMA grouped GEMM, then FP8-resident weights decoded in the GEMM (LDS staging, scale reuse across 16x16 tiles).
4. FP8 decode GEMV (M): lane_moe_gemv_fp8 beside lane_moe_gemv; fp32 accumulate over decode_e4m3(code) * scale * activation; no WMMA for one-row decode.
5. Performance and integration (M): benchmark decode and prefill separately; then Qwen3.6-35B-A3B end to end through the worker with the R9700 engine and the XTX workers.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Flash-Next W4 replay stays bitwise identical at every step; FP8 GPU output agrees with moe_fp8_ref within the recorded tolerances; probe accepts FP8 only once steps 1-4 pass; Qwen3.6-35B-A3B end to end: correct output and acceptance, then multi-stream stream-bench.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Unverified by the review: an existing compatible FP16 grouped-GEMM plugin registration and the exact plugin tensor order; inventory the native kernels before step 3. Pitfalls on gfx1100: no FP8 matrix instructions, E4M3 decode and scale multiply can dominate short GEMMs, LDS expansion halves tile capacity, FP16 overflow when scaling before WMMA, scale indexing must follow logical N/K after any relayout. Related: RCS01, RCS05, RCS09, RAD11.

## Change Log

- 2026-10-19T12:23:26.909976+00:00 (created-by): Created by agent
- 2026-10-19T12:23:41.009876+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:notes

## Findings (2026-10-20, Qwen3.6-35B-A3B FP8, R9700 engine + two XTX workers)

Built on radiance 1.8.0's own `moe_gemm_fp8a8` row and the plain quantisers; the container's planes are the kernel layout as stored (the row-major hook answers `RAD_E_UNSUPPORTED` on purpose, meaning "use the checkpoint bytes"). The worker reads the model's shape from the container (hidden 2048, 256 experts, expert width 512, top-k from `n_expert_used` = 8).

- GPU lane vs a CPU reference decoded from the container planes: rel_l2 4e-5 to 6.9e-4, max_abs <= 4.9e-4, on synthetic routes (layers 3, 20) and on 12 real captured requests (version 3 captures, `R3_MOE_WORKER_CAPTURE`). Flash-Next W4 replay stays bitwise identical on every step.
- End to end (stream-bench, workers on vs off on the same build): engine decode tokens/s 121.7 vs 104.2 (1 stream), 179.7 vs 247.9 (2), 276.9 vs 142.3 (4), 374.1 vs 216.0 (6); client totals 88.2 vs 79.6, 120.9 vs 189.5, 157.8 vs 124.7, 183.6 vs 185.3. Worse at 2 streams, flat at 6 on the client total (prefill / first-token cost not yet split out). The workers-off arm shows MTP draft acceptance collapsing at 4 and 6 streams (0.05, 0.03; about 0.6 with the workers on): an engine-side multi-stream speculation issue to isolate separately.
- Quality (KL mode, 4 documents, 8240 positions) against the engine alone: KLD mean 0.042 (median 0.0076, p99 0.75), top-1 agreement 91.97%, PPL ratio 1.005. Isolation: the engine-alone control is exactly 0; shadow mode (workers running, engine keeps its local results) is exactly 0; the exact merge control (`R3_MOE_WORKER_MASK_ONLY=1`, merge kernel fed the engine's own rows) is exactly 0, so the merge kernel and the worker's presence change nothing. The f16-widening flags (`R3_MOE_FP16`, `R3_MOE_AHALF`, `R3_MOE_NATIVE`) do not touch the fp8a8 path (identical to six digits). Layer bisection (`R3_MOE_WORKER_LAYERS`): layers 0-9 0.042, 10-19 0.020, 20-40 0.006; layers 0-2 0.041, 3-5 0.034: no single bad layer, error grows toward the early layers. Shadow compare per layer: worker vs the engine's local routed sum differs by a uniform 0.23% relative L2 on every layer (max_abs <= 1e-2, no outlier rows).
- Reading: the worker matches exact math (about 1e-4) while the R9700's local FP8 path sits 2.3e-3 from it, so the difference is the engine's native gfx1201 FP8 arithmetic (matrix-unit accumulation), not a worker defect; the W4 path was bit-identical to the engine and had no such gap. The corpus perplexity of the workers' arms is within 0.5% of the engine's (the 10-19 arm is below it), so this is a different rounding path, not a quality loss. KL against the engine alone is therefore a consistency gauge with a floor for FP8, not a pass/fail for W4-style identity; thresholds for FP8 should be set from this floor (and PPL ratio) and ideally checked against a higher-precision reference.
- Tooling added: `fastbuild.sh` (incremental worker and plugin rebuild in an existing lane build: seconds instead of 25 minutes), capture version 3 with `capture_upgrade.py`, `R3_MOE_WORKER_LAYERS`, the KL, bisect, shadow and mask-only configs under tools/lab/radiance/qwen35b-*.
- Open: FP8 decode GEMV (RAD14), the 2-stream and prefill/first-token cost, the engine's MTP acceptance collapse, KL floor calibration against a higher-precision reference, R9700-vs-worker per-route comparison to confirm the accumulation hypothesis.

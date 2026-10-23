---
id: RWQ01
order: 1
plan: patching-radiance-wire-quality
state: pending
created-at: '2026-10-15T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: M
---

# Freeze Flash-Next wire baseline and attribute model divergence to the actual fused operations

## Description

Measured Flash-Next two-XTX exact vs WHT6: 42,761 teacher-forced positions, WHT6 mean KLD 0.0746 and 87.18% top-1, versus exact-wire self-reference 0/100%. Separate 7.2K-prompt profiling found 1,263 tok/s exact versus 1,343 tok/s WHT6 (~6.3% improvement). Qwen3.8-27B behaves differently: exact baseline 0.003665 mean KL versus R9700, WHT6 0.004422 (+0.000757) and +38% prefill. Before redesigning the code, identify WHICH Flash-Next collective, layer, and activation distribution explains that sensitivity. These results are distinct comparisons (the Flash-Next KL uses itself as reference). Sources: engines/radiance/README.md sections 'The lossy two-card wire', 'The cross-card sum, timed', and 'Flash-Next ... first runs'.

## Steps

1. Freeze pinned engine build, libr3 build hash, exact .rad file size+SHA, settings and GPU order; use the flashnext-2xtx profile and model's exact wire. Use same BF16/FP8 KV, MTP disabled for KL and independently repeated with MTP for performance.
2. Re-run exact vs exact (repeatability), exact vs current WHT6, WHT6 vs WHT6. Preserve per-position KL, top-1, corpus segment, p50/p99/p99.9, PPL, and layer-agnostic performance.
3. Instrument entry-path counts and shape/bytes for ar_rmsnorm_quant_fp8, ar_hc_write, ar_gather_hc_write and standalone all_reduce; record fused activation selection, wire=0/6, rows M, width K, layer/phase where available, disabled forms, byte count and correlation with outliers. Instrumentation must not call hipStreamSynchronize in production timing.
4. Establish per-path and per-layer exact-vs-lossy ablations (RWQ03 provides the selector if current graph cannot do it); isolate a single change at a time. Count all declaration/issue sites including MTP, vision and disjoint embedding gather.
5. Export versioned summary with an honest missing-data list to docs/evidence/RWQ01; raw logs to artifacts/RWQ01-<run>. If host data cannot expose layer identity, explicitly record the gap and make it RWQ03's prerequisite.

## Detailed Solution & Technical Design

The reduction is not a generic all_reduce alone. Pinned libr4d/r4d_ar_entry.hip drives fused ar_rmsnorm_quant_fp8 and both HC fused forms; the engine arch/common/rad_block_hc.h declares distinct exact and WHT6 ops, and arch/common/rad_block_moe_fp8.h may hand off the gather. Uncompressed embedding all_reduce is intentionally exact, even under --tp-wire wht6 (arch/qwen35_fp8/qwen35_fp8.cpp). Inspect declaration-versus-issued matrix to avoid attributing bytes to a kernel that did not execute. Two-rank no-P2P path uses GPU system-scope scratch/flag synchronization via mapped host memory. Large-message link timings at 27B: WHT6 7.17 MB median ~2.0-2.15 ms; exact 18.35 MB ~5.0 ms. Flash-Next is not necessarily wire-bound, so do not project a 27B speedup to it.
Offline prework includes a synthetic Hadamard codec proxy, tested locally without GPU; this reports no model KLD. Evidence snapshot contains its JSON report.

## Code Samples & Guidance

Emit records keyed by {engine_pin, build_hash, model_sha256, profile, layer, op, phase, rows, width, wire, rank, bytes, case_id}. For each position, calculate KL(softmax(exact_logits)||softmax(candidate_logits)) using float64 stable logsumexp, retaining direction and reference identity; do not compare different prompt tokenisations. Do not infer logits equality from greedy text.

## Files

engines/radiance/README.md (append measured outcomes only); tools/lab/radiance/wire-quality/codec_mock.py, README.md; tools/tests/radiance/test_wire_quality_mock.py; docs/evidence/RWQ01/; optional scoped R3_AR_WIRE_TRACE instrumentation in engines/radiance/components/libr3/native/r4d_ar_entry.hip.rw; docs/reference/engines/RADIANCE_WIRE_QUALITY_DESIGN.md.

## Validation

Offline: seven deterministic unittest cases for pack/unpack, BF16 ties, Hadamard inverse, rank exchange, zeros, input rejection and sample accuracy (already locally passed; NOT GPU evidence). On dev: exact-repeat KLD 0, top-1 100%, effective wire route counts consistent with graph; two rank device kernels checked against reference, no difference between same-mode rank outputs; 42,761-position KL report from existing Radiance scoring machinery; record log hashes and telemetry overhead.

## Effort & Risk

Low implementation risk for offline mock; medium for instrumentation, high for instrumentation at every graph issue site. Never mistake per-op timing sync instrumentation for production speed. Data dependence is potentially nonstationary across layers, lengths, MTP verification and QSA.

## Standards

- Target Radiance 1.3.0 pinned at 89cee7ce414486c717aa2085c99e1f522e4c0ea6; no unpinned edits.
- Engine edits are explicit Radiance patch packages; gfx1100 kernels and settings belong to the libr3 component. Keep opt-ins OFF unless independently promoted with quality evidence.
- Use engine-specific recipes, source/build identities, runtime profiles containing only settings different from declared defaults, and existing bigcherry ab-benchmark/engine_quality evidence facilities. No second campaign or loader.
- Follow docs/reference/tooling/TOOLING.md, docs/reference/engines/radiance/RUNBOOK.md, patch mechanics, relevant device checker, and main-branch code review requirements.
- Changes must preserve both-rank numerical agreement, graph replay and no-P2P mapped-pinned-host protocol; fail closed on missing kernel support or disagreement.

## Acceptance Criteria

- Frozen reproducible controls and per-path issue counts published, or explicit evidence-backed instrumentation blocker recorded.
- Per-position and per-section KLD/top-1/PPL and correlation with operation family recorded.
- Separate 27B control reproduced; never conflate model-weight quantisation KLD with wire-induced KLD.
- Mock and test results clearly identified as synthetic; status remains pending until hardware characterization.

## Notes

Dependencies: none; blocks RWQ02 quality attribution, RWQ03 selector scope, RWQ05 hardware campaigns. The first baseline comparison is already recorded; this item is NOT completed by that historical evidence alone.

## Change Log

- 2026-10-15T00:00:00+00:00 (created-by): Initial plan; unimplemented GPU work remains pending.

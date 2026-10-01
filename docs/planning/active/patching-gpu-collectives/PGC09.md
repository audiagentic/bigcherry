---
id: PGC09
order: 0
plan: patching-gpu-collectives
state: pending
created-at: '2026-09-30T04:03:45.724752+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# AllReduce: adaptive provider as production default for dual XTX, gated by wire accuracy

## Description

Dual-XTX 27B Q8_0 provider A/B (ab-27b-allreduce, 6 balanced rounds, complete separation): RCCL wins prefill (host pp4096 -8.7%), host pipeline wins decode (tg512 +7.5%, tg2048 +2.5%). 0840 adaptive (host below the size threshold, RCCL above) should take both. The host path compresses to bf16 by default (lossy), so no lossy wire may become a default until it passes accuracy gates.

## Steps

1. Measure 0840 adaptive vs ccl vs host (ab-27b-adaptive, queued).
2. Measure 1272 host f32/bf16/f16/q8_0 (ab-27b-ar-wire, ab-27b-ar-wire-q8, queued).
3. Accuracy gates (tools/lab/ar-accuracy): KLD vs exact RCCL reference on a fixed corpus; MTP acceptance equality across A/B arms.
4. Pick the fastest configuration that passes; if f32 host keeps the decode win, prefer it (no tolerance needed).
5. Make it the production default (0860 --allreduce default / recipe) with contract evidence; tune the adaptive threshold (PGC05).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Gates (tools/lab/ar-accuracy/gates.py): mean KLD <= 0.001, p99 KLD <= 0.01, same top token >= 99.5% vs RCCL f32 reference (27B Q8_0, 32 x 2048-token chunks); MTP draft acceptance within 0.5 pp across A/B arms. Throughput: balanced A/B with CI95 excluding zero, no regression on prefill vs RCCL beyond noise.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Yardstick: Q8_0 weight quantization itself is ~0.001-0.003 mean KLD vs full precision; the AllReduce wire should add at most ~10% of that. Upstream b11233 already defaults the internal path to bf16 (GGML_CUDA_AR_BF16_THRESHOLD=1).

2026-09-30 promotion plan AGREED with dev-gpt-agent (req_f76c6c7f145a4fc8 + amendments): adaptive default only for HIP + tensor split + exactly two participating gfx1100 devices; closure 0860 + 1225 + 0840 (0830 dependency removed in 50a2ff57; default scoped in 952b920e); 1272 experimental; 1275 only as a frozen digest after its A/B + stress; N=3 RCCL-only until 1276 qualifies. Gates: tg512/tg2048 gain vs ccl CI95 low > 0; pp1024/pp4096 CI95 low > -0.5%; MTP acceptance within 1.5 pp (exact/exact spread 1.2 pp); closure overhead (old production binary vs closure forced ccl); KLD vs host-f32 reference (mean <= 0.001, p99 <= 0.01, same-top >= 99.5%; pristine RCCL is bf16 >= 32768 elements so not a reference); AR stress; single-GPU/non-HIP unchanged; contract ALLREDUCE-ADAPTIVE-DEFAULT with llama-server preflight. Evidence so far: adaptive vs ccl tg512 +4.1% (CI 3.8..4.5), tg2048 +4.3%, pp -0.1..-0.2%, acceptance 84.28 vs 84.29%. P2P (1252) gives nothing over host staging and does not change RCCL; excluded. Review req_93102df2391e4495: do not promote yet — fix 1225 non-HIP compile safety, 0830 provider_name in explicit RCCL branch, add RCCL admission to the auto-adaptive eligibility test, snapshot switch_bytes per comm context; plus 0860 provider marker missing under llama-server.

2026-09-30 owner: BigCherry is AMD/HIP-only; NVIDIA/CUDA builds are not a consideration. Dropped: 1225 non-HIP compile-safety fix and the 'non-HIP default unchanged' gate. HIP-specific code needs no CUDA fallbacks.

2026-10-01 pp1024 regression found and fixed by threshold. Plain decode (no MTP, ubatch 512), 6 rounds vs RCCL: adaptive 1 MiB pp1024 -4.2%/-3.6%, tg +6.9..+7.1%; adaptive 16 KiB = RCCL on everything; adaptive 64 KiB pp1024 -0.07% [-0.58,+0.60], pp4096 -0.25%, tg512 +6.80%, tg2048 +6.92%. Cause (1277 size trace, ar-size-trace-1): the server splits a 1000-token prompt into 512 + 484 + a 4-token tail; the tail's 128 ARs are 80 KB and took the host path at 1 MiB; decode ARs are 20 KB. Mechanism of the ~50 ms cost not yet explained (128 host calls should cost a few ms). Pending: MTP depth 5 A/B (ab-27b-adaptive-switch-mtp) since MTP verify ARs are ~100-120 KB and move to RCCL at 64 KiB. Review items from req_93102df2391e4495 checked at HEAD: RCCL admission present in 0840 init_hybrid, switch_bytes snapshotted per comm context, 0860 marker WARN-level, 0830 out of the closure. Decode KLD: adaptive-f32 = 0 vs exact f32 (same as RCCL).

2026-10-01 MTP depth 5, ubatch 2048, 6 rounds vs RCCL: adaptive 1 MiB pp1024 -0.20%, pp4096 -0.14%, tg512 +7.98% [7.71,8.21], tg2048 +2.17% [1.95,2.41], acceptance 78.45 vs 78.74% (gate PASS); adaptive 64 KiB tg +0.5% only (MTP verify ARs ~100-120 KB go to RCCL). Decision (owner): promote at the 1 MiB default for the MTP production setup; plain-decode pp1024 tail cost and phase-aware routing tracked in PGC12. Contract ALLREDUCE-ADAPTIVE-DUAL-GFX1100 sessions t-0840-gfx1100-s1..s4 running.

## Change Log

- 2026-09-30T04:03:45.724752+00:00 (created-by): Created by agent
- 2026-09-30T12:21:22.319473+00:00 (updated-by): Updated: section:notes
- 2026-09-30T12:23:40.352140+00:00 (updated-by): Updated: section:notes

## Ledger-events

- chg_20260930_125153_reviewed-and-consolidated-toda_3870
- 2026-09-30T12:52:13.375927+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-01T05:33:24.886015+00:00 (updated-by): Updated: section:notes
- 2026-10-01T06:26:12.665601+00:00 (updated-by): Updated: section:notes

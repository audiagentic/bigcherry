---
id: PWR13
order: 0
plan: run-power-efficiency
state: in_progress
created-at: '2026-11-26T01:45:56.843677+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
priority: P1
---

# Expand independent GPU qualification workloads and fail-closed proof

## Description

Extend existing platform gpu-ai-screen beyond repeated BLAS and sequential memory. Independent coverage only; no engine dependency or duplicate controller/profile/store. PWR12 retains physical research; RCS26–28 own native replay; PWR07/08/RSC02 own admission/recovery; PWR10/PWR02 own production.

## Steps

1. Review owner admission/restore before unattended mutation. 2. Typed cases, diverse CPU oracles and verifier fault checks. 3. Capability-proven instruction, memory, transfer, mixed and transition cases. 4. Managed evidence; observe-only hardware cohort then approved finalist tests.

## Detailed Solution & Technical Design

docs/design/gpu-qualification-code-guide.md sections 1–12.

## Code Samples & Guidance



## Files

tools/bigcherry/hardware/ai_probe.cpp, ai_probe.py, cli/ai_probe_cmds.py, tools/tests/hardware/test_ai_probe.py; existing power owners by coordination.

## Validation

CPU vectors, fault injection, malformed coverage, target compile/ISA proof, leased observe-only cases on supported cards, exact restore and matched finalist receipts.

## Effort & Risk



## Standards



## Acceptance Criteria

Required cases complete with positive verified work; missing/unsupported/thermal/recovery nonpass. No unattended writes until recovery gates pass. No auto promotion; native and serving gates separate.

## Notes

2026-11-26 re-review and first implemented slice: rotating dense/sparse/signed-diagonal exact corpus, guarded output and per-corpus host/device verification, input readback, comparator faults at three positions, native protocol 2 / observation schema 3; old reference and stale binary refused before candidate transaction. ROCm 7.2.4 builds gfx1030/gfx1100/gfx1201, CPU self-test, 197 hardware tests pass (one skipped), 774373 clean GEMMs over four physical cards and 28 full 128 MiB memory passes. Six observe-only managed captures harvested: 20261126T015614Z-gpu-ai-screen-512f35e587ee, 20261126T015751Z-gpu-ai-screen-82c5b9fae077, 20261126T015805Z-gpu-ai-screen-5cb973591371, 20261126T015818Z-gpu-ai-screen-835094a5e951, 20261126T015836Z-gpu-ai-screen-6c7afecb7598, 20261126T015921Z-gpu-ai-screen-4880e38ee7de. Detailed counts in AI_POWER_SCREEN.md. PWR13 stays in progress: varied shapes, direct instruction/wave/LDS, mixed traffic and hot/transient scopes outstanding; P0 owner admission and exact restoration fixes required before unattended mutation. No control or native owner edits.

2026-11-26 direct instruction slice: bc-platform 1.78.0 base, native protocol 3/screen schema 4; independent alu/wave/lds cases, exact CPU readback every launch, changing epochs, output guards/input integrity and strict geometry/count/headroom validation. New cases refuse profile application until owner recovery hardening. Actual code objects on gfx1030/gfx1100/gfx1201 show mul_hi/lo, bit extraction/rotate, ds_bpermute, explicit LDS read/write and barriers; initial volatile LDS emitted generic flat accesses and was corrected. 202 hardware tests passed, one skipped; CPU golden self-test passed. All 24 observe-only sustained/burst cases across four cards passed with 39635648512 verified 32-bit words and unchanged controls; four receipts harvested, IDs in AI_POWER_SCREEN.md. Host-readback-limited coverage, not hot or saturation proof. Remaining varied matrix geometry, native matrix/FP instructions, mixed residency/traffic and phase/transient/thermal work; P0 owner admission and native serving remain separate. No writer or production changes.

2026-11-26 rectangular geometry slice: explicit M,N,K,LDA,LDB,LDC NN layout, three exact corpora and full logical/padding comparison, strict geometry identity, protocol 4/schema 5. Initial missing native geometry serialization was correctly rejected (four harvested inconclusive receipts); serializer fixed and native validation failures now retain per-case diagnostic/output. Final four observe-only captures passed 15 requested cases: 2158560 GEMMs, 2082375360 GPU and 195222690 CPU padding comparisons; eight captures harvested, IDs in AI_POWER_SCREEN.md. CPU golden self-test and 210 hardware tests passed (one skipped), all three offload targets compiled. No power changes; explicit geometry refuses apply pending P0. Remaining mixed residency/traffic, direct matrix/vector instruction and phase/transient coverage; native and production gates separate.

2026-11-26 mixed-residency slice: protocol 5/schema 6, explicit resident budget and full known allocation headroom, independent nonblocking memory stream, 4 MiB changed-chunk exact host checks, initial/final full verification and complete compute-time sweep required. Three offload targets and CPU self-test passed; 213 hardware tests passed, one skipped. Five observe-only cases passed (129 MiB partial-chunk smoke plus 20992/20992/29184/12800 MiB full cohort); 671744 verified GEMMs and 66035122176 resident words across full-size runs, unchanged controls, all captures harvested (IDs in AI_POWER_SCREEN.md). Job-monitor restart interrupted two queued launches, resubmitted only after lease/process check. No actual overlap, hot floor or serving proof. New review confirms distinct gfx1100/gfx1201 WMMA layouts, no native WMMA on gfx1030, historical gfx1030 fdot2 builtin issue. Remaining native matrix/vector/packed-dot, peer traffic, phase/transition/health and P0 owner recovery/matched finalist gates. No control or native owner edits.

2026-11-26 tensor ISA slice: protocol 6/schema 7, vector FP, signed dot4/dot2, nonnegative dot2, FP16/BF16 WMMA signed/nonnegative, mixed IU8 WMMA. Three target code objects and CPU selftest passed; 222 hardware tests passed, one skipped. ISA proof as documented. Seventeen harvested observe-only captures: 70 passing cases, 7 strict mismatch cases, 28578021376 host-verified words in passing cases, all controls unchanged. gfx1100 both cards reproduce signed FP16/BF16 WMMA residue (0x3f7ffffa vs exact 1.0) and small signed dot2 discrepancies; nonnegative and mixed IU8 pass. gfx1201 passes signed and nonnegative; gfx1030 no native WMMA. Strict signed nonpasses retained; no tolerance or factory/production claim. Remaining recipe/transfer/phase/health/owner gates and separately reviewed precision contract.

## Change Log

- 2026-11-26T01:45:56.843677+00:00 (created-by): Created by agent
- 2026-11-26T01:52:35.520482+00:00 (state-transition): State: pending → in_progress
- 2026-11-26T02:00:20.973947+00:00 (updated-by): Updated: section:notes
- 2026-11-26T03:02:56.388933+00:00 (updated-by): Updated: section:notes
- 2026-11-26T03:10:45.923234+00:00 (updated-by): Updated: section:notes
- 2026-11-26T03:21:55.716861+00:00 (updated-by): Updated: section:notes
- 2026-11-26T03:42:09.263099+00:00 (updated-by): Updated: section:notes

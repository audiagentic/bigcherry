---
id: RAD16
order: 16
plan: patching-radiance-rdna3
state: pending
created-at: '2026-10-21T02:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# The RX 6900 XT (gfx1030) on dev: what it can usefully do

## Description

The 6900 XT (RDNA2, 16 GB, no WMMA, no FP8; has v_dot2c_f32_f16, v_dot4c_i32_i8, v_perm) sits idle behind a PCIe 3.0 x4 link (~3.5 GB/s). Our packed-f16 FP8 and W4 GEMVs compile for gfx1030 unchanged (checked 2026-10-21); the moe-remote worker cannot run there today because it loads a radiance kernel plugin (libr4d/libr3) for its grouped GEMM and weight relayout, and none exists for gfx1030.

Assessment (2026-10-21, with GPT, numbers from this week's measurements):
- Third expert worker: 0-8%. Flash-Next's engine already holds only 77 MiB in host memory with two XTX workers, and the worker GEMV is not on the decode critical path (exposed wait ~16 us a layer at 1 stream; a 24% faster worker kernel did not move decode). For the 35B the bottleneck is the engine's own MoE GEMM, which another worker cannot speed.
- Logits or sampling offload: about neutral. A 248320 x 2560 head would take ~2.5 ms on the 6900 against 1.06 ms on the R9700, and full logits cost 0.14-1.7 ms a step on the link.
- MTP drafter offload: the only promising use, +5-20% at 1 stream and 0-12% at 6, and only if drafting overlaps target verification (today the drafter runs 3 serial passes on the R9700, about 0.8 ms of drafter kernels plus its head passes a step). Hidden-state transfer is cheap (2-9 us a step); gfx1030 compute efficiency and the serial draft dependency are the real limits. Needs an engine patch. Reference: llama.cpp's --spec-draft-device; SGLang's overlapping SpecV2.
- Default: nothing, until the experiment below says otherwise.

## Steps

1. Decisive experiment, no engine change: time the MTP draft chain's GEMMs and head on gfx1030 standalone (rad-kbench cannot load a gfx1030 plugin, so a small HIP harness from the worker's kernels), measure a 5-30 KB PCIe round trip to the 6900, and simulate serial vs overlapped drafting with measured acceptance lengths.
2. Go only if the projection is at least +10% end-to-end at 1 stream; then design the engine patch (drafter placement, overlap, reconciliation and rollback).
3. Optional, low value: a gfx1030 worker in GEMV-only mode (no grouped GEMM; relayout from the gfx1100 plugin's host hooks) for models whose experts do not fit the XTX.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Projection from measured kernels and link; if built, matched stream A/B with acceptance unchanged.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Related: RAD15 (R9700 rows), RCS07 (gfx1030 backend), RR03 (superseded plan for a drafter on a separate card).

2026-10-21 DECISION. Radiance's drafter "runs as extra passes after the trunk step" (core/engine_draft.cpp, SERIAL mode for
MTP): a draft depends on the verified step's result, so relocating it to the 6900 cannot overlap verification. Overlap
would need branch-parallel drafting of every possible acceptance length (4 chains x ~4.5 ms = ~18 ms on the 6900 against a
~20 ms step): borderline at 1 stream, infeasible as streams grow, an engine patch, for at most ~16% at 1 stream.
- Current models (Flash-Next, Qwen3.6-35B): the 6900 stays off the decode critical path. Every candidate role is slower
  than the R9700 (logits, sampling), serially dependent (MTP drafting) or already hidden (expert workers).
- Its role: overflow expert worker for models whose cold experts do not fit the two XTX (GLM-5.3 from radiance 1.9.0,
  bf16 or larger containers). Needs a GEMV-only worker mode on gfx1030 (our packed-f16 FP8/W4 GEMVs compile for it; no
  grouped GEMM; weight relayout from the gfx1100 plugin's host hooks). Built when such a model is deployed.
- Branch-parallel drafting: shelved with the numbers above; revisit only if a simulation shows >= 10%.
2026-10-21 measured on the 6900 XT (HIP device 3; harness /tmp/g1030/rt.hip, to move into tools/lab): host round trip
(H2D + kernel + D2H) 38 us at 5 KB, 51 us at 30 KB, 212 us at 300 KB; device read bandwidth 505 GB/s (R9700: ~630).
So a drafter on the 6900 that takes the hidden state and returns draft tokens costs ~50 us a step in transfers (negligible
against a 20-25 ms step). Its draft chain would take ~4.5 ms a step (three MTP passes; the int8 head alone is ~1.2 ms a pass
at 505 GB/s against 1.06 ms on the R9700), well inside a step, so drafting can hide behind verification in time. Serial
relocation gains nothing (the head is slower on the 6900); the win needs pipelined speculation: drafts for step t+1 start
from the state assumed after step t's verification and are discarded on a mismatch. That is the engine change to design
(scheduler + MTP state + rollback), with acceptance as the risk. Next: a trace-driven simulation of pipelined drafting with
measured acceptance (Flash-Next 0.63-0.66 at MTP 3) before any engine patch.

2026-10-21 step 1 decisive experiment run (tools/lab/radiance/rdna2-draft-bench.hip, packed-f16 GEMV via __builtin_amdgcn_fdot2 at Flash-Next's MTP/head shapes, measured on the real 6900 XT, ROCR_VISIBLE_DEVICES=3): lm_head (248320x2560) alone costs 2556.9 us a call -- gfx1030 has no WMMA/FP8 matrix cores, so a GEMV this large at plain packed-dot-product rates is the dominant cost by far. One MTP layer (q/o/kv/gate_up/down, approximate shapes from rad-info --meta) totals 69.8 us; 3 serial draft passes without the head is 209.5 us, but a real draft loop needs the head at least once a pass to sample a token, so the real total is 3*(layer+head) = 7880.3 us. Against the R9700's current full drafter chain (~0.8 ms today per RAD16's own description), that is roughly 10x WORSE, not a marginal miss of the +10% go bar.

Answer to 'determine how to use the RDNA2 card' for MTP drafting: NO. Measured, not projected from guesswork -- gfx1030's lack of matrix cores makes the lm_head alone prohibitive for a per-pass drafter role. This closes step 1/2 of RAD16 with a real result: do not design the engine patch for drafter placement/overlap (step 2 is now moot). Combined with the existing doc's own assessment (third expert worker 0-8%, logits/sampling offload ~neutral), the 6900 XT has no currently-viable compute role for Flash-Next or the 35B's decode/prefill on this hardware. The only remaining option (step 3, 'optional, low value': a GEMV-only worker mode for models whose experts do not fit the XTX) stays exactly that -- optional, low value, not pursued.

2026-10-21 correction to the step-1 result above: the lm_head figure (2556.9 us) used plain f16 weights in the harness, not the real model's int8 lm_head -- double the bytes a bandwidth-bound GEMV actually reads. This plan item already has an independent bandwidth-based estimate from the same day ('the int8 head alone is ~1.2 ms a pass at 505 GB/s'), which the harness's own bandwidth-bound behaviour confirms: 248320*2560 elements at 2 bytes (f16) / 505 GB/s predicts ~2.52 ms, matching the measured 2556.9 us almost exactly, so the harness IS bandwidth-bound, just on the wrong (2x too large) weight size. Corrected total: 3*(69.8us layer + ~1200us int8-equivalent head) ~= 3.8 ms against the R9700's ~0.8 ms -- about 4.75x worse, not the originally reported ~10x. The go/no-go conclusion is unchanged (clear no-go for serial MTP offload to the 6900), only the stated magnitude was overstated; correcting the record rather than leaving the inflated number standing.

## Change Log

- 2026-10-21T02:00:00+00:00 (created-by): Created by agent
- 2026-10-21T04:06:29.285042+00:00 (updated-by): Updated: section:notes
- 2026-10-21T04:07:06.334581+00:00 (updated-by): Updated: section:notes

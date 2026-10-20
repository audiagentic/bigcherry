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

# The RX 6900 XT (gfx1030) on Brutus: what it can usefully do

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

## Validation

Projection from measured kernels and link; if built, matched stream A/B with acceptance unchanged.

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

## Change Log

- 2026-10-21T02:00:00+00:00 (created-by): Created by agent

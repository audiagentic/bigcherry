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

## Change Log

- 2026-10-21T02:00:00+00:00 (created-by): Created by agent

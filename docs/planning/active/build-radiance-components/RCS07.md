---
id: RCS07
order: 7
plan: build-radiance-components
state: pending
created-at: '2026-10-18T12:57:54.768641+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P3
work: L
---

# RDNA2 (gfx1030) MoE backend: capability probe and kernel plan

## Description

RDNA2 has no WMMA and no FP8, so the gfx1100 FP16/AHALF paths do not carry over. Write the gfx1030 backend against moe_backend.h: weights stay in the same W4 layout; activations quantised to int8 (not FP8) with dot4/packed-int8 or plain FMA accumulation; wave64/32 and LDS size from the capability bits. Start with the replay harness (correctness vs the CPU reference, per-kernel timing) before any model load.

## Steps

1. Confirm an RDNA2 device is available on a host (none known on dev); otherwise stop at the design.
2. Capability probe and backend skeleton that returns a clear unsupported error per lane.
3. Small GEMV lane first (decode), then the grouped lane.
4. Per-arch settings and evidence: nothing validated on gfx1100 is claimed for gfx1030.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Replay agreement with the CPU reference on real captures; then lightweight-tier evidence on the new arch.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Blocked on hardware and RCS05. Activation format differs (int8 vs FP8), so captures need a version bump or a per-backend activation plane.

Updated 2026-10-19: hardware exists, the RX 6900 XT (gfx1030, 16 GB) on dev, so this is ready once RCS05 lands. Reference data: Strata (Niko1221/Strata, HIP backend for gfx1031/gfx1201) reports an RX 6700 XT 12 GB serving Flash-Next IQ3_XXS hybrid at 35-38 tok/s decode and 640-700 tok/s prefill; its RDNA2 path uses the signed integer dot instruction for quantised kernels and plain packed FP16/SGEMM for prefill (320 -> 690 tok/s); its PR #1897 K-quant MMQ prompt kernels take RDNA2 prefill 633 -> 1,036 tok/s. Kernel prior art, numbers unverified (GPT deep dive requested). The 6900 XT is tier L3 in the tiered expert cache (RHP07). First step stays the isolated feasibility test: capability probe plus replay of captured requests against the CPU reference, before any model load.

Link note (2026-10-19): the 6900 XT is behind a PCIe 3.0 x4 hop (about 3.5 GB/s to host memory) though its own link is x16; it suits permanently resident experts, not staging. The topology descriptor and swap cache are in RHP07.

## Change Log

- 2026-10-18T12:57:54.768641+00:00 (created-by): Created by agent

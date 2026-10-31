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

## Progress 2026-10-31 (RCS07 step 1-3 first cut)

- A device is available: GPU3, RX 6900 XT (gfx1030, 16 GiB, PCIe 3.0 x4). Measured with ROCm 7.2.4: v_dot2c_f32_f16 21 T MAC/s, v_dot4c_i32_i8 22 T MAC/s (no throughput reason to requantise activations to int8), device copy 421 GB/s.
- moe_backend.h: gfx1030_backend (W4NL64 only, no WMMA required) and backend_for(arch); the helper admits gfx1030 and loads libr3 on it; --describe takes an ARCH.
- r3_compat.h: r3_mma16_f16 / r3_mma16_bf16 emulate the 16x16x16 tile on gfx1030 (8 dwords of each row by ds_bpermute, f32 FMA); on gfx11 they are the WMMA builtins. Taking halves from a v16h fragment with a vector bit_cast gave wrong tiles on gfx1030; the dword form is checked.
- moe_lane_reduce.h: the store drain takes the gfx11 spelling on gfx1030.
- libr3 for gfx1030 builds with R3_NATIVE=OFF and R3_ONLY="r4d_moe;r4d_fused_quant_fp8;r4d_quant_act_fp8"; the integer-WMMA GEMM and attention units do not build and the helper does not need them.
- Checked against the host reference with r3_moe_worker --compare (Flash-Next, layer 0, 12 GiB budget): 8 and 64 rows bit-identical to a 7900 XTX; 256 rows rel_l2 1.1e-3.
- Kernel time (rocprofv3, layer 0): decode-sized lane at 64 rows gate-up 3.83 ms vs 3.03 ms, down 1.30 vs 1.31 ms (6900 XT vs XTX); grouped GEMM at 256 rows 18.3 ms vs 4.1 ms (emulated tile, about 4.5x slower).
- Not done: a whole-model run, FP8 block-scaled experts, a faster prefill tile (the emulation uses 64 cross-lane exchanges a tile), and a third helper in the engine, which needs RAD23 (kMaxRanks = 2 is both the engine-rank axis and the helper bound).


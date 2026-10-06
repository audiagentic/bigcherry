---
id: FKE01
order: 1
plan: patching-fused-kernel-equivalence
state: pending
created-at: '2026-10-06T15:29:25.180488+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# Validate every fused GPU kernel against its unfused path (bit-exactness, fidelity, speed)

## Description

Fused launches are on by default in the production build, and which candidates fuse depends on real memory addresses (ggml_cuda_check_fusion_memory_ranges refuses a fusion whose output overlaps a live input). MSM02 showed the consequence: changing only the buffer layout changed about 45 of ~19,000 fusion decisions and the greedy text, while GGML_CUDA_DISABLE_FUSION=1 made the two layouts identical. So at least one fused kernel is not bit-identical to the unfused sequence it replaces, and the output of production depends on allocator layout. We do not know which fusions are bit-exact, how far the inexact ones are from the unfused path and from the CPU f32 reference, or whether each one is actually faster.

## Steps

1. Whole-set measurement on production (Flash-Next, 3-GPU tensor split): fusion on vs GGML_CUDA_DISABLE_FUSION=1, ABBA prefill / decode at 8K and 98K, probes of each arm against the other and against the CPU f32 reference (which arm is closer to the reference). 2. Inventory: list every fusion in the production tree (upstream ggml_cuda_try_fuse / can_fuse paths and our patches 1205, 1206, 1237, 1241, 1274, 1309-1313 and later) with its off switch, or add a per-fusion mask (one env bitmask) where none exists. 3. Per fusion: an op-level test in the style of tests/test-mul-mat-id-range.cpp - same inputs through the fused launch and through the unfused node sequence on each GPU arch (gfx1100, gfx1201, gfx1030): bit-exact yes/no, max abs / rel difference, both against the CPU reference. 4. Per fusion on hardware: speed with only that fusion off (ABBA). 5. Classify: bit-exact (keep), inexact but closer to or as close as unfused to the reference and faster (keep, record the tolerance), inexact and no faster or further from the reference (fix or remove). 6. Make fusion selection independent of buffer layout where cheap (e.g. decide from graph structure and let the kernel handle overlap, or make the allocator keep fusion outputs off live inputs), so output no longer depends on allocator decisions.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

vendor ggml-cuda/ggml-cuda.cu (fusion predicates, read), patches/12xx-13xx fusion packages, patches/1339_meta_memory_report (fusion_overlap counters), tools/lab/flash-next/queue-env-ab.sh + flash-fidelity.sh, new tests/test-fused-equivalence.cpp via a patch package

## Validation

A table per fusion: bit-exact or max difference, distance to the CPU reference with and without it, speed delta, decision. Production output identical across two buffer layouts (common arena vs 1340) once inexact fusions are fixed or selection is layout-independent.

## Effort & Risk

Low risk (measurement and tests first). The op-level harness is the main work; per-fusion off switches may need small edits inside the owning patches.

## Standards



## Acceptance Criteria



## Notes

Owner question 2026-10-07: 'Should we validate the fused kernels against unfused'. Does not invalidate the promoted fusion patches' proofs (their evidence was taken with the path on); this adds an equivalence check that was never part of those proofs. Unblocks an identical-output gate for memory-layout changes (MSM02).

## Change Log

- 2026-10-06T15:29:25.180488+00:00 (created-by): Created by agent

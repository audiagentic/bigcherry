---
id: TRVP16
order: 0
plan: tuning-rocm-vulkan-provider
state: pending
created-at: '2026-10-02T12:35:11.094367+00:00'
breadth: ''
skill: ''
created-by: agent
priority: P3
work: L
---

# Vulkan collective and mixed-architecture qualification

## Description

Qualify Vulkan AllReduce provider thresholds and homogeneous-Vulkan mixed-RDNA roles. HIP + Vulkan inside one meta backend is not a supported topology.

## Steps

1. Compare stock meta fallback vs PRVP03 provider by reduction byte size and device count.
2. All-Vulkan XTX+R9700 and XTX+6900 only after homogeneous dual-XTX correctness.
3. 6900 as a Vulkan draft device only if the draft-device selector resolves it under Vulkan.
4. No expert-specific placement unless upstream exposes a precise expert-device primitive.
5. No HIP target plus Vulkan draft/expert backend in one meta communicator.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Correctness first; statistically supported end-to-end gain required for promotion; changed ICD/device/topology invalidates tuning evidence.

## Effort & Risk

L / high.

## Standards



## Acceptance Criteria



## Notes

## Change Log

- 2026-10-02T12:35:11.094367+00:00 (created-by): Created by agent

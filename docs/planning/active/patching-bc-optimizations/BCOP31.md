---
id: BCOP31
order: 31
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-05T21:45:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: S
---

# Track native AMD backend qualification

## Description

Action/disposition ledger for evaluating AMD-native backend mechanisms without creating a parallel BigCherry runtime. `build-rocm-vulkan-provider` (BRVP) owns backend/provider integration. Existing HIP kernel plans own kernel-specific ports. RPL01 may consume measured backend costs but does not own backend implementation.

## Actions

1. BRVP inventories current AMD-native/HRX-style candidates and classifies support for gfx1100/gfx1201 and BigCherry-required operations.
2. For a credible candidate, run the smallest equal-work correctness/performance comparison against the promoted HIP/Vulkan path. Separate kernel-only wins from backend-wide wins.
3. Route kernel-only mechanisms to their existing HIP/attention/MoE owner rather than introducing a new backend.
4. Create/extend a BRVP technical item only if a backend-wide capability survives the cheap qualification gate.
5. Record one terminal disposition here: `kernel-only`, `BRVP candidate`, `wait-upstream`, or `rejected`.

## Gate

Do not create a new loader, scheduler, allocator, placement policy or backend fork from BCOP31. Backend-wide work requires correctness parity plus either >=5% end-to-end gain in a material workload or a required capability unavailable through existing providers without >2% regression elsewhere.

## Related

BRVP01-03; RPL01; existing HIP/autotune/kernel owners.

---
id: BCOP56
order: 56
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-08T01:16:37.463Z'
created-by: agent
priority: P2
---

# PRBE48 HRX external-backend eligibility disposition

## Discovery/change

The old RD57/PRBE48 plan assumed HRX source could not yet be pinned and proposed a new BigCherry backend build/benchmark harness. Current AMD staging already supplies a pinned source pair and its own benchmark tooling. The staging `find_dispatch_registry` supports gfx1100/gfx1151 but not gfx1201/gfx1030, and the model manifest marks Qwen3.5-35B-A3B-MTP PPL as expected failure. Compile target availability does not prove graph execution eligibility.

## Authoritative owner(s)

PRBE48 is the single BigCherry HRX eligibility/evidence owner; AMD HRX owns its implementation, JIT, scheduler, buffers and graph replay. Existing HIP/Vulkan, MTP/MoE and qualification owners remain unchanged.

## Subsequent work already acting on it

Upstream draft PR #27218 remains unmerged; ROCm/ggml-staging-automation pins AMD-Ecosystem/llama.cpp `02f2880c9f422dd1df9db11e3a0bb442bc5f11f5` and ROCm/hrx-system `c0b135a778cc1e133001ee77a5a3f74536b3f483`, and already contains build, PPL and throughput benchmark runners. No independent BigCherry PRBE48 implementation/plan activity was found after September 24.

## Unresolved action / terminal gate

No BigCherry HRX patch, harness, GPU queue or broad comparison campaign. First rerun PRBE48 Gate 0 on a newer exact integration pin: architecture dispatch, production-model op coverage and no known XFAIL. Only then one disposable gfx1100 correctness smoke; only after that four-session matched HIP/Vulkan E2E. Fail closed as `no-campaign` or `external-smoke-rejected` on unsupported device, failed PPL/greedy/multi-ubatch/long-context, or ambiguous offload. External evaluation only after CI95-low >=3% E2E vs faster baseline and <=1% controls. Multi-GPU/MTP/Flash-Next require distinct upstream capability, not inferred support.

## Dependencies / blockers

gfx1201 and gfx1030 have no registered HRX dispatch in the inspected staging pin. MTP model has a known PPL XFAIL. HRX remains an external, experimental backend and the staging repo warns it is temporary. Static fixture: 8/8 assertions passed. No hardware/build/benchmark this run.

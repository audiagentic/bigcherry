---
id: BCOP26
order: 26
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-05T08:14:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Qualify user-code MoE cache integration seam

## Description

Action ledger for the 2026-10-05 MET01 optimisation audit. Technical authority remains MET01. Upstream llama.cpp #29943 moves selective host-expert copying out of generic scheduler internals into `ggml_backend_sched_copy_callback`; #29887 is intended to become user-code-only on that seam. BigCherry should qualify that boundary on HIP before carrying any private scheduler-core MoE cache implementation.

## Actions

1. Check the current llama.cpp pin for #29943. If absent, use only its minimal callback/refactor delta as the qualification vehicle; do not add MET policy to `ggml/src/ggml-backend.cpp`.
2. Build an observation-only callback mock that counts host-weight callback invocations/bytes and always returns false. Prove greedy/logit identity with callback disabled and verify non-weight inputs are available before host-weight callbacks.
3. Enable upstream selective expert copying/#29887 policy only after the transparent-control test passes. Record selected experts, copied bytes, cache hits/misses and target backend so silent fallback is impossible.
4. Run gfx1100 and gfx1201 correctness/integrity lanes, including multi-request, scheduler-copy reuse and MTP. Then run equal-VRAM ABBA for whole-layer, LRU, static-hot and hybrid policy; keep pp512/2048/8192 separate from <=32-token decode.
5. Dispose the qualification as `adopt upstream/user-code`, `wait for upstream`, or `reject on AMD`. Do not leave a second cache/scheduler implementation behind.

## Related

MET01 is authoritative policy/accounting owner. MET05/1328 owns auxiliary 6900 execution/staging. MET06 remains blocked behind its measured placement-gap gate. QFP28 owns Flash-Next performance-integrity qualification.

Upstream: llama.cpp #29943 and #29887.

## Acceptance Criteria

- The copy-callback seam is classified for the current pin and HIP backends.
- Observation-only callback is semantically transparent before cache testing.
- Any cache result proves nonzero expert-copy/cache activity and passes multi-request correctness/integrity gates.
- AMD promotion is based on equal-VRAM end-to-end results, not NVIDIA hit-rate numbers.
- No duplicate scheduler cache, residency solver, auxiliary transport or loader policy is introduced.

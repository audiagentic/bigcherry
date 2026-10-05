---
id: BCOP25
order: 25
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-05T07:24:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: S
---

# Execute bounded D=72 HIP Flash-Attention root-cause qualification

## Description

Action ledger for the PHA08 implementation audit. PHA08 is the technical authority. This item exists only to ensure the b11402 qualification is executed and disposed without creating a duplicate FA dispatch or leaving a speculative diagnostic patch behind.

The audit updated PHA08 from the historical `050439614` baseline to current b11402, supplied a bounded helper-level tail-safe diagnostic prototype, added byte-range telemetry/stop gates, and classified fresh upstream #29435 (`d89651a7`) correctly: its whole-tile scheduling change is gated to NVIDIA DGX Spark two-stage async-KV FA and is not an AMD D=72 fix.

## Actions

1. Execute PHA08's b11402 reproduction matrix on gfx1100 before authoring a permanent patch.
2. If reproducible, implement the temporary tail-safe helper guard plus byte-range telemetry exactly as a diagnostic; build the normal HIP targets and run D=64/72/80 controls.
3. Remove the diagnostic if it does not change the fault or telemetry proves tail accesses are in range; proceed only to PHA08's workspace/padded-sequence isolation.
4. If the guard proves the defect, replace it with the smallest compile-time helper fix and run the 20x high-resolution VLM stress/parity gate.
5. If b11402 does not reproduce, record the stress evidence and close PHA08/BCOP25 without creating a patch.
6. Do not port #29435's DGX-Spark condition to AMD. Any later AMD whole-tile/Stream-K experiment requires profiling evidence and belongs to HIP-autotune policy, not PHA08.

## Authoritative owner

`docs/planning/active/patching-hip-autotune/PHA08.md`.

Likely implementation touch points are `ggml/src/ggml-cuda/fattn-tile.cuh` and its D-tail vector helper; `tools/mtmd/clip.cpp` is fallback-only. No new dispatch registry is permitted.

## Acceptance Criteria

- b11402 D=72 reproduction/non-reproduction is documented on actual gfx1100 hardware.
- If reproduced, diagnostic build/test distinguishes tail-vector addressing from workspace/padded-sequence addressing.
- Any permanent fix passes PHA08's 20x 2048/2560px correctness/parity gate and healthy D=64/80 non-regression gate.
- Diagnostic code is removed if disproven; no unfinished experimental mechanism is handed downstream.
- Item closes with one disposition: fixed, narrow fallback retained, or current-pin non-reproduction.

## Related

PHA08; llama.cpp #28608/#28664; upstream #29435 / commit `d89651a7`.

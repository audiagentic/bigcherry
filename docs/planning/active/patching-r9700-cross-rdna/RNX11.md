---
id: RNX11
order: 11
plan: patching-r9700-cross-rdna
state: pending
created-at: '2026-10-03T01:33:52.132134+00:00'
breadth: ''
skill: advanced
created-by: codex
work: L
priority: P1
---

# R9X11 — Custom AllReduce fixed-grid, sequence-counter, and graph-replay hardening

## Description

Audit and harden custom collectives whose scratch/epoch state can be reused across calls of different sizes or graph replays. The invariant is fixed maximum grid plus sequence advancement by every block on every call, with inactive blocks doing no payload work.

## Steps

- Map r9k fixed-grid protocol and race reproducer to BigCherry 1252/1291 provider sections.
- Record generation source, graph-capture behavior, scratch parity, size-dependent grid, inactive block progression, wraparound comparison, and stream/system ordering for each provider.
- Add a lab reproducer under tools/lab/rccl alternating 10 KB/3.3 MB/40 KB/6.5 MB and dynamic serving-like joins; run RCCL, cpu-root, adaptive/internal, and P2P with graph replay where supported.
- Fix only affected providers with fixed-grid advancement or an equally proven invariant; add debug trace/assertions without production host sync.
- Preserve 1291 device-derived small-path generation and audit the large path separately.

## Detailed Solution & Technical Design

Before any implementation, require a graph-safe collective protocol: a device-derived monotonically advancing sequence on every replay, no host-generated progress in captured kernel arguments, fixed rank participation, exact zeros for uncomputed PARTIAL outputs matching the 1291 lesson, fixed-grid behavior that does not depend on host progress, and specified teardown/re-entry/wraparound semantics. Acceptance must be an in-model prefill using the 96-AR pattern; standalone latency remains diagnostic only.

## Code Samples & Guidance



## Files

- kernels/r9k_ar.hip
- r9700_vllm/comm/r9k_ar.py
- tests/test_ar_race.py
- ggml/src/ggml-cuda/ggml-cuda.cu
- tools/lab/rccl/

## Validation

Thousands of alternating-size calls, wraparound-near counters, 2/3/4 ranks, dynamic joins/leaves, graph capture/replay, restart, mixed gfx11+gfx12, exact CPU/RCCL comparison after every call, and sanitizer/debug checks.

## Effort & Risk



## Standards

Do not impose fixed-grid changes on RCCL; preserve provider-specific ownership and fail closed on ambiguous evidence.

## Acceptance Criteria

- Complete only when an affected provider fails pre-fix and passes post-fix, or every custom provider is proven not vulnerable and the regression test is retained.
- Promotion of RNX07 is blocked until this item is resolved for the relevant protocol.

## Notes

Original source alias is R9X11. Proposed slot 1311. Current Flash-Next evidence identifies 1291 large path as not graph-safe and default-off; this is an open audit input.

Verbatim legacy source retained during R9X→RNX migration:

# R9X11 — Custom AllReduce fixed-grid, sequence-counter, and graph-replay hardening

Status: planned / correctness-first
Proposed patch: `1311_r9x_collective_fixed_grid_sequence`
Depends on: R9X01; must be reviewed before promoting R9X07 or any custom provider with per-block epochs/double buffering
External source: `kernels/r9k_ar.hip`, `r9700_vllm/comm/r9k_ar.py`, `tests/test_ar_race.py`
Existing BigCherry owners: `1252_nro03_allreduce_p2p_provider`, `1275_ar_small_latency`, `1276_ar_adaptive_nway`, `1277_ar_size_trace`, `1291_ar_cpu_root`; related graph correctness: 1233/1216



GPT design pass (req_83e7cdc000be4b9e): no 1311 package now. The 1291 large CPU-root path uses a host-side generation counter and is not graph-capture safe as-is; audit provider-by-provider and resolve RNX07 transport ownership before proposing a successor.

## Goal

Audit and harden every BigCherry custom collective whose scratch buffer or epoch/sequence state can be reused across calls of different sizes or graph replays. The motivating r9700 bug is protocol-level and portable across RDNA generations: changing active block count between calls can make different blocks advance different sequence parity and select different halves of double-buffered scratch.

## Source behavior to reproduce as a regression test

`r9700_vllm/comm/r9k_ar.py` documents the fix: every collective launches a fixed maximum grid and **all blocks advance their sequence counter on every call**, while blocks outside the payload do no data work. `tests/test_ar_race.py` reproduces serving corruption when a prompt joins running decode sequences. `kernels/r9k_ar.hip` is the device-side protocol reference.

## Exact BigCherry mapping

Audit `ggml/src/ggml-cuda/ggml-cuda.cu` sections inserted/modified by 1252 and 1291. For each custom provider record:
- where generation/epoch is produced (host or device);
- whether graph capture freezes a host-side value;
- scratch buffer parity/slot selection;
- launch grid as a function of message size;
- whether inactive ranks/blocks still advance protocol state;
- wraparound comparison semantics;
- stream/system-scope ordering.

1291 already deliberately derives its small-path generation on device for graph replay; preserve that design and audit its large-path generation/ring separately. Do not impose a fixed-grid change on RCCL, which owns its own protocol.

## Implementation

1. Add a lab reproducer under the existing `tools/lab/rccl/` area: alternate 10 KB/3.3 MB/40 KB/6.5 MB calls, then dynamic serving-like sequences where a large prefill joins repeated decode reductions.
2. Run it against RCCL, cpu-root, adaptive/internal and P2P providers, with graph capture/replay where supported.
3. If any BigCherry provider has per-block sequence state and size-dependent grid, convert to fixed-grid counter advancement or an equivalently proven invariant. Payload work remains bounded to active data blocks.
4. Verify uncomputed meta ranks still contribute zero and participate in protocol progression exactly as required.
5. Add trace counters/assertions in debug/evaluation mode for epoch divergence; no per-call host synchronization in production.

## RDNA adaptation

This is `common` protocol correctness. gfx12/gfx11/gfx103x must obey the same sequence invariant; only system-scope atomic primitives/launch tuning may differ. Mixed-generation ranks are mandatory because protocol agreement matters more than local ISA.

## Validation/acceptance

Stress thousands of alternating-size calls, wraparound-near counters where practical, 2/3/4 ranks, dynamic request joins/leaves, graph capture/replay, context/server restart, and mixed gfx11+gfx12. Compare outputs to RCCL/exact CPU reduction after every call, not only final generation. Run under sanitizer/debug checks where supported.

1311 completes only when either (a) an affected provider is fixed and the reproducer fails pre-fix/passes post-fix, or (b) every custom provider is proven not vulnerable and the regression test is retained as evidence. Promotion of R9X07 requires this item resolved.

## Change Log

- 2026-10-03T01:33:52.132134+00:00 (created-by): Created by codex
- 2026-10-03T01:39:34.483874+00:00 (updated-by): Updated: section:notes
- 2026-10-03T02:18:47.155191+00:00 (updated-by): Updated: section:detailed_solution, section:notes

## Reviews

- RV4210

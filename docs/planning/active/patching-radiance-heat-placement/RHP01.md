---
id: RHP01
order: 1
plan: patching-radiance-heat-placement
state: pending
created-at: '2026-10-15T01:35:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: M
---

# Prove complete asynchronous Radiance routing evidence under long prefill

## Description

Radiance already asynchronously records each MoE router's `expert_count`, but `core/engine.cpp::tiering_dispatch` explicitly reports that ~2% of layer histograms were credited in one 2,048-token long-prefill experiment because the CPU ran ahead of GPU completion. Determine whether the uncredited reports are delayed then consumed or overwritten in the bank ring. Do **not** add a second GPU counter or force stream waits to make heat fresh. Preserve existing `HeatEngine` accounting and semantics.

## Steps

1. Pin baseline, inspect `core/runtime/ctx.cpp::{route_report_impl,route_flush,next_histogram,mark_credited,last_histogram_reports}`, `core/runtime/ctx.h` route ring and `core/engine.cpp::tiering_dispatch`; map bank lifecycle across trunk, MTP draft and mixed steps. Inspect `HeatEngine::enabled` for all-VRAM/remote-unplaced models; distinguish heat disabled from histogram absent.
2. Add **diagnostics to the existing producer/consumer**, not another route collector: reports produced, completed, credited, overwritten, still in flight, weighted pass count, oldest-uncredited age, phase (prefill/decode/mixed), source step. No read of unlanded data or false credit on duplicate/recycled banks.
3. Reproduce 2K, 4K and 8K prefill / 64-token verification / decode, with host enqueue ahead, asynchronous DMA late, ring wrap and shutdown; bound overrun. Test event/query ordering; do not block compute/host per layer.
4. If samples are lost, implement a bounded lossless-or-accounted scheme at the existing `route_flush`/drain seam: retired-bank staging queue, later draining, overflow counted and explicit backpressure **only if necessary and measured**. Do not retain every token/route forever. Preserve original order and `reports` weight for HeatEngine decay. Define shutdown drain without indefinite GPU wait.
5. Validate a synthetic oracle using exact per-layer counts for prefill/decode/MTP against the existing BigCherry diagnostic histogram on Brutus. Record coverage rather than claim always complete if capacity is insufficient.

## Detailed Solution & Technical Design

The GPU already produces per-layer integer `expert_count` and `route_flush()` transfers banked contiguous rows to pinned host storage. Host later calls `Ctx::next_histogram(...)` and `HeatEngine::observe(..., reports)`. A 30-minute ownership policy tolerates stale heat; exact one-time accounting and bounded memory are more important than freshness. Run the old diagnostic `R3_MOE_HELPER_HISTOGRAM` only as a **temporary parity oracle**. Test that a bank cannot be overwritten while logically uncredited without an explicit loss counter; the correct fix may be retirement accounting, a larger ring or a deferred queue, determined from measurement.

## Files

Pinned upstream `core/runtime/ctx.{h,cpp}`, `core/engine{,_priv}.h/.cpp`, `core/place/heat.{h,cpp}`; radiance patch package to be allocated under `engines/radiance/patches/`; `tools/tests/patch/test_radiance_patch_packages.py`; `tools/lab/radiance/` diagnostics.

## Validation

Card-free deterministic tests for same-step multi-pass, missed/late events, ring overwrite, output ordering, reconnect/restart, all-VRAM static mode, deterministic mode and overflow counters. Hardware: no new per-router H2D/D2H; no mandatory device/stream synchronisation; histogram coverage and lag by phase measured; existing output/KLD unchanged; A/B throughput no meaningful regression.

## Effort & Risk

Risk: a lossless drain may require bounded backpressure under unlimited CPU run-ahead. Explicitly account for unavoidable losses; never make an unbounded queue or silently block decode.

## Standards

- Source of truth: pinned standalone Radiance 1.3.0 `89cee7ce414486c717aa2085c99e1f522e4c0ea6`; BigCherry `engines/radiance/`. Engine edits MUST be packaged as `engines/radiance/patches/radNNNN_*/{patch.py,patch.toml,SUMMARY.md}` with code-line anchors (patcher strips comments), tests with pinned blobs; helper policy lives in `libr3moe`. Re-check patch IDs before allocation.
- Follow `engines/radiance/recipes.toml` and `components/libr3/component.toml`; runtime profiles set ONLY non-defaults. Use existing BigCherry quality/ABBA/campaign/evidence facilities; no parallel scheduler, telemetry backend, loader, or compatibility alias. Disabled behaviour byte/quality/performance equivalent.
- Related active work: `RAD02` (R9700 trunk + 2 XTX helpers), `RAD03` (GPU mailbox), `RAD06` (prefill MoE), `RAD09` (merge/reconcile helper and rad0002). Integration depends on verified `RAD09` baseline. Do not overwrite another agent's branch or modify a shared patch without merging latest main.

## Acceptance Criteria

- Source and fate of >98% uncredited prefill layer reports diagnosed with timestamps/counters; sample fate classified as pending/credited/dropped, not guessed.
- Histogram seen at most once, in original order with correct `reports` weights; end-of-run accounting reconciles produced = credited + dropped + pending.
- Representative 7K and 32K prefill reaches >=99% **completed-report coverage** after final drain or documents a measured justified bounded loss that RHP04 must gate on; no GPU sync/quality regression.

## Notes

P0 prerequisite for reusing Radiance heat as the remote ownership oracle. RHP02 must not remove the BigCherry collection diagnostic until this gate passes.

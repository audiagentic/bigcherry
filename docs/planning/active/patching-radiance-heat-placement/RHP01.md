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

The GPU already produces per-layer integer `expert_count` and `route_flush()` transfers banked contiguous rows to pinned host storage. Host later calls `Ctx::next_histogram(...)` and `HeatEngine::observe(..., reports)`. A 30-minute ownership policy tolerates stale heat; exact one-time accounting and bounded memory are more important than freshness. Run the old diagnostic `R3_MOE_WORKER_HISTOGRAM` only as a **temporary parity oracle**. Test that a bank cannot be overwritten while logically uncredited without an explicit loss counter; the correct fix may be retirement accounting, a larger ring or a deferred queue, determined from measurement.

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

## Implementation-grade notes (2026-10-15)

### Actual pinned call path, ownership and ordering

| Stage | Existing function | Required invariant |
|---|---|---|
| Router produces counts | `Ctx::route_report_impl(layer, RadRouting*)` | Count array indexes are model-global expert IDs, not compact runtime slots |
| Device staging | `route_dev_` | Row complete before asynchronous copy/flush event |
| Host staging | `Ctx::route_flush()` | Copy contiguous reported layers once per pass; event records end-of-copy |
| Discovery | `Ctx::next_histogram(layer, after_step, before_step, landed)` | Only completed banks strictly older than issuing step |
| Consumption | `Engine::tiering_dispatch` -> `HeatEngine::observe` | Original temporal order, exactly one credit, `last_histogram_reports` weight |
| Reuse | `Ctx::mark_credited` and bank reclamation | Old-generation event must not masquerade as new slot |

Do not assume a single layer row is a single router pass: MTP can report repeatedly during one step, and the engine's actual `reports[bank]` captures multiplicity. Do not mistake TP mirrored routes for separate traffic. Use rank-local accounting during collection, then define aggregation with an explicit rank selector or shard-weight contract.

### Concrete telemetry contract

Emit monotonic aggregates, not a log entry per request: `issued_histograms`, `completed_histograms`, `credited_histograms`, `overwritten_histograms`, `stale_event_rejected`, `pending_banks`, `credit_delay_steps_p50/p95`, `buffer_occupancy_high_water`, `prefill_coverage`, `decode_coverage`. A report is **covered** only if its full device histogram landed, the bank was not overwritten, and its multiplicity was applied; distinguish `issued`, `completed` and `credited`.

Reconcile `issued = credited + dropped + pending` after all known GPU events settle (when source semantics support count-per-step). `completed` is a status subset, not an extra addend. Final-drain budget and shutdown timeout must be explicit. If a report is lost, record step/layer/rank/phase and count, but do not silently fill a missing observation with zeros.

### Fix selection decision tree

1. Instrument and reproduce; compare outstanding host issue depth against device completion delay and bank count.
2. If banks are merely late, consume them in order when ready; prove eventual coverage after a finite run.
3. If banks overwrite before drain, evaluate draining completed banks more often, then increasing bank depth; use bounded ring + overflow accounting.
4. Only if data loss remains performance-significant, consider deferred recycling or bounded producer backpressure at a **safe boundary**, with separate latency A/B. Never put a blocking per-layer copy or `hipDeviceSynchronize` in the routing path.
5. The shared RHP02 consumer must reject heat windows below coverage minimum, not pretend interpolation is ground truth.

### Offline fixtures / probes

Use `tools/lab/radiance/heat-placement/heat_mock.mjs::Collector`: delayed completion; out-of-order completion; ring wrap before completion; already-completed bank overwritten; same-step duplicates; steps marked complete only after the source step is finished; shutdown accounting. Extend with variable 1/8/64/1024/4096 rows and dynamic bank size; record observed counts independent from calculated heat.

### Hardware scripts, measurement and exit gate

Capture pinned SHA, patchset, model digest, run ID and exact `--max-num-batched-tokens`. A/B disabled/instrumented, then long prefill 2,048/4,096 chunks, 32K prompt, mixed and MTP 64-row verification. Report per-phase `issued/credited/dropped/pending` and coverage **after settling**, not during run-ahead; compare old temporary diagnostic GPU histogram using same rank/step window. Exit only after no correctness regression and measured scheduling overhead within noise. If coverage still falls below 99%, RHP02 may land a hook but RHP04 automatic proposals must remain disabled.

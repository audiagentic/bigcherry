---
id: RHP02
order: 2
plan: patching-radiance-heat-placement
state: pending
created-at: '2026-10-15T01:35:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: L
---

# Single Radiance heat observation feed to libr3moe; remove duplicate route collection

## Description

Expose a **read-only, one-to-many** observation from Radiance's existing histogram consumption, not a parallel GPU-side aggregation. Feed both the current `HeatEngine` and a separate cross-GPU ownership *policy consumer* from the same source events. Remove production dependence on `R3_MOE_WORKER_HISTOGRAM`/`R3_MOE_WORKER_HEAT` only after matching coverage and ranking; avoid compatibility aliases.

## Steps

1. After RHP01, define a narrow versioned `abi/rad_moe_heat.h` **optional hook** in a new Radiance patch package (or extend rad0002 if consistent with ownership ABI): immutable {model-global layer/expert domain, rank, step/epoch, phase or step-kind, counts, count length, multiplicity, completeness/overrun markers}; document buffer lifetime (valid during callback only). No pointers or blocking across processes.
2. Publish from the existing CPU `tiering_dispatch` completed-bank loop exactly once per credited histogram; do not dispatch callbacks from `rad_route_report`, alter router math, copy counts again from GPU, or wait for future samples. Keep the existing `HeatEngine::observe` order and `last_histogram_reports`.
3. Make observation work if `HeatEngine::enabled()==false` (all resident or helper-owned units `Tier::Mapped`) without turning **placement** on; `--deterministic` suppresses adaptation but may permit explicit read-only diagnostics. Avoid observing twice if multiple ranks share equivalent global counts; define rank selection/aggregation and MTP protected-layer semantics.
4. Keep callback constant-time and bounded: queue or accumulate small immutable host snapshots out of the inference dispatch loop; no filesystem I/O, sorting, mutex contention with helpers, RPC, CPU callback on GPU critical stream or extra copy. Document queue full/restart policy and coverage counters. Consumer may use the same score formula or phase-specific window, but must not fork the *collection path*.
5. Wire `libr3moe` ownership policy to the shared observations, compare its (layer,expert) counts and ordering against current `R3_MOE_WORKER_HISTOGRAM` diagnostic on matched runs; test prefill/decode split and protected slots. Once parity is accepted, remove duplicated GPU allocations, route counting kernel and heat file env vars from production component/settings/tests/docs in one migration, with no deprecated aliases.

## Detailed Solution & Technical Design

`HeatEngine` is a policy over *movable local units*, not a universal GPU-owner model. Do not force it to treat remotely owned experts as locally resident or invent movable mapped units. The shared **input histogram** is the invariant; `libr3moe` may maintain a bounded CPU score/window for remote ownership and reuse upstream's decay/hysteresis algorithm rather than a duplicate device telemetry stack. Ensure `rad0002` map/owner hooks still refer to exact runtime slot IDs; protected experts are never remapped merely from container IDs.

## Files

New/extended package in `engines/radiance/patches/` touching upstream `abi/`, `core/engine.cpp` (and `ctx` only when required); `engines/radiance/components/libr3/worker/{moe_remote_hook.cpp,moe_remote_gpu.h,r3_moe_worker.cpp}`; `component.toml`, patch tests, module parity tests and docs.

## Validation

Card-free callback lifetime, version mismatch, no callback, disabled-static mode, double-credit, phase attribution, MTP multiplicity and partial coverage tests. Hardware shadow comparison against existing histogram with same consumed step set, including 32K prefill and two helpers; check no CPU critical-path stalls and unchanged KL/greedy outputs.

## Effort & Risk

High: long-prefill host run-ahead and remote-only mapped units can hide holes in the normal HeatEngine path. Keep a diagnostic comparison gate but no permanent second collector.

## Standards

- Source of truth: pinned standalone Radiance 1.3.0 `89cee7ce414486c717aa2085c99e1f522e4c0ea6`; BigCherry `engines/radiance/`. Engine edits MUST be packaged as `engines/radiance/patches/radNNNN_*/{patch.py,patch.toml,SUMMARY.md}` with code-line anchors (patcher strips comments), tests with pinned blobs; helper policy lives in `libr3moe`. Re-check patch IDs before allocation.
- Follow `engines/radiance/recipes.toml` and `components/libr3/component.toml`; runtime profiles set ONLY non-defaults. Use existing BigCherry quality/ABBA/campaign/evidence facilities; no parallel scheduler, telemetry backend, loader, or compatibility alias. Disabled behaviour byte/quality/performance equivalent.
- Related active work: `RAD02` (R9700 trunk + 2 XTX helpers), `RAD03` (GPU mailbox), `RAD06` (prefill MoE), `RAD09` (merge/reconcile helper and rad0002). Integration depends on verified `RAD09` baseline. Do not overwrite another agent's branch or modify a shared patch without merging latest main.

## Acceptance Criteria

- Exactly one authoritative GPU routing histogram producer and existing Radiance host bank transfer; both consumer policies see the same credited events.
- Old diagnostic vs new feed agrees **exactly** on matched (rank,step,layer,expert) counters, including MTP weighting, with full explicit coverage report. No silent expert-ID remapping.
- Default-off path unchanged; remote helper still reproduces established quality/speed baseline; no new serving GPU sync or duplicate GPU heat allocation in production.
- Existing duplicated collection/storage variables removed when parity passes, no legacy path retained.

## Notes

Depends on RHP01 and merged/verified RAD09. Does not implement ownership migration; that is RHP05/06.

## Implementation-grade notes (2026-10-15)

### ABI: proposed, NOT implemented

```cpp
// abi/rad_moe_heat.h, versioned opt-in observer, no ownership mutations.
typedef struct RadMoeHeatRowV1 {
    uint32_t abi_version;       // 1; struct_size checked before every callback
    uint32_t struct_size;
    uint64_t engine_step;       // monotonically assigned by source engine
    uint32_t rank, layer, n_expert, pass_reports;
    uint32_t phase;             // 1 prefill, 2 decode, 3 mixed; provenance required
    uint32_t flags;             // coverage / overflow / report-kind
    const int32_t* counts;      // borrowed; only valid until callback returns
} RadMoeHeatRowV1;
typedef void (*RadMoeHeatObserverV1)(void*, const RadMoeHeatRowV1*);
void rad_moe_heat_set_observer_v1(void* context, RadMoeHeatObserverV1 fn);
```

This is a design sketch; verify ABI and registry/layout conventions in pinned Radiance before choosing the header location. `count[e]` uses model-global expert `e`, not `slot_of_expert[e]` nor helper-local packed IDs. Distinguish *histogram report phase* from row count: a mixed batch may contain both prefill/decode tokens; if the existing row aggregates across both, mark `phase=mixed` rather than inventing a split. A true per-phase split is a separately measured API enhancement, not a guessed label. Same for speculative draft reports.

### Registration / ownership / lifetime

- Register the callback at engine/plugin initialization through the existing explicit `rad0002` ABI wiring; use fixed `abi_version/struct_size`, no implicit symbol lookup or default-on behavior.
- The engine retains ownership of the histogram bank. Callback has **no right** to retain `counts` pointer after return.
- Implement a fixed-capacity SPSC CPU ring per engine rank, or copy/accumulate bounded compact host data in a preallocated per-rank window. For each callback, O(n_expert) copying might be required; measure versus existing dense HeatEngine pass; avoid allocation and system calls.
- Release/acquire publication from callback to CPU worker. Consumer thread may be behind and coalesce only **mathematically equivalent** reports; never reorder decayed-LFU updates or count the same MTP report twice. Queue overflow increments shared `lost_observation`, reduces confidence, and disables replacement when coverage insufficient.
- Callback failures, unsupported ABI or helper absence leave Radiance heat/residency unchanged and suppress only new ownership planning. Do not call a C++ exception through C ABI.
- `HeatEngine::enabled()==false`: RHP02 can drain route evidence independently of promotion; no dummy movable weights and no forced `HeatEngine::step()` calls. In deterministic mode read-only diagnostics may be enabled explicitly, but adaptation must remain off.

### Parity fixture

For every matched tuple `(rank,step,layer,model-global expert)`, compare source count and report multiplicity to the legacy `R3_MOE_WORKER_HISTOGRAM` diagnostic. If the old histogram is cumulative rather than step-scoped, compare exact sums over an identical known-complete window and report the difference in coverage. Count a route only once across mirrored TP ranks; don't erroneously halve partitioned tensors. Protected experts must appear in **routing evidence** even though never remote-eligible.

### Patch-engine mechanics and placement

Keep engine package separate from libr3 component modifications. Use exact code-line anchors in `patch.py` and `PIN_BLOBS` in `tools/tests/patch/test_radiance_patch_packages.py`; the patch matcher strips comments. If extending `rad0002`, do so in a single cohesive package after merging `RAD09`; otherwise choose next free `radNNNN`, never assume 0004 is unused. Component source builds only when new ABI header exists, with no legacy callback shim.

### Gate

Offline tests: ABI mismatch, null observer, early unregister, borrowed-pointer lifetime, no histogram after helper init, phase=mixed, MTP reports >1, dropped event and deterministic/all-VRAM. Hardware: parity across full 32K prefill, four-doc KL unchanged, no extra GPU-side counter or D2H copy and no blocking in critical path. Remove `R3_MOE_WORKER_HISTOGRAM` and `R3_MOE_WORKER_HEAT` only when full parity passes.

## Revised 2026-10-31 (GPT review of the plan against radiance 1.14.0 and RAD23; source claims checked)

Export immutable, credited histograms to the remote ownership policy through a versioned, optional CPU observer. HeatEngine keeps local residency; HeatEngine::enabled() is not permission to collect remote heat. HeatParams::decay (0.95) is per credited step: do not decay again per callback.

Steps:
1. Verify the 1.14.0 engine ABI and the remote hook registration (bc_moe_remote.h, rad0002).
2. Add a versioned borrowed-row callback with global layer/expert ids, rank, step, multiplicity, phase and coverage status, published from the completed-bank path with no extra GPU transfer.
3. Accumulate into bounded per-rank CPU state (r3::heat::Heat in worker_heat.h is the accumulator seed); report overflow and incomplete observations.
4. Compare with the static diagnostic on matched steps before retiring R3_MOE_WORKER_HISTOGRAM and R3_MOE_WORKER_HEAT in production.

Validation: Callback lifetime, ABI mismatch, disabled heat, deterministic mode, rank duplication, MTP, mixed phases, histogram parity; no new GPU synchronisation.

## What --record-heat costs today (2026-11-01, flashnext-4gpu, the 42K flashnext-probe)

Without recording 14.28 / 14.31 / 14.35 s (three runs); with `--record-heat` 15.09, 17.69, 15.0, 15.1 s (period 30 s) and 16.06 s
with the snapshot thread off (`--record-period 0`): 5-24%, irregular. Not the per-route global atomics: counting the step in
an LDS histogram and adding once an expert (at most 512 atomics a step instead of 20,480) gave the bit-identical histogram
(21,532,920 routes, 0 cells differing) and the same 15.0 s. Not the snapshots either. The cause is open; until it is found,
record heat in a window (minutes of real traffic) and serve without it, which is what RHP05 stage A needs anyway.

Clue (2026-11-01, engine logs of the same probes): the mover is identical with and without recording (2,669 promotions,
8.43 GiB host-to-card in every run), so it is not extra swapping; but the engine card settles with 173-174 MiB free
without recording, 153.6 MiB with recording and no snapshots, 141-145 MiB with 30 s snapshots: 20-33 MiB more than the
256 KiB histogram. Hypothesis to test: recording brings extra HIP runtime objects onto the R9700 (the snapshot thread's
own stream, the runtime's default stream for the zeroing hipMemset), i.e. extra hardware queues, which RDNA time-slices
(RAD20 found the same on the XTX helpers). Test: count the engine process's KFD queues (/sys/class/kfd/kfd/proc/<pid>/queues)
with and without --record-heat; if they differ, zero and copy the histogram on the engine's own stream instead.


# Shared KV capacity: code-level implementation guide (RSC16)

Proposed, not implemented. Read radiance-shared-kv-capacity.md first. Source anchors
checked against the local materialized Radiance tree used for recent work and
BigCherry main e79e736c2; re-resolve against the actual target pin before edits.
Keep upstream changes as managed patches under engines/radiance/patches, following
existing pin/composition gates. Do not edit a materialized source or generated
fixture by hand. RSC17 owns startup allocation; RSC18 owns live accounting;
RSC16 integrates and qualifies. ENG04 owns format/quality evidence.

## A. Reproduce and diagnose before changing behavior (RSC17)
- core/engine_bringup.cpp Engine::plan: capture each VramFacts, WeightFootprint,
  KVManager probe group geometry, kv_fixed, kv_one, kv_ceiling, staging and
  actual rank/device BDF. Record before common split, after refit, and after
  vram_budget_rank_split. Record available memory again at allocation.
- core/mem/vram_budget.{h,cpp}: resolve(), vram_budget_need(), rank_split().
  Existing common plan mixes min capacity with max statics, arena, queues,
  staging and kv costs. Do not assume that conservative mix caused the failure:
  compare each rank's actual feasible floor against it.
- rad0011_plan_memory/patch.py bc_plan_memory_note: refit occurs before per-rank
  split; record all refit_serving_pools inputs, calibrated ppm and exact deficit
  on refusal; no safety relaxation. Preserve original failure and before/after
  receipts in managed capture. Compare 256K baseline and 512K 4/8/16 slots.

## B. One native configuration and policy (RSC17)
- core/rad_core.h Config; core/config.cpp option table, parser, given flags,
  serialization/reconstruction; core/fleet/config.{h,cpp} YAML and fleet
  inheritance. Proposed typed CapacityIntent {legacy, auto_max, minimum_tokens}.
  Omitted -> legacy; auto -> maximize within safe placement policy; integer
  positive -> minimum feasible, not a per-sequence size. Reject wrong types,
  negatives, overflow, impossible target and unsupported capability combinations.
  Model families with nonstandard context geometry retain legacy until qualified.
- native_fleet_config.cpp inspect settings; serving.cpp native_value; session.cpp
  native admission and launch verification; manifest_writer.cpp and
  v2/native_config_snapshot, engine_admission, runtime_receipt, manifest.
  Carry authored intent and resolved measurements through existing sealed
  contract; do not add a parallel x-bigcherry setting or new resolver.
- Existing calibration fraction is legacy policy. Shared mode requires an
  explicit schema/versioned capacity-policy migration; stale policy/plan must
  be rejected or remeasured, not silently read as auto. Preserve old readers
  and old release config behavior. Add capacity mode/value, measured geometry,
  topology/rank order, dtype, rotary, slot count, max context, batch and
  placement/tier choices to existing plan/replay/cache identity. Invalidate
  when a relevant value changes; avoid new cache namespaces.

## C. Exact feasibility, per-rank (RSC17)
Use KVManager::plan_groups, fixed_bytes, ceiling, paged_split and
pool_mib_for_tokens (core/mem/kv.{h,cpp}). ceiling currently computes
S*ceil(L/block_size) plus fixed and streamed maps: retain as upper bound,
not floor. Pool inverse includes state and rounding. Split fixed and marginal
cost; never infer bytes/token from kv_one_session.bytes / L. Per-rank ledger:
claimable = live measured capacity/grant minus measured driver use, drift,
headroom, arena, queues, staging, prefetch and applicable pinned charge.
Weights floor = aligned statics plus required residency. Check independent MiB
rounding and actual staging dependence; don't double-charge fixed KV.
Illustrative pseudocode (interfaces provisional):
    feasible(C):
        for rank in ranks:
            kv = probe[rank].pool_mib_for_tokens(C, limit[rank])
            if kv < 0: return false
            weights = fit_weights(rank, kv, placement_floor)
            if !fits_exact_ledger(rank, weights, kv): return false
        return shared_resource_ledger_fits()
    lower = full_request_floor(geometry, L, S)
    require feasible(lower)
    C = largest_feasible_checked_search(lower, nominal_ceiling)
    require C >= explicit_minimum_if_present
    commit all rank budgets only after final geometry proof

Do not assume monotonicity with dynamic placement/stager changes: freeze the
placement regime or enumerate bounded breakpoints, then verify both sides of
each boundary. Check C+1/block boundary against solver result, overflow and
alignment. For non-uniform group caps, solve the common logical geometry
vector, not min of unrelated token estimates. Zero-page groups don't bind;
state slots, windows, rings and draft groups do. Account for worst per-sequence
tail fragmentation (up to S*(block_size-1) logical tokens per full group)
and page/slot guard; report packed capacity versus guaranteed mixed capacity.

## D. Integrate memory facts (RSC17)
Engine::plan computes probes before vram_budget_resolve; keep one source of
physical accounting and pass exact per-rank plan to configure_kv, rank receipts
and fleet broker. Current vram_budget_rank_split only saves on cheaper ranks
after the common plan; shared mode replaces that allocation decision, not just
the log. Verify the same logical block table geometry across rank order,
different bytes/block, and the final actual pools before accepting requests.
Review core/fleet/broker.h and fleet.cpp reservation/grant paths; shared/unified
memory uses one ledger, not independently reusable RAM on every rank.
core/kvstream (mirror, carve) and core/mem/kvtier/prefix: active mirror vs
cold retention; stream window and host mapping costs must be separately
accounted. No implicit host offload. Defer shared mode for unproven groups
rather than approximating their feasibility.
rad0011 bc_plan_memory_note / note_ranks, serving/v2/admission.{h,cpp}
refit_serving_pools, kv_required, check_engine_plan, admission_margins:
introduce shared-mode exact per-rank floors and typed refusal via existing
S5 room payload. Keep the one-full-request constraint and exact rounding.
Audit native_measurement.cpp, p1_measurement.cpp, manifest_writer.cpp:
remove shared-mode assumption of integral kv_one/L and S*L as aggregate
target. Carry explicit group facts and measured capacities. Existing legacy
checks unchanged. Explain exact minimum, fitted max, deficit per rank/tier,
fixed vs elastic bytes. Actual allocations exceeding ledger fail closed;
no silently shrinking explicit minimum or disabling safety margin.

## E. Live capacity and scheduler (RSC18)
Existing Scheduler::fresh_prefill_can_fit already checks actual whole prompt
plus decode position beside runners and same-step admissions in FULL groups;
WINDOW/RING recycle. Scheduler::preempted_can_prefill protects whole recompute,
preempt_wait_overdue/preempt_back bound progress. Preserve these rather than
inventing a second policy. Audit preempted_can_prefill for window/ring
accounting and prefix-hit clamping. Use one pure group-demand helper with
actual per-request length, existing adopted prefix, partial-block COW,
window geometry, state and draft/verify needs. Use checked arithmetic and
preserve existing bounded preemption semantics. Don't reserve each request's
max output or L up front; account for actual next step plus bounded progress
reservation; explicit user output limit still enforced. A queued request
must not consume an active slot. Whole-prompt + next-token feasibility
prevents admit/preempt loops.
KVManager::ensure already has preflight/commit, but recycle_window and
grow_for may change backing in preflight: audit failure paths before calling
it atomic. Preserve semantically valid old sequence if a later group/rank
fails, no leaked refs/state or pending COW, rollback any tentative physical
reservation or reconcile safe elastic backing. Reuse can_grow, can_ever_fit,
available_blocks, free_blocks, free_states; distinguish usable free versus
lent/reclaimable pages, avoid double count. Reuse scheduler mutex and existing
rank coordination; no independent per-rank logical block ID assignment.
Check cancellation/free_kv, deferred release after in-flight steps, prefix
adopt/fork, cache eviction, restore, checkpoint and speculation. Do not count
in-flight pages free until device completion. Bound preemption/queue age,
fail requests that cannot ever fit; no livelock when all runners want growth.

## F. Fixed overhead and visibility (RSC18, coordinated with RSC17)
core/sched/batch.cpp BatchBuilder layout/device_bytes and init reserve
max_seqs rows at max_ctx/block_size, but build stages actual step pitch.
Keep existing fixed allocation first and measure it; if material, design
bounded aggregate table storage using actual per-step rows, max pitch,
per-group geometry and offset range. A rectangular kernel ABI may still
require S*max pitch; never size a ragged allocation then pass rectangular
strides. Any table redesign needs separate kernel ABI, shape and performance
proof; not a prerequisite unless the ledger shows it blocks desired fit.
core/sched/metrics.h existing counters, scheduler metrics snapshot and existing
server metrics/explain: add mode, configured L/S, active/queued counts, per
group total/usable/lent/evictable/refcounted pages, state free, committed
physical bytes, peak, pressure waits/preemptions and limiting rank. Snapshot
under existing synchronization; no per-token full scan or new output stream.
Expose model-wide logical capacity only where comparable; retain group facts.
Power profile unchanged; no runtime memory expansion into other GPU leases.

## G. Test ladder and release (RSC16 integrator)
Reuse upstream tests/config_test.cpp, fleet_config_test.cpp, mem_test.cpp,
kv_window_test.cpp, sched_test.cpp, sched_fakes.h, checkpoint_test.cpp,
server_test.cpp, engine_e2e_test.cpp, lifecycle_test.cpp, yarn_test.cpp.
Extend existing serving v2/s5_refusal_test, native_measurement_test,
p1_measurement_test, p1_live_room_test, runtime_receipt tests and
stored_manifest_cache_test; add synthetic heterogeneous allocator table
(min card and max costs on different ranks), exact boundary +/-1 MiB,
overflow, zero groups, window/ring/state/streamed, unified and fleet grants.
Cross-check inverse against actual carve and every rank allocation; fail at
one page beyond physical limit, no double counting. Legacy parity required.
Runtime deterministic tests: 1 long + variable short, shared prefix COW,
tail fragmentation, simultaneous admissions, partial failure, deferred
release, preempted recompute, long wait vs short progress, speculative
rollback, cancellation, exhaustion and recovery, no lost state or livelock.
Use existing tests and harness managed records, not new standalone framework.
GPU: retain stock/efficient separation; actual tokenizer-counted prompt +
output at <=512K, overlapping 512K +128K + smaller if feasible; exact page
ledger before/after, retrieval depths, short-context controls, performance,
memory pressure; use ENG04/ENG03 and MEN17 release gates. Compare predicted
allocation vs actual at allocation granularity; measured driver uncertainty
reported separately, no unsupported percentage guarantee. Don't promote a
new default until cross-family gates and reviewed thresholds pass.

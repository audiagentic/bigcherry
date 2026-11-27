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

## H. Second review: patch ownership and minimum change first (2026-11-27)

The first design missed an important reuse seam. rad0029_rank_own_split
(VTP33, validated) already computes an independent dense rank pool from
its own weights and additional free memory, uses KVManager::paged_tokens
and pool_mib_for_tokens, chooses the least common token count, and trims
each rank to it. It deliberately excludes experts, unified memory,
explicit gpu-memory-mib, and weights not fitting the common pool.
It runs AFTER rad0011 bc_plan_memory_note and its refit_serving_pools gate,
which is the observed refusal. Therefore it cannot rescue the refused
512K launch. VTP33 records successful 8 x 128K prompts and capacity gains;
those are historical, not a current 512K proof. Do not add a third split
algorithm or replace the allocator wholesale.

Stage A first: establish the effective calibration, exact rank facts,
pool inverse and current rad0029 candidate on the same input without
allocating or bypassing admission. The checked-in calibration seeds
for kv_backing_floor are scoped to FlashNext-4gpu latency/balanced;
do not infer a nonzero dense floor from their existence. Capture the
selected provenance (including local calibration) and the actual one-
session minimum. If the existing algorithm suffices once correctly ordered,
refactor it into a pure reusable feasibility step before final admission.
Only extend the solver where evidence or generic policy requires it.
The failed experiment used the short qwen27b-tp3 base with changed L,
S, dtype and YaRN; it did not include long-profile prefill-governor or
prefill-interleave settings. Reproduce on that exact archived input,
then compare matched long-bf16i8 baseline and extended variant with
only intended axes changed. Do not conflate those results.

### Patch/package boundaries

- rad0029: retain legacy behavior, refactor the existing dense own-split
  seam into a proposal/solver reused by explicit shared mode; account for
  actual per-rank overhead, not just free-memory delta. Preserve VTP33
  tests and promotion evidence. A new generic upstream-shaped patch may
  supersede this block only after legacy parity; do not delete the old
  evidence or leave two competing writers of rank_cfg_.
- New narrowly scoped generic engine patch(es): typed native capacity
  option and shared allocation orchestration, plus separate runtime
  correction only if the audit proves a deficiency. Use the next free
  patch ID allocated at implementation, declared requires/conflicts and
  plan-ids; no arbitrary fixed number in this plan. Prefer upstream
  acceptance; retire overlapping local edits after equivalent gates.
- rad0011 owns instrumentation, plan producer, sealed admission and
  rank receipts. Separate initial fact capture from final admission:
  evaluate the final candidate rank vector before pools or weights,
  not the common intermediate. Do not move live checks after allocation.
  Existing legacy common checks stay for legacy mode; shared mode must
  validate a complete rank ledger, not call both competing splitters.
- rad0070 native managed entry: check inspect/measure/start parity and
  owner verification; avoid overlapping native-entry edits. Respect
  rad0036 governor and rad0028 rank-consistent draft dependencies.
  No new runtime library, fork or alternate launch entrypoint.
- Composition gate: patch-lint, existing package tests incl.
  test_vtp33_rank_own_split.py, production and affected experiment
  patch-chain application/rebase checks, anchor uniqueness, compile
  checks and standard build/qualification gates. Update patch metadata
  and owned-unit boundaries using the existing composition standard.

### Startup accuracy and invariants

The present rad0011 rank receipt computes overhead from arena, staging,
queue count*queue bytes and a common stage; it does not explicitly report
all terms used by vram_budget.cpp (queue scratch, prefetch, drift,
headroom, pinned charge etc.). Do not reuse that partial receipt as
the authoritative new per-rank ledger. Derive accounting from the same
budget routine, expose both total capacity and measured free and their
relationship; do not subtract already-held bytes twice from free.
Keep units exact checked integer bytes, MiB pool rounding, and checked
wide sums. Verify zero/nonzero weights-pool minimum, explicit memory
caps, fleet grants and staged residency. Compare free at plan/start,
revalidate sealed minimum, bounded retry through existing owner on drift.
Auto is maximum under selected placement policy, NOT permission to
discard all optional expert residency; declare the placement trade-off
and preserve MoE quality/performance. No assumption that CPU offload
is available or desirable. For the first implementation constrain
shared mode to qualified full-paged dense discrete GPU paths including
TP1/TP2/TP3; capability-driven refusal for unsupported groups. Broaden
in separate qualified steps, not a model name switch.

Engine::configure_kv caps every rank at the least paged_tokens before
carve. Its current kv_planned_tokens_ check is conditional on a rank
pool being smaller than cfg_'s common pool; shared mode must verify
EVERY rank against the sealed target, including enlarged/equal pools,
then verify the actual post-carve block geometry and state slots.
Compare rank order and group handles/kind/block size/table bounds,
not merely the scalar token count. Use exact group vector where needed;
a packed token capacity is not a guarantee for arbitrary fragmented
sessions. For C=auto expose achieved capacity, not hypothetical maximum
from stale measurement; if sealed plan cannot fit live, remeasure/refuse.
For an explicit minimum do not silently shrink. Retain generator
branch minima and no-paged-group sentinel behavior.

### Runtime review: avoid unnecessary change

Scheduler::init already binary-searches the maximum one-sequence
capacity using can_ever_fit; add_locked refuses an impossible prompt,
fresh_prefill_can_fit uses whole prompt plus next position and actual
runner/same-step demand, not L*S. Reuse the existing progress mechanisms.
Treat RSC18 as audit-and-fix-only: test existing behavior first; if no
defect, deliver accounting/observability and tests, not a scheduler
rewrite. KVManager::can_grow uses ceil_div and COW whereas ensure uses
window_table_len plus recycle_window for runtime windows; use shared
read-only demand geometry or explicitly qualify the differences before
claiming equivalent admission. Preserve recoverable preflight elastic
growth semantics, avoid expensive rollback or refcount scans on hot path.
Keep context ceiling vs physical one-request fit distinct in API and
error codes; test exact L including output and speculative boundary,
including prefill next-position requirement. Existing scheduler can
truncate output to context room: don't silently clamp requested length
under new contract; retain documented legacy semantics, explicitly
define shared-mode context overflow behavior. Distinguish permanently
impossible request from transient resource pressure, queue/timeout
with existing front protocol; no assumption of a new HTTP status.

Qualification should include a deterministic synthetic two-rank case
where min capacity and max KV/weights occur on different ranks, where
common refit rejects and same-rank proposal fits; a physical-shortfall
control; old rad0029 successful case; 1 page/1 MiB boundary; rank
mismatch and drift; unchanged legacy configurations. Add negative
tests for 512K+128K if the physical budget cannot back it. Do not label
a smaller or sequential workload as passing the mixed-resident goal.

## I. Consolidated cleanup work packages (third review, 2026-11-27)

RSC17 owns physical memory planning, native geometry, host retention and
calibration/contract cleanup; RSC18 owns actual runtime demand and metrics;
RSC16 owns acceptance, compatibility cutover and superseded-patch retirement.
No extra plan IDs: these are work packages with independent evidence and
commits inside existing owners. Do not reopen completed RSC03/10/11/14 or
change in-progress RSC12 (variants/reuse), RHP16/20 (live expert policy).
RSC15 owns deferred site/profile TOML/--set retirement; do not pull it into
this change. ENG04 owns format/context quality, ENG03 concurrency speed,
ENG07 QSA quantization, RAD38 mixed precision, MEN17 release acceptance;
PVPS14 checkpoint correctness investigation remains independent. VTP33 is
historical proof of rad0029, not another implementation. Preserve all existing
acceptance criteria and archive prior code/results through standard lifecycle.

### I1. Consolidate memory facts, proposal and verification (RSC17)

Confirmed duplication: Engine::plan builds throwaway KVManager probes,
then cheaper-rank resplit and rad0029 each reconstruct them; configure_kv
plans again on actual pools. rad0011 common refit has its own rounded floor.
Separate the necessary live remeasurement from redundant policy calculations.
One declared geometry per rank per planning attempt, one pure proposal,
one committed rank_cfg vector, one independent live verifier. Reuse native
VramFacts, VramBudget, KVManager inverse, rank_split and rad0029 seam; do not
create a second BigCherry allocator. Geometry changes (stream cap, placement,
model/config, rank order) invalidate the probe; actual configure/carve remains
an independent check, not a reused stale answer. Separate device/group
geometry identity from mutable free bytes and from placement policy.
Use the native checked budget ledger for device capacity/grant, driver-held,
headroom, drift, arena, queues including scratch, prefetch, host-pinned,
fixed weights, expert floor, staging and fixed KV. Retain role-specific
placement floors. Derive physical allocated bytes from rounded pools;
statics and expert floor are components of weights, not additional pool
charges. BigCherry keeps independent validation of identity, caps, typed
shortfall, actual measurements and lease/owner; it must not solve a second
allocation policy. Test an identical rank-fact fixture through proposal,
sealed measurement and live verifier, with mutations to each field, swapped
BDFs, changed live room and one-MiB rounding boundaries. Assert unchanged
legacy proposals, no overlap/double subtraction of held bytes, and no
allocation after refusal. Preserve typed S5 refusal and existing one bounded
retry, never weaken physical checks to remove apparent duplication.

### I2. Replace retained-host scalar with existing entry geometry (RSC17)

Confirmed producer: native_measurement.cpp and p1_measurement.cpp derive
kv_bytes_per_token from rounded kv_one_session/L; manifest_writer.cpp
and s8_replay.cpp propagate it. resources.cpp::calculate and
v2/s8_rules.cpp independently multiply retained_tokens by that scalar;
p1_producer.cpp and measurement contracts cap retained target at L*S.
This is not exact marginal or host bytes: kv_one includes fixed state for
all slots, may include windows/drafter, and is a limiting rank's pool.
The host tier actually packs a whole engine entry, with replicated group
bytes once and unequal shards concatenated. A scalar extrapolation may
under- or over-budget; no measured current error amount claimed.
Reuse core/mem/prefix.cpp PrefixCache::configure entry_groups and block_size
(LCM of eligible groups; excludes window; streamed entries not demotable).
Reuse core/runtime/tierexec.{h,cpp} TierExec::slot_layout: slot_bytes is
whole engine host slot, entry_bytes is largest rank transfer/staging entry.
Engine::configure_kv uses slot_bytes for IdleTiers::Config.entry_bytes and
checkpoint store separately. Extract/reuse a pure pre-allocation layout
helper from these existing owners, with checked LCM, alignment and shape
validation; do not infer from GPU page cost or duplicate entry inclusion.
For requested retention of R packed tokens and Q tokens/entry, require
ceil_div(R,Q) complete slots times aligned slot bytes; derive actual
entry count from HostArena::open capacity/rounding, report packed vs
per-session guaranteed retention and tail slack. R=0 -> zero host KV.
Distinguish independent active capacity and cold retention; no S*L upper
bound on total retained histories unless explicit policy requires it.
Check unified or streamed disabled/demotable cases before allocating.
Snapshot/checkpoint slots and staging remain separate, not multiplied
by retained tokens. Keep prefix-cache-mib semantics as byte cap and
avoid silently turning a cap into a guarantee. Host pool, RAM, cgroup,
pin, helper footprints and reserve gates remain existing owners.
Carry versioned measured retention geometry and independent intent through
contract/serving.toml, native_measurement, p1_measurement, p1_producer,
manifest_writer, s8_rules, s8_replay and resources.{h,cpp}. Audit actual
callers/link targets: resources::calculate search found parity tests, not
proof that production calls it; p1/native paths have distinct targets.
Consolidate shared pure checked host arithmetic only if reachable; otherwise
retain historical fixture reader/tests until legacy retirement is proven.
Never delete a contract field based only on text search. No new resolver,
calibration store or launch envelope. Old records remain legacy version;
no mixing old scalar with new geometry. Reuse RSC12 cache identity seam;
coordinate its active owner before touching shared files. Keep RSC15 deferred.

Card-free tests: heterogeneous ranks with replicated + sharded + zero
blocks, different block sizes, window exclusion, streamed/disabled tier,
zero R, R=Q-1/Q/Q+1, >S*L retention if policy permits, overflow, alignment,
checkpoint store separate, per-rank staging vs total slot. Compare exact
slot layout and actual HostArena open/IdleTiers capacity against proposed
bytes, including MiB rounding; reject mismatched geometry, insufficient
pin/cgroup/host reserve by one unit; run resources_parity_test, native/P1
measurement, S8 replay, S5 and cache tests. Physical host gate requires
measured actual bytes on supported deployment before promotion; compare
legacy mode against existing fixtures and RSC03/14 contention evidence.
### I3. Calibration and compatibility cleanup (RSC17; RSC16 retirement)

Existing calibration.toml seeds kv_backing_floor.fraction_ppm at 50% and
60% for FlashNext latency/balanced; they encode a measured placement
trade-off, not universal cache demand. Record selected source and value,
keep legacy ppm unchanged; new shared mode uses explicit minimum target
and placement policy, requires migration/requalification of matching
calibration (no silent zeroing or generic replacement from dense test).
Measure performance/quality of expert residency choices using RHP16/20
outputs, not duplicate dynamic expert policy. For auto capacity, define
objective: maximize safe page backing within chosen residency policy,
not unconditional max VRAM at the expense of latency. Existing host
reserve and margin, measured replay keys and checks remain. Test selected
seed/cache/site precedence, stale version, changed capacity mode/target,
dtype/rotary/slot/group shape/layout/build, legacy replay and mixed schema
refusal. Scope change to affected memory contract, not deferred resolver.

### I4. Reuse demand calculations and make pressure observable (RSC18)

Confirmed existing behavior: KVManager::available_blocks includes usable
free plus recallable loan, limited by flex room; held_blocks=live-free.
Scheduler metrics computes blocks_free=total-held (includes lent), not
immediately usable. adapters.h uses most-utilized non-window paged group
for tokens total/used, min group capacity for tokens_max, max utilization
across groups; bytes are summed across groups. These are different axes,
not interchangeable. Existing reporting is useful; do not overwrite metric
semantics silently or sum token capacity across groups/ranks. Audit
pool_bytes/used/carved and rank aggregation; rank-0 scheduler view is not
whole-engine physical bytes. Add explicit scope and per-group/rank facts
through existing adapters, iface, dashboard, metrics and history only as
needed. Keep existing API/Prometheus names for compatibility, version
changed/new semantics, migrate existing consumers rather than new stream.
Reuse KVManager::blocks_for, can_grow, ensure, available_blocks and actual
runtime window geometry; pure demand helper should account for COW and
in-flight/deferred release, with eligibility and conservative progress
reserve explicit. Keep existing fresh_prefill_can_fit and recompute guards;
prove equivalence on full groups, ring/window and state groups; avoid
assuming state slots or window groups never constrain runtime. No whole
context reservation for a short client; no new scheduler. Test before
changing, avoid added per-step allocation or refcount scan. For each paged
group assert total=held+usable_free+lent, actual available<=free+lent,
cache-only reclaimable<=held (exclusive of shared/in-flight), COW counts
exactly once, state slots separate. Include failed recall, cancellation,
preempt/recompute, window wrap, prefix sharing, group limit changes,
zero/sentinel and state-only cases. A synthetic example total=100,
live=70, free=20 gives held=50, lent=30, reported legacy free=50,
usable free=20; flex may reduce recallable below 30. Assert no false
pressure and no promise that every lent page can be immediately used.
Reuse sched_test, mem_test, kv_window_test, server/dashboard tests;
check metrics snapshots and exact physical allocations with no extra
runtime work on hot path. Confirm bounded forward progress and no leaked
pages under fault injection; compare existing governor throughput/TTFT.
### I5. Controlled retirement and deferred work (RSC16)

Retire superseded split/refit/helper code only after caller inventory,
legacy default and historical replay/fixture parity, patch application
and compile gates, measured installed acceptance. Keep patch README,
metadata, evidence, rejected alternatives and history in managed archive
with supersedes/requires links; no deletion of evidence or worktree until
merge and verification. Do not mark RSC03/11 historical achievements
superseded wholesale. Use standard patch-lint, composition and affected
experiment chain checks; no new IDs before implementation allocation.

Conditional investigations, not parallel implementation plans: measure
rectangular BatchBuilder staging and fixed state; if dominant, RSC18
records a separately reviewed bounded task for ragged ABI or dynamic
state (requires kernel ABI/lifetime proof). Streamed mirror in
core/kvstream/mirror.cpp/carve.cpp uses max_ctx*max_seqs by design;
explicitly refuse unqualified shared mode rather than silently resize it.
If material, RSC16 opens a capability-specific follow-up with host map,
transfer and quality gates. Live VRAM growth and tensor ownership changes
are separate fleet/placement work; do not touch until evidence demands
it. Site/profile cleanup RSC15 remains deferred. No new plans merely
for speculative improvements; each investigation produces measured
reason to proceed or a documented no-change decision.

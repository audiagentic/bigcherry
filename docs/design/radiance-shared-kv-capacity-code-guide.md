# Shared KV capacity: code-level implementation guide (RSC16)

Proposed, not implemented. Read radiance-shared-kv-capacity.md first. Source anchors
originally checked against Radiance 2.2; sections K-L record verified 2.3 target
source and overrides stale implementation anchors/assumptions. Recheck the final
merged recipe and source identity before edits.
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
The failed experiment used the short qwen27b-tp3-q8 base with changed L,
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

## J. Fourth review: exact guarantees and qualification boundaries (2026-11-27)

This section refines H/I, not a new owner or additional allocator. Source
review used materialization e3d02821f12bd25b34e58ef4d0a0c37b, upstream
b6b8751cccb0e834a13ae9e5afca824ee9a487aa (2.2.0), classified experimental,
tree 61fe7eb3708e6727e50cba98e1b424c8833019d7. Archived .1 release has same
upstream but source slice b6bd832be944a4ab31cc29906dece6c5, not the reviewed
slice e699f40e7a8b5a8efaea63818c239798. Exact .1 slice was not found in
local source metadata. Review confirms source mechanisms, not byte-for-byte
production equivalence. RSC17 must materialize the exact target production
patch chain and record release/build/source/patch identities before tests;
check bump branch changes separately and reuse upstream replacements.
Archived four-slot log says 24% and '1.0 full-length sessions' before
refit refusal. Rounded display is NOT one full-context proof, and refit
failure does not identify a physical limiting rank. The initial probe
need=-1 path may leave f.kv_one unset; validate all probe return values
and distinguish no paged group from failed geometry or impossible inverse.

### J1. Exact inverse and page demand (RSC17)

KVManager::paged_split sums double bytes_per_block/block_size then truncates
to scalar tokens; out[g]=tokens/block_size truncates again. Its inverse
pool_mib_for_tokens checks scalar t>=target, not the out vector. E.g. t=65,
B=16 yields four pages=64 tokens, not five. Do not use either scalar as
proof of a guaranteed minimum. Keep the native planner, factor out a
checked group-vector calculation and exact feasibility predicate using
actual block counts and carve layout; reuse the inverse/search only if
the predicate is monotone in the chosen fixed placement regime.

For ordinary full groups and a packed token target T, require at least
ceil_div(T,B_g) logical pages per group, not floor(T/B_g). A cold request
of length L must fit every group including resolved state, window/drafter
requirements, and required next-position/verify scratch for its legal
final step. Use checked integer arithmetic, actual per-reservation
RAD_ALIGN_UNIT padding, per-pool MiB rounding, and actual ID/offset type
bounds (carve_into uses int32_t IDs; HostArena uses int32_t slot IDs).
Compare the vector to carve_into's reserve sequence, including state
copies, checkpoint stage, window maps and zero groups. Fixed_ already
includes per-group padding and ring costs: do not charge them twice.
Avoid changing float heuristic on legacy path without parity tests.
Test fixed==budget (inverse currently accepts equal in some no-page
cases; carve_into rejects), missing/zero groups, invalid geometry,
65/16, unequal group page sizes, one byte/page/MiB over, overflow,
and plan/carve disagreement. Refuse incompatible geometry rather than
fall back to arbitrary scalar. All ranks must have same logical ID
space while their byte costs and actual backing differ.

### J2. Capacity contract and elasticity (RSC17/RSC18)

Distinguish max individual L, active slots S, packed logical capacity
C_packed (min of comparable full-group block_count*block_size), currently
backed pages, reclaimable/lent pages and actual workload fit. Explicit
kv-cache-capacity target is minimum packed capacity, not a promise that
any mix totalling C tokens fits. For lengths l_i, full group demand is
sum(ceil_div(l_i,B_g)) less proven shared references plus COW/in-flight
and progress reserve. E.g. 65+1 tokens with B=16 consume six pages
(96 token slots), not five. If offering a worst-case guarantee, specify
the number of sequences and reserve policy, and derive it from exact
group vectors; do not conflate it with packed capacity. No independent
per-client reservation for unused contexts. S is scheduler slots, not
number of idle/retained conversations or outstanding queued requests.
Guarantee one feasible L on an otherwise available pool, not immediate
admission of a long client regardless of currently active work.
Report transient pressure, bounded queue/preemption, and permanent
impossibility separately; external harness bounds do not replace safety.

Fleet already has flex: Engine::plan broker sets vram_claim and
kv_reserve from ask, and configure_kv calls set_flex; the manager may
reserve a larger address space than it physically backs. Do not disable
existing legacy elasticity by wording 'no live expansion'. For shared
mode initial qualification use fixed backed pools, not optimistic
virtual reservations or fleet growth; existing fleet mode remains legacy
until exact grant/backed/reserved/recall semantics are proven. A future
shared flex capacity must differentiate guaranteed grant from potential
additional broker-approved room and avoid counting it twice. Auto is
bounded by hard caps, explicit grants and nominal addressable geometry.

### J3. Host geometry, cap semantics and checkpoint policy (RSC17)

TierExec::slot_layout computes slot=sum own_bytes+rep_bytes and
entry_bytes=max(rep_bytes+own_bytes), no arbitrary per-slot padding.
HostArena::open computes floor(cap_bytes/slot_bytes) slots, commits
that product and bitmap separately, refuses cap smaller than one slot.
Use exact native slot bytes, not a newly invented alignment. Native
prefix-cache-mib is an authored cap, not an independently requested
retention minimum. manifest_writer currently derives retained target
from this cap using kv_one/L and clamps to L*S. Preserve cap authority:
if cap H is supplied, report floor(H/slot)*entry_tokens; do not round
a requested minimum UP past an explicit cap. If a minimum R is ever
explicitly supported, require ceil(R/Q)*slot <= cap and safe host room,
or refuse; only derive whole MiB when the contract permits it.
Account for real committed bytes, metadata, pin/cgroup and reserve
margins once. Retention target can represent packed histories, not
active slots. Do not change authored cap semantics or add retention
knobs simply to retire scalar. Preserve zero and disabled-tier cases.

checkpoint_slots_want in prefix.cpp uses S*(policy.held(L,interval)+1),
with independent floor/want/room selection in checkpoint_slots_auto.
Record floor, want, actual slots and bytes separately; do not scale
snapshot bytes with C or delete this legitimate upper-bound policy.
If host pressure is dominant, review auto checkpoint retention
independently using current policy and prefix-resume correctness;
no weakening of minimum state/replay safety. Test cap below one slot,
exact slot, slot+1, MiB cap rounding, replicated/sharded/zero,
snapshot and bitmap costs, and shared-mode zero host tier even on a
state-only or non-demotable model. Preserve legacy fixture semantics.

### J4. Measurement and staged acceptance (RSC16/RSC17/RSC18)

Engine::plan may configure pools; bc_plan_memory_done runs after plan,
before load_weights and configure_kv. A plan-memory record alone does
not prove that actual group carve, staging, runtime metadata, late
allocations, drafter, or server initialization fit. Split evidence:
(1) preallocation proposal + live gate on complete rank facts,
(2) actual configured pools and final KV carve/group/slot geometry,
(3) all late startup allocations complete and scheduler/front ready.
Verify before listener admits traffic; capture failure at each stage
and release all reservations. Do not report plan-only as actual
capacity. Preserve existing admission and typed refusal paths.
Source-inspection and card-free tests do not substitute for installed
GPU qualification; read-only planning may still need device allocations,
so use lease and do not change power settings in RSC17 diagnostics.

Initial scope is one text engine on discrete GPUs, fixed backed pools,
full-context trunk plus its declared recurrent/conv and bounded
drafter/window groups (Qwen27B IQ8R is hybrid, not full-paged-only).
Qualify by group capability, not model name. TP1/TP2/TP3 as supported;
unsupported generator/streamed/flex/unified paths retain legacy or
explicit refusal. Extend to MoE and remaining group policies after
qualification, not model-specific forks. Use stock for capacity
comparison; efficient separate and only under existing qualification.

For mixed long+short acceptance use actual tokenized prompt+output
length and simultaneous active/resident full-group page receipts:
long decode stays active while shorter prompts are admitted, then
short decode and long decode progress. A long idle cached prefix or
preempted/recomputed request is not proof of simultaneous residence.
Use unique prompts first (no sharing hides pressure), then intentional
prefix sharing control; test 512K+128K+additional if feasible, 16
slot target and exhaustion. Check long output, short progress and
numerical quality with matched context/YaRN controls. A capacity
refusal is valid physical evidence, not successful target acceptance.
Set explicit performance/no-regression criteria before run.

## K. Radiance 2.3 reuse map and execution order (2026-11-27)

This section is the target-specific implementation map; A-J retain the requirements,
historical evidence and mathematical tests. Do not implement each review as a separate
work package. RSC17 owns startup and retention facts; RSC18 owns runtime verification;
RSC16 owns integration and retirement. The bump is independent of shared mode.

### K1. Verified source, not assumed upstream parity

Reviewed controller `e481494914bfa1d3195e947c76876fd5ad17ce2f`,
branch bump/radiance-20261127, pin `3b13fd32663ef52252a7f0be255523480aa22ffe`
(v2.3.0), compared pristine Git revisions b6b8751 and 3b13fd3. Base materialization
plan 445dc21264c89b571a8077b7f573b9af, source slice
90bca121c9dcee4cac6cf2b41a8d7b3e, tree 4725962c4b5da8b9506b5427e56d771015c78a07:
all 34 recorded patch implementation digests matched the branch's
`bigcherry.patch.patchset.catalog`. Alternate slice c0f67098ba8c37faf9d82d7d4cb7230a
differed in rad0046 only; compared 14 host source/test files byte-for-byte, all equal.
This verifies the reviewed base chain, not every recipe/overlay or installed binary.
Recheck final merged pin, recipe selection, source identity and dependent patches
before edits; do not freeze the bump or duplicate its canonical build.
Bump CANONICAL-LANE evidence records 504 syntax passes at an earlier controller,
not shared-capacity acceptance. No new builds, device runs or tests executed here.

### K2. What changed natively from 2.2 to 2.3

| Native seam | Verified behavior | Use / extension |
|---|---|---|
| `mem/vram_budget.cpp` | New headroom lookup per device; rank room subtracts max(configured, host reserve); common room uses minimum after each rank's reserve. | Reuse these rules, not raw free or an independent reserve formula. Common split still combines worst costs; prove actual limiting rank. |
| `vram_budget_rank_spare_mib` | Native named-card surplus; rad0029 adds defaulted named-only switch and calls with false for dense own split. | Preserve default expert behavior, explicit cap/claim/broker/unified exclusions. Extract proposal, do not fork helper. |
| `KVManager::can_grow` | Now caps non-runtime windows and uses window_table_len for runtime rings; includes COW. | Remove assumption it has old uncapped window behavior. Test real keep_from/recycle geometry rather than impose a new window rule. |
| `Scheduler::blocks_still_needed/refill_cap` | New shared runner-demand and drafter hit-cap helpers. | Reuse both; do not add an independent prefill/recompute demand implementation. |
| `IKVManager::ensure_admission` | Explicit waiting_for_fleet propagated through KVBinding, ensure and grow_for. | Keep distinct local FULL, external capacity wait and allocator failure. No new status taxonomy for the same condition. |
| `TierExec::quiesce/report_pass/close` | Native transfer-fence failure propagation and fatal guard before freeing unsafe buffers. | Preserve; no early release on cancellation or failed restore. |
| `Engine::drafter_window_snap` | New checkpoint-draft-window option allows drafter snapshots beside linear state. | Probe with resolved policy on all ranks; include snapshot geometry in cache identity. Do not reimplement checkpoint policy. |

### K3. RSC17: one geometry and one proposal

Confirmed materialized anchors in engine_bringup.cpp: 2338-39 calls
vram_budget_resolve then bc_plan_memory_note; bc_plan_memory.cpp:51-75 runs
refit_serving_pools, exits on error, writes common cfg pools. Only then 2368-2514
sets rank_cfg_, native expert redistribution and rad0029 dense own-split.
The latter requires multiple ranks, no broker/claim/explicit GPU cap, no experts
or unified memory, and sufficient common static-weight pool. It derives each
weight MiB, adds returned weight room and native spare room, finds least rank
token count, uses inverse to trim pools, applies only if improved. It cannot
recover a refusal before that block. Native expert re-split has different
placement constraints; do not broaden it or remove tier floors.

- Reproduce the archived managed refusal with the matched long profile against
  the merged 2.3 pin. Record each rank's full VramFacts, actual reserve, program
  declarations, refit inputs/ppm/deficit and candidate own-split, including failed
  probe status; distinguish actual cause from possible conservative rejection.
- Preserve legacy ordering and calibration when C omitted. For explicit shared
  mode, extract existing rank proposal into a side-effect-free computation ahead
  of admission; use native rank_room/reserve/spare semantics, existing
  vram_budget_trial/resolve and rank-split primitives. One writer of final
  rank_cfg_, no second resolver, do not merely move a mutating hook later.
  If a supported dense proposal fits, admit on its sealed per-rank facts; if not,
  report rank/group/byte shortfall. Unsupported expert/fleet/unified/streamed
  combinations fail closed initially; legacy remains supported. Qwen hybrid
  state/window groups within the fixed-backed discrete text path are required.
- mem/kv.cpp:459-540: paged_split divides remaining bytes by double cost,
  then floors each group to tokens/block_size; inverse checks scalar t >= target.
  Both unchanged in pristine 2.3. At 65 tokens/B16, 4 pages cannot hold the
  target. Zero ranks inherit a common table, not an independent capacity.
  Extend native geometry with checked integer group vector and exact reservation
  sequence: ceil target/B for full groups, existing window/stream/state rules,
  group-by-group RAD_ALIGN_UNIT, checkpoint stage, precharged maps, ID and
  slot-offset limits. Compare actual reservation plan and carve; no guessed scalar
  margin. Keep old scalar API/behavior for legacy unless parity proves safe
  replacement; one geometry producer, not competing byte formulas.
  Fixed == budget is refused by carve_into (547-558), while inverse can pass
  fixed equality; invalid/empty/unbounded and overflow must be explicit.
- Preallocate gate before first pools.configure at 2530-2543; plan() already
  allocates physical pools, so bc_plan_memory_done at 5591-5595 is NOT this
  gate. Recheck live per-rank facts before allocating; note_ranks currently
  runs before configure, not proof of success. configure_kv:4910-4953 probes
  actual pool availability and caps scalar least, but only tests planned
  tokens when rank pool shrank. Verify ALL rank/group actual block vectors,
  backed bytes and table compatibility after carve, then later front readiness.
  Distinguish proposal, allocated pools, carved/backed pages and ready receipts.
- Keep native plan_groups (99 onward), packed page bytes via declaration,
  state copies and window rules. Reuse set_drafter_window_snapshot and
  declare_ring_windows consistently. No kernel/ABI change merely to set C;
  preserve rad0048 ABI45 and rad0049-54 packed layout contracts.

### K4. RSC17: retention and measurement, not another allocator

Engine::configure_kv:5097-5132 calls prefix.configure then entry_groups (LCM
entry geometry, repeated short-block groups), TierExec::slot_layout and
IdleTiers::configure. slot_layout:113-168 sums replicated identical-width
groups once and per-rank owned shards, skips streamed groups, rejects invalid
replication; entry_bytes is max per-rank staging width, slot_bytes the full
engine slot. Preserve this and HostArena::open floor(cap/slot), checkpoint
store separately, native unified collapse. Use planned entry geometry from
same declarations and verify actual configured entry layout; if no safe
pre-carve reuse seam, expose a pure native helper and call it in both places.
Do not infer retained byte cost from kv_one/L or round minimum up past cap.
Use resolved chunk_geometry_resolve/checkpoint_slots_want, including new
drafter snapshot policy, and independently verify actual host slots/room.
Keep RSC12 configuration/measurement cache and historical readers; migrate
reachable scalar S8/P1/manifest callers in one versioned shared-mode contract,
only after inventory of actual callers. No separate retention configuration.

### K5. RSC18: validate unchanged scheduler first

At scheduler.cpp:1985-2018 fresh gate follows hit adoption, evicts bounded
cache rounds, distinguishes impossible empty-pool fit, idle refusal and busy
wait. At 2147-2275 blocks_still_needed counts remaining prompt/next position
and one additional block for runners, fresh also includes took_ this step;
recompute credits only actual capped/grid/checkpoint-adoptable prefix. Full
groups alone consume full-context demand; WINDOW/RING recycle. Preserve
rad0036 governor after room gate, rad0028 draft rules, native
AdmissionFailure::ExternalCapacity and existing exact error propagation.
Do not add a redundant admission gate or reserve max output per slot.

ensure:1537-1627 recycles windows in pass 0, calls grow_for in pass 1 (may
recall backing and change loan/short_peak), then commits table/COW/state
refs and pending copies in pass 2. Therefore the comment "leaves sequence
as it was" is not literally all-state atomic. Safe window recycling and
monotone loan recall may be legitimate on failure: test retained readable
keys, unchanged committed tokens and ownership, valid pending/copy fences,
conservation and retry before requiring rollback. can_grow is now
window-aware but ensure without runtime window uses actual recycled first
position, not the same min(window_seq_cap) expression. Verify at real
chunk boundaries/keep_from and partial COW; share read-only helper only if
a demonstrated mismatch requires it. Preserve native transfer error fences.
No full allocator transaction or hot-path scan unless failure proves need.

Metrics: kv.cpp:850-920 has free_usable, held=live-free, total and available
free+recallable (bounded by flex_room). cache_only_blocks tracks references
held solely by cache. Scheduler::metrics_locked:4520 uses total-held
including lent, adapter:50-100 selects binding nonwindow utilization, min
capacity across groups, sums group bytes not ranks. Preserve those legacy
semantics and expose actual backed/free/held/lent/cache-only with existing
metrics surface; avoid summing logical tokens across groups/ranks. Distinguish
backed from reserved even in fixed/elastic paths. Test 65+1 at B16 = 6 pages,
not 5, plus COW, retained cache, state slots and progress headroom; C is a
packed target not arbitrary mixed-demand promise. Leave block-table layout
alone unless measured S*L fixed overhead is dominant.

### K6. Focused test ladder; one integrated build

1. RSC17: card-free mem_test exact inverse cases
   kv_pool_mib_for_tokens_is_the_inverse_of_paged_tokens,
   kv_pool_mib_for_tokens_holds_across_block_sizes,
   kv_pool_mib_for_tokens_with_a_zero_group_beside_a_paged_one and
   kv_ranks_with_unequal_pools_carve_the_least_ranks_blocks_under_a_cap:
   retain legacy assertions, extend actual vector/rounded carve boundaries.
   Reuse test_vtp33_rank_own_split and its real vram_budget.cpp fixture for
   asymmetric room, named/unnamed reserves, explicit cap/claim and one-byte
   shortfall. Test preallocation drift and actual carve, not only inverse.
   Retention use mem_test replicated/zero, snapshot and arena tests plus
   existing native measurement/S5/cache replay fixtures; explicit stale
   policy/source/rank receipt failures. No GPU for synthetic geometry.
2. RSC18: existing sched_test fresh/preempt whole-prompt and running-headroom,
   fresh window, refill-capped split/windowed drafter, allocator error,
   external-capacity wait/cancel, backing failure and ahead-step cancel tests;
   mem_test rollback/window/COW/cache random interleavings; paging_fence_test.
   Add only missing mixed-length and multi-group failure/retry cases, exact
   metric invariants and supported shared-mode path tests. Preserve governor,
   draft and existing legacy tests. Tests named here are present at 2.3;
   their execution remains pending, not inferred from syntax.
3. RSC16: lint/composition for affected patch chain and targeted host/device
   syntax only when touched. Reuse bump canonical lane and its pinned build
   workflow; one integrated candidate build after card-free gates. Lease GPU
   only for actual startup/allocation and mixed long active decode + short
   progress, unique then shared prompts, baseline legacy and supported family
   matrix. Use existing ENG03/04/MEN17 acceptance and qualify no changed format
   twice. Recheck actual rank pages and host retention receipts; refusal is
   actionable evidence, not a successful 512K test. No full-engine test after
   each host change; do not skip required final integration gates.

## L. Implementation readiness: exact launch contract (2026-11-27)

K is the source/reuse map; L closes implementation choices, not a new work
package. Reviewed bump 1977b71a5: since K's e48149491 only libr2 unit inventory
and evidence changed, not the host patch chain. Main includes PR1727
(ecb354643). Reviewed the same verified 2.3 materialization and current
main serving consumers. No engine execution, build or GPU qualification.

### L1. Scope and configuration

Start with a single discrete fixed-backed text engine, TP1 or TP across
distinct devices, dense weights, ordinary full paged groups plus native
bounded sliding windows and linear/conv state. Qwen27B hybrid is required.
Decline generator branches, runtime reference/ring, streaming, unified,
expert placement/lending, broker/claim, explicit GPU cap and duplicate BDF
until their contracts are qualified. This is an explicit shared-mode
capability gate only; legacy must continue unchanged. Do not infer a
supported path from a model name or dtype. Check resolved declarations,
placement and rank facts, not YAML strings alone. Native ordinary windows
recycle; KVManager::blocks_for (1096) adds a ring to the full reference for
runtime_win, while scheduler full-group gate excludes WINDOW/RING. Thus
reference-plus-ring is not safely equivalent to bounded lookback.

Use the proposed native kv-cache-capacity option once: omitted = legacy,
auto = greatest safe backed packed capacity within native full-service
ceiling, integer = minimum packed target, not a hard cap nor guaranteed
arbitrary mixed demand. Require C >= L and reject explicit targets above
supported logical service ceiling (initially checked S*L); bound block
vectors by native per-group ceiling S*ceil(L/B), with existing bounded
window cap, not ceil(S*L/B). Leave excess VRAM unclaimed; idle retention
beyond ceiling is separate future work. Actual physical rounded capacity
may exceed requested logical minimum. Do not reduce L/S, change dtype,
offload weights or relax reserves to make a target fit. For auto with
measured-cache replay, remeasure on changed proposal or refuse; no silent
reseal in serving mode. Same native snapshot and resolved policy at measure
and start. Keep explicit cap semantics for legacy; no new raw-pool knobs.

Implement typed absent/auto/minimum in rad_core.h Config, config.cpp option
parser/given/report, native YAML route and native_fleet_config.cpp inspector
(protocol bigcherry.native-fleet-inspect/v2) paired with session.cpp strict
exact_keys runtime reader (around 2535), serving settings and native
manifest/cache contract. Version inspector producer/consumer together
where schema changes; test fake engine and native fleet config fixtures.
Do not add an x-bigcherry alias or parse YAML twice. Native snapshot digest
already participates in plan_cache v3 identity; add resolved policy and
geometry version to measurement identity; keep old readers explicitly.
Legacy option --vram-kv-mib is not a public override (config.cpp:873).

### L2. Exact rank and geometry handoff

VramFacts contains queue_scratch, prefetch, held_out and host_reserve.
Native resolve counts n_queues*queue_bytes + queue_scratch, prefetch,
arena/staging, drift, reserves, unified host_pinned; weights rounding and
stage funding occur later. Do not copy a partial ledger from
bc_plan_memory_note_ranks: overhead there is arena+staging+queues+stage,
omits queue_scratch/prefetch and reports raw f.free as claimable. That
record is consumed by p1_producer and plan_cache (rank_demand kind).
bc_plan_check::check selects least raw free BDF, not limiting margin.
Reuse native reserve/driver/held-out semantics in an extracted per-rank
budget calculation; account terms exactly once at the correct sampling
boundary. Maintain raw free separately. Report limiting rank by minimum
usable margin on same-rank terms; bind ordered BDF, reject duplicate
physical device. Test asymmetry where smallest free is not limiting,
nonzero queue scratch/prefetch/held_out, exact MiB and reserve shortfall.

Do not call vram_budget_resolve on shared mode then assume its conservative
common-floor rejection can be recovered; instrument both common native
refusal and rad0011 refit refusal. Trial on copied config preserves legacy
resolver behavior; shared proposal must consume exact same-rank facts and
native placement predicates. Reject unsupported combinations before
refit/physical pools. Single proposal contains resolved L/S/C policy,
per-rank weights/KV MiB, exact group IDs/kinds/block sizes/counts/bytes/
offsets, fixed state and checkpoint stage, usable room and limiting margin.
Use checked arithmetic on all multiplication/addition/alignment including
wire integer bound 2^53-1, int32 IDs and slot expressions; no float proof.
Pool::reserve (pools.cpp:96) aligns current cursor before each reservation:
mirror exact order, avoid counting both conservative fixed padding and
actual padding. Preserve alignment, reserve order and stream-map semantics
in legacy. Inverse acceptance uses vector and actual reserve cursor, not
a returned scalar. Test max feasible target plus next page/MiB, zero groups,
fixed==budget, failed group planning, multi-block geometry, cap and overflow.
Use KVManager plans not separate format-specific byte math. Initial fixed
backed path must actually commit all advertised usable pages.

### L3. Managed plan/replay chain

bc_plan_memory_note runs pre-pool, mutates cfg and then calls bc_plan::check;
note_ranks runs before pools.configure, done after plan and before
load_weights/configure_kv. Separate (1) immutable proposal + live check
before first pool, (2) successful allocation receipt, (3) post-carve
geometry/backing receipt and (4) front readiness. Use existing managed
receipt path; measure-only exit at done is not (3) or (4). No artifact
called actual capacity from plan alone. Ensure late startup failure cannot
produce a ready receipt or publish a successful deployment.

Version the shared EnginePlan alongside contract/serving.toml (v1
literal), v2/admission.cpp check_engine_plan/admission_margins and
refit_serving_pools, bc_plan_memory/check, engine_admission envelope,
bc_serving_engine, plan_cache, native_measurement/p1_measurement,
p1_producer, s8_rules/s8_replay, manifest_writer and existing fixture
consumers. Both native/P1 measurements currently demand version 1 and
divide fixed-inclusive kv_one_session by L, and P1 rejects retention>S*L.
Do not pass shared facts through these validators disguised as v1.
Keep legacy historical readers, strict expected/live field validation,
owner provenance/digests, no silent fallbacks. Shared retention uses
actual whole-engine entry slot size and floor(cap/slot), independent
checkpoint slots. Keep native prefix-cache-mib byte cap, no second
retention resolver; if unsupported native helper/remote-expert path
requires scalar-only facts reject explicitly before use. Check actual
reachable consumers before implementation; retain existing code guide
I2/I3 migration inventory. No arbitrary new receipt stream.

### L4. Native admission and context boundaries

Config.prefill_room_gate defaults false (rad_core.h:938); fresh gate in
scheduler.cpp:1991 is conditional. Shared mode must resolve it on
(omitted -> enabled for shared only; explicit off -> typed refusal for
shared mode), preserve legacy behavior. Inspect resolved gate in receipt.
Reuse existing governor, no alternate demand gate. Scheduler::init derives
ctx_limit from builder and pool_tokens via can_ever_fit. add_locked
checks prompt alone; step at 2770 refuses generation prompt == L, permits
embedding == L. Fresh prefill currently demands prompt+1 even for embed;
test if a full-capacity embedding is blocked and correct only if needed,
with no extra decode reservation for embedding. Generation test L-1
prompt + 1 output, L prompt refusal, overflow, exact L limit, speculative
headroom trimmed by existing context guard. Do not reserve maximum
requested output upfront. C >= L is packed cold-one guarantee, not
promise for arbitrary concurrent outputs; ensure runner extra block
headroom and bounded progress do not strand an otherwise feasible lone
request. Reject context overflow distinctly from temporary pressure.

At scheduler.cpp:2660 running ensure_from loop compares only <0; it evicts
then preempts or reports RAD_E_FULL even if allocator status was different.
2.3 admission path has exact failure status; extend/test same distinction
for running growth, preserving legacy behavior unless separately qualified.
Use existing error/close_request and external-capacity machinery, no new
error namespace; ensure serious device failure follows native fatal/fence
policy, never evict live data in response to a non-capacity error. Test
allocation error with another runner and cached prefix, retryable FULL,
deferred release/cancel, window recycling and partially recalled loan
in later-group failure. Retain existing tests; only add failing gaps.

### L5. Definition of ready for implementation

No further design plan needed. Work in order: pin and capability/typed
config fixture, extract/verify exact rank geometry and proposal, version
measure/check receipts and retention, run unchanged native scheduler with
shared geometry then minimal fixes, one integrated candidate and managed
GPU qualification. RSC17 first two stages and receipts/retention; RSC18
runtime; RSC16 final qualification and retirement. Implement in small
reviewable commits; no second allocation engine. Required pre-code gate:
confirm final merged 2.3 source/patch identity and reconcile any changed
anchors. Required before release: card-free tests for every named boundary,
actual post-carve and readiness receipts, cold L and overlapping mixed
long-active/short progress, legacy parity and supported family coverage.
No engine/GPU validation has occurred in this review.

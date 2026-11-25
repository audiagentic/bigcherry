---
id: RSC03
order: 3
plan: run-serving-configuration
state: in_progress
created-at: '2026-11-02T00:53:06.703369+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: L
---

# One derived start-up resource plan: host RAM, pinned memory and VRAM budgeted together

## Description

Replace independent memory flags (host pool, n-gram placement, host KV tier, checkpoint slots, helper VRAM budgets) with one resource ledger resolved at start-up from discovered RAM/VRAM/pin and cgroup limits, model geometry and the actual helper owner map, plus a safety reserve and a performance objective (not 'the most that fits'). Measured 2026-11-02 on flashnext-4gpu: the hand-picked 22,528 MiB host pool used ~7.1 GiB; a 9 GiB pool loses nothing fresh (44K prefill 3,825 vs 3,837 tok/s) and the freed RAM as a 12 GiB host KV tier removes the long-running-server slowdown (6-stream decode after an ~1M-token session fill 46.9 ms vs 52.8; 8 sessions cold 3,026 vs ~2,450 tok/s) with the n-gram table kept in RAM; a 32 GiB tier with the n-gram table on disk helped prefill but left 0 GiB free RAM - a derived plan must reject that. Reuse radiance's budget machinery (core/mem/vram_budget.cpp resolve, core/engine.cpp loan_tick, core/mem/kv.cpp spare_tokens/grow_for, core/engine_bringup.cpp resolve_table_placement and size_checkpoint_store); add a managed patch so the engine sizes its host pool after the helper owner map is known (the planner must use the real ownership planner or its published map digest, not a Python approximation).

## Steps

Lead-approved plan (bigcherry-1, 2026-11-02). Work on branch feat/rsc03-resource-plan, cut from feat/rsc06-helper-manifest. Each slice gets its own PR once RSC06 has merged.

1. Slice 1, host ledger (card-free). DONE on the branch:
   - detect: memory cgroups, memlock with CAP_IPC_LOCK, amdgpu pin cap
   - calibration store (calibration.{toml,h,cpp}; seeds carry evidence; cache via --calibration)
   - memory.host_reserve_mib derived from max(floor_mib, total_fraction x MemTotal), raisable only by site hard_floors; engine.host_headroom_mib = reserve, so there is one reserve
   - ledger.{h,cpp}: limits MemAvailable, cgroups, pin cap and memlock; the checkpoint store planned at calibration.checkpoint_share = want and shrunk toward the floor before the tier falls to 0; strict v1 engine-plan contract
   NOT wired into resolve() until 1b and 2 supply real sizes (lead: no placeholders).
2. Slice 1b (B1): managed radiance patch for `radiance --manifest M --plan-memory --json`. It builds the Program with the kernel code target set by the manifest's engine-card arch name, with no device, no HIP init, no allocation, pin or mailbox, and no weight payload reads. It emits the v1 plan JSON (host consumers with bytes, residence and engine function; checkpoint slot/floor/want; kv bytes per token; expert unit sizes). Thin upstream edits only. Arch with no code objects fails closed. Report wall time; cache by container digest plus planning inputs if slow.
3. Slice 2: move select_owned (moe-remote worker_owners.h) and the Container reader into a GPU-free shared library linked by bc_serving and r3_moe_worker, behaviour unchanged (test: identical owner sets before and after, real container and heat if card-free readable). Manifest carries the owner-map digest; the helper refuses on mismatch. Head budget = vram_available - arch reserve - helper overhead. Tail from the objective's KV backing floor (calibration.kv_backing_floor.fraction, latency 0.50 / balanced 0.60 seeds; throughput no floor). Latency still sends what doesn't fit engine-local to the tail before the host pool. Host pool = embedding (ram) + unowned non-local expert units + margin. Wire the ledger into resolve().
4. Slice 3: engine patch makes the manifest budgets authoritative; fail closed at start-up if short, never grow.
5. Slice 4 (GPU, gpu lease, ask the lead first): RHP12 fresh and retained-session cohorts for throughput and latency, decode and prefill separately; hardware diff of --plan-memory against the real start() sizing (must be identical); remove the literal 9216/12288/22528/14336 origins.

## Detailed Solution & Technical Design

B1 findings (2026-11-02):
- Footprint: core/place/planner.cpp weight_footprint(prog, ...) is pure arithmetic over a built Program: embedding, ngram, statics, experts, expert_unit_max, mapped/hosted. It depends on resolve_sites (host_ok/device_ok per weight), which depends on which kernel rows the registry kept.
- Device dependency: core/plugin/loader.cpp drops a kernel plugin's device rows when rad::dev_code_runs(code_from, code_to, code_device_) == 0. That calls the AQL backend a_code_runs (core/device/aql.cpp:3868), which checks each registered bundle with bundle_code_object(bundle, G().dev[device].isas) (aql_code.cpp:74). isas come from hsa_agent_iterate_isas (agent_isas, aql.cpp:272).
- Bundles are recorded by the interposed __hipRegisterFatBinary (aql.cpp:4006) even when HIP returns no handle, so they exist with GPUs hidden.
- The arch-keyed path therefore needs: a patch-owned code-runs check taking an ISA list built from the arch name (amdgcn-amd-amdhsa--<arch> plus whatever generic ISAs HSA reports for that arch; the hardware check must prove the list matches agent_isas); a registry setter beside set_code_device; the planning entry calling the engine's existing read_metadata/load_plugins/declare and weight_footprint, checkpoint_slots_want x slot and arena_plan_bytes through a friend declaration, never copied bodies.
- Open risk to verify on hardware: whether dlopen of a kernel plugin plus HIP's __hipRegisterFatBinary creates a KFD process context (check no /dev/kfd fd and no /sys/class/kfd/kfd/proc/<pid> during --plan-memory).
- Upstream main.cpp has no hook yet; rad0009 added one for --manifest (bc_serving_engine_argv). The plan hook goes beside it as a one-line call.
- Found and handed off: rad0009 read flat keys while bc_serving nests values; bigcherry-1 is fixing it on fix/rad0009-nested-manifest. Nested is the contract.


Slice 1b = R3 (lead decision 2026-11-02; B1 as specified needs a plan() refactor, see R2 in notes):
- Managed patch hook: one thin call after Engine::plan() in start() writes the consumer sizes as the strict v1 plan JSON (ledger.h EnginePlan: host consumers with bytes, residence and the engine function each came from; checkpoint slot/floor/want; kv bytes per token; per-card footprint). A `--plan-memory` mode runs start() as far as plan() and exits before weights load or arenas allocate. The exit logic lives in patch-owned code.
- Cache, under the work root (not $HOME), with provenance (when measured, which build). Key: container digest; engine build identity (build_plan_id or binary hash, plus the kernel/plugin set); engine card arch and VRAM; the manifest's planning inputs (max_sessions, max_context, checkpoint_interval, kv dtype, placements, layout). Any change is a miss.
- On a miss: `bc-serve --explain` never touches a GPU. The native session owner measures with the engine and its helpers, stores the plan, and supplies a sealed native configuration, frozen plan key and owner-attempt identity to `bc-serve --write-manifest OUT`. The writer only reads that session-owned cache; a cache miss exits 10. Raw plan-file import options have been removed.
- Slice 3's start-up check compares the live plan() against the manifest's budgets, failing closed when short. Same function, same numbers.

## Code Samples & Guidance



## Files



## Validation

Budget arithmetic tests (host, pin cap, cgroup, n-gram, KV tier, checkpoints, expert pool, staging, dual generations); RHP12 fresh and retained-session cohorts pass; configurations leaving no safety reserve rejected with an explanation.

## Effort & Risk



## Standards



## Acceptance Criteria

No fixed memory values in the default launch path; the retained-session cohort matches fresh-server step time within the RHP12 gate; explain shows why each budget has its size.

Native YAML measurement and serving use same measured resource/ownership facts; no full manifest required for plan derivation, cache or child launch. Verify structured live-room refusal before weights, bounded remeasurement/retry and measured provenance, same native plan and actual bytes, exact ordered BDF, per-rank units, queue/host/cgroup/pin/VRAM reservations, helper RSS and no uncovered expert. Cache fresh admission and compatible carry, no replay of stale settings. Preserve existing RHP12 cohort and no fixed resource constants.

## Notes

Depends on RSC01/RSC02; feeds RHP13 (runtime) and RCS10 (derived settings). Runtime reclaim between finished-session KV and expert residency stays RHP13.


R2, draft upstream proposal for radiance (codeberg StillDeadcode/radiance). The owner must approve before anything is posted; do not post.
Title: Split Engine::plan() into a pure sizing step and a device measurement step.
Why: plan() (core/engine_bringup.cpp) is the only place a configuration's memory needs are computed: weight footprint, n-gram and embedding tables, checkpoint slots, host arenas, KV floor, the VRAM split. The device measurements it uses (queue VRAM from two stream creations, rad_dev_props capacity/free, rad_xp_supported, already-held VRAM) are taken in the same per-rank loop, after kernel inits. A supervisor or capacity planner therefore cannot ask "what would this model and these options need on a gfx1201 card with N GiB" without initialising the device. Serving tools end up hand-picking --host-pool-mib, --prefix-cache-mib and helper budgets instead of deriving them.
Proposed interface:
- struct DeviceFacts { arch, capacity, free, already_held, queue_bytes, xp_supported, integrated, host_reserve } per rank, with HostFacts { mem_available, cgroup rooms, pin cap, memlock } beside it.
- int plan_sizes(const Program&, const Config&, const DeviceFacts[], const HostFacts&, PlanSizes* out): pure arithmetic, everything plan() computes from those inputs, with no device calls.
- plan() = measure_device_facts() + plan_sizes() + apply. Behaviour is unchanged: the same numbers, by construction.
- A device-free route to a declared Program for a named architecture, which requires the kernel registry's code-object check (dev_code_runs) to take an ISA list rather than a device index (set_code_isas beside set_code_device). Kernel inits needed only for device residency are deferred until after sizing.
Benefit upstream: `radiance --plan` / capacity checks without a GPU, deterministic sizing tests (plan_sizes on synthetic facts), and the startup report's decisions become testable without hardware.
Ask: whether upstream wants the split, and which name and API shape they prefer. BigCherry would contribute the refactor and tests.


Pending v1 placement. Owner direction 2026-11-02: one specified manifest v1 format, under design in GPT request req_main_c69d1519f282. Until then RSC03 adds no new manifest shapes or top-level sections and renames nothing. Each new value sits under the nearest existing section and is listed here for the migration:
- values.memory.host_reserve_mib: derived, the calibrated host reserve (serving.cpp schema field; provenance origin "calibrated").
- values.engine.host_headroom_mib: derived alias of memory.host_reserve_mib, passed to radiance --host-headroom-mib.
- facts.host.memory_cgroups / facts.host.memlock / facts.host.amdgpu_pin: detected (detect()).
- site input table hard_floors.memory.host_reserve_mib: site-only, may only raise (a new top-level input table beside hard_caps; v1 should decide whether floors and caps share one constraints namespace).
- Provenance origin value "calibrated" with source "calibrated (seed|cache: <date>) <key>: <evidence>".
- The calibration store, self-contained in components/serving/calibration.{toml,h,cpp}: seeds compiled in, cache via bc-serve --calibration FILE (an Options field, not a manifest key).
- The owner map, self-contained in components/serving/experts.{h,cpp} (selection) and owner_map_digest in serving.cpp. It is planned to sit in the manifest as values.handoff.owner_map_sha256 (whole map) and process_helpers[i].owned_sha256 (each helper's own set, which the helper verifies at start-up), both pending v1 placement.


- The rad0011 --plan-memory JSON (plan_version 1) is a second machine format and falls under the manifest v1 design too. Its shape stays as it is for now; v1 is expected to fix its name and placement (probably a `plan` document or section next to the manifest).

Plan cache key and the circularity rule (slice 1b). Some plan() outputs depend on what the plan itself derives: vram.experts, vram.kv and vram.expert_floor depend on host_pool_mib (the expert floor is what the host pool cannot hold), and cards[].engine_expert_plane depends on the helper budgets (rad0002 marks helper-owned units out of the footprint). The cache key therefore excludes the derived memory outputs (memory.*, engine.host_pool_mib, engine.prefix_cache_mib, engine.host_headroom_mib and the helper budgets), and the derivations read only the budget-independent fields: host consumers, checkpoints, kv_one_session, vram.claimable / statics / elastic / stage / kv_ceiling, cards[].capacity / expert_unit_max. Slice 3's start-up check compares the live plan() (with the real budgets) against the manifest.


resolve() wiring: deferred to the v1 migration (lead decision 2026-11-02, option b). It will be done once, in the v1 shape. This list is an input to the v1 migration plan. The wiring adds:
1. Options fields and bc-serve flags (all explicit, no defaults):
   - Options.plan_cache (dir) / --plan-cache DIR: the measured-plan cache under the work root.
   - Options.engine_binary (path) / --engine-binary PATH: for engine_identity, beside engine.radiance_home.
   - Options.calibration_cache / --calibration FILE: exists already.
   - resolve() reads engine.model (the container: owner map and container_identity) and handoff.heat (the owner map), which become real files at resolve time.
2. The four derived memory values replace their interim literals:
   - memory.helper_budgets_mib: heads = vram_available - archs.toml reserve-mib - helper overhead (overhead per arch, calibrated from helper logs); the tail from the objective's KV backing floor (calibration.kv_backing_floor.fraction; throughput none), with L/T/P split as in the slice plan.
   - memory.tail_budget_mib: = the tail helper's budget.
   - memory.host_pool_mib: embedding (when ram) + unowned experts not engine-local x expert unit bytes, from plan_owners and the plan.
   - memory.host_kv_mib: host_ledger's tier.
   - Provenance: "derived" with inputs naming the plan key, the owner map digest and the calibration entries. On a plan cache miss, --explain shows each as "needs a measured plan (run: bc-serve --plan-memory)" and --dry-run refuses.
   - The ledger's checkpoint_slots is passed to the engine as --checkpoint-slots (engine.checkpoint_slots), so plan and engine agree.
3. Owner map in the manifest:
   - values.handoff.owner_map_sha256: owner_map_digest over all helpers.
   - process_helpers[i].owned_sha256: the helper's own set.
   - r3_moe_worker recomputes its set with bc_experts::select_owned and refuses to start on a mismatch.
   - read_manifest's exact_keys for process_helpers gains owned_sha256.
4. Test migrations:
   - serving_test's Fixture::standard() and every site literal using /model.rad and /heat.hot-first move to the synthetic container (test_rad_fixture.h write_rad) and a written heat file, plus a fixture plan-cache entry (plan_cache_store with a test plan JSON).
   - The same for moe-remote/helper_settings_test.cpp (RSC06's) and anything resolving through bc-serve.
   - New: the explain miss text, the dry-run refusal, the derived values' provenance, the owned_sha256 mismatch refusal in the helper, and the launch-equivalence test updated for the derived memory values (no longer the launcher literals).


5. Start-up plan check (slice 3), also deferred to v1. The manifest carries the expected plan, the measured plan document the budgets were derived from (or its cache key plus the document), and the engine runs bc_plan_memory_check(expected, live) at start-up, right after plan(), failing closed when any live need exceeds the expected value. The check itself is written and tested: rad0011 core/bc_plan_memory_check.{h,cpp}, a pure function of two plan documents, compiled and run by test_radiance_patch_packages.TestPlanMemoryPatch.test_plan_check_compiled_and_run. v1 only supplies the expected document and the one thin call.


6. Design question for the v1 review: planning without live helpers. The engine's plan() footprint depends on the helper-owned expert units (rad0002's bc_moe_remote_mark_footprint reads the helpers' published map through the libr3moe mailbox), so measuring a plan needs the helpers up. Interim (lead pre-decision): if the 2026-11-02 hardware run's 1a confirms this, `bc-serve --plan-memory` becomes "start the helpers from the manifest, then the engine with --plan-memory, then stop all", sharing RSC02's supervision code rather than duplicating it. Target: the helper-owned set is known before start-up (plan_owners, the same select_owned the helpers run), so v1 lets the engine take it from the manifest (owner_map_sha256 plus the owned sets) for planning, and uses the live shm only to verify it at start-up. Then --plan-memory needs only the engine card, and the start-up check compares like with like.


   Confirmed evidence for item 6 (hardware, 2026-11-02, run rsc03-plan step 1a): `bc-serve --plan-memory` with the engine alone (no helpers) failed closed after 4.9 s. The engine reached plan(), but start-up stopped "out of memory" ("weights-from-disk: 33.13 GiB of weights a rank are read from the container on disk"): with no helper-owned units, plan() computes a different plan. A plan exists only with the owner map, which supports "the engine plans from the manifest's owned sets".

Decisions after the hardware run (lead, 2026-11-02):
- Slice 3's claimable check treats claimable as a live fact. Live claimable must be at least the planned statics + kv + experts + stage, with no tolerance constant. The run showed free VRAM moving by 8192 bytes between two starts. The other consumers stay exact.
- calibration.checkpoint_share is seeded tier_first, with calibration.host_kv_tier.target_mib = 12288 for flashnext-4gpu. This reproduces today's launch, which gives 43 snapshots (4.72 GiB) of a want of 72 with the 12 GiB tier. Want and floor stay visible in explain. Refit criterion: RHP12's retained-session cohort decides whether to move toward want.


Owner decision (2026-11-02, via bigcherry-1): v1 goes ahead as proposed in GPT review req_main_c69d1519f282, with curated engine options, a manifest that owns the initial owner map, and a one-release cutover, refined as it goes. This settles item 6: the engine plans from the manifest's owned sets, and the live shm only verifies them. RSC03's pieces (calibration store, host ledger, owner map via bc_experts, measured-plan cache, rad0011 --plan-memory and bc_plan_memory_check) are inputs to v1's plan section. The wiring list above (items 1 to 6) is the migration's checklist for RSC03.


v1 contract decisions affecting RSC03 (lead, 2026-11-02; v1 slice 1 design approved):
- Manifests carry integers only. Fractions use fixed-scale integers with the scale in the field name (e.g. calibration fractions as *_ppm, balance speeds as *_milli). Calibration evidence keeps decimals only in its free-text string. The calibration seeds' fractional values (host_reserve.total_fraction, kv_backing_floor.fraction) move to that form in v1.
- ServingInputV1 has one `constraints` namespace holding site-only, tighten-only caps (max) and floors (min). This replaces today's hard_caps and hard_floors tables.
- The engine binary is not an input: bc-serve resolves it relative to its own installed location (radiance_home/../bin/radiance), covered by the package identity. The plan cache directory is derived from the work root in site config (<work-root>/plan-cache). Neither is a CLI flag in v1, so today's `bc-serve --plan-memory --plan-cache DIR --engine-binary PATH` flags (and wiring list item 1's Options.plan_cache / --engine-binary) are replaced at the v1 cutover.

2026-11-25 owner direction: no launch manifest, all authored serving configuration YAML, no compatibility shims or legacy backports. docs/design/radiance-native-yaml-launch.md supersedes earlier manifest-launch instructions; completed RUM04/05/06/09/12 evidence is historical, not proof of the new launch path. Existing capture manifests may remain output-only. No new programme or second resolver. Existing RSC03 remains sole resource planner. Extract reusable measured plan, P1/rank-demand, S8 admission and runtime guards from Manifest dependency; same arithmetic and owner map, no new allocator. rad0011 check expected typed plan at actual pre-weight native Engine boundary; bind derived pool/prefix/headroom/checkpoint and ordered BDF before Fleet reservation/Engine construction; verify per-rank vs aggregate MiB, native queue reserve/staging and helper RSS. Preserve RUM12 measured fast reuse, no stored launch input. RSC11 transferred RSC10 bounded margin refit remains, RSC14 owns independent host race. Old --manifest/--write-manifest examples are historical.

2026-11-25 bump integration: docs/design/radiance-bump-integration.md and MEN24 gate order. Candidate admission uses existing typed planner and native pre-weight check. Reuse measured data across pins only under explicit semantic compatibility of actual native defaults, ordered devices, resource geometry, selected providers, host/VRAM/queue reservations and fresh live room; record carried-from evidence. Changed power settings invalidate performance calibration, not unrelated memory arithmetic. Wrong candidate or resource identity refuses or triggers bounded remeasurement; RSC14 host-race gate remains separate. No executable manifest, no second cache/planner.

2026-11-25 second integration review: docs/design/radiance-integration-review-20261125.md is the exact-source review and controls conflicting earlier wording. Shipping pin remains b6b8751 (2.2.0), upstream observed main 1c104de4 is a development candidate, 119 commits and 327 changed files; no automatic repin or backport. No new compatibility layer, launch manifest, bump driver or identity machinery. Candidate 5e0cd1e2 per-card TP budget maps cfg.devices[rank], least(free - max(per-card headroom, host reserve)) common split, expert_tiered spare room; explicit claims separate. 4865bd6b queue-reserve 1–24 default8, 80e094b3 pinned NUMA, a8c8066d mlock. Reconcile at actual native Fleet reservation and Engine::plan; native authority, avoid double subtract reserve/queue bytes or copied old scalar planner. Test actual queue opening, memlock/cgroup and host budgets, unequal ranks/reversed order/explicit claim/exact fit, helper reservation and bounded remeasurement. Cache carry only native semantic compatibility+fresh room. Keep RSC14 race gate.

2026-11-25 serving/configuration closure sweep: docs/design/radiance-serve-config-review-20261125.md. Current CachedPlan stores v2 EnginePlan, rank demand and measuring_manifest_sha256. Extract measured domain data/owner attempt receipt from launch reader; preserve existing model/build/native YAML/root/ordered devices/helper/budgets cache checks; only actual memory-affecting controls belong to plan key, time/energy calibration separate. Replace launch-coupled cache version with measurement-only data, incompatible old entries remeasure, no dual reader. Keep atomic tracked successful measurement publication and fresh cgroup/pin/host/VRAM recheck, structured preweight one retry under RSC11; RSC14 tests contention, no second retry.

docs/design/radiance-native-yaml-launch.md §4.2 (code seam review, 2026-11-25): session.cpp:3398–3630 has TWO distinct recovery mechanisms: existing measured host-pool convergence (kMaxHostPoolChanges=2, at most three attempts, repeated pool refuses, measuring helpers must retire segment before remeasure) and S17 structured preweight live-room one retry. Preserve both; do not collapse convergence into one retry or add a third recovery loop. Replace writer exit 11 with existing typed requested-pool result in owner. Keep S8/P1 arithmetic, draft host KV ceiling -> smaller final measured tier, same pool/reserve, rank receipt for ngram auto, helper RSS/baseline release, rad0011 plan/rank output and expected plan/KV floor/BDF checks. Apply derived per-rank host_pool_mib before Fleet computes aggregate reserve, native final allocator authoritative. Keep one named measurement reuse path using existing plan_cache/stored cache indexing, load only measured fields not stored config; incompatible launch-coupled entry remeasure. No new cache. Preserve preweight/quality distinctions (offline scorer does not open serving prefix/checkpoint store).

2026-11-25 current-pin implementation decision: docs/design/radiance-native-yaml-launch.md §4.3 controls. Finish S18 on approved Radiance b6b8751 (2.2.0); do not wait for a bump, cherry-pick newer Fleet or extend scope. Native single-model Fleet is already present; reuse actual source behavior. Pinned Fleet use_broker() is models_.size()>1; load_op host reservation and VRAM guess/grant are broker-only. Do not enable broker to solve per-card headroom. Existing rad0011 records live Engine::plan and checks before load_weights; it checks/refuses, not a per-card VRAM allocator. Native vram_budget.cpp resolve takes min free, max configured scalar/host reserve and largest rank costs; Engine::plan measures queue bytes per rank and budgets actual 2/3 (+prefetch where relevant) queues, with existing expert_tiered rank split. Keep that arithmetic and rad0016 ordering. Fleet::reopen_queues (837) returns immediately for <2 models: no eight-queue Fleet reserve to recover on our single-model serving. No duplicated subtraction or local queue knob. First measure using actual selected device/plan and same native settings then final check; carry only compatible existing measurements, preserve both pool convergence and one preweight shortfall retry. If real current layout cannot meet an existing requirement, show exact rank facts/refusal; prefer existing scalar native control where it meets requirement without regression, otherwise smallest independently removable managed change at native budget boundary after owner review. Do not backport 5e0cd1e2 or reproduce whole per-card splitter; explicit per-card map unsupported until qualified. No artificial headroom limit for currently supported configs.

## Status 2026-11-19

The derived start-up resource plan is in production use (C++ RAD/BDF ledger: `resources: derived host pool N MiB, host KV N MiB`, then the measuring start re-measures the pool, for example 1033 -> 2247 MiB on flashnext-4gpu). Observed today: the engine's own host admission is the last word (n-gram table 47.68 GiB needs about 74.8 GiB available including the 4.59 GiB headroom; a start with 74.1 GiB available was refused), so a start right after another heavy process can fail for memory the ledger cannot see; consider reporting MemAvailable at the ledger step. The policy-calibration key no longer includes the host pool or helper budgets (#1169), so a re-measured pool does not orphan a calibration record.

## Change Log

- 2026-11-02T00:53:06.703369+00:00 (created-by): Created by agent

## Ledger-events

- chg_20261102_005338_planning-one-serve-command_5259
- 2026-11-02T00:53:52.338745+00:00 (updated-by): Updated: section:ledger-events
- chg_20261102_012258_long-running-flash-next-server_6339
- 2026-11-02T01:23:02.605425+00:00 (updated-by): Updated: section:ledger-events
- 2026-11-02T08:32:40.131206+00:00 (updated-by): Updated: section:steps, section:detailed_solution
- 2026-11-02T08:32:43.990265+00:00 (state-transition): State: pending → in_progress
- 2026-11-02T08:34:08.493382+00:00 (updated-by): Updated: section:detailed_solution, section:notes
- 2026-11-02T08:59:59.182507+00:00 (updated-by): Updated: section:notes
- 2026-11-02T09:08:56.161715+00:00 (updated-by): Updated: section:notes
- 2026-11-02T09:13:27.207054+00:00 (updated-by): Updated: section:notes
- 2026-11-02T09:16:58.106092+00:00 (updated-by): Updated: section:notes
- 2026-11-02T09:35:39.506296+00:00 (updated-by): Updated: section:notes
- 2026-11-02T09:46:22.035505+00:00 (updated-by): Updated: section:notes
- 2026-11-02T09:56:31.372057+00:00 (updated-by): Updated: section:notes
- 2026-11-02T09:58:14.772754+00:00 (updated-by): Updated: section:notes
- chg_20261102_155533_the-start-up-memory-check-no-l_8617
- 2026-11-02T15:55:37.776302+00:00 (updated-by): Updated: section:ledger-events
- chg_20261102_193311_the-engine-now-checks-at-ever_1908
- 2026-11-02T19:33:16.115864+00:00 (updated-by): Updated: section:ledger-events
- chg_20261102_185212_bc-serve-and-the-moe-helpers-n_5105
- 2026-11-02T18:52:16.705767+00:00 (updated-by): Updated: section:ledger-events

## RSC11 S7: offline v2 resource derivation (2026-11-10)

The preparatory GPU-free Python ledger is `tools/bigcherry/engine_adapters/radiance_resources.py`. A caller obtains `radiance_inventory.Layout` from bc-serve's BDF inventory and calls `Layout.derive_resources(model, targets, host, worker_footprints_bytes=...)`. The result carries a host pool, host KV tier, per-helper expert budgets and a byte-exact explanation. Inputs are mandatory: no guessed model footprint, ownership plan, or worker footprint. Re-resolve on device/owner-map/model/context/RAM change.

The shipped `engines/radiance/layouts.toml` now omits all literal helper `budget-mib` settings, allowing the existing v1 bc-serve worker-footprint-based budget derivation to take effect. The v1 manifest schema, C++ resolver, manifest writer and generated registry have not been changed.

**Required v2 handoff, for S10/S13:** bc-serve must publish the engine-measured host embedding, unowned expert, staging, n-gram, checkpoint and other resident bytes; the actual owner-map digest; host-pool margin calibration; KV bytes per token and the context/retention target; live MemAvailable, effective cgroup headroom and pin availability; and each selected helper BDF's complete non-expert worker memory (lanes, pointer banks, slot map, incoming slot) from `worker_footprint.h`. The output must carry provenance and be invalidated/recomputed with inventory, owner map, model or targets. The v2 writer should use those values to populate the existing resource budget concepts, enforce RAM/pin admission and compare with engine `--plan-memory`; fail closed if any required measurement is missing. No v2 contract fields are introduced by S7.

**Unresolved until the single v2 cutover:** `serving.cpp` still holds the v1 host-pool/host-KV defaults and engine-only recipe-specific pool sizes remain in layouts.toml. The v1 Bash helper launcher now calls the GPU-free Python launch adapter on every start: bc-serve selects actual BDF roles, helper footprints and owner map; the RAD directory and current host memory facts feed the shared resource ledger. The adapter prints a breakdown and injects computed host-pool and host-KV MiB without restoring the old constants. The C++ v1 defaults remain for non-launcher callers until the single S13 cutover, when model/plan sizes and KV geometry must be carried through a typed v2 resource interface. The v1 manifest schema is unchanged; S16 removes the Bash surface.
- 2026-11-19T07:37:03.328605+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4554
- 2026-11-22T12:18:29.092969+00:00 (updated-by): Updated: section:detailed_solution
- RV4591
- 2026-11-25T01:37:00.647185+00:00 (updated-by): Updated: section:notes, section:acceptance_criteria
- RV4622
- 2026-11-25T02:13:25.537739+00:00 (updated-by): Updated: section:notes
- RV4639
- 2026-11-25T02:43:57.354274+00:00 (updated-by): Updated: section:notes
- RV4656
- 2026-11-25T03:25:50.227755+00:00 (updated-by): Updated: section:notes
- RV4673
- 2026-11-25T05:40:50.133015+00:00 (updated-by): Updated: section:notes
- 2026-11-25T05:53:35.312402+00:00 (updated-by): Updated: section:notes

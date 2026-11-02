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
- On a miss: `bc-serve --explain` and `--dry-run` never touch a GPU. They report each affected budget as "needs a measured plan (run: bc-serve --plan-memory)" and refuse to emit a launch manifest. The serve path (RSC02) runs the --plan-memory step on its engine card under the launch's lease semantics, then resolves. `bc-serve --plan-memory` warms the cache explicitly.
- Slice 3's start-up check compares the live plan() against the manifest's budgets, failing closed when short. Same function, same numbers.

## Code Samples & Guidance



## Files



## Validation

Budget arithmetic tests (host, pin cap, cgroup, n-gram, KV tier, checkpoints, expert pool, staging, dual generations); RHP12 fresh and retained-session cohorts pass; configurations leaving no safety reserve rejected with an explanation.

## Effort & Risk



## Standards



## Acceptance Criteria

No fixed memory values in the default launch path; the retained-session cohort matches fresh-server step time within the RHP12 gate; explain shows why each budget has its size.

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

# Radiance upstream migration review — 2026-11-18

## Decision and scope

Do not perform a pin-only update. Moving from BigCherry's v1.17.0 source to
current upstream is a configuration, runtime-lifecycle, plugin-ABI and kernel
migration. Preserve the current production source while porting in an isolated
source/build lane. This review changes no pin, patches, runtime configuration,
GPU state or acceptance status.

Reviewed BigCherry snapshot: `98d481cc836d71685ee9c0e57214dde7a9ba6529`.
Pinned upstream v1.17.0 resolves to `19d4e8fe` (the mirror's annotated tag object
is `8ac9a79c99b0f9cb5fe3a9283d3c402a71bd8378`; it is not the source commit).
Target upstream main: `b8bca7885a5a118bfa65a62dde74e0a6e2d4e42e`, fetched from
https://codeberg.org/StillDeadcode/radiance.git. Target identifies itself as
2.1.0 but includes subsequent commits; pin the SHA rather than assuming the
2.1.0 tag contains all reviewed changes.

Delta: 160 commits, 898 changed files, 178,954 added and 60,533 removed lines
(including vendored dependencies and generated documentation). The full commit
and file lists are retained alongside this report. These totals are not a
measure of handwritten engine complexity.

## Review method and limits

Used a fresh upstream mirror and separate pristine base/target worktrees. Loaded
BigCherry's production set through `resolve_exact`, then used the existing
`apply_patch(..., dry_run=True, texts=...)` API in dependency order. No patch
outputs were written into source trees. Base: all 30 patches apply without failed
edits. Target: 14 patches encounter failures, totaling 166 failed edits; 16 have
no anchor failures. Detailed per-edit failures and target files are in
[applicability.json](applicability.json).

After a failed edit, the exploratory engine continues with the partial in-memory
composition. Later failures can be consequences of earlier failures, so 166 is
an observed repair surface, not 166 independent upstream defects. Clean anchors
prove neither compile success nor semantic compatibility. Optional patches were
tried separately against that partial production composition; their results are
leads, not proofs that their intended experimental dependency sets compose.
No compilation, GPU benchmark, numerical parity, KLD or serving acceptance was
performed. Validation below is required migration work, not a claim of success.

## Manifest and extension redesign proposal

See [the concrete YAML, placement and extension proposal](MANIFEST-AND-EXTENSIONS.md).
Recommendation: reuse native upstream YAML for authoring, add a closed top-level
`x-bigcherry` intent namespace, and retain generated immutable runtime contracts.
Move injected implementations into owned components with narrow typed hooks;
share one placement plan and upstream allocator instead of duplicating policy.

## Findings requiring design decisions first

### P0 — server entry point is incompatible with our manifest-to-argv bridge

Upstream `core/main.cpp` now accepts `--config FILE` and optional `--check`;
old server flags are refused with a YAML translation. It loads `FleetConfig`
and normally calls `fleet::run`. Offline modes run one model. See upstream
[main.cpp](https://codeberg.org/StillDeadcode/radiance/src/commit/b8bca7885a5a118bfa65a62dde74e0a6e2d4e42e/core/main.cpp)
and `docs/MULTIMODEL.md` §4. BigCherry rad0009 currently translates its typed
manifest into engine argv and marks supplied options. That behavior no longer
launches the upstream server correctly, even if most insertions find anchors.

Choose explicitly: retain BigCherry's manifest as authority and adapt it to
upstream `Config`/`FleetConfig`, or emit validated YAML as a derived artifact.
Do not add a second independently editable policy file. Preserve strict unknown
key rejection, CLI/config precedence, model paths, stable device ordering,
helper role binding and measuring/serving separation. Port every typed option
mapping and the engine plan exporter before enabling fleet serving. Update
`engine.toml` serve metadata, smoke/accept arguments, native bc-serve launch,
release fake engine fixtures, scripts and documentation that still use old flags.
Keep helper configuration on its existing contract; upstream YAML does not
replace the external helper transport or BigCherry placement plan.

### P0 — fleet lifetime conflicts with process-global extension state

Upstream 2.0 introduces multiple `Engine`s per process, shared VRAM accounting,
parking, unload/reload, priority reclaim and hot reload. Architecture instances
now index `g_model[rad_build_slot(ctx)]` / `rad_slot(c)`, not just rank.
`RadBuildCtx::slot_base`, `RAD_MAX_SLOTS=32`, `rad_arch_multi_instance` and
`rad_plugin_freed` are concrete new interfaces. `Registry` data now lives in a
shared set (`set_->...`).

rad0002 injects a process-wide callback/layer-map interface and caches policy
answers in function-static values; helper settings and tuning pins also need
lifetime auditing. A clean anchor does not make these fleet-safe. First port
single-model operation with a fail-closed restriction to one live engine for
BigCherry remote MoE. Full fleet support then needs engine/TP-group identities
on registration, tickets, layer maps, telemetry and helper ownership; unregister
and drain on unload; lifetime-scoped policy; and no outstanding callback into
freed arenas. Validate reload, partial startup failure and same-rank concurrent
models. Do not opt a plugin into multi-instance support merely to bypass a guard.

### P0 — plugin ABI and collective synchronization need a complete rebuild

Upstream `RAD_ABI_VERSION` moves 35 -> 41. `RadBatch` gains fields including
`anchor_rows`; `RadBuildCtx`, probe fields and runtime exports change. ABI layout
fingerprints cover structure fields. Rebuild all upstream and BigCherry plugins
against the same patched headers and update fingerprint rows for added fields.
Never carry old shared objects into a new release home.

`libr4d/r4d_ar_entry.hip` replaces a single rendezvous with one per `tp_group`,
with timeout arrival rollback. rad0017 currently tries to inject a shared libr3/
libr4d rendezvous and poison handling into the old singleton. Port its behavior
onto group-scoped state and preserve rollback. Registry row selection now also
tracks `chain_fenced`: preserve it when restoring per-rank kernel eligibility.
Recent AQL/kernel commits change acquire fences, scalar invalidation, doorbell
flushing and idle-peer wake stores. Treat all these as one ordering contract;
replay, pipelining and two-shot fast paths must not remove required visibility.

### P0 — serving configuration cannot build against removed JSON dependency

Upstream removes `vendor/nlohmann/json.hpp` in favor of its newer JSON/YAML
stack (`yyjson`, `rapidyaml`). BigCherry serving CMake explicitly refuses a
`RADIANCE_SRC` without that header; serving and helper code widely uses
`nlohmann::json`, including strict manifest parsing and plan-cache decoding.

Either supply a separately pinned BigCherry-owned nlohmann dependency and correct
its declared build identity inputs, or port the component to a new JSON API with
strict-parser parity. The first route has the smaller migration scope. Audit
`bc_serving_v2.cmake`, serving/helper CMake, sdk-check, artifact packaging and
container dependency provisioning. Do not restore a header inside the upstream
vendor directory as an unrecorded local edit.

### P1 — the upstream KV contract replaces the base our KV patches extend

Upstream replaces narrowed fp8 quant-layout declarations with dtype-based paged
layouts. `rad_kv_row_elems(dtype, head_dim)` adds four bytes for I8 per-token,
per-head f16 K/V scales; `kv_quant`, older quant capability fields and fp8-prefill
fields are removed/reworked. Qwen4exp removes its old fp8_v4/k6_v4 declaration.
Most rad0037/38/40/41 edits therefore cannot be reanchored mechanically.

Compare our `bf16_i8` numeric/store format against upstream I8 before retiring
anything: equal byte size is insufficient evidence of equal scales, rounding,
stride, supported heads, merge math and mixed prefill/decode behavior. Consolidate
the runtime layout description around upstream row sizing while retaining only
measured local formats that still provide value. Cover paged and streaming
allocation, KV transfer, prefix reuse, speculative rollback, all supported GQA
ratios, and model probe refusal. Migrate manifest enum validation and fingerprints
with an explicit old-name translation or actionable rejection. Rebuild the KV
selftests against the new layout before attempting numerical acceptance.

### P1 — upstream replay and scheduler work overlap local enhancements

`7f5218ee` records Qwen4exp decode/verify/draft passes keyed by split cap, directly
covering rad0013's reason for existence. Newer pass keys additionally distinguish
indexer-bound/grid decisions; retire the local replay override only after proving
coverage of its exact geometry and verify cases, rather than overwriting the new
key. Latest commits add token/block/traversal verification, adaptive coupled MTP
depth and suffix proposals (`d0fd4aa1`, `e78d484e`, `b8bca788`). They change rows,
acceptance and pass shapes relevant to ticket admission and issue-ahead.

Upstream adds adaptive prefill interleave (default auto:1), mixed-step targeting,
preemption/prefix adoption and queue fairness fixes. rad0036 and experimental
rad0034/35 overlap this scheduling policy. Establish one policy authority; map
manifest fields to upstream equivalents where semantics match, explicitly preserve
or retire priority/order differences, and prevent two controllers adjusting the
same cap. Initial parity runs should explicitly pin old behavior rather than
inherit new adaptive defaults unnoticed.

## Upstream changes and migration consequences

| Area | Upstream changes / representative commits | Required BigCherry response |
| --- | --- | --- |
| Fleet and configuration | 2.0.0 `ddcb9c69`; one process/multiple engines, YAML, shared flex VRAM, parking and reload | Port launcher/config bridge; single-engine restriction first; separately design helper/fleet ownership and memory-plan admission. |
| Hardening and dependencies | 2.1.0 `2a2b7563`; bounded file reads, parser depth/range limits, profile validation, path confinement, sanitized logs, admin guards; cpp-httplib `79cc319d` | Keep these checks when porting local diagnostics and handlers. Retest authentication and generated-file semantics; decouple JSON dependency. |
| Memory accounting | Resident queue scratch charged `7bd195d3`, staging ring 3 -> 16 `b2228c17`, fleet fixed/flex pools | Remeasure plan-memory and all layout budgets. Include queue/staging allocations and fleet reservations; old plan-cache budgets are not transferable. |
| Placement | Fractional expert-pin shares corrected `8dcd8fdc`, profile count and file validation | Reconcile heat-driven placement, helper skip-owned bytes and ownership totals with new profile rounding. Preserve deterministic plan hashes. |
| Speculation | Recorded MTP `7f5218ee`, chained MTP `e2e94168`, launch/prefetch changes `ca7a7212`, new verification/depth/suffix modes | Recheck rank-consistent decisions, per-parent tickets, variable verify rows, replay keys, cancellation and accepted history. |
| GPU synchronization | `8f2e511f`, `01f08152`, `946a6f9e`, `e912b04f`, collective wake store `30459797` | Re-audit custom collective/plugin chain declarations and memory visibility; benchmark as an integrated system. |
| Quantization and kernels | W6NL down/U4 gate `d1b5a45e`, U5/W5 `ba3ebd38`, rotated int8 output `2e558e2d`, short IQR verify `1d94e416`, hc_read tile arms | Update helper encoding admission and kernel coverage; refuse unsupported recipes. Recalibrate local kernel pins rather than assume the old winning tile remains best. |
| Attention/KV | Unified int8 paged layout; fp8 latent GLM cache `d858aa72`; h128/gqa16, gqa7 and window rows; split-KV causal merge fix `d6109d7e` | Rework local KV family atop new contracts; keep causal correctness fixes in all retained formats. |
| Model families | Nemotron-3.5-Bolt `0c4292e4`, LFM3 `6c7bcefe`, Ling hybrid `33f96447`, Muse/DFlash2 additions and Gemma SWA changes | Add explicit capability gates; current Flash-Next helper assumptions are not validation for these models. |
| Server/text/media | Control-vector variants `157959fd`, reverse proxy `742191a2`, tool delimiters `38c182a2`, host-readable mixed encoder operands `590dbcc1`, media-disabled vision skip `73f94b43` | Verify bc-serve routes, model aliases, health/metrics, reverse proxy, media capability and memory planning. Retain FFmpeg compatibility only if still needed. |

## Per-patch migration disposition

Numbers below are observed failed edits in target composition. Zero means anchors
apply, not that a patch is ready to ship. Patch IDs link to source packages.

### Production serving set

| Patch | State | Failed edits | Required work |
| --- | --- | ---: | --- |
| [rad0002_moe_remote_handoff](../../../engines/radiance/patches/rad0002_moe_remote_handoff/patch.toml) | validated | 0 | Keep remote-MoE functionality. Scope plugin registration, layer maps, cached policy and parent tickets per engine/TP group; audit new draft/verify paths and unload drains despite zero anchor failures. |
| [rad0005_mp4_ffmpeg6_writer](../../../engines/radiance/patches/rad0005_mp4_ffmpeg6_writer/patch.toml) | validated | 0 | Retain callback signature compatibility if the new media writer still needs it on FFmpeg 6. Compile both callback signatures; remove only after checking upstream minimum FFmpeg version. |
| [rad0008_hc_upmix_rt_sweep](../../../engines/radiance/patches/rad0008_hc_upmix_rt_sweep/patch.toml) | validated | 0 | Recalibrate the retained upmix sweep against changed kernel/launch behavior. Keep tuning choices manifest-owned and per engine. |
| [rad0009_serving_manifest](../../../engines/radiance/patches/rad0009_serving_manifest/patch.toml) | validated | 1 | Replace manifest-to-argv startup with typed Config/FleetConfig bridge or derived YAML. Update supplied-option tracking and offline-mode handling; old CLI is refused. |
| [rad0011_plan_memory](../../../engines/radiance/patches/rad0011_plan_memory/patch.toml) | validated | 1 | Port explicit embedding host-room handling to new bringup, queue/staging costs and fleet budget reservations. Preserve measuring-mode semantics through new startup; invalidate old plan-cache entries. |
| [rad0012_helper_kernel_config](../../../engines/radiance/patches/rad0012_helper_kernel_config/patch.toml) | validated | 0 | Keep helper kernel-config contract but scope its install-once behavior to the intended plugin/engine lifetime. Revalidate supported encodings and new ABI. |
| [rad0013_mtp_pass_replay](../../../engines/radiance/patches/rad0013_mtp_pass_replay/patch.toml) | validated | 1 | Candidate retirement: upstream records draft/verify/decode passes. Compare split-cap, QSA-bound and grid key semantics, instrument replay coverage, then remove the override if equivalent. |
| [rad0016_vtp14_rank_code_selection](../../../engines/radiance/patches/rad0016_vtp14_rank_code_selection/patch.toml) | validated | 5 | Move per-rank kernel eligibility into shared Registry set ownership. Preserve chain_fenced and reference flags, actual device order, loader rejection and isolation between engine groups. |
| [rad0017_vtp14_n3_shared_rendezvous](../../../engines/radiance/patches/rad0017_vtp14_n3_shared_rendezvous/patch.toml) | validated | 2 | Replace singleton rendezvous injection with per-tp_group shared libr3/libr4d rendezvous. Carry group keys through init/fini; preserve timeout arrival rollback and poison propagation. |
| [rad0018_vtp14_n2_tp_ownership](../../../engines/radiance/patches/rad0018_vtp14_n2_tp_ownership/patch.toml) | validated | 1 | Reinsert uneven TP ownership fields into the new build context and ABI fingerprint. Keep ownership sums, zero-width slices and per-instance slot identity coherent. |
| [rad0019_vtp14_n2_tp_ownership_fp8](../../../engines/radiance/patches/rad0019_vtp14_n2_tp_ownership_fp8/patch.toml) | validated | 2 | Update Qwen/DFlash2 geometry hooks for new build slots, replicated KV heads and split gather/head layout. Preserve coverage for heterogeneous rank widths. |
| [rad0020_vtp14_tp3_pipelined_prefill](../../../engines/radiance/patches/rad0020_vtp14_tp3_pipelined_prefill/patch.toml) | validated | 1 | Port pipelining context fields to ABI 41 and group-scoped buffers. Recheck launch barriers and fixed/flex scratch accounting. |
| [rad0021_vtp14_weighted_twoshot_3rank](../../../engines/radiance/patches/rad0021_vtp14_weighted_twoshot_3rank/patch.toml) | validated | 0 | Revalidate link-weighted two-shot slices with group rendezvous and updated wake-store/acquire semantics. Rebench three-rank imbalance; no anchor repair currently observed. |
| [rad0022_tune_pin](../../../engines/radiance/patches/rad0022_tune_pin/patch.toml) | validated | 0 | Retain tuning pin interface only after checking new kernel names, geometries and model lifetime. Old selected variants require new performance evidence. |
| [rad0023_ar_timeline](../../../engines/radiance/patches/rad0023_ar_timeline/patch.toml) | validated | 0 | Scope AR timeline records by engine/TP group, preserve sampling overhead controls, and include new chained/fenced ordering in interpretation. |
| [rad0024_ts3_decode](../../../engines/radiance/patches/rad0024_ts3_decode/patch.toml) | validated | 0 | Recheck TS3 decode against per-group collectives and latest chained launch fences. Confirm shape/tail coverage and ordered rank progress. |
| [rad0025_multiseq_pipelined_prefill](../../../engines/radiance/patches/rad0025_multiseq_pipelined_prefill/patch.toml) | validated | 2 | Extend the new RadBatch layout without discarding anchor_rows. Replace obsolete sizeof==472 assumption and update ABI fingerprint; validate prefix/anchor handling across multi-sequence chunking. |
| [rad0026_mixed_pipelined_prefill](../../../engines/radiance/patches/rad0026_mixed_pipelined_prefill/patch.toml) | validated | 0 | Revalidate mixed-sequence pipelining with new RadBatch fields and scheduler decisions. Verify chunk boundaries, prefix reuse and heterogeneous ranks. |
| [rad0027_mixed_one_pipelined_prefill](../../../engines/radiance/patches/rad0027_mixed_one_pipelined_prefill/patch.toml) | validated | 0 | Revalidate one-sequence mixed path under upstream interleave/default changes and variable speculative verification rows. |
| [rad0028_rank_consistent_block_drafts](../../../engines/radiance/patches/rad0028_rank_consistent_block_drafts/patch.toml) | validated | 0 | Preserve rank-consistent block-draft decisions across traversal/adaptive/suffix forms. Test all ranks see identical draft depth, parent identity and accept/reject outcomes. |
| [rad0029_rank_own_split](../../../engines/radiance/patches/rad0029_rank_own_split/patch.toml) | validated | 0 | Audit rank-own split against slot_base and updated build-context geometry. Preserve full tensor coverage and helper ownership at every supported TP width. |
| [rad0030_int8_pipelined_prefill](../../../engines/radiance/patches/rad0030_int8_pipelined_prefill/patch.toml) | validated | 0 | Update int8 pipelined prefill coverage for rotated int8 trunk/activation formats; keep encoding and stride checks explicit and reject unsupported new recipes. |
| [rad0031_ts_timeline](../../../engines/radiance/patches/rad0031_ts_timeline/patch.toml) | validated | 0 | Scope TS timeline to TP groups/engine instances and compare timing overhead under new launch order. |
| [rad0032_ts3_duplex](../../../engines/radiance/patches/rad0032_ts3_duplex/patch.toml) | validated | 0 | Revalidate duplex mailbox/fence reuse across independent TP groups and engine unload; benchmark after integrating upstream coherence fixes. |
| [rad0033_ts3_once](../../../engines/radiance/patches/rad0033_ts3_once/patch.toml) | validated | 0 | Check once-only collective state for model restart, TP-group reuse and partial init failure. No stale process-global completion state may survive unload. |
| [rad0036_prefill_governor](../../../engines/radiance/patches/rad0036_prefill_governor/patch.toml) | validated | 10 | Production governor needs policy reconciliation with upstream adaptive controls, followed by Config/YAML/manifest plumbing and scheduler tests. Do not run two independent cap controllers. |
| [rad0037_bf16_fp8_kv](../../../engines/radiance/patches/rad0037_bf16_fp8_kv/patch.toml) | validated | 56 | Reimplement bf16/fp8 hybrid KV using the new dtype row-sizing/probe contract. Update allocation, store/load/merge kernels, manifest mapping and every advertised GQA/head shape; do not restore removed structs verbatim. |
| [rad0038_bf16_i8_kv](../../../engines/radiance/patches/rad0038_bf16_i8_kv/patch.toml) | validated | 60 | Compare upstream I8 per-token/head scale format against local bf16_i8. Prefer reuse where numerically equivalent; retain local hybrid behavior only with explicit layout/store/decode guarantees. |
| [rad0040_kv_layout_table](../../../engines/radiance/patches/rad0040_kv_layout_table/patch.toml) | validated | 10 | Replace old KV enum expansion with a unified layout registry aligned to rad_kv_row_elems. Use one source of truth for row bytes, probes, allocation and kernel dispatch; preserve explicit refusals. |
| [rad0041_kv_b16_flashnext](../../../engines/radiance/patches/rad0041_kv_b16_flashnext/patch.toml) | validated | 14 | Port Flash-Next b16 format and its paged GQA8/GQA12/GQA6-bs4 kernels onto the new layout API. Reapply causal split-merge fixes and test MTP-shared KV group use. |

### Optional / experimental / diagnostic inventory

| Patch | State | Failed edits | Required work |
| --- | --- | ---: | --- |
| [rad0001_drain_mover_before_arena_free](../../../engines/radiance/patches/rad0001_drain_mover_before_arena_free/patch.toml) | superseded | 0 | Recheck mover drain/arena destruction against upstream fleet unload and per-engine shutdown. Preserve drain-before-free; retire only on equivalent upstream cleanup evidence. |
| [rad0003_mtp_full_head_switch](../../../engines/radiance/patches/rad0003_mtp_full_head_switch/patch.toml) | untested | 0 | Reevaluate full-head switch against new coupled MTP and token/block/traversal verification. Enforce rank-consistent choice and supported batch forms. |
| [rad0004_moe_remote_diagnostics](../../../engines/radiance/patches/rad0004_moe_remote_diagnostics/patch.toml) | diagnostic | 1 | Reattach after-down shadow hook to the new rotated/codebook routed arm; compare identical tensors and keep diagnostic-only admission. |
| [rad0006_heat_freeze_diagnostic](../../../engines/radiance/patches/rad0006_heat_freeze_diagnostic/patch.toml) | superseded | 0 | Audit heat-freeze state and snapshot lifetime per engine. Revalidate placement under shared flex reclaim and reload. |
| [rad0007_hc_pf2_fp8_down](../../../engines/radiance/patches/rad0007_hc_pf2_fp8_down/patch.toml) | untested | 0 | Rebench local hc prefill fp8-down arm against updated hc_read/IQR kernels and widened encoding support; retain only demonstrated benefit. |
| [rad0010_routing_evidence](../../../engines/radiance/patches/rad0010_routing_evidence/patch.toml) | diagnostic | 0 | Scope routing histogram evidence by model/engine and preserve bounded/authenticated report output. Recheck tensor labels after model-instance changes. |
| [rad0034_prefill_concurrency](../../../engines/radiance/patches/rad0034_prefill_concurrency/patch.toml) | superseded | 5 | Experimental concurrency cap overlaps upstream mixed-step targeting/interleave. Map semantics and acceptance evidence before retaining or retiring; repair startup/preempt/observe hooks together. |
| [rad0035_prefill_order](../../../engines/radiance/patches/rad0035_prefill_order/patch.toml) | superseded | 14 | Experimental priority/order rules overlap upstream fairness and prefix adoption. Reconcile ordering explicitly; preserve request age, preemption eligibility and deterministic defaults. |
| [rad0039_moe_heat_observer](../../../engines/radiance/patches/rad0039_moe_heat_observer/patch.toml) | untested | 0 | Keep observer-only behavior, add engine identity and lifecycle-scoped storage, and verify routed arm coverage for new quantized formats. |
| [rad0042_kv_b16_selftest](../../../engines/radiance/patches/rad0042_kv_b16_selftest/patch.toml) | untested | 9 | Rebuild optional b16 selftest cases against the migrated store and attention kernels, new strides and upstream selftest restructuring. Include speculative rollback and long-prefix splits. |
| [rad0043_expert_stats_protected](../../../engines/radiance/patches/rad0043_expert_stats_protected/patch.toml) | untested | 0 | Compare protected expert stats to upstream report/parser hardening. Retain any extra privacy/correctness guarantees and engine-scoped counters; do not retire on anchor success. |
| [rad0044_expert_route_record](../../../engines/radiance/patches/rad0044_expert_route_record/patch.toml) | untested | 1 | Move route-recorder cleanup to the new context/model lifetime. Validate request token positions and routed-layer naming under adaptive/suffix/traversal and multi-instance execution. |

## Component and enhancement migration beyond patch anchors

- **libr3 / libr2 / bc-gfx12:** rebuild against ABI 41 and recompute exported
  layout fingerprints. Audit init parameters, `tp_group`, chain/fence declarations,
  RadBatch additions and row schemas. Validate per-device code filtering across
  gfx1100/gfx1201/gfx1030; a plugin that loads is not proof every rank has code.
- **moe-remote:** preserve helper handshake, encoding admission, admission ledger,
  cancellation, issue-ahead and memory bounds. A new upstream quantization must
  not be silently interpreted as an existing wire format. Thread engine identity
  through callbacks before supporting multiple fleet models; drain the helper
  before freeing the model's layer maps or weight storage.
- **serving / bc-serve:** retain typed manifest authority, resolve/store/inspect,
  measurement plans, load budgets, strict JSON and release installation behavior.
  Add target-engine capabilities and explicit mappings for upstream YAML and new
  defaults. Update launch command construction and health/metrics expectations.
- **ngram-helper / draft integrations:** distinguish the external helper from new
  upstream suffix proposals. Decide when each can activate, retain deterministic
  mode selection, and propagate actual verify-row bounds into memory/admission.
- **model recipes and layouts:** retest Flash-Next mixed-TP and helper layouts
  first. Recompute per-card weight/KV/staging/queue reservations. New upstream
  families/encodings remain unsupported until an explicit capability and test
  matrix admits them; avoid claiming support by accepting their model names.
- **build/release tooling:** update pin, pinned-version, stale 1.14 recipe comments,
  source identity receipts, SDK include inputs and runtime plugin packaging only
  after the port is buildable. Keep new runtime homes separate from old release
  homes. Rerun install/relocation/SDK fingerprint checks on the migrated bundle.

## Ordered implementation slices and acceptance

1. **Freeze identities and comparison workloads.** Keep this target SHA and the
   current BigCherry source/build identity. Capture explicit single-model config,
   device order, data/weight hashes and old defaults for a controlled comparison.
2. **Build foundation.** Resolve JSON ownership and compile upstream plus all local
   components against one header set. Extend ABI layout fingerprints and update
   Registry/device filtering. Gate: component configure/build, SDK/plugin ABI
   acceptance and expected negative mismatch cases.
3. **Launch one model.** Port manifest startup and measuring mode, preserve the
   external helper contract, and fail closed on multiple remote-MoE engines.
   Gate: manifest unknown/unsupported options, CPU smoke, native launch/health,
   shutdown and plan export. Do not quietly fall back to old argv parsing.
4. **Restore heterogeneous TP and remote handoff.** Port group rendezvous, ownership,
   pipeline and ticket state before enabling scheduling experiments. Gate: two-
   and three-rank layouts, uneven tails, helper on/off, admission backpressure,
   cancel/partial init, unload drain and no cross-group state sharing.
5. **Consolidate KV formats.** Choose retained local formats and explicit aliases,
   update stores/readers/allocation/transfer together, then selftests. Gate:
   prefill/decode/verify equivalence, long-prefix split-KV, multiple sequences,
   streaming, prefix reuse, unsupported shape rejection and memory-plan accuracy.
6. **Reconcile speculation and scheduling.** Remove rad0013 only on upstream replay
   coverage proof. Test token/block/traversal, fixed/adaptive depth, suffix on/off,
   prefix/preemption and local governor behavior. Gate: rank agreement, correct
   accepted history, deterministic reference cases and mode-specific KLD.
7. **Performance and release acceptance.** Run paired/interleaved comparisons on
   fixed prompts, contexts, sequence counts and layouts; report prefill/decode,
   TTFT/ITL, draft acceptance, replay hit rate, memory peaks, energy and failure
   behavior. Use existing project acceptance thresholds rather than inventing
   thresholds here. Revalidate tuning pins, package runtime/SDK, and test bundle
   install/relocation before switching production.
8. **Fleet enhancement is a separate acceptance slice.** After single-model parity,
   add engine-scoped remote policy, helper sessions, observers and collective
   state. Test co-residency, priority reclaim, park/unpark, model reload, group-id
   reuse and concurrent shutdown. A single-model migration is not fleet support.

## Reproduction and evidence index

- `upstream-commits.txt`: every commit in v1.17.0..target, newest first.
- `upstream-files.txt`: upstream file status delta.
- `applicability.json`: canonical production dependency order, file/edit errors,
  target paths, base control and exploratory optional-patch results.
- Local mirror/worktrees and raw scratch stay under project-local
  `work/radiance-upstream-review/`; no production source was repatched.

To repeat: resolve v1.17.0 and the recorded target SHA from a fresh mirror; create
pristine detached trees; load recipes' serving-core IDs through
`bigcherry.patch.patchset.resolve_exact`; load each resolved module with
`load_resolved` and call the maintained patch engine with `dry_run=True`, sharing
one `texts` dictionary through the ordered composition. Verify the base control
has no failures before interpreting target failures. Do not use recipe list
order directly: rad0011 depends on rad0009 despite its position in that list.

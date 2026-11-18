# Source consolidation and future-bump design

> **Astra refinement:** [ASTRA-CONSOLIDATION.md](ASTRA-CONSOLIDATION.md) and the current Planning items govern execution. They supersede earlier proposals for a normalized YAML writer, bundle index, parser-tree extension API, full native option mirror and initial fleet-broker reservations. Keep original YAML bytes bound to existing admission, delete native-only duplication before extracting survivors, and use disjoint helper cards initially. Earlier source inventories remain evidence, not implementation acceptance.

Scope: static source analysis at BigCherry `cd0581fc9` and upstream
`b8bca788`. No builds, GPU runs or numerical acceptance. This extends the
per-patch migration review; it does not certify every runtime path.

## Measured coupling

[patch-footprint.json](patch-footprint.json) inventories all 42 actual declarative
patch definitions: 8,896 definition lines, 580 edits and 25 created-file payloads
containing 2,974 text lines. Counts include tests/comments and are not net compiled
code or guaranteed deletion savings. The JSON lists every target and all patches
sharing it, so maintainers can inspect composition order before editing.

| Upstream target | Local patches touching it |
| --- | ---: |
| core/engine_bringup.cpp | 12 |
| core/config.cpp | 11 |
| core/rad_core.h | 11 |
| tests/config_test.cpp | 11 |
| arch/qwen35_fp8/qwen35_fp8.cpp | 10 |
| libr4d/r4d_rows.cpp | 7 |
| core/engine.cpp | 6 |
| core/runtime/ctx.cpp | 6 |
| abi/rad_builder.h | 6 |
| libr4d/r4d_ar_entry.hip | 5 |

These are the first seams to stabilize. Do not respond by making every anchor
more permissive: that hides semantic conflicts and retains the repeated edits.

## Foundations that remove work

### 1. Generate mappings, not another configuration framework

Actual existing assets: serving/contract/serving-v2.toml -> contract/generate.cpp
-> generated/v2/serving_types.h (`kFields`, `FieldInfo`, `kEnvBindings`).
rad0009 already iterates `v2::kFields` to produce flags; it is not an entirely
handwritten option mapping. Preserve that investment instead of adding a new
registry parallel to it. Upstream fleet Reader -> ConfigFlag/config_parse owns
native value validation.

Target: each owned extension field declares its native binding or extension
consumer in the existing registry. Generate the small binding/provenance data
from it. A native engine setting should have one authored spelling, one upstream
validation path and one normalized value. Keep schema generation about data;
custom cross-field/resource semantics remain normal typed C++.

Delete the obsolete argv materialization and `g_emitted` option relabeling after
all launchers use admitted native configuration. Delete replaced TOML/native
alias maps, not schema metadata that still defines runtime contract semantics.
Do not generate an assignment switch for every upstream Config member: reuse
the option table and only bind owned extensions. One fixture with a real option
and a representative extension plus unknown-field rejection proves the boundary.

### 2. Share semantic inputs without conflating cache purposes

Source correction to the broad earlier recommendation: stored_manifest_cache.cpp
already has one static projection for BOTH writer and stored replay. It explicitly
removes derived host_pool, KV and per-start shared-memory values. plan_cache.cpp
`plan_key_inputs` instead excludes `kNonPlanEngineKeys`, captures engine geometry,
rank-card shapes, helper budgets and tuning. These caches are not duplicates
merely because both hash JSON.

Introduce at most one small typed effective-input projection consumed by both
builders; preserve separate measured-memory and static-resolution key functions.
Incorporate target normalized native settings and extension semantics there.
Keep fresh admission outside static reuse. No universal cache service, blanket
hash-of-entire-manifest, or removal of model identity validation. Share existing
JCS/hash functions without changing established digest bytes accidentally.

Removal condition: a field-list/projection is deleted only when its exclusion
rules have a named replacement and writer/replay parity still holds. Minimal
fixtures: same semantic settings with different authoring order, one changed
memory-affecting setting, and changed live room with an unchanged static key.

### 3. Link owned implementation; keep only semantic upstream hooks

Created payload clusters include rad0002 (remote header/registry/place/block/engine
adapter), rad0009 (372 lines header/source), rad0011 (430 lines memory/check
header/source), rad0016 (rank-device implementation and test), rad0017 (shared
slot and test), rad0010 (296-line evidence header), rad0036 (115-line governor),
and rad0039 (heat ABI/sidecar/registry/observer).

Move CPU implementations and tests to their existing owning components and
link through declared CMake targets. Do not use a configure-time copy to recreate
the same untracked source injection. Preserve public ABI exports where plugins
need them. Keep the few architecture-dependent templates as owned headers with
small typed views; do not spread upstream model structs through a generic hook API.

Needed seams are concrete: resolve/admit configuration, bind engine/group state,
declare remote ops, publish after routing, combine remote output, drain/unregister,
and report observed events. Give each one an exact call site and failure rule.
An extension event bus or service locator would add more future compatibility work.

Extraction improves patch maintenance but does not reduce total executable LOC.
Actual code reduction requires deletion of duplicated maps, redundant overrides
and obsolete adapters. Report moved lines separately from deleted lines.

### 4. Bind one engine state and one placement view

Trace: `bc_moe_remote_register` / `g_plugin` -> admit/bind -> rank TLS ticket ->
before_pass -> architecture after_route/combine -> host_done/rank_done ->
shutdown. `bc_moe_remote_skip_unit` and layer-map publication connect placement
to the same ownership. Function-static policy caches are lifetime hazards when
upstream starts multiple engines, even if their anchors still apply.

Reuse admission_ledger.h, engine_admission_gate.h and engine_rank_coordinator.h.
Bind state once from the admitted engine generation and TP group. Keep the parent
ticket, placement epoch and plugin ABI version distinct; no new duplicate epoch
protocol. An immutable owner/local-slot/encoding view should be shared by planner,
helper setup and execution rather than reconstructed from JSON or copied per step.

The initial plan must reproduce the current ownership decision. RHP04/RHP09 own
future calibrated improvement; upstream Fleet::grant/load_op owns allocation.
External helper reservations add demand to that accounting. Do not make a second
allocator or new solver in the name of foundational cleanup.

Minimal proof: same-rank/different-group separation, unused-ticket cancellation,
partial startup teardown and drain-before-free. The existing short production
canary then proves the live route/combine path; no all-topology campaign.

### 5. One KV descriptor, generated backend adaptation

rad0040 already centralizes packed layout names, row sizes and capability checks
with `RAD_KV_LAYOUTS`/`rad_kvq_from_name`; extend/reconcile it with target dtype
row sizing instead of building a second table. Native I8 must reuse the target
format only when scale granularity, rounding, store/read stride and quality agree.
Planning/stream transfer must derive bytes from the same descriptor as dispatch.

Important downstream caller found: libr3/CMakeLists.txt runs `r3_rewrite.py`
against upstream libr4d to generate the gfx1100 adaptation and registers
`r3_rewrite_check`. Kernel changes therefore have a generator consumer as well
as libr4d. Edit the generator/source ownership, never checked build outputs or
both backends by hand. Include its input bytes in build identity.

Retain kernel specializations where layout/ISA requires them. Do not replace
compile-time kernels with generic per-element callbacks to reduce source lines.
One retained-format stride/roundtrip and causal split case plus generator check
is the initial gate. Expand shapes only when their production support is claimed.

### 6. Upstream capability reuse is the main deletion opportunity

rad0013's replay override overlaps upstream Qwen4exp recording. Prove split-cap,
QSA-bound and grid-key equivalence, then remove the override and production-set
entry together if equivalent. Do not keep a redundant override as insurance.
rad0036's created governor and optional rad0034/35 controllers overlap upstream
interleave/mixed-step policy, but differing objectives cannot be declared equivalent
from names. Choose one production controller and delete replaced parsing/state/tests.

Keep current fixed-depth production behavior for initial acceptance. Newly added
suffix/adaptive/traversal features are not prerequisites. Consolidate observations
from rad0010/rad0039/rad0043/rad0044 only where they truly share data: route identity,
model/generation and bounded sink plumbing. Histograms, protected expert statistics
and per-token route records have different cost/privacy/retention requirements;
keep their output contracts distinct and avoid always-on recording.

### 7. Similar helper functions are not automatically safe to merge

Both session.cpp and release_install.cpp define `write_atomic`, but their guarantees
differ: session uses a fixed .tmp + ostream/rename; release install uses exclusive
no-follow creation, fsync and directory handling. Do not replace the durable install
writer with the simpler session helper. If shared after the bump, make a narrow
byte-writer with explicit durability/error requirements and migrate only compatible
callers. This is not necessary to qualify the pin and must not grow into an I/O
framework. Likewise, resource/cache/provenance identities must retain purpose.

## Future bump contract

Maintain a small seam manifest in the existing patch metadata/report tooling,
not a second patch engine: patch owner, upstream attachment symbols, owned source
inputs, ABI/schema touched and required focused proof. Generate its file-overlap
and created-payload report from FilePatch definitions, as this review does.

For each bump: resolve exact SHA -> summarize changes only in affected seams ->
canonical ordered composition -> changed seam builds/fixtures -> one admitted
production canary -> release. Reuse existing registry and build identity APIs.
Keep semantic assertions for a handful of critical invariants; do not snapshot
whole upstream functions or add brittle line-count correctness tests.

Success measures: fewer independently edited upstream sites per feature, fewer
manually maintained projections, removal of superseded runtime branches and lower
review/build effort. Track actual owned/generated/upstream-touch line changes;
there is no justified numerical LOC-reduction promise before implementation.

## Coverage and remaining uncertainty

All 42 patch definitions and target overlaps were inventoried, and the shared
configuration/cache, remote admission/ownership, KV generator and collective
interfaces above were inspected. Lexical target overlap does not establish full
semantic call-graph coverage. Before deleting a path, use RSC13's callable inventory
and search its concrete consumers, then perform only that path's proof gate.
Compilation, device memory visibility and approximate-format parity remain future
implementation gates, not claims made by this static review.

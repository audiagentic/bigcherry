# ServingManifest v1

Status: **locked wire contract**, 2026-11-02, as amended in §11 (the integer-only rule). This repository copy is the source of truth; the review draft it came from is history. Implementation requires a coordinated producer/consumer migration; this is not the JSON currently emitted on main.

Normative companions:
- `serving-manifest.v1.schema.json` — closed JSON Schema Draft 2020-12.
- `example.flashnext-4gpu.json` — synthetic four-GPU production fixture.
- `example.lab-experiment.json` — synthetic lab fixture with shape overlays.
- `errors.md` — stable error codes, machine/human rendering and exit behavior.

Terms **MUST**, **MUST NOT**, **SHOULD** have their usual requirements-document meanings. A manifest is a complete, immutable startup artifact, not a CLI/env converter. The shared C++ `bc_serving` loader validates the entire artifact before selecting a typed engine or helper view. Unknown keys and unsupported versions fail closed. TOML is for human input; JSON is for the machine artifact. One release supports exactly one manifest wire version.

## 1. Envelope and identities

```text
apiVersion: "bigcherry/v1"
kind: "ServingManifest"
metadata:
  instance_id
  producer: {name, build_sha256}
  contract_version
  content_digest
spec:
  intent, facts, calibration, resources, handoff, processes, plan
  experiment?                  # lab only
provenance: JSON Pointer -> provenance record
```

- `apiVersion` is exactly `bigcherry/v1`; `kind` is exactly `ServingManifest`. `metadata.contract_version` is exactly integer `1` and MUST match the `apiVersion` suffix. It is an identity assertion, not an independent setting.
- `metadata.producer.name` is `bc-serve`; `build_sha256` is a lowercase 64-hex SHA-256 of the *installed bc-serve executable*. Producer provenance identifies the particular shipped executable, not the upstream Radiance build. The Radiance binary/runtime identity lives in `spec.plan.engine_memory.engine_identity`.
- `instance_id` is a canonical lowercase UUIDv4, created once with a cryptographically secure source **per resolved serving instance**, not per process or read. All processes in one launch use the same ID. Never reuse the same ID for a changed manifest. Regenerating an otherwise identical run creates a different ID and therefore a different manifest digest.
- There is **no `created_at`** in v1. A timestamp needlessly destabilizes otherwise identical content; file timestamps and run receipts provide event time. The instance UUID supplies uniqueness. Calibration evidence has its own `seeded` date.
- `content_digest` is `sha256:<64 lowercase hexadecimal characters>`, over the complete manifest with the `metadata.content_digest` member **removed**, as canonicalized by RFC 8785 JCS. It covers the entire `spec`, selected calibration, measured plan, owned sets, metadata, and provenance.
- The examples use **synthetic build/model identities and deliberately short owned sets** for contract tests. Their machine digest and internal hashes are real. They are not executable plans for a production .rad model; the actual producer MUST obtain those identities, owned sets and plan measurements from the installed software and model.

## 2. Taxonomy and authoritative ownership

| Path | Meaning | Authority |
|---|---|---|
| `spec.intent` | Model, layout selection, service objective and policy | Typed operator/site/profile input |
| `spec.facts` | Detected host limits, CPU and GPU inventory | KFD/sysfs/proc/cgroup/memlock/pin discovery |
| `spec.calibration[]` | Selected fitted parameters and supporting evidence | Versioned calibration seed/cache |
| `spec.resources.host` | Derived pool, KV tier, reserve and checkpoint allocation | RSC03 ledger and calibrated policy |
| `spec.handoff` | One shared remote-MoE transport, placement and telemetry configuration | Resolver |
| `spec.processes.engine` | Engine-only device, runtime, scheduler, HTTP and execution decisions | Resolver + Radiance capability contract |
| `spec.processes.helper_common` | Full grouped defaults shared by helper processes | One generated type |
| `spec.processes.helpers[]` | Ordered role/GPU selection and per-role differences | Resolver and architecture catalogue |
| `spec.plan.owner_map` | Immutable **initial** expert assignment and digests | RSC03 GPU-free `plan_owners` / `select_owned` |
| `spec.plan.engine_memory` | Content-addressed Radiance measured-plan receipt and start-up admission target | rad0011 and plan cache |
| `spec.experiment` | Explicit lab-only placement tests, shape overlays and diagnostics | Lab-only inputs |

No process repeats a shared fact as another serialized configuration field. For example: helper count is `processes.helpers.length`; helper index is its array position; engine rank count is `processes.engine.scheduler.tp`; host pool is `resources.host.pool_mib`; per-helper VRAM budget is `helpers[i].vram_budget_mib`. A tail role does not create a second `tail_budget_mib`.

The one intentional **receipt echo** is `spec.plan.engine_memory.expected.host_pool_mib`: rad0011's strict `plan_version:1` format requires it. It is a measured-plan assertion, **not** a second mutable configuration source, and MUST exactly equal `spec.resources.host.pool_mib`; mismatch is a schema/semantic error. Likewise owner `count` equals the owned-set length and digests verify the set; these are integrity fields rather than alternative settings.

Consumers read only typed views assembled from authoritative paths, never raw string-key lookup dictionaries.

### 2.1 Concern-oriented groups

`processes.engine` groups:
- `device`: ordered GPU BDF references;
- `runtime`: `radiance_home`, ordered `kernels`;
- `server`: `port`;
- `scheduler`: `tp`, `max_num_batched_tokens`, `checkpoint_interval`;
- `execution`: `placement`, `ngram_placement`, `kv_cache_dtype`, `num_speculative_tokens`, `rope_scaling`, `prefix_resume`.

`handoff` groups:
- `transport`: `shm`, `max_rows`, `small_rows`, `gemv_large_rows`, `skip_owned`;
- `placement`: `strategy`, `heat_file`;
- `telemetry`: `histogram_file` (null disables), `period_s`;
- `kernels`: `moe_down_tile` integer mode in `0,1,2`.

`helper_common` and the per-role `settings` patch use exactly these seven groups:

| Group | Fields and semantics |
|---|---|
| `cpu` | `reference` (CPU reference path), `affinity` (`auto` or `all`) |
| `runtime` | `power_save`, `queue_warm` |
| `gemv` | `mode` (0/1/2), `wide`, `h2` (0/1/2/4), `w4h2`, `mr` |
| `lanes` | `small_lanes` (up to four sorted thresholds), `rows`, `rows_small` |
| `combine` | `enabled`, `min_rows`, `threads` |
| `kernels` | `native`, `fp16`, `ahalf`, `moe_form` (0/1/2), `moe_long_rows`, `fuseroute`, `vecio`, `sparse_act`, `lane_quant` |
| `placement` | `balance`, optional array `balance_speeds_milli` (integers ≥ 1; 1000 = speed 1.0) |

All seven groups and all base fields except `balance_speeds_milli` are required under `helper_common`. `balance_speeds_milli` MUST be absent when balance is false, and MUST contain precisely one value per selected helper when balance is true. Cross-role coordination flags (e.g. balance, combine enabled) MUST yield a consistent effective protocol configuration for every participating helper; nonuniform values are rejected unless explicitly supported by a reviewed future contract. This is validated **after** applying per-role and lab overlays.

### 2.2 Per-role patch semantics — exact rules

`processes.helpers[]` is ordered `helper0`, `helper1`, etc., with unique roles and BDFs. Required role fields are `role`, `gpu`, `vram_budget_mib`, `plugin_file`, `formats[]`, `rows_kernel`, `tier`, `share`. Optional `settings` is a **partial object of the seven known helper concern groups**, and each contained group is a nonempty partial field map. No other role-level configuration override is allowed.

The typed merge is **recursive at known object boundaries only**:

1. Start from fully validated `helper_common`.
2. Apply that role's `helpers[i].settings`: for each explicitly present leaf, replace the original leaf. An absent leaf inherits. An explicitly present `false`, `0`, or empty array is a real value, **not** an instruction to inherit.
3. If profile is lab, apply `experiment.shape.helper_common`, then the matching `experiment.shape.helpers[j].settings` patch, in that order.
4. Objects recurse by schema-declared field; arrays and scalars **replace in full**, never concatenate, index-merge or toggle. Null is allowed **only** for fields explicitly nullable in the schema. Unknown keys, wrong types, empty override objects, unknown/duplicate lab role patches and inconsistent cross-role effective results fail.
5. Validate every final `HelperView` and every cross-helper invariant. No process serializes the merged copies back into the manifest.

For example, base `gemv:{mode:1,h2:2,...}`, role `helper1.settings.gemv:{h2:4}` produces mode 1, h2 4 for helper1 without repeating all other settings. In the lab example, experiment common `gemv.h2=4` overrides role/base h2 for every helper, while experiment helper2 `gemv.mode=0` overrides only helper2's mode. The override layers express intentional baseline versus experiment differences; they are not competing independent sources of truth.

`rows_kernel` is a resolved architecture capability and `lanes.rows` is operator policy; the effective typed view sets active rows when either requires it. The schema does not embed a second `rows = rows || rows_kernel` field.

## 3. Naming, units, enums and native Radiance binding

Only the conventional envelope key `apiVersion` uses camelCase. All BigCherry-defined names use `snake_case`. Units: `_mib` (2^20 bytes), `_bytes`, `_s`, and `_tokens` when ambiguous. Raw exact memory-plan fields retain upstream names (`slot_bytes`, `kv`, `claimable`, etc.), measured in **bytes** per rad0011. JSON Booleans are true/false, never string or integer flags. No `off` magic string stands for an optional path; use null for disabled histogram/capture paths.

**Reviewed engine option naming rule:** where a manifest field maps to a Radiance option, use the Radiance option spelling with its leading `--` removed and each `-` replaced by `_`, e.g. `--max-num-batched-tokens` ↔ `max_num_batched_tokens`. The field can sit within a concern group; it need not be a flat `engine.*` value. Non-option intent/resource fields retain their own conceptual names and use explicit bindings. The rule does not automatically expose every Radiance flag.

The current **curated v1 engine surface** is:

| Manifest authoritative field | Radiance option / native `rad::Config` destination |
|---|---|
| `intent.model` | `--model` / `Config.model` |
| `intent.service.max_sessions` | `--max-num-seqs` / `Config.max_seqs` |
| `intent.service.max_context_tokens` | `--max-model-len` / `Config.max_ctx` |
| `resources.host.pool_mib` | `--host-pool-mib` / `Config.host_pool_mib` |
| `resources.host.kv_mib` | `--prefix-cache-mib` / `Config.prefix_cache_mib` |
| `resources.host.reserve_mib` | `--host-headroom-mib` / `Config.host_headroom_mib` |
| `resources.host.checkpoint_slots` | `--checkpoint-slots` / native checkpoint limit |
| `processes.engine.runtime.radiance_home` | `--radiance-home` / `Config.radiance_home` |
| `processes.engine.runtime.kernels` | `--kernels` / `Config.kernel_hierarchy` (ordered list adapter) |
| `processes.engine.server.port` | `--port` / native port |
| `processes.engine.scheduler.tp` | `--tp` / `Config.tp` |
| `processes.engine.scheduler.max_num_batched_tokens` | `--max-num-batched-tokens` / `Config.max_tok` |
| `processes.engine.scheduler.checkpoint_interval` | `--checkpoint-interval` / native interval |
| `processes.engine.execution.placement` | `--placement` / native placement mode |
| `processes.engine.execution.ngram_placement` | `--ngram-placement` / native table placement |
| `processes.engine.execution.kv_cache_dtype` | `--kv-cache-dtype` / native dtype |
| `processes.engine.execution.num_speculative_tokens` | `--num-speculative-tokens` / `Config.n_spec` |
| `processes.engine.execution.rope_scaling` | `--rope-scaling` / `Config.rope_scaling` |
| `processes.engine.execution.prefix_resume` | `--prefix-resume` / `Config.prefix_exact` |

The generated, compile-time typed `Config` adapter owns these explicit assignments; it reuses Radiance's native validation helpers for nontrivial values and propagates manifest `given` provenance to the startup report. The engine patch invokes `bc_serving::load_manifest` and the adapter **directly**. It MUST NOT construct argv, strip JSON path prefixes, reparse option strings, or re-read settings from environment. `--manifest PATH --role engine` and verbosity flags are startup routing only. No `--engine-arg` fallback. Runtime GPU visibility is permitted solely for device isolation.

Any engine key **outside** the generated curated contract is rejected as `BC_CFG_UNKNOWN_FIELD`, status `RAD_E_SCHEMA=-9`, exit 2. An upstream option lacking a reviewed native binding is not an implicitly permitted manifest key. Upstream changes must update binding coverage and corresponding tests. Radiance's own option metadata remains authoritative for CLI meanings, not for selecting all available manifest fields.

## 4. Initial owner map: explicit sets, digests and epochs

`spec.plan.owner_map` contains `owner_map_sha256` and `helpers[i]` aligned by index with `processes.helpers[i]`. Every helper record contains:
- `experts`: the **ordered** list of `[layer,expert]` unsigned-integer pairs, no duplicates across the entire map;
- `count`: exactly `experts.length`;
- `bytes`: the exact device-layout bytes returned by the shared GPU-free `bc::experts::select_owned` path, with the model's actual expert formats;
- `owned_sha256`: the helper's digest.

**Digest encoding is frozen to the existing RSC03 `owner_map_digest` convention** (not JSON, not JCS):

```text
segment(i) = "h" + decimal(i) + "\n" +
             concat(decimal(layer) + ":" + decimal(expert) + "\n"
                    for each pair in helpers[i].experts, in list order)
owned_sha256[i]    = lowercase_hex(SHA256(UTF8(segment(i))))
owner_map_sha256   = lowercase_hex(SHA256(UTF8(concat(segment(i) for i=0..N-1))))
```

There is no locale dependence, zero-padding, whitespace beyond the shown newlines, or sorting after selection. The owner order comes from the actual selector; the digest includes helper order. These two names use bare 64-hex hashes for compatibility with the existing RSC03 owner-map contract; the manifest's `metadata.content_digest` uses `sha256:` notation.

At resolution, `bc-serve` loads the model and heat file, runs the **same** GPU-free `plan_owners` used by helpers, and emits all owned pairs, bytes and digests. The complete map is needed: a digest by itself cannot tell the engine which expert units to exclude from its footprint.

The engine MUST compute its pre-allocation footprint using **these manifest-owned sets** (or an exactly reconstructed set from the included pairs), not the mutable mailbox contents and not a zero-helper assumption. It checks the corresponding initial map against the helper-published runtime map before admitting the run. Every helper recomputes its expected owned set/size and refuses an inconsistent role. No live helper is required to calculate the engine's *expected* footprint. On a mismatch, all startup is refused.

The manifest owns **only initial placement**. After startup, RHP11's mailbox placement epochs are the live authority; epoch flips and later owner sets MUST NOT rewrite or rehash the startup manifest. The startup digest remains the audit identity of the initial state.

Evidence: RSC03 `experts.h` defines `Owned{expert,bytes}`; `experts.cpp:347-384` plans ownership; `serving.cpp:751-758` computes map digest. [tracked RSC03 hardware check](../../evidence/RSC03/PLAN_MEMORY_HARDWARE.md) confirms engine-only `plan()` without helper ownership computed a different footprint and failed out of memory.

## 5. Strict expected engine memory plan and cache identity

`spec.plan.engine_memory` has `cache_version=1`, `cache_key`, `model_identity`, `engine_identity` (each lowercase SHA-256 hex), and `expected`.

`expected` is the full strict **rad0011 `plan_version:1` engine-plan JSON**, unmodified in field names and units:

```text
expected:
  plan_version: 1
  producer: "radiance --plan-memory (rad0011)"
  host: [{name, bytes, residence, from, remedy}, ...]
  checkpoints: {slot_bytes, floor_slots, want_slots, from}
  kv_one_session: {bytes, from}
  vram:
    claimable, statics, elastic, experts, kv, stage, kv_resolved
    expert_floor, kv_ceiling, expert_ceiling, from
    cards: [{capacity, free, engine_expert_plane, expert_unit_max}, ...]
  host_pool_mib
```

All sizes inside `expected` are **bytes**, except `host_pool_mib`. Names of host consumers are unique, and all fields are required and closed by the schema. `checkpoints.floor_slots <= checkpoints.want_slots`. The planner decides the actual `resources.host.checkpoint_slots` within these bounds; it is **not** identical to `want_slots`.

RSC03's selected calibration now uses `checkpoint_share=tier_first` and `host_kv_tier.target_mib=12288`, supported by its latest 2026-11-02 evidence. The real run produced 43 allocated snapshots of a corrected 72-slot want (112.48 MiB each) while preserving a 12 GiB host KV tier. The pre-fix hardware JSON had an incorrect 16-slot want; **do not** promote that stale want to a v1 fixture. See RSC03 `calibration.toml`, `RSC03.md:139-145`, `rad0011_plan_memory/patch.py:88-139`, and the hardware RESULT.

### 5.1 Cache key, no circularity

`model_identity` is RSC03's SHA-256 over the container size and relevant header/directory/plane/encoding tables. `engine_identity` is SHA-256 over the installed engine binary plus installed runtime-home files by sorted relative path and contents. They are distinct from the bc-serve producer identity.

`cache_key` = lowercase hex SHA-256 of **JCS canonicalized** `key_inputs`, built in memory exactly as follows (never duplicated as another manifest field):

```json
{
  "cache_version": 1,
  "model": "<model_identity>",
  "engine_build": "<engine_identity>",
  "card": {
    "arch": "<engine GPU's discovered arch>",
    "vram_total_mib": "<engine GPU's discovered total MiB as INTEGER>"
  },
  "layout": "<intent.layout>",
  "owner_map_sha256": "<plan.owner_map.owner_map_sha256>",
  "planning_options": {
    "objective": "<intent.service.objective>",
    "retention": "<intent.service.retention>",
    "max_sessions": "<intent.service.max_sessions as INTEGER>",
    "max_context_tokens": "<intent.service.max_context_tokens as INTEGER>",
    "tp": "<engine.scheduler.tp as INTEGER>",
    "max_num_batched_tokens": "<engine.scheduler.max_num_batched_tokens as INTEGER>",
    "checkpoint_interval": "<engine.scheduler.checkpoint_interval as INTEGER>",
    "placement": "<engine.execution.placement>",
    "ngram_placement": "<engine.execution.ngram_placement>",
    "kv_cache_dtype": "<engine.execution.kv_cache_dtype>",
    "num_speculative_tokens": "<engine.execution.num_speculative_tokens as INTEGER>",
    "rope_scaling": "<engine.execution.rope_scaling>",
    "prefix_resume": "<engine.execution.prefix_resume>",
    "kernels": ["<ordered engine.runtime.kernels>", "..."],
    "handoff_max_rows": "<handoff.transport.max_rows as INTEGER>",
    "helpers": [
      {
        "role": "<helper role>",
        "arch": "<helper GPU's discovered arch>",
        "plugin_file": "<helper.plugin_file>",
        "formats": ["<helper formats>", "..."],
        "rows_kernel": "<BOOLEAN>",
        "settings": "<FULL EFFECTIVE helper settings after all overlays>"
      }
    ]
  }
}
```

The quotation marks enclosing angle-bracket placeholders above document value sources, not the wire types. Use the stated integers, Booleans, arrays and objects as JSON values. `helpers` is the ordered array of role projections.

This key includes the **initial owner map** and **effective per-helper shapes**, fixing the interim RSC03 cache-key omission. It excludes derived resource outputs (`resources.host.*`, helper VRAM budgets, checkpoint allocation, etc.), live free/claimable VRAM, temporary runtime paths and server port. Otherwise plan measurements would depend circularly on their own outputs. The new v1 cache-key inputs are not accepted as old interim cache entries; on a miss, require a fresh measured plan, not a guess.

The cache entry stored outside the manifest may retain RSC03's `{cache_version,key,key_inputs,measured_at,plan}` format. Its key, inputs and strict measured plan MUST all match. `measured_at` remains cache/run evidence, not manifest metadata. A bad cached entry is an error, not a silent miss.

### 5.2 Startup admission

After applying the manifest-owned sets, Radiance computes the **live** `plan()` and compares it against `expected` before weights or arenas are allocated.

- **Stable facts** must match: every stable host consumer's name, exact byte count, residence and complete inventory, the checkpoint slot size and floor/want geometry, the one-session KV bytes, `vram.statics`, `vram.expert_floor`, `vram.expert_ceiling`, `vram.kv_ceiling`, and each card's expert-plane geometry (`engine_expert_plane`, `expert_unit_max`). Each card's physical capacity is the one minimum-bound exception: the live capacity must be at least the measured one. A new consumer or missing expected consumer is a mismatch. The expected plan freezes the **resolved** placement: where `ngram_placement = auto` or the embedding's RAM-to-VRAM fallback lets placement follow live resources, the placement measured into the plan is the one admitted, and a start whose automatic placement resolves differently (changing `vram.statics` or a host consumer) is a deliberate `BC_CFG_PLAN_MISMATCH`, remedied by remeasuring and resolving a new manifest. Machine-relevant byte sizes use equality; optional explanatory `from`/`remedy` strings are diagnostic text, not memory arithmetic. A mismatch is `BC_CFG_PLAN_MISMATCH`.
- **Live terms** are never compared against the measured snapshot: `vram.cards[].free`, `vram.claimable`, and everything the engine derives from claimable — `vram.elastic`, `vram.experts` (expert residency), `vram.stage` (the prefill staging region), `vram.kv` and `vram.kv_resolved`. The plan's measured `kv` is never an admission input. The live resolved KV must be at least `kv_required`.
- **`kv_required`** `= max(expected.kv_one_session.bytes, ceil(expected.vram.kv_ceiling * fraction_ppm / 1000000))`, where `fraction_ppm` is the value of the `kv_backing_floor.fraction_ppm` entry selected in `spec.calibration`, or 0 when none is selected (the throughput objective has no backing floor). The floor is a fraction of the **total** `kv_ceiling`, which includes the fixed (recurrent and convolution) state; it is not the engine's reported paged-backing percentage, which excludes that state. The ceiling is computed without an intermediate beyond 64 bits (quotient and remainder by 10^6).
- **Admission** has no arbitrary tolerance, no reserve constant, and three checks, each refused with `BC_CFG_ADMISSION_REFUSED` before allocating, naming the live bytes, the required bytes and a remediation:
  1. `live.vram.experts >= expected.vram.expert_floor` — the experts the host pool cannot hold stay on the card;
  2. `live.vram.claimable >= expected.vram.statics + live.vram.experts + live.vram.stage + kv_required`, naming `kv_required`;
  3. `live.vram.kv_resolved >= kv_required`, naming both numbers. `kv_resolved` is the **common KV budget** the engine resolves for every rank: `vram_kv_mib × 2^20` after the staging region is paid for and the cache is rounded to whole MiB. On a single rank (the production layout) it is the cache the engine allocates. On an uneven tensor-parallel world the engine may then give a cheaper rank fewer bytes for the same paged-token capacity (`vram_budget_rank_split`); the admission checks guarantee the common budget, and each rank's split must cover the same paged-token capacity as `kv_required` (RSC08 acceptance).
- Every VRAM term the checks read is a byte count in `[0, 2^53 - 1]`; anything else, a missing `kv_resolved` included, is refused as invalid input.
- The shared vectors in `admission-vectors.json` (generated by `tools/tests/core/serving_admission_model.py`, which models the engine's split and reproduces this directory's examples byte for byte) are normative examples of these checks; the contract test and the engine's check test both run them.
- The initial owner-map hash and all owned-expert sets must match the live helper map before serving. No fallback to planning with zero helpers.

Evidence (§11, amendment 5): in four hardware measurements of the same configuration on 2026-11-02 (RSC03 run rsc03-plan, live1..live4), `claimable` read 31,198,251,000, 31,198,242,808, 31,198,234,616 and 31,198,242,808 bytes, moving in 8 KiB steps over a 16 KiB range, and `statics + kv + experts + stage` equalled `claimable` exactly every time (margin 0), because the engine's KV takes the remainder. Admitting on the measured split would refuse any start whose reading came in lower than at measurement time.

## 6. Lab experiment namespace

`spec.intent.service.profile` is exactly `production` or `lab`. A lab profile is selected **explicitly at input resolution**, never inferred from the presence of a diagnostic setting.

- Production MUST omit the entire `spec.experiment` key, even when its values would be false, zero or otherwise apparently harmless. Reject explicit experimental values from site/profile/CLI before producing an artifact as well as when reading a forged/rehashed artifact.
- The error is **`BC_CFG_EXPERIMENT_FORBIDDEN`**, mapped to **`RAD_E_SCHEMA=-9`**, exit **2**. Stable message: **`spec.experiment is forbidden when spec.intent.service.profile=production`**. Hint: `select profile=lab explicitly and regenerate the manifest`.
- Lab `experiment.placement` has `mode=observe` (no activation change) or `mode=forced_churn` (requires the `forced` schedule; §11 amendment 10 replaced `at_step`), and an optional `epoch_cycle_s`. The mutation is a **live RHP11 protocol operation**; it must not rewrite the initial owner map.
- Lab `experiment.shape` mirrors the seven helper concern groups, with partial `helper_common` and per-role `helpers[{role,settings}]`. Apply it by the exact merge order in §2.2, and include the **resulting effective settings** in the plan cache key. Any plan-sensitive change must be measured before serving; an old plan is not reused.
- Lab `experiment.diagnostics` is typed by concern: `gemv.{probe_mode,refusal_trace}`, `lanes.{capture_file,capture_every,capture_max,trace_level,test_stop_at}` and `kernels.{plugin_override,poison,sweep_fast}`. Diagnostic probe values that alter correctness are lab-only; tests must mark non-comparable outputs.
- Unknown experiment fields, invalid role references, unsupported mutation modes and incompatible settings fail closed; lab is **not** an escape hatch from schema, memory or ABI validation.

The `example.lab-experiment.json` shows an observe placement experiment, common GEMV H2 and kernel changes, a helper2 GEMV/lane override and typed diagnostics. Its initial owner sets intentionally equal the production fixture; the lab plan cache key differs because effective shape differs. A real lab run must measure a matching engine plan.

## 7. Provenance coverage and explain

**Choose sparse provenance with enforced complete coverage**: a record is required for every *non-schema-default* effective leaf, but a record on an ancestor covers its descendants **only when every covered descendant has the same origin/source/class/lifecycle/confidence**. Schema-default leaves inherit from the generated field registry and the required root `""` entry; the writer MUST NOT conceal a site, derived, calibration or experiment value under the root default.

Records use RFC 6901 **JSON Pointer** keys (e.g. `/spec/resources/host/pool_mib`), with `~0` and `~1` escaping. A record has exactly `{origin,source,class,lifecycle,confidence,inputs[]}`; inputs are JSON Pointers to real fields or metadata identity fields, and the dependency graph must be acyclic. `source` is a human-referenceable file/section or named derivation rule, not a second copy of the setting. Metadata digest and provenance records themselves are not covered by a recursive provenance tree.

**Identity exemption (§11, amendment 6).** These leaves identify the document and are written only by the producer; they need no provenance record of their own, the required root `""` record covers them: `/apiVersion`, `/kind`, `/metadata/instance_id`, `/metadata/contract_version`, `/metadata/producer/name`, `/metadata/producer/build_sha256`, `/metadata/content_digest`. The list is by name, with no wildcard. Every other leaf under `/apiVersion`, `/kind` or `/metadata` needs a record at the leaf or at a proper ancestor other than `""`, so a metadata field a later version adds is uncovered (refused with `BC_CFG_PROVENANCE_INVALID`) until it is given one. There is no provenance `class` for identity fields.

This provides exact auditability without thousands of identical provenance objects for `owner_map.helpers[].experts`. A single `/spec/plan/owner_map` record can cover that whole homogeneously derived subtree; a per-role site override needs its own more-specific pointer. Reject nonexistent pointers, incorrect default claims, uncovered non-default leaves and cycles.

Align the startup `given` report with this source model. `bc-serve --explain PATH` can render all leaves with their effective origin and dependency chain, or emit the same structure as JSON; both use the generated registry and shared library. Do not maintain an independent Markdown explanation of values.

## 8. Canonical JSON, serialization, verification and evolution

Use JSON Schema Draft 2020-12 for **structural** validation and one C++ semantic validator for cross-field invariants. The schema requires closed object types. **Every JSON number in a v1 document is an integer** (I-JSON-compatible, at most 2^53−1); a fraction is a fixed-scale integer with the scale in its name (`_ppm`, `_milli`). There are no non-integer numbers, NaN or Infinity, no duplicate JSON properties, deterministic arrays and UTF-8 text.

`serving-manifest.v1.schema.json` is **generated**: the field registry `engines/radiance/components/serving/contract/serving-v1.toml` is the one source of the schema, the typed C++ model, the field reference and the contract fingerprint (`contract/README.md`). Edit the registry and regenerate; CI fails when a committed copy differs from the generator's output.

**JCS SHA-256 procedure**:
1. Serialize the fully validated manifest as JSON.
2. Temporarily remove `metadata.content_digest` (do not replace it with null).
3. RFC 8785-canonicalize the remaining JSON: UTF-16 property-name sorting, ECMAScript number rendering, canonical escaping, UTF-8 bytes, no whitespace.
4. SHA-256 those exact bytes and insert lowercase `sha256:<hex>`.
5. All consumers independently recompute before applying configuration.

Sorted-key `nlohmann::json::dump()` is **not by itself** a JCS implementation. Use a tested implementation or a small narrowly scoped conformant encoder with RFC 8785 test vectors. With integers only, ECMAScript number rendering is plain decimal, and the encoder refuses any non-integer number rather than format it. The two example files have computed digests using UTF-16-sorted property keys; their content deliberately uses only ASCII and safe integers. Any change to either file requires regenerating its digest.

The manifest must be written atomically to a private, non-overwritten run path, treated as immutable, and loaded before HIP/weights. Content hash is integrity, **not** a signature. Filesystem permissions and trusted release artifacts enforce the trust boundary.

V1 is **closed**. Adding a field or accepted enum, changing a type, moving/renaming/removing a field, changing a derivation's meaning, or altering the canonical digest procedure requires `bigcherry/v2`. No runtime v1→v2 translator, legacy aliases, or partial field acceptance. Producer, readers, plugins, bindings, fixtures and docs migrate together. The build pins one supported contract/schema fingerprint; missing or incompatible contracts fail.

## 9. Validation and executable responsibilities

The same `bc_serving` C++ library is linked into `bc-serve`, Radiance (via a thin managed patch) and `r3_moe_worker`. It owns strict parsing, JCS hashing, structural + semantic validation, provenance, projection and structured errors. The engine adapter only applies the curated typed fields to `rad::Config`, using shared native Radiance validators. The helper kernel ABI remains size/version checked; no C++ object crosses the plugin C boundary.

Startup order: open and size-limit JSON → reject duplicate JSON members → check `apiVersion`/`kind` → verify JCS hash → schema and semantic checks → resolve typed role view → check role/device/plugin identities → check initial owner map → run engine admission check → initialize GPU/model. Once the starting set is valid, RHP11 live epochs carry later placement changes.

**Card-free release gates:** schema meta-validation; producer-generated fixture regeneration and diff; each consumer loading the same fixture; JCS known-answer vectors; mismatched-hash and correctly rehashed invalid-semantic negatives; lab-only enforcement; exact role/owner-map checks; calibrated ledger invariants; cache key invalidation when owner map or helper shape changes; real Radiance native Config parity; no remaining application `R3_*` getters or synthetic argv; patch rebase tests. GPU performance/identity checks are separate leased tests and were **not** run to author this specification.

Reference inputs reviewed: `RSC03.md:94-145`; `plan_cache.{h,cpp}`; `ledger.{h,cpp}`; `rad0011_plan_memory/patch.py:88-263`; `experts.{h,cpp}`; [RSC03 hardware check](../../evidence/RSC03/PLAN_MEMORY_HARDWARE.md).

## 10. Stable errors and process exits

See `errors.md` for the full `BC_CFG_*` enum, mappings onto Radiance `RAD_E_*`, machine error object, human rendering and all process exit codes. The library returns typed errors without logging; each executable renders an error once with `<executable>: config:` and exits before GPU access. Unknown fields use `BC_CFG_UNKNOWN_FIELD / RAD_E_SCHEMA / exit 2`; production experiments use `BC_CFG_EXPERIMENT_FORBIDDEN / RAD_E_SCHEMA / exit 2`; a live plan shortfall uses `BC_CFG_ADMISSION_REFUSED / RAD_E_NOMEM / exit 4`.

## 11. Amendments against the review draft

The review draft (GPT review, 2026-11-02) is superseded by this copy. Approved amendments (lead, 2026-11-02):

1. **Calibration fractions become fixed-scale integers.** `host_reserve.total_fraction` (0.05) is `host_reserve.total_fraction_ppm` = 50000; `kv_backing_floor.fraction` (0.6) is `kv_backing_floor.fraction_ppm` = 600000, each in `[0, 1000000)`.
2. **Helper balance speeds become thousandths.** `placement.balance_speeds` (positive numbers) is `placement.balance_speeds_milli`, an array of integers ≥ 1 (1000 = speed 1.0), in both `helper_common` and the per-role and lab patches.
3. **Calibration values are integer, string or boolean.** The schema's `number` alternative is removed, and the contract generator rejects any non-integer number type in the field registry, so it cannot return.
4. **Both examples are re-emitted** with these fields and their `content_digest` recomputed, cross-checked by an independent canonicaliser in the contract tests.

5. **Admission uses what the plan requires, read from the live split** (lead, 2026-11-02, after RSC03's hardware check A3 and the GPT review of this amendment).
   - *The measured KV share is not an input.* §5.2's inequality was `live.vram.claimable >= expected.vram.statics + expected.vram.kv + expected.vram.experts + expected.vram.stage`. Since the engine's measured `kv` is the remainder of the elastic plane, that sum equalled the measured `claimable` exactly (margin 0 in all four measurements: 31,198,251,000 / 31,198,242,808 / 31,198,234,616 / 31,198,242,808 bytes, 8 KiB steps over a 16 KiB range), so a start reading even 8 KiB lower would be refused at random. Admission now uses `kv_required = max(kv_one_session, ceil(kv_ceiling × kv_backing_floor.fraction_ppm / 10^6))`. A calibrated VRAM reserve was rejected: it would turn a measurement artefact into a constant and spend real KV on it.
   - *Staging and expert residency are live terms, not stable facts.* In Radiance's `vram_budget_resolve` (`core/mem/vram_budget.cpp:150-178`) expert residency is what the split leaves the experts — `min(max(expert_ceiling, expert_floor), elastic − kv)` when the cache lends, `min(expert_ceiling, elastic − kv)` when it does not — so it moves with claimable whenever the ceilings do not both fit. The staging region (`:198-222`) is `2 × align_4KiB(ceil(f × layer_max))` with `f = (expert_ceiling − experts) / expert_ceiling`, so it moves with residency, and it is paid by the experts while they stay at or above their floor, otherwise by the KV (`b.kv -= b.stage`) when the cache lends. The review draft compared `stage` (and named "stage bytes" and the expert footprint) as stable; an 8 KiB lower reading could grow the stage by 4–8 KiB and be refused as plan drift. Both are now live terms; the stable expert facts are `expert_floor`, `expert_ceiling` and the cards' plane geometry.
   - *The resolved KV budget must meet the requirement.* When the KV pays for staging, its share drops by the whole region with no change in `stage`; and the engine rounds the cache down to whole MiB after the weights' MiB round-up (`cfg.vram_kv_mib`). The plan therefore records `vram.kv_resolved` (`vram_kv_mib × 2^20`, the common budget for every rank), and admission requires `kv_resolved >= kv_required`. Per-rank splits on an uneven tensor-parallel world are checked for paged-token coverage, not bytes (re-review S1; RSC08).
   - *What each check catches.* With every elastic term taken live, the capacity sum (check 2) reduces to `live.kv + unused >= kv_required`, where unused is what the split leaves unassigned; the engine always splits all of claimable, so it is close to an identity and is kept as a fail-closed guard against a resolver that over-allocates. The checks that bite in practice are `kv_resolved >= kv_required` (check 3, which adds the MiB rounding and any staging the KV paid) and `experts >= expert_floor` (check 1).
   - *Shared vectors.* `admission-vectors.json` carries resolver-shaped pairs (the A3 readings, 8 KiB steps, the MiB truncation edge, a staging payer switch alone and with the MiB rounding, residency moving above and falling below its floor, no lender with and without experts paying the staging region, the full-ceiling round-up) and boundary pairs (equality, one byte short, ppm 0 / 999999 / −1 / 10^6, the schema's largest byte count, stable-fact drift). `vram.kv_resolved` is added to the schema and both examples, whose `content_digest` is recomputed.

6. **Document-identity leaves are exempt from per-field provenance, by name** (lead, 2026-11-02, after the GPT review of the contract registry). §7 exempted the metadata digest and the provenance records, but every other identity field (`instance_id`, `producer.*`, `contract_version`, `apiVersion`, `kind`) would have needed a record whose `class` has no fitting value. The seven paths listed in §7 are exempt and covered by the root record; any other identity-envelope leaf needs its own record, so the rule fails closed for future metadata. No `class` value is added. The shared reader enforces it (`check_identity_coverage`).

7. **Every pattern-constrained string has a `maxLength`** (lead, 2026-11-02, after the GPT review of the shared reader). libstdc++'s `std::regex` recurses once per repetition, so a long enough string crashes the reader before any error is reported: a provenance key of 100,000 `/a` segments (200,000 characters, far under the 16 MiB document limit) segfaults it, while 10,000 segments (20,000 characters) match. Every string with a `pattern` therefore has a `maxLength`, counted in characters as JSON Schema counts it, and the reader checks it before the pattern runs: `jsonPointer` 1024 (at most 4,096 bytes of UTF-8, a 5x margin under the 20,000 bytes measured safe even when every character takes four bytes; real pointers are a few dozen ASCII characters), the shared-memory name 255 (POSIX `NAME_MAX`), a GPU `arch` 32, a helper `role` 32, and the fixed-shape strings their exact lengths (`hex64` 64, `digest` 71, `bdf` 12, `instance_id` 36). The contract generator refuses a pattern without a `max_length`, or with one above 1024 characters, so at most 4,096 bytes ever reach a pattern and no future field can reopen the crash.

8. **A document nests no deeper than the closed schema allows** (lead, 2026-11-02, after the same review). 1,000,000 nested arrays, well within the 16 MiB limit, took the strict parser quadratic time and memory (it builds a path per level) and would recurse through canonicalization. Because v1 is closed, its deepest legal nesting is exact: the contract generator computes it from the registry (every object, array, map or tuple is one level; a ref to an object type continues into it) and emits it as `kMaxDepth`. The reader refuses deeper input with `BC_CFG_INVALID_FORMAT` while parsing. A type that refers to itself would make the depth unbounded, so the generator refuses a recursive ref; the cap always has a derivation.

9. **Schema defaults are a registry field, verified against a real source, and §7 coverage is enforced** (lead, 2026-11-03, closing the provenance-coverage gate #536 left open). §7 requires a record for every non-schema-default leaf, but the registry had no notion of a default, so the root `""` record (`origin: schema_default`) could hide any site, derived, calibrated or experiment value. A `[[field]]` may now carry `default` and `default_source`, together or not at all: a literal of the field's kind that its schema accepts (a field that refs a scalar def is checked against the def), from a named place, `radiance:Config.<member>` (the field's own `radiance_config`; the `<member> = <literal>;` initialiser in `struct Config` of radiance's `core/rad_core.h`) or `setting:R3_<NAME>` (the `default` of `[setting.R3_<NAME>]` in a component's `component.toml`). The generator and the CI check (`bc_contract_check`) read those sources and fail on any divergence; an initialiser that is not a simple literal is no source, and that field leaves its default absent. A default is never copied in by hand. The reader (`check_provenance_coverage`, run by `read_manifest`) then enforces §7 on every leaf under `/spec`: a leaf resting only on a schema-default claim (the root record, or a record whose origin is `schema_default`) must have a registry default and equal it; any other leaf needs a record at itself or a proper ancestor, and that record's `class` must be the class the leaf has at its document position (a shared type takes the class of where it is used, by the longest matching `[[class_default]]`), so a record covers only homogeneous leaves. Nothing else changes on the wire: `default` is a JSON Schema annotation, so the validation fingerprint does not move.
   **The semantic fingerprint moves once**, because a default decides which leaves need a provenance record and so is contract semantics: it is hashed per field, only where present. v1 had not shipped to any consumer, which is the only reason this is allowed under §8; **after this amendment any change to a default, its source or the coverage rule is a `bigcherry/v2` change.** Validation fingerprint `sha256:e0ec5b5d6d05808b3eb3fd1f69e160a408a534cfb8f749676f610a266e696fe3` (unchanged); semantic fingerprint `sha256:f9c9198530061c43a5f5e8926e59c634ab224a2aea362954f51c02d4a98b08a0` before, `sha256:ab22236abc50a05ac7ff2bf3484ea0602bcfe90276d55fe44378c5d99be60b49` after. Defaults sourced so far: engine `max_sessions` (256), `reserve_mib` (4096), `scheduler.tp` (1), `max_num_batched_tokens` (2048) from radiance's `Config`; helper `gemv.h2` (2), `gemv.w4h2` (true), `gemv.mr` (false), `runtime.power_save` (true) from the moe-remote settings. Both examples gain two records each (`/spec/handoff/transport` and `/spec/handoff/kernels`, origin `profile`, class `derived`, confidence `estimate`) because the five launcher literals they cover (`max_rows` 4096, `small_rows` 8, `gemv_large_rows` 32, `skip_owned` true, `moe_down_tile` 1) differ from the setting defaults (64, 64, 0, none, none) and were only ever hidden under the root claim; their `content_digest` is recomputed.

10. **The helper's remaining settings, the plan state and the measuring manifest** (lead and owner, 2026-11-05, closing RSC06). The helpers still read 22 environment settings v1 did not model: the live placement policy (`R3_MOE_POLICY_*`, 18 with its enable flag), the forced-move test controls (`R3_MOE_PLACEMENT_FORCE_MOVE`, `_FORCE_SCHEDULE`) and the epoch-cycle diagnostic (`R3_MOE_WORKER_EPOCH_CYCLE_S`). The owner's rule is one source: a process reads its whole configuration from the manifest, with no mixed environment and no allowlist. The contract therefore gains, once:
   - `spec.handoff.placement.live_policy` (optional, **production-allowed**: hot-expert deployment from the runtime heat map is the production goal). Absent: no policy thread. Present: every member is explicit, none has a default: `effect_gate`, `accept_unwatermarked_heat` (booleans); `cadence_ns`, `max_moves` (at most 4294967295), `max_moved_bytes`, `min_residency_ns`, `max_heat_age_ns`, `min_samples`, `abort_cooldown_ns`, `learned_eligibility_ttl_ns`, `effect_window_samples`, `effect_min_quiet_windows`, `effect_revert_ttl_ns` (positive integers); `margin_ppm` (1 to 999999), `half_life_observations_milli`, `effect_noise_multiplier_milli` (positive; the former decimals as fixed-scale integers, amendment 1's rule); `protected_layers` (strictly ascending model-global layer ids; empty means none). There is no enable flag: presence is the policy. The reader (`validate_semantics`) refuses a live policy together with `experiment.placement.forced` (one placement driver), with `processes.engine.scheduler.tp` other than 1, and without `handoff.telemetry.histogram_file`: the checks the helper made at start-up now fail the same way in every consumer. The policy's shares are `processes.helpers[].share`.
   - `spec.experiment.placement.forced` replaces `at_step` (never implemented by any consumer): an object `{repeat, steps[]}`, each step `{kind: move | swap | auto_swap, a_layer, a_expert, destination (move only), b_layer, b_expert (swap only)}` (an `auto_swap` carries its x, y in `a_layer`, `a_expert`). Mode `forced_churn` requires it and `observe` forbids it. An inline schedule is hashed with the manifest, where the former schedule file was an unhashed input. `spec.experiment.placement.epoch_cycle_s` (optional positive integer) is the epoch-cycle diagnostic. All of it is lab-only (`BC_CFG_EXPERIMENT_FORBIDDEN` under a production profile).
   - `spec.plan.state` (required, `measured` or `unmeasured`). The measuring start (the launcher's cold plan-cache run: the helpers come up, the engine runs to `--plan-memory`) needs a manifest before any plan exists, and the plan needs the manifest's owner map. An **unmeasured** manifest carries the owner map and every setting, no `plan.engine_memory`, and no `resources.host.checkpoint_slots` (derived from the plan: both are optional and present exactly when `plan.state` is `measured`). It is hashed like any manifest. `read_manifest` and `load_manifest` take a required `PlanUse` (`serving` or `measuring`; there is no default): a serving start refuses an unmeasured manifest and a measuring start refuses a measured one, both `BC_CFG_PLAN_MISMATCH` at `spec.plan.state`; `engine_view` of an unmeasured manifest is `BC_CFG_INVALID_STATE`. A measuring start is asked for in one place per binary: `radiance --plan-memory FILE` for the engine and `r3_moe_worker --measuring` for the helpers; bc-serve writes the manifest with `--write-manifest OUT --unmeasured`.

   **Both fingerprints move once.** v1 has not been put into service as of this change: bc-serve and the engine's rad0009 reader exist (they are built into the radiance lanes), but no serving start consumes a v1 document yet (the launcher starts the engine from flags and, until this change, the helpers from their environment), and nothing outside this repository reads or writes one (checked 2026-11-05: the launcher, the lab and campaign tools, the release and CI workflows). That is the only reason this is allowed under §8, as for amendment 9; **after this amendment any change to the registry, a default or a rule is a `bigcherry/v2` change.** Validation fingerprint `sha256:e0ec5b5d6d05808b3eb3fd1f69e160a408a534cfb8f749676f610a266e696fe3` before, `sha256:471535ef547f6bbdb9c93d22efe416e3bfdc04790826b7be5ccfe3daf708db06` after; semantic fingerprint `sha256:ab22236abc50a05ac7ff2bf3484ea0602bcfe90276d55fe44378c5d99be60b49` before, `sha256:8319ac928d6c011b882f9afcea07e66478fb0d4442b3dfcee228a8030193d98a` after. The two examples gained `plan.state`, the production example a live policy and the lab example an `epoch_cycle_s`; their digests were recomputed.

11. **The engine's remaining flags, so no raw engine argument survives** (lead, 2026-11-05, closing RSC08 S3; **confirmed by the owner 2026-11-06**, reported by the lead: "you can add the fields to v1 but you need to actually get this locked down"). With the engine started from its manifest alone (`--manifest M --role engine`, which refuses every other flag), the launcher's `--engine-arg` pass-through and the lab and campaign arms' direct flags need a manifest field or they cannot run. The inventory of radiance flags used in the tree and not yet bound (`tools/lab/radiance/*.json`, the launcher, `rad32-perf.sh`): `--profile-ops`, `--debug-selection`, `--deterministic`, `--kv-resident`, `--no-prefix-cache`, `--tp-wire`, `--gpu-headroom-mib`; every other flag in use is already a registry field, and `--prefix-cache-mib 0` does not stand for `--no-prefix-cache` (the first sizes the host KV tier, the second disables prefix caching), so that one is its own field. Added, all optional, none defaulted, integers or enums only:
   - `spec.experiment.engine` (lab only: `production` refuses it with the rest of `spec.experiment`, `BC_CFG_EXPERIMENT_FORBIDDEN`): `profile_ops`, `debug_selection`, `deterministic` (each the one value `"on"`: present is the flag; there is no second spelling of off) and `kv_resident` (a positive token count).
   - `spec.processes.engine.execution.tp_wire` (`exact` or `wht6`) and `no_prefix_cache` (the one value `"on"`), and `spec.processes.engine.device.headroom_mib` (an unsigned MiB count, schema default 96 from `Config.gpu_headroom_mib`), all profiles: a production tensor-parallel layout can need each.
   - The registry gained `radiance_switch`: a bound flag that takes no value, valid only on an enum whose one value is `"on"`; the engine reader emits the flag without a value. It is part of the semantic fingerprint where set.
   - A calibration key, `host_ledger_margin.mib` (an unsigned MiB count; a **seed** of 1024, labelled `source: seed` in `spec.calibration` with its evidence, a judgement of about 3.6 x the largest unexplained loss observed in the hardware gate, not a derivation; a refit cache entry replaces it, shows `source: cache`, moves the manifest digest and the derived slots, and is the subject of RSC10, which makes the engine's own host-room refusal store its shortfall and retry once): memory the host ledger leaves unplanned beyond the reserve, for what the host loses between the manifest being written and the engine's own host-room check. It is listed in `spec.calibration` when selected and counted by the ledger as a pageable consumer, so the derived `resources.host.checkpoint_slots` leave room for it. Found by the S3 hardware gate: a plan filled to the brim by the measured figures stopped the engine's own check by 205 MiB and a 49-slot start died under six concurrent streams.
   Unknown engine flags are not passable at all after this: the manifest is the engine's whole configuration. v1 is still consumed only by this repository's own launcher (the lab path; no released artifact and no outside reader exists, checked 2026-11-05), which is the same reason amendments 9 and 10 were allowed; **this amendment supersedes the closing sentence of amendment 10 for this one change. v1 is now closed: there is no amendment 12, and any further change to the registry, a default, a rule or a calibration key is a `bigcherry/v2` change.** The two fingerprints below are pinned by `tools/tests/core/test_serving_manifest_v1_contract.py` (core-offline), which fails on any change and names `bigcherry/v2` as the only path. Validation fingerprint `sha256:471535ef547f6bbdb9c93d22efe416e3bfdc04790826b7be5ccfe3daf708db06` before, `sha256:a6539bdfbb8ea182b7023a73e64167e7e181a8387e8a13b578556f9e58824996` after; semantic fingerprint `sha256:8319ac928d6c011b882f9afcea07e66478fb0d4442b3dfcee228a8030193d98a` before, `sha256:227546581a84a5a8534ac0d664bdb06d0a9ed183e2411657a97ff580594ad2d6` after. The lab example gained `experiment.engine`, `tp_wire`, `no_prefix_cache` and `device.headroom_mib`; its digest was recomputed.

No other part of the draft changed.

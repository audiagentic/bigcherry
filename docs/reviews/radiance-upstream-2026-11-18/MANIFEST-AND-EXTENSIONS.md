# Proposal: reuse upstream YAML, resolve once, keep extensions small

Status: design proposal; no runtime or contract implementation changed.
This supplements the [migration review](README.md). It incorporates the request
to refactor placement and extension code rather than port large injected blocks.

## Recommendation

Use upstream fleet YAML as the operator-facing configuration for Radiance.
Retain upstream names and semantics for settings it owns. Add one closed,
versioned top-level `x-bigcherry` namespace for BigCherry-specific intent.
Continue generating an immutable, strictly validated runtime contract for
helpers, measured admission, ownership and provenance. Do not ask operators to
maintain equivalent YAML and JSON settings independently.

Use the upstream YAML parser and `Config` validation for native settings. Resolve
BigCherry extensions once during preparation, then bind typed runtime state at
engine creation. Keep engine-specific adapters in BigCherry components and use
small upstream lifecycle/build hooks. No YAML traversal, JSON lookup, device
probe, policy search or string-based dispatch belongs in token execution.

This gives us one authoring surface while preserving the execution guarantees
that upstream YAML does not implement. YAML is a syntax and authoring decision;
it does not replace measured plans or transport/admission protocols.

## What the code already supports

Upstream `core/fleet/config.h` exports `load` and `parse(text, origin, out)`.
`config.cpp` checks duplicate keys and correct levels, merges defaults, and
passes native options through the existing `config_parse` table. The root walk
explicitly skips keys beginning `x-`. This is an available authoring namespace,
not an extension registration or execution API. Nested model keys are closed;
putting arbitrary `handoff` keys directly under a model is not supported.

Upstream provenance records YAML paths for option sources. BigCherry's current
v2 runtime reader additionally checks contract identity, canonical content digest,
generated structural decoding, pointer coverage/provenance, cross-field semantics,
measuring versus serving state, owner maps and live admission. These are distinct
requirements. Preserve the reader and generated registry machinery. The design
spec's heading still calls v2 inactive, but actual `main.cpp`, rad0009 and shared
v2 library integrations consume strict-v2 paths: reconcile that stale prose as
part of the implementation; do not design from the heading alone.

## Ownership and mapping

| Setting or artifact | Authority after migration | Migration action |
| --- | --- | --- |
| HTTP bind/auth/timeouts, aliases/model IDs | Native `server` and `models` YAML | Use upstream validation; eliminate duplicate authored BigCherry copies. Keep secrets referenced through environment/secret providers, not resolved into public reports. |
| Model path, TP degree, KV dtype, context/sequence limits, native speculation | Native model YAML | Map once into the effective per-model snapshot; validate local capability limits; pin defaults explicitly for baseline comparisons. |
| Fleet cards, priority, flex, park, groups and load/drain policy | Native YAML | Upstream remains fleet scheduler/allocator. BigCherry validates admitted bounds and augments lifecycle, without a second allocator. |
| Exact rank/device binding and heterogeneous TP ownership | `x-bigcherry.models.<id>` intent plus resolved topology | Compile into a typed engine-creation binding. Fleet `cards` is an allowed set, not a stable rank assignment. Extend allocation constraints rather than silently rewriting device IDs. |
| External helper roles, transports, kernel choices and admission limits | Typed BigCherry intent and resolved per-model helper contract | Keep absent by default. Validate complete helper ownership/encoding support before launch. |
| Heat-driven placement and calibration selection | Named BigCherry policy plus model/engine-scoped measured inputs | Produce immutable ownership/placement plan; route its local-expert portion through upstream placement interfaces. |
| GPU power policy | Named BigCherry fleet/device policy | Resolve per physical card once, not independently per model. Refuse incompatible concurrent model requirements. |
| Measured host/VRAM plans, owner-map digest, build identity | Generated runtime bundle | Never authored inside YAML; recompute when relevant model/build/topology/layout changes. |
| Live counters, actual applied power, timing samples | Runtime receipts/telemetry | Do not rewrite the immutable manifest to represent observations. |
| Optional diagnostic experiments | Explicit lab-only typed extension | Enforce production/lab admission; do not allow generic arbitrary keys or environment overrides. |

## Proposed authoring shape

Illustrative schema only: the `x-bigcherry` fields below are proposed, not
accepted by current bc-serve. Native keys are upstream keys. File paths are
project-relative examples; the implementation must define a single path base.

```yaml
server:
  host: "127.0.0.1"
  port: 8000
  api-key-env: RADIANCE_API_KEY
cards: [0, 1]
memory:
  gpu-headroom-mib: 160
models:
  flashnext:
    model: models/flashnext.rad
    aliases: [default]
    tp: 2
    pin: true
    kv-cache-dtype: i8
    max-model-len: 32768
    max-num-seqs: 1
x-bigcherry:
  schema-version: 1
  profile: production
  power-profile: stock
  models:
    flashnext:
      layout: flashnext-with-helper
      placement-policy: calibrated
      helper-profile: remote-moe
```

Named profiles refer to existing typed policy/catalog definitions, not opaque
files that inject arbitrary native flags. The layout can resolve a helper card
outside the engine's `[0,1]` card set; helper reservations must still be included
in global physical-device accounting. The policy must refuse overlap or explicitly
support shared use. If a named layout contains native TP/device values, those
are constraints: native YAML must agree, or resolution fails naming both sources.
Do not silently overwrite operator values.

Start with this small authoring extension, using existing policy types and
per-role overrides only when an actual use case requires them. Do not expose the
whole generated plan as editable YAML. A helper-free user can omit the section
and use normal upstream YAML unchanged.

## Resolution and launch

1. Parse once into a location-bearing YAML tree using upstream's existing parser
   stack. Native keys follow upstream validation. Validate `x-bigcherry` through
   a closed typed schema; unknown keys, orphan model references, duplicates and
   unsupported versions fail. Share the parsed tree through a small parser hook,
   or factor the upstream parser's tree-walk interface; do not build a second
   native-settings parser or parse the same settings under different YAML rules.
2. Resolve defaults, profiles, device identities and explicit rank bindings.
   Normalize native values after upstream parsing, preserving absent/default/
   explicit provenance and source locations. Record the pinned upstream identity
   and option-table/capability identity so defaults cannot drift unnoticed.
3. Measure resource needs and compile ownership, helper sessions and admission
   bounds. Feed external helper reservations into the fleet's accounting before
   either side allocates. Keep queue, staging, pinned-host and KV consumers in the
   same physical resource accounting; do not add independent overlapping budgets.
4. Produce an immutable bundle: resolved native fleet configuration, per-model
   typed BigCherry contracts, and a bundle index binding their digests and build/
   model identities. The bundle index references leaf artifacts; leaves need not
   hash the index, avoiding circular digests. Hash normalized semantics for cache
   keys; retain authored bytes/hash for provenance. YAML comments and key order
   should not invalidate a measured plan.
5. Start helpers from their resolved typed contracts. Bind the matching model
   extension state during engine construction; enforce the expected native
   config digest and placement identity before admitting serving work.
6. Launch through upstream `FleetConfig`/fleet lifecycle. A generated YAML view
   may be kept for inspection, but executable input must be the same immutable
   snapshot that was admitted, not a mutable authored file reopened later.

Existing v2 is a single-engine deployment contract; do not pretend it already
represents arbitrary multi-model fleets. First reuse it per single-model bundle
where semantics remain valid. A fleet envelope and shared resource contract need
an explicit versioned addition; change v2 only through its existing registry,
codec and fingerprint process. Prefer a new contract version for incompatible
lifecycle/resource semantics. Provide an explicit legacy-config import command
that resolves new artifacts; do not make current strict readers accept old wire
identities or add a silent runtime v1/v2/v3 fallback.

## Small integration surfaces instead of injected implementations

| Current pressure point | Proposed interface and ownership | Execution consideration |
| --- | --- | --- |
| rad0009/rad0011 inject startup and plan machinery | BigCherry-owned config adapter and plan validator, linked through a declared CMake target; upstream receives a narrow config-resolution/engine-creation hook | Validate once before allocation; no argv reconstruction or second parsing pass. |
| rad0002 creates policy and model-specific mailbox logic inside upstream paths | Owned remote-MoE adapter implements a typed execution extension; upstream retains declare, after-route, combine and lifecycle call sites | Declare/fix up handles once, bind immutable step context; remove mutable global policy lookup from hot path. |
| Placement skip-owned rules and layer-map copies spread across planner/runtime | A shared typed placement plan provides owner, local slot and encoding for each expert. Planner and runtime consume views of that plan | Use dense indexed maps/spans with stable lifetime; measure mapping bytes and locality rather than copying maps per step. |
| Shared libr3/libr4d collective patches modify singleton internals | Shared BigCherry collective-group context keyed by upstream TP-group identity, with a small adapter for each backend | Allocate scratch once per admitted group; preserve upstream chain/fence/wake ordering and cleanup. |
| KV fields/enums/row-size switches duplicated across patches | One owned layout descriptor table, extending upstream dtype/row sizing through a narrow capability query | Resolve descriptor at build time; use explicit formats or compile-time kernel specialization, not an indirect function call per element. |
| rad0034/35/36 inject competing scheduler controllers | Use native controllers where equivalent; otherwise one typed scheduler-policy hook with supported decisions and bounded state | Compute decisions once per scheduler step outside GPU critical kernels; avoid two controllers modifying one limit. |
| Diagnostics embed ad hoc file/tensor handling | Owned model-scoped observer sink invoked at stable milestones | Off path has no allocation, serialization or locks beyond an explicitly measured gate. |

Do not introduce a universal callback bus. Start with the concrete hooks these
components need, carrying typed engine/TP-group identifiers and explicit owning
lifetimes. Use direct internal C++ calls for linked engine adapters; preserve a
versioned C interface where a shared-object/plugin boundary actually exists.
Interfaces need documented failure/drain ordering, not only signatures.
BigCherry remains a thin integration patch on upstream plus owned components;
large string literals that create implementation files inside `patch.py` should
be migrated to normal owned source files with declared build identity inputs.
The hook patches still require anchor/compile/semantic validation, but implementation
changes no longer need to replace upstream functions or duplicate source files.

## Smarter placement without a second allocator

Separate **plan selection** from **allocation** and **execution**:

- Construct one immutable candidate plan from geometry, expert heat, qualified
  kernel throughput and measured host/peer link costs. Include helper memory,
  local expert slices, quantization quality policy and actual encoding capability.
  Optimize a named service objective such as latency or throughput under explicit
  host/VRAM/pinned-memory constraints; avoid a universal 'fastest' heuristic.
- Treat measured costs as calibrated estimates with an identity and validity
  envelope. Account for contention and uncertainty; refuse infeasible plans before
  allocation. Compare a bounded candidate set with the current admitted plan.
  A more elaborate heuristic must demonstrate benefit over the existing policy.
- Hand upstream its local placement constraints and external reservations; let
  the fleet allocator own shared fixed/flex resources. Helper-owned experts can
  be omitted locally only when the admitted execution paths all have guaranteed
  helper coverage. Prefill, variable verify rows, fallback, calibration and failed
  helpers must either be supported by that plan or refuse/replan before execution.
- Resolve rank ownership and expert-to-slot maps once and reuse them for planner,
  kernel declaration, transport setup and observers. This removes divergent
  interpretations and repeated conversion work, rather than merely shrinking code.
- Keep ownership fixed during a parent step. Later adaptive placement can select
  a new plan only at a drain/quiescence boundary with a new generation, helpers
  acknowledged and reservations transferred. Add hysteresis and minimum residency
  only after measurements justify them; no automatic per-token migration proposal.

Potential execution wins are fewer repeated configuration/policy/map operations,
better placement under measured links and fewer unnecessary allocations/copies.
None is a promised speedup. Profile allocation count, CPU step time, queue wait,
transport volume, GPU overlap, TTFT/ITL and energy against the current path.

## Reload, capability and failure semantics

Upstream `Fleet::reload` compares native model `entry.args` to decide when to
unload. `x-bigcherry` changes do not enter that comparison. Bind each model's
extension/config digest into its effective runtime identity; changes to ownership,
helper topology, encoding or memory policy require drain, remeasurement and a
new admitted bundle. No raw SIGHUP/admin reload of an authored file may bypass
that sequence on the managed path. Listener/card changes keep upstream's
restart requirement. A future lightweight observer change can have a narrower
reload rule only if implemented and tested explicitly.

Stock upstream silently ignores `x-bigcherry`. It therefore cannot enforce a
BigCherry helper plan. Managed launch must check extension capability/version
and bundle identity before starting helpers or accepting work. Clearly label
managed YAML and reject a stock binary in bc-serve when extensions are required;
no schema trick can force an unmodified stock binary to process ignored keys.
A patched binary should also fail closed when it sees required extension intent
without its admitted contract. Do not enable the multi-instance extension until
engine-scoped ownership and helper lifetimes are validated.

## Implementation sequence and acceptance

1. Extract existing injected implementation into owned components without changing
   behavior; keep API and compiled-source identity explicit. Test old-pin parity
   before mixing a refactor with the upstream bump where feasible.
2. Pin native JSON separately, implement YAML authoring/closed extension decoding,
   and generate an inspectable per-model bundle. Reuse registry generation and
   current measured admission; do not hand-maintain parallel field lists.
3. Integrate native config and one-engine lifecycle, then group-scoped transport
   and shared placement-plan views. Use an explicit capability gate throughout.
4. Migrate KV/scheduler policies through the small hooks; retire demonstrated
   upstream duplicates. Add fleet envelopes/reservations and multi-model lifetime
   support as a separate slice after single-model parity.
5. Validate duplicate/unknown keys, defaults/merge precedence, malformed aliases,
   model names, YAML/JSON provenance, digest mismatch, changed defaults, stock
   binary rejection, mismatched topology, helpers failing mid-start, resource
   overcommit and reload changes. Check equivalent native settings resolve exactly
   as upstream does, and helpers/engine consume the same plan generation.
6. Benchmark refactored and prior execution with identical model/layout/policy.
   Gate improvements on measured results; retain correctness and quality gates
   from the migration review. Report hook footprint and migrated owned source
   separately from total code size so moving code cannot masquerade as deletion.

This is a proposed design for review, not authorization to replace runtime
configuration, change live ownership, or claim fleet-safe remote execution.

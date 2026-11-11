# ServingManifest v2: inactive S9 contract

This is the **inactive** `bigcherry/v2` registry and generated codec scaffold. No production executable consumes v2 before the single S13 cutover. Shipped v1 remains pinned to its original registry, generated artifacts, semantics and strict reader; v1 is not an accepted alternative to v2, and there is no v1↔v2 translator.

## Identity and authoring

The single registry is `engines/radiance/components/serving/contract/serving-v2.toml`. Its generated JSON Schema, C++ types, strict C++ codec, Python codec, field reference, and schema/semantic fingerprints are committed under `engines/radiance/components/serving/generated/v2/`. The schema and field reference also have byte-identical documentation copies alongside this spec. Generated Python `decode(text)` rejects duplicate JSON members, non-integer numbers, unexpected properties and the v1 wire identity; `encode(document)` revalidates before serializing. Generated C++ `decode` and `encode` use v2 typed structures, with the shared error vocabulary from v1 (not a version translator). The S13 semantic validator remains the authority for canonical digest, complete provenance coverage, business cross-field invariants and full production startup validation.

The v2 wire identity is `apiVersion=bigcherry/v2`, `metadata.contract_version=2`, and schema id `urn:bigcherry:serving-manifest:v2`. Every field not explicitly changed remains a v1 field with its original meaning. All objects are closed and all numbers are I-JSON-safe integers; offsets use signed integers.

## Power (PWR08)

`spec.intent.power` requires a named `profile` and `role_overrides` (an empty map is valid). Overrides are nonempty patches of the four settings `vddgfx_offset_mv`, `sclk_max_mhz`, `sclk_offset_mhz`, and `power_cap_w`. `spec.plan.power` requires `mode` (observe/apply), `profile`, `profile_digest`, `devices[]` with `role`, `bdf`, `settings`, and `cpu.settings` (a closed empty object, no CPU controller in v2). The C++ resolver S10 must match the plan's role/BDF multiset with the engine TP BDFs plus selected helper BDFs, disallow duplicate BDFs and unknown role overrides, ensure plan profile matches intent, and insist the stock observe plan has empty settings. The generated Python scaffold already checks these structural/semantic relationships. The full qualification, PWR07 probe ranges, profile digest construction, apply-vs-observe safety rules and provenance checks belong to S10/S12/S13.

No static device power/clock ranges are invented in the manifest schema: signed offsets and positive setting integers are typed, while S10/PWR07 must use a fresh per-BDF capability probe and qualified plan limits. A schema bound is not evidence that a device can accept a setting.

Named profile **selection** is policy; the per-BDF resolved settings are derived from a qualified profile. A profile file changing after manifest resolution cannot change the immutable plan. Live readings, power writes and actual values belong to PWR09 receipts, not the manifest. Memory geometry/cache identity does not change with power policy; performance comparisons must include the resolved power plan digest.

## Measured resource handoff (S8/S10)

`spec.plan.resource_inputs` is a required object (not guessed constants) with engine-measured bytes for host embeddings, unowned experts, staging, n-gram, checkpoints and other residency; the actual owner-map digest; KV bytes per token; context and retention targets; live MemAvailable, effective cgroup headroom and pin availability; the selected host-pool-margin calibration key/source/evidence/MiB; and complete selected helper non-expert worker footprints per BDF (lanes, pointer banks, slot map, incoming slot, total). Every field is defined in `serving-fields.md`. This is not a second resource-budget or margin policy: S10 derives the existing `resources.host` and helper budgets, enforces admission and compares with the engine plan. `host_ledger_margin.mib` remains exactly one calibration key, with `seed` or `cache` and evidence naming a measurable refit.

The registry declares `origins=[derived,calibrated]` and `confidences=[exact,measured]` on RSC03 host budgets, RSC09's five handoff settings, and resource inputs. Both restrictions are included in the semantic fingerprint and exposed in the generated C++ field reference and Python codec. S10/S13 must enforce these constraints against effective covering provenance records, including records inherited from ancestor paths. No `profile`/`estimate` literal is a valid measurement.

## Live placement decision (RHP18)

`spec.handoff.placement.live_policy_state` is a typed required startup decision, separate from the runtime placement snapshot: requested on/off/not_applicable, configured, reason_code, reason_detail, calibration_key, record_identity, effect_gate, histogram_period_s, cadence_ns and max_heat_age_ns. S10 must create this decision after the policy preflight and correctly distinguish no qualified calibration record, stale heat feed and not-applicable layouts. S13 must display it and compare it with runtime degraded/placement-stopped evidence in receipts.

## S8/S10 integration interface and S5 refusal ownership

S8/S10 consume the committed `generated/v2/serving_types.h` / `serving_codec.{h,cpp}` under `engines/radiance/components/serving/`, namespace `bc::serving::v2`. The concrete members are `Manifest.spec.plan.resource_inputs` (`MeasuredResourceInputs`), `Manifest.spec.plan.power` (`PowerPlan`), and `Manifest.spec.handoff.placement.live_policy_state` (`LivePolicyStartup`). Python exposes `generated/v2/serving_codec.py:decode(text)` and `encode(document)`, raising `ManifestV2Error`. S8 supplies measured bytes and qualified handoff settings; S10 builds and validates those sections. S13 is the sole production cutover.

The admission refusal is an out-of-band S5/S11 wire protocol, **not a manifest field**. Its one authoritative consumer is `tools/bigcherry/serving/measured_plan.py:RoomRefusal.from_record` and `EngineRoomRefused.from_process_exit` (S11, #840). S5 (#844) emits `BC_SERVING_ADMISSION_REFUSAL=` followed by the JSON from `engines/radiance/components/serving/v1/error.cpp:room_refusal_json`, and exits 4. The `schema_version=1` on this error envelope is not the serving-manifest version. S10/S13 must reuse that exact contract, not clone the refusal schema in v2 or infer it from human error text.

## Validation, fixture limits and activation

Two **synthetic card-free** fixtures demonstrate v2 shape and deterministic JCS content digests. They borrow example device/host identities from v1 and intentionally use zero helper worker-byte components where no measured worker footprint was provided; **they are not hardware qualification evidence or admission defaults**. No GPU lease is required for S9. The generator diff check verifies both registries and rejects any drift from committed output; Python tests exercise version isolation, duplicate/unknown fields, type/bounds, role/BDF plan, margin and round-trip. S8/S10 must replace fixture placeholders with actual engine-derived values and complete the single-reader semantic checks before S13.

The v1 spec's canonical JSON/digest and release evolution requirements (section 8) continue to apply: all 2020-12 schema objects are closed, duplicate names are invalid, the content digest omits only `metadata.content_digest`, and a one-release reader pins exactly one schema and semantic fingerprint. This scaffold does not activate a second serving reader.

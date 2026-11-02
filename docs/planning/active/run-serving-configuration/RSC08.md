---
id: RSC08
order: 8
plan: run-serving-configuration
state: pending
created-at: '2026-11-02T12:14:33.492000+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: L
---

# Serving manifest v1: generated contract, shared loader and one-release cutover

## Description

Replace the interim serving manifest with the locked v1 wire contract (docs/design/serving-manifest-v1: spec, errors, schema and examples, integer-only per the spec's §11 amendments). One C++ field registry generates the typed model, the strict reader and writer, the JSON Schema, the engine and helper bindings, the field docs and a contract fingerprint. One bc_serving library is linked into bc-serve, radiance and r3_moe_worker. The engine takes native typed configuration (no synthetic argv), helpers take typed role views, and the cutover happens once, in one release, with no shims. Source: GPT design review 2026-11-02 (req_main_c69d1519f282, gpt-response-1.md) and the owner's decisions: a curated engine option surface; the manifest owns the initial owner-map selection and digest while live protocol epochs own later placement; a one-release incompatible cutover.

Fed by RSC01 (serving component, schema and loader), RSC02 (bc-serve as the only start-up path), RSC03 (resource plan: detection, calibration store, host ledger, owner map, measured-plan cache, rad0011), RSC06 (typed helper settings and plugin ABI) and RSC07 (the engine's manifest entry point).

## Steps

1. Contract and generator: the canonical field registry, a C++ generator (rejects float types), the complete v1 schema (checked against the committed docs/design/serving-manifest-v1 copy), the typed model, a schema fingerprint and generated docs, plus the JCS encoder with RFC 8785 vectors. No serving behaviour changes. The contract documents landed in #528.
2. Shared strict reader: size-limited open, duplicate-member rejection, apiVersion/kind check, JCS digest verification, structural and semantic validation, provenance coverage, typed engine and helper views, and structured ConfigError (BC_CFG_* on RAD_E_*, numerics checked against abi/rad_types.h).
3. RSC03 resource core on v1: detection (cgroups, memlock, pin cap), the calibration matcher (integer seeds), the host ledger (tier_first), the owner map (spec.plan.owner_map) and the measured plan (spec.plan.engine_memory with the v1 cache key). Evidence-backed budgets only; the receipt echo expected.host_pool_mib is verified equal to resources.host.pool_mib.
4. RSC06 helpers rebased onto v1: typed helper views replace WorkerConfig and manual extraction; the size- and version-checked plugin ABI is kept; helpers verify their initial owned set.
5. Native radiance integration: a managed patch links bc_serving into rad_core (thin CMake edit, no fallback); a generated config_apply_manifest replaces rad0009's argv converter; engine-side R3_* reads move to typed plugin configuration; the engine plans from the manifest's owned sets, and the startup admission check (§5.2) runs before weights.
6. bc-serve supervisor and tooling cutover: one immutable manifest; engine and helpers start with manifest and role only; the engine binary and plan cache come from the package and the work root; the shell launcher, the Python layout duplicate and the raw engine-arg path are deleted and lab callers migrated.
7. Release gate and documentation: schema fixtures, upstream rebase tests, no-legacy checks (no application getenv("R3_*"), no synthetic argv), RHP12 equivalence, generated inventory coverage, clean README and component docs.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria

Cutover criteria (GPT review §F):
- The exact same v1 manifest loads successfully in bc-serve, radiance and every helper role.
- No application-level R3_* setting influences execution through environment inheritance.
- The engine's resulting native configuration matches the accepted baseline for all supported fields.
- The host and GPU budget plan is derived or calibrated with evidence; no interim 9,216/12,288/22,528 MiB constants remain as unqualified universal defaults.
- The RSC03 resource ledger and the actual engine allocation agree, including model residency, n-gram placement, checkpoint memory, pinned consumers and reserved capacity.
- RHP12's fresh and retained-session cohorts, step-time checks and numerical-identity checks pass on hardware in the normal integration workflow.
- Invalid, old-version or partly migrated manifests always fail before any GPU work.
- The release packages exactly one supported schema and its matching binaries.


Slice 2 (shared strict reader) gates. Each is a required negative test, each failing with its stated code before any configuration is applied:
- Duplicate JSON members are refused at parse time with BC_CFG_DUPLICATE_FIELD (RAD_E_DUPLICATE) and the holding object's pointer. That includes escape-equivalent names that collide after decoding (e.g. "a" and "a"), duplicates nested and inside arrays, and not same-named members of sibling objects. A document whose only fault is a duplicate reports the duplicate, never a digest mismatch.
- metadata.content_digest of the wrong type (not a string), or of the wrong format (not exactly `sha256:` followed by 64 lowercase hex digits), is refused with BC_CFG_DIGEST_MISMATCH.
- A correctly re-hashed but semantically invalid manifest (owner map, memory plan, role or GPU references) fails with its semantic code, not a digest error.
- The input size limit and the UTF-8 checks are enforced before the document is accepted: oversize input, and invalid UTF-8 anywhere including member names, fail with BC_CFG_INVALID_FORMAT.



Admission wiring (slice 5 / RSC03 A2; deferred from the #534 review, S3):
- bc_plan_memory_check (rad0011) returns a structured result, not bool plus text, that separates invalid input (BC_CFG_CALIBRATION_INVALID or BC_CFG_INVALID_FORMAT for a bad ppm or byte count), stable-plan drift (BC_CFG_PLAN_MISMATCH, RAD_E_SCHEMA, exit 2) and resource refusal (BC_CFG_ADMISSION_REFUSED, RAD_E_NOMEM, exit 4).
- A production caller runs it at engine start-up, before weights: it takes the expected plan from the validated manifest's spec.plan.engine_memory.expected, and kv_backing_floor_ppm only from the selected spec.calibration entry kv_backing_floor.fraction_ppm (0 when none is selected). The live plan comes from the same plan() call that sizes the engine.
- The engine's check and the contract agree on docs/design/serving-manifest-v1/admission-vectors.json: every vector gives the same verdict and code class through the production entry point, not only through the card-free check test.



Admission, from the #534 re-review (GPT, 2026-11-02; lead decisions):
- Multi-rank (S1, hard criterion): kv_resolved is the common KV budget the admission checks guarantee. After vram_budget_rank_split, every rank's split KV must cover the same paged-token capacity, and the check refuses, naming the rank, when any rank's split falls short of kv_required's token count.
- Placement and boundary (S4): the check runs after the final live budget is resolved and before Pools::configure() (Engine::plan() already configures the pools, so "after plan()" is too late). All input validation (ppm, byte-count ranges, calibration selection) precedes the stable-plan and admission comparisons.
- Stable facts (S3): the host and checkpoint comparisons become the exact equalities §5.2 specifies, each with a negative test: a live host consumer with fewer bytes, a missing expected consumer, a changed residence, a changed checkpoints.want_slots and a changed host_pool_mib are each PLAN_MISMATCH. An automatic placement (ngram_placement = auto, embedding fallback) that resolves differently from the measured plan is PLAN_MISMATCH (S2).
- Vectors still to add to docs/design/serving-manifest-v1/admission-vectors.json once the model covers them (S5): a multi-rank uneven split, the kv_streams round-up of kv_one, and all_vram / layer_offload placement (no staging region).

## Notes

Feeds: RSC01, RSC02, RSC03, RSC06, RSC07.
Slice 1's contract documents: PR #528 (docs/design/serving-manifest-v1 and tools/tests/core/test_serving_manifest_v1_contract.py); its ledger event was recorded against RSC01 before this item existed and is relinked here.
Decisions: v1 numbers are integers only (fractions as *_ppm / *_milli); one site-only, tighten-only `constraints` namespace in ServingInputV1; the engine binary resolves relative to bc-serve's installed location and the plan cache from the work root, neither a CLI flag; generated files are committed and CI regenerates and byte-diffs them.


Slice 2 (shared strict reader) requirement, from the lead's #530 review (2026-11-02): duplicate JSON member names must be rejected while parsing. nlohmann::json::parse silently keeps the last of duplicate members, but I-JSON/JCS input must have none, and a duplicate must never reach hashing. The v1 reader therefore parses with a parser callback or SAX handler that tracks the member names of each open object and fails on the first repeat with BC_CFG_DUPLICATE_FIELD (RAD_E_DUPLICATE, -11) and the JSON Pointer of the object holding it. Before any digest check, a duplicate fails at parse time (errors.md: duplicate-member tests fail at parsing, before hashing). Tests: a duplicate at the top level, inside a nested object, and inside an array element, each naming its pointer; the same member name in sibling objects is not a duplicate; a document whose only fault is a duplicate fails with BC_CFG_DUPLICATE_FIELD, not BC_CFG_DIGEST_MISMATCH.

## Change Log

- 2026-11-02T12:14:33.492000+00:00 (created-by): Created by agent

## Ledger-events


- chg_20261102_121239_the-serving-manifest-v1-format_6543
- chg_20261102_121824_the-serving-component-can-prod_8129
- 2026-11-02T12:18:29.161474+00:00 (updated-by): Updated: section:ledger-events
- chg_20261102_124002_every-components-card-free-te_9438
- 2026-11-02T12:44:46.508959+00:00 (updated-by): Updated: section:notes
- 2026-11-02T13:01:00.152027+00:00 (updated-by): Updated: section:acceptance_criteria
- 2026-11-02T16:24:11.913385+00:00 (updated-by): Updated: section:acceptance_criteria
- chg_20261102_162418_engine-start-up-admission-no-l_6822
- 2026-11-02T16:24:22.385758+00:00 (updated-by): Updated: section:ledger-events
- 2026-11-02T16:40:58.669519+00:00 (updated-by): Updated: section:acceptance_criteria
- chg_20261102_170003_the-serving-manifest-v1-schema_4690
- 2026-11-02T17:00:07.344057+00:00 (updated-by): Updated: section:ledger-events

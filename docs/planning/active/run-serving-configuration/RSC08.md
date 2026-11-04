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
- Vectors (S1b, 2026-11-03): added to docs/design/serving-manifest-v1/admission-vectors.json (now 58): all_vram and layer_offload placements with no staging region (stage 0), and an uneven world of cards with different capacity, expert plane and unit (a per-card capacity minimum, per-card stable plane and unit, a card-count change, and one common resolved KV). Each is checked by the Python model, the C++ checker (bc_v1_test) and the engine's production entry point (plan_flags_test).
- Named gates still open (lead, 2026-11-03; production is tp=1, so none is exercised on hardware yet):
  - TP>1 hardware coverage of the admission check. The plan carries one common `kv_resolved` and no per-rank KV; adding a per-rank field would change what validates (a v2 change). The check for TP>1 must be per-rank paged-TOKEN capacity after `vram_budget_rank_split` (each rank's `probes[i]->paged_tokens(rank_cfg_[i].vram_kv_mib)` covering kv_required's token count, refusing naming the rank). It must NOT be a minimum over ranks of KV MiB: upstream re-splits a cheaper rank to FEWER MiB so that it pages the same token count, so a min over MiB would refuse a valid uneven world. The hook runs before the rank split today, so this needs the call moved after the split (engine_bringup.cpp, between the rank split and `Pools::configure`) and a TP=2 hardware run.
  - The kv_streams round-up of kv_one is the engine's computation (`VramFacts::kv_one`, reported as `kv_one_session.bytes`), not a checker rule; its coverage is an engine-side test, still to be written.



Slice 2 gates left open by #536 (lead, 2026-11-02). Each blocks the cutover (A4, slices 5-6):
- DONE (S1c, 2026-11-03; spec section 11 amendment 9) Provenance coverage (spec section 7). Registry defaults with a verified source, the reader's coverage rule, and the two examples' honest records are in. Remaining: more defaults can be sourced (only 8 fields have one so far); each addition after this is a v2 change, so sourcing is complete only when the owner says the set is. Original text:
- Provenance coverage (spec section 7). Every non-schema-default effective leaf is covered, and a schema-default claim never hides a site, derived, calibrated or experiment value. This needs schema defaults as a registry field, taken only from the spec text or a measured source and never guessed. Until then the reader checks pointers, input cycles and amendment-6 identity coverage only.
- Exact owned-set bytes. spec.plan.owner_map.helpers[].bytes equals the GPU-free bc::experts::select_owned sizing for the model's expert formats. #536 refuses only bytes that contradict emptiness.



Candidates to derive or calibrate (lead, 2026-11-03): the five handoff literals the flashnext profile sets, `max_rows` 4096, `small_rows` 8, `gemv_large_rows` 32, `skip_owned` true and `moe_down_tile` 1, are hand-picked tunables. The examples record them as profile-origin, derived-class, confidence `estimate` (they differ from the component settings' defaults, so they are not schema defaults); the profile must not be treated as their final source. Derive or calibrate them like RHP16.

Interim engine plan-check flags (A3, lead decision 2026-11-02). Removed at the cutover (slices 5-6), not before and not after:
- The flags are `--plan-check FILE --plan-check-floor-ppm N` on radiance (rad0011), `bc-serve --plan-check-args`, `--plan-store` and `--plan-key`, and the launcher's plan step (tools/lab/radiance/lib/plan-check.sh). They exist because no producer writes a v1 manifest with spec.plan.engine_memory yet.
- At the cutover the engine reads its expected plan and floor with engine_view(manifest); all of the above is deleted in the same change, with no shim.
- tools/tests/lab/test_interim_plan_flags_removed_at_cutover.py fails as soon as an engine-side caller uses engine_view while any flag site still names plan-check-floor-ppm, so the flags cannot outlive the cutover.
- bc-serve alone cannot measure the plan (without helper-owned experts the engine's plan is a different one, RSC03 hardware step 1a). Until the v1 owner map exists, the launcher's measuring start (helpers up, engine --plan-memory) is the only producer. The v1 producer replaces it.

Open correctness item from the A3 hardware gate (2026-11-02, #538 at 296bda44b), unresolved:
- The first #538 start of a cold plan cache (measuring start, then the real start, one launcher invocation) served garbage: every token `!`, draft acceptance 1.00, 4.0 tok/step, 6 streams identical (text hash 6bcedd0e). The next start of the same build and flags (cache hit) and every main arm were correct (8ba4f122, decode within noise). Evidence: runs/a3-gate (work root) (off2/off3 results, keep/ logs).
- Its engine and helper logs match the correct start's apart from timing noise and the checkpoint slot count; no warning separates them.
- Seven cold-cache launches afterwards (immediate, after a 20 s pause, after a main arm) all served correct text, every PID of the measuring start was gone and the shm inode changed each time. No first-run state (files, caches) was found. The failure is therefore not reproduced and its cause is unknown; #538 stays failed.
- Hardening added regardless: fresh helper logs per start, fail-closed stop (no continuing while a process of the measuring start lives), and a first-serve text assertion in the gate. The re-gate on the fixed build (a new build id, so a true first run) must serve the reference text on every launch, the first cold one included.

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
- chg_20261102_172348_bc-serve-the-engine-and-the-h_2822
- 2026-11-02T17:23:52.577413+00:00 (updated-by): Updated: section:ledger-events
- 2026-11-02T17:53:14.387976+00:00 (updated-by): Updated: section:acceptance_criteria
- 2026-11-02T19:29:34.542005+00:00 (updated-by): Updated: section:acceptance_criteria
- chg_20261102_193311_the-engine-now-checks-at-ever_1908
- 2026-11-02T19:33:20.254177+00:00 (updated-by): Updated: section:ledger-events
- chg_20261103_033321_the-serving-manifests-owned-e_3536
- 2026-11-03T03:33:25.449820+00:00 (updated-by): Updated: section:ledger-events
- chg_20261103_042644_the-serving-manifest-now-enfor_8446
- 2026-11-03T04:26:49.257727+00:00 (updated-by): Updated: section:ledger-events
- chg_20261104_075438_the-serving-configuration-tool_6545
- 2026-11-04T07:54:42.951709+00:00 (updated-by): Updated: section:ledger-events

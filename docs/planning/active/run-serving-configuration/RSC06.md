---
id: RSC06
order: 12
plan: run-serving-configuration
state: in_progress
created-at: '2026-11-02T03:52:00.963006+00:00'
breadth: ''
skill: intermediate
created-by: agent
work: M
priority: P0
---

# Helpers start from native YAML and narrow derived transport facts; no R3_* config

## Description

Existing typed worker/kernel settings and strict loader replaced R3_* getenv. Finish hardware parity and remove full launch manifest input. All authored helper intent comes from the same closed x-bigcherry YAML as owner/engine, with assigned role and genuinely derived budget/ownership/transport facts from existing bootstrap. No separate helper configuration file, no duplicate native parser or env fallback.

## Steps

Reuse one shared BC YAML decoder and typed HelperSettings; derive per-role effective settings from YAML and assigned role, validate against actual selected physical device, model/build, shared generation and independent plan_owners. r3_moe_worker --config same sealed bytes + role, use existing bootstrap/transport for derived owner map/budget, only bounded role-specific admission facts if indispensable. Migrate engine remote/typed plugin settings (rad0002/rad0012) and manifest_identity generation to scoped actual ownership/admission digest; preserve head/tail, formats, lanes, combine/affinity, heat/live policy/calibration/forced lab gate, clean drain. Remove --manifest loader and all actual campaign/lab callers together; no R3_* config reads or legacy ABI fallback. Preserve RHP16/20 owner and per-layout outstanding gates.

## Detailed Solution & Technical Design

docs/design/radiance-native-yaml-launch.md sections 3–4 and 8. components/moe-remote/r3_moe_worker.cpp, worker_bootstrap.h, worker_policy*, moe_remote_shm.h, engine_rank_coordinator.h, serving/v2/helper_settings, libr3/r3_moe_settings.h, rad0002/rad0012. No whole Manifest reader in worker or engine. Reuse existing shared transport, no new scheduler or generic launch descriptor.

## Code Samples & Guidance



## Files



## Validation

Card-free strict YAML/role/derived identity/unknown/missing/range/duplicate, wrong BDF/model/root/generation, mismatch owner-map and unsupported format refuse before HIP; generation reuse/cancel/drain, compare typed settings and exact owner/byte sums with baseline. Actual helper layout same-build text/step time and relevant numerical gates, RHP12 cohorts; grep no R3_* configuration reads or --manifest launch. Existing per-layout live swap gates stay with RHP20.

## Effort & Risk



## Standards



## Acceptance Criteria

Helpers consume same native YAML and assigned role, independent ownership validation and derived facts only, no full manifest launch reader or env fallback. Actual helper hardware parity, stable lifecycle and quality, old direct manifest input rejected. Preserve existing no R3_* reads and typed kernel ABI.

## Notes

Depends on the RSC01 schema. The engine side is RSC07; the supervisor is RSC02.

2026-11-05: redone on the v1 contract (supersedes PR #527). Helpers read their whole configuration through the shared strict reader: r3_moe_worker --manifest FILE --role helperN (v1::helper_view projected by serving/v1/helper_settings.h); no helper R3_* getenv and no allowlist (tools/tests/lab/test_helper_reads_no_environment.py); the one environment write is GPU_MAX_HW_QUEUES. The 22 settings v1 did not model became manifest fields in spec section 11 amendment 10 (live_policy production-allowed under handoff.placement; forced schedule and epoch_cycle_s under experiment.placement; plan.state with the unmeasured measuring manifest, one measuring flag per binary). rad0010 helper kernel config renumbered rad0012 (weak symbol, install-once). The launcher writes the manifests (unmeasured for the measuring start, measured for the real start). Open: hardware identity and step-time validation against main (needs the gpu lease); campaign arms that start helpers need a manifest per arm (BC_SERVING_MANIFEST in engine_adapters/radiance.py) until the S4c campaign slice.

2026-11-25 owner direction: no launch manifest, all authored serving configuration YAML, no compatibility shims or legacy backports. docs/design/radiance-native-yaml-launch.md supersedes earlier manifest-launch instructions; completed RUM04/05/06/09/12 evidence is historical, not proof of the new launch path. Existing capture manifests may remain output-only. No new programme or second resolver. Historical v1/v3 helper startup, 22 fields and 2026-11-05 delivery are retained as evidence, not a reason to rerun the environment migration. Existing helper hardware identity/time gate remains outstanding.

2026-11-25 cross-programme bump review: docs/design/radiance-bump-integration.md is the shared implementation-order and compatibility specification; MEN24 owns sequence and timing, domain implementation remains here. No duplicate checker, bump driver, manifest launch or legacy fallback. Replace moe-remote/component.toml manifest-path coupling and old protocol flags with actual shared YAML extension/typed role transport boundary, explicit wire generation/identity and callback signatures, update tests and selected patch dependencies atomically. Exact candidate helper row/format/owner-map/transport/readiness/drain checks, wrong generation/BDF/format fails before HIP. Keep existing RHP20 per-layout qualification and RCS23 provider identity, no new helper protocol just for bump.

2026-11-25 second integration review: docs/design/radiance-integration-review-20261125.md is the exact-source review and controls conflicting earlier wording. Shipping pin remains b6b8751 (2.2.0), upstream observed main 1c104de4 is a development candidate, 119 commits and 327 changed files; no automatic repin or backport. No new compatibility layer, launch manifest, bump driver or identity machinery. Candidate native quiescent callback serialization 896ab80f and Fleet failed-handoff/staging ownership changes affect helper drain, not new transport. Reuse upstream semantics where target has them, actual provider-specific packed layout/row and independent owner map per rank, exact copy completion and group stop before power restore. No new helper config/ABI shim; test old and new target seams, RHP20 remains owner of live policy qualification.

## Change Log

- 2026-11-02T03:52:00.963006+00:00 (created-by): Created by agent

## Ledger-events

- chg_20261102_035356_serving-start-up-plan-one-str_4985
- 2026-11-02T03:54:11.557458+00:00 (updated-by): Updated: section:ledger-events
- 2026-11-05T06:03:06.682889+00:00 (updated-by): Updated: section:notes
- 2026-11-05T06:03:10.546906+00:00 (state-transition): State: pending → in_progress
- chg_20261105_060310_the-expert-helper-processes-no_8907
- 2026-11-05T06:03:14.921955+00:00 (updated-by): Updated: section:ledger-events

## Reviews

- RV4593
- 2026-11-25T01:38:00.583347+00:00 (updated-by): Updated: section:title, section:description, section:steps, section:detailed_solution, section:validation, section:acceptance_criteria, section:notes
- RV4612
- 2026-11-25T02:07:52.888903+00:00 (updated-by): Updated: section:notes
- RV4641
- 2026-11-25T02:44:58.836045+00:00 (updated-by): Updated: section:notes

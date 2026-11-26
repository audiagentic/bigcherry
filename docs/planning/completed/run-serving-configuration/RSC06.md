---
id: RSC06
order: 12
plan: run-serving-configuration
state: completed
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

Reuse current HelperSettings and one BC YAML decoder for authored intent, assigned role and only indispensable derived bootstrap facts (budgets, shares, topology and shm namespace before worker creates segment). Same raw YAML and root. Use existing per-attempt session/segment identity shared by all roles, shape/topology and independent owner-map/generation checks, no new digest or launch document. Install existing typed engine/plugin settings before registration, migrate manifest presence marker and actual callers together. Preserve all helper settings, measuring/lab gating, formats, live/calibration/forced schedules, heat and drain. Remove manifest loader, no R3_* fallback. See design §4.2.

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

2026-11-25 serving/configuration closure sweep: docs/design/radiance-serve-config-review-20261125.md. helper_profile.toml embedded via manifest_writer.cpp and archs.toml capability/reserve data must be classified before removing writer. Move authored defaults into one typed BC YAML policy, retain facts/calibration in existing owner. Helpers same admitted YAML plus only assigned derived role/transport facts. No per-helper generated config or manifest inheritance; actual format/row/owner map/readiness and copy drain before stop/restore. Preserve existing measured handoff literals S17 proof and RHP16/20 independent policy acceptance.

docs/design/radiance-native-yaml-launch.md §4.2 (code seam review, 2026-11-25): v2/helper_settings.cpp::helper_settings currently projects all WorkerHelper/WorkerHandoff settings and model/home, role-specific plugin/format, budget, all helper budgets/shares, balance speeds, lanes, live/calibration/forced lab and diagnostics. Retain existing HelperSettings and typed kernel ABI; replace its producer with same BC YAML decode plus actual selected role/derived facts. Critical boot order: r3_moe_worker loads these BEFORE HIP and before v26::init/map, so the helper cannot obtain initial budget/topology/shm name solely from a segment it must create; pass only indispensable derived bootstrap facts using existing owner child handoff, not full manifest or JSON copy of YAML. v26 header manifest_digest currently comes from content_sha256; use existing unique owner-selected segment/session identity shared before init (nonzero, same for roles of each measuring/final attempt), keep shape/topology/independent owner-map/generation checks; no new digest protocol. bc_engine_settings.h manifest flag is a presence marker and DSO getter gates on it; change together with rad0002/rad0012 and actual consumers, preserve absent standalone defaults and install before native Fleet/Engine plugin registration; never turn managed missing settings into silent standalone defaults. Retain independent owner-map agreement, lab/heat/phase trace, layout and actual drains.

2026-11-26 closeout: helpers read typed HelperSettings from one sealed descriptor (`--settings-fd FD --role helperN`), never a manifest (#1618); the shared-memory header identity is the owner-selected segment identity, not a manifest hash (#1617); the helper registration test is deterministic (#1621). The helper cohort (flashnext-4gpu: three helpers + engine) started, served, stopped and restored through the native path in the shared acceptance (docs/evidence/2026-11-26-rsc02-b-acceptance) and in the RSC14 and Gate E runs. `bc-serve helper-exec` and the lab scripts that use it still read the owner's recorded manifest (lab tooling; kept, RSC11 note).

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
- RV4658
- 2026-11-25T03:26:05.495951+00:00 (updated-by): Updated: section:notes
- RV4674
- 2026-11-25T05:41:21.769599+00:00 (updated-by): Updated: section:notes
- 2026-11-25T05:43:45.066176+00:00 (updated-by): Updated: section:steps
- 2026-11-26T05:42:56.854858+00:00 (updated-by): Updated: section:notes
- 2026-11-26T08:35:57.019357+00:00 (state-transition): State: in_progress → completed

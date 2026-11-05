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

# Helpers load their manifest section with a strict C++ loader; R3_* getenv removed

## Description

The moe-remote helpers and our engine-side components read about 80 R3_* environment variables through scattered getenv calls. A misspelt variable is silently ignored, and defaults are inline literals. Owner decision 2026-11-02 (recorded on RSC01): each process loads its section of the resolved manifest through a strict loader.

## Steps

1. A typed settings struct per component (worker, GEMV, small-lane, combine, heat, n-gram and the engine-side hook), generated from the same schema RSC01 defines, or checked against it.
2. `--manifest <path> --role helperN`: parse with nlohmann::json, reject unknown keys and missing required keys, and check types and ranges. Fail before any GPU work.
3. Replace every getenv("R3_...") read with the typed struct, passed explicitly; no global getenv cache.
4. Delete the environment-variable path completely, with no fallback. Lab experiment knobs become manifest fields in the experiment namespace.
5. Print the manifest hash and the role at start-up. Activation markers (e.g. W4H2) stay.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Card-free C++ tests: an unknown key fails, a missing key fails, an out-of-range value fails, and a valid manifest round-trips. On hardware, a helper started from a manifest produces byte-identical text and step time within noise against the current env-launched build. `grep getenv\(\"R3_` over the components is empty.

## Effort & Risk



## Standards



## Acceptance Criteria

No R3_* environment variable is read anywhere. Helpers start only from a manifest, and an invalid manifest fails closed with the offending key named.

## Notes

Depends on the RSC01 schema. The engine side is RSC07; the supervisor is RSC02.

2026-11-05: redone on the v1 contract (supersedes PR #527). Helpers read their whole configuration through the shared strict reader: r3_moe_worker --manifest FILE --role helperN (v1::helper_view projected by serving/v1/helper_settings.h); no helper R3_* getenv and no allowlist (tools/tests/lab/test_helper_reads_no_environment.py); the one environment write is GPU_MAX_HW_QUEUES. The 22 settings v1 did not model became manifest fields in spec section 11 amendment 10 (live_policy production-allowed under handoff.placement; forced schedule and epoch_cycle_s under experiment.placement; plan.state with the unmeasured measuring manifest, one measuring flag per binary). rad0010 helper kernel config renumbered rad0012 (weak symbol, install-once). The launcher writes the manifests (unmeasured for the measuring start, measured for the real start). Open: hardware identity and step-time validation against main (needs the gpu lease); campaign arms that start helpers need a manifest per arm (BC_SERVING_MANIFEST in engine_adapters/radiance.py) until the S4c campaign slice.

## Change Log

- 2026-11-02T03:52:00.963006+00:00 (created-by): Created by agent

## Ledger-events


- chg_20261102_035356_serving-start-up-plan-one-str_4985
- 2026-11-02T03:54:11.557458+00:00 (updated-by): Updated: section:ledger-events
- 2026-11-05T06:03:06.682889+00:00 (updated-by): Updated: section:notes
- 2026-11-05T06:03:10.546906+00:00 (state-transition): State: pending → in_progress
- chg_20261105_060310_the-expert-helper-processes-no_8907
- 2026-11-05T06:03:14.921955+00:00 (updated-by): Updated: section:ledger-events

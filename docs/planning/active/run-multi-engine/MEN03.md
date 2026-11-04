---
id: MEN03
order: 3
plan: run-multi-engine
state: pending
created-at: '2026-10-08T20:36:14.144983+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: M
---

# Engine adapter for the lab: start, health, stop, metrics behind one interface

## Description

The lab scripts and campaign producers launch llama-server directly and read its logs and flags (28 modules under tools/bigcherry reference it; long-ctx-profile.sh builds its command line inline). A second engine needs the same four operations without each script knowing which engine it drives.

## Steps

1. Define the adapter contract: start(config) -> endpoint, wait_healthy, stop (graceful, never kill -9 first), collect(metrics, logs, VRAM). ServerRunner (tools/bigcherry/tuning/server_runner.py) is the starting point for llama-server.
2. Implement two adapters: llama-server (existing behaviour) and container (docker start / health / stop of a pinned image, as run-radiance.sh does today).
3. Normalise what a run reports: prompt tokens, time to first token, decode t/s, drafted and accepted tokens (llama-server timings; vLLM /metrics spec_decode counters).
4. Migrate tools/lab/reference-vllm first, then one Flash-Next A/B script, to prove the contract. No compatibility wrapper: migrated scripts call the adapter only.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

tools/bigcherry/tuning/server_runner.py, tools/lab/reference-vllm/, tools/lab/flash-next/long-ctx-profile.sh, tools/lab/flash-next/flash-prefill-env-ab.sh

## Validation

The same queue row runs the same depths against either engine and writes the same result schema; unit tests for the adapters with a fake process and a fake container.

## Effort & Risk



## Standards



## Acceptance Criteria

One A/B script and the reference lane run through the adapter on dev for both engines with identical output schema.

## Notes

Only worth starting if MEN01 or MEN02 says vLLM stays in use.

2026-10-09 retarget to radiance: the second engine is a native server binary with a llama-server-compatible API, so the second adapter is a process adapter with different flags and log patterns, not a container adapter. The container adapter is only needed if the vLLM image is kept as a reference lane; decide after MEN02 step 3 shows how much of the llama-server request path already works.



BACKLOG TRIAGE 2026-11-04 (SUPERSEDED; revalidated against main 6eb31a708, verdict CONFIRM): tools/bigcherry/engine_adapters/ has the radiance adapter (start/health/stop/metrics) and radiance_bump.py; regression tests belong to MEN17. Evidence: tools/bigcherry/engine_adapters/ has the radiance adapter (start/health/stop/metrics) and radiance_bump.py; regression tests belong to MEN17. Still referenced by: BCOP83, MEN05, MEN06, MEN04, MEN09. Original content kept below for history.



REOPENED 2026-11-04: an independent revalidation (against main 6eb31a708) rejected the closure recorded above, which had been based on a lighter check. It stays open. Why: tools/bigcherry/engine_adapters/base.py:180-308 and engine_adapters/radiance.py:38 provide substantial process/sidecar adapters, and campaign/benchmark.py:529-659 uses them. MEN03.md:39-53 additionally requires a reference-vllm lane and one Flash-Next A/B script both migrated, with identical cross-engine outputs; that specific dev fire-test and script migration were not evidenced.

## Change Log

- 2026-10-08T20:36:14.144983+00:00 (created-by): Created by agent
- 2026-10-08T20:47:55.960817+00:00 (updated-by): Updated: section:notes

## Ledger-events

- chg_20261009_041445_each-engine-now-declares-how-i_5202
- 2026-10-09T04:14:52.459208+00:00 (updated-by): Updated: section:ledger-events
- 2026-11-04T03:45:50.712872+00:00 (updated-by): Updated: section:notes
- 2026-11-04T03:45:55.047676+00:00 (state-transition): State: pending → superseded
- 2026-11-04T04:43:28.133968+00:00 (updated-by): Updated: section:notes
- 2026-11-04T04:43:37.184501+00:00 (state-transition): State: superseded → pending

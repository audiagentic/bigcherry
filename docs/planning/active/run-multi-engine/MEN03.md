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

One A/B script and the reference lane run through the adapter on Brutus for both engines with identical output schema.

## Notes

Only worth starting if MEN01 or MEN02 says vLLM stays in use.

2026-10-09 retarget to radiance: the second engine is a native server binary with a llama-server-compatible API, so the second adapter is a process adapter with different flags and log patterns, not a container adapter. The container adapter is only needed if the vLLM image is kept as a reference lane; decide after MEN02 step 3 shows how much of the llama-server request path already works.

## Change Log

- 2026-10-08T20:36:14.144983+00:00 (created-by): Created by agent
- 2026-10-08T20:47:55.960817+00:00 (updated-by): Updated: section:notes

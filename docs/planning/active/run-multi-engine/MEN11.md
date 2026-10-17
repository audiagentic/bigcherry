---
id: MEN11
order: 0
plan: run-multi-engine
state: pending
created-at: '2026-10-14T22:16:15.217144+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Radiance measurements and patch status on the campaign and evidence tooling

## Description

Radiance speed and quality runs are ad-hoc tools/lab/radiance scripts: no ab-benchmark, no evidence records, no contracts, no promotion. The helper work (libr3moe, R3_MOE_HELPER_* settings) lives in components with settings in component.toml, outside the patch lifecycle entirely. Bring radiance under the same management as llama.cpp: order-balanced A/B through the maintained harness, evidence attached to the patch or component setting it proves, and promotion/default-on decisions made the same way (lightweight tier per owner policy).

## Steps

1. Extend ab-benchmark (or engine-bench) to drive a radiance layout: engine + helper processes, graceful teardown, activation evidence (helper call counts, stage markers).
2. Quality producer: KLD against a named reference with the full metric set (RAD08).
3. Give component settings a lifecycle: each experimental setting in component.toml carries state, evidence pointer and SUMMARY like a patch package, or decide components are patches' peers in the registry.
4. Migrate the current helper evidence (rows/GEMV/heat A/Bs) into evidence records.
5. Retire lab scripts the harness replaces; keep the rest documented.

## Detailed Solution & Technical Design

Survey 2026-10-17 (main 08441aa4): ab-benchmark's server comparison (campaign/benchmark.py run_server_comparison_capture) already accepts engine = radiance with execution_evidence 'observe' and stock mode; experiment/engine_quality.py has parse_radiance_kld, quality_gate and setting_activation (component.toml activation regex). Missing for the helper layout: (1) per-arm environment (arms differ by a component setting, e.g. R3_MOE_HELPER_ROWS); (2) sidecar processes - the r3_moe_helper processes must start before the server, be health-checked, and stop gracefully after it, per arm and per round; (3) the llama.cpp-specific branches (dispatch modes, build inspection, execution attestation) live in benchmark.py as `engine == LLAMACPP` checks.

Plan:
a. Engine adapters: tools/bigcherry/engine_adapters/{llamacpp,radiance}.py implementing one small interface - dispatch_modes(), inspect_build(build_dir), parses_execution_attestation, sidecars(arm, env, run_dir) -> list of SidecarSpec(argv, env, ready_line, stop_signal, log). benchmark.py calls the adapter; the LLAMACPP branches and its engine-literal allowlist entry go.
b. Server comparison config: per-arm `environment` overlay (merged over the shared environment; device selectors stay shared) and per-arm `settings` (component settings) the adapter turns into sidecars/env.
c. Radiance adapter: when R3_MOE_HELPER is set, launch HELPER_DEVS helpers (r3_moe_helper from the build's radiance_home/helpers, R3_MOE_HELPER_INDEX/COUNT, MIB) as in tools/lab/radiance/fn-r9700-engine.sh; ready on the helper's 'runtime map owns' line; SIGTERM + wait on stop; shm segment removed before start.
d. Activation: component setting_activation on the server log for each setting under test; quality: radiance KLD via the existing parser when a reference is configured.
e. Fire test on Brutus: R3_MOE_HELPER_ROWS 0 vs 1 (known result: ROWS=1 is ~38% slower prefill) as the harness acceptance run, then retire fn-r9700-engine.sh / fn-bench-once.py / fn-helper-gate.sh in favour of the config.
f. Component settings promotion path: settings in component.toml get state + evidence pointer like patches (separate PR).

## Code Samples & Guidance



## Files



## Validation

One real A/B (e.g. R3_MOE_HELPER_ROWS 0/1) run through the harness end to end on Brutus, producing an evidence record that patch-verify-evidence (or the component equivalent) reads; offline tests for the radiance adapter.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Owner policy: promotion stays lightweight - reuse mechanics tests, activation evidence and a balanced A/B with identity; no heavyweight per-patch campaigns.

Correction 2026-10-15: main already has an engine-neutral activation and KL quality evidence path with a radiance adapter (#223, RAD01 PR 7) and build identity binding components (#225). Start from those: the gap is that the helper-layout runs (two processes, R3_MOE_HELPER_*) were done with lab scripts instead of through them, and component settings have no promotion path.

2026-10-17 acceptance (steps a-e): ab-benchmark --server-config tools/lab/radiance/helper-ab.json --pairs 4 on Brutus at c95d49e2 (run men11-fire-10171108): 8/8 cells clean, both helpers started ready and stopped exit 0 every cell, arm_variables record the single varied setting. R3_MOE_HELPER_ROWS=1 vs 0: pp4096 472-485 vs 1989-1991 tok/s, pp1024 459-480 vs 2035-2037, tg512 76.3-77.2 vs 128.3-128.6, tg2048 76.5-77.2 vs 141.2-143.4 - complete separation, same direction as the lab measurement (-38% prefill on the 7.3K prompt). Bugs found and fixed: helper device selection (#248); ROWS had no activation marker (#250). Remaining: retire fn-r9700-engine.sh / fn-bench-once.py / fn-helper-gate.sh in favour of the config (step e cleanup); KLD through the harness; component-setting promotion path (step f).

## Change Log

- 2026-10-14T22:16:15.217144+00:00 (created-by): Created by agent
- 2026-10-14T22:21:10.146038+00:00 (updated-by): Updated: section:notes
- 2026-10-16T17:51:41.625681+00:00 (updated-by): Updated: section:detailed_solution
- 2026-10-17T00:47:06.344774+00:00 (updated-by): Updated: section:notes

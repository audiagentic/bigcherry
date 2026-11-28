---
id: MEN15
order: 1
plan: run-multi-engine
state: pending
created-at: '2026-10-17T02:58:13.549094+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# RAM-only engine smoke tests: tiny models on CPU for llama.cpp and radiance, in the bump and change gates

## Description

Owner, 2026-10-17: quick, RAM-only smoke tests for both engines so bumps and other changes catch basic engine breakage without needing the GPUs (which may be occupied). Each engine builds a CPU-capable variant (or runs its normal build with no device visible), loads a tiny model, serves a few requests, and checks output sanity and determinism - seconds to a couple of minutes, no GPU, small RAM.

## Steps

Reuse existing generic runner, tiny-model SHA and health/completion/RSS/stop checks. For Radiance use actual native CPU-only YAML fixture (libavx/libref, one rank, small context and one sequence), existing harness fills only ephemeral model path/port in test fixture; no production flags-to-YAML shim. Actual argv --config and candidate binary/home, no GPU visible or inherited config env, verify logs/device and deterministic nonempty output/clean shutdown. Separate existing card-free managed owner tests exercise same sealed YAML, observe-only power and admission. Preserve llama.cpp CPU smoke unchanged; wire to MEN16 existing lane and docs.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Both engines pass the CPU smoke on dev with all GPUs busy (or devices hidden) in under ~2 minutes each; a deliberately broken patch (e.g. a crash on load) fails it; offline unit tests for the runner and adapters.

## Effort & Risk



## Standards



## Acceptance Criteria

Both engines retain pinned tiny-model integrity and actual no-GPU smoke, Radiance candidate exec uses native --config (no manifest/legacy flags), fails wrong build/hash/missing or invalid YAML/GPU opened/load failure, passes real deterministic outputs and clean stop. Fake runner and actual installed candidate receipt required, not fake owner only. No duplicate smoke harness.

## Notes

Sequence: after MEN11 leftovers, before the bumps (MEN10 step f uses it as the first bump smoke).

2026-11-25 cross-programme bump review: docs/design/radiance-bump-integration.md is the shared implementation-order and compatibility specification; MEN24 owns sequence and timing, domain implementation remains here. No duplicate checker, bump driver, manifest launch or legacy fallback. radiance.py cpu_smoke currently uses engine_args flags plus engine.toml smoke.cpu.args, base.build_cpu_smoke and campaign.cpu_smoke use shared ServerRunner. Actual argv may already translate elsewhere; audit rather than claim failure. Native cutover removes any Radiance legacy arg-to-YAML projection, preserves llama.cpp flags.

2026-11-25 second integration review: docs/design/radiance-integration-review-20261125.md is the exact-source review and controls conflicting earlier wording. Shipping pin remains b6b8751 (2.2.0), upstream observed main 1c104de4 is a development candidate, 119 commits and 327 changed files; no automatic repin or backport. No new compatibility layer, launch manifest, bump driver or identity machinery. Native --check is parser-only; actual native CPU --config launch with pinned tiny artifact and candidate library must serve deterministic output. Separate fake owner test, no GPU visible, no flag-to-YAML legacy adapter. Candidate new native config keys tested only when supported; no duplicate parser.

2026-11-27 real 2.3.0 base-runtime CPU smoke: checked-in native YAML fixture with only port/model substitutions; legacy flags removed. Old tiny.rad refused retired block FP8, same source converted to independently named pinned IQ8R tiny fixture. Three actual passes, final six nonempty completions in three identical pairs, clean exit, no GPU; 32 offline tests pass. Evidence docs/evidence/2026-11-27-radiance-230-bump/FIRST-RUNTIME-TESTS.md. Complete candidate bundle repeat and llama.cpp actual smoke/remaining acceptance still required.

## Change Log

- 2026-10-17T02:58:13.549094+00:00 (created-by): Created by agent

## Reviews

- RV4616
- 2026-11-25T02:09:58.080009+00:00 (updated-by): Updated: section:notes, section:steps, section:acceptance_criteria
- RV4650
- 2026-11-25T02:49:39.074949+00:00 (updated-by): Updated: section:notes
- 2026-11-27T08:00:41.998472+00:00 (updated-by): Updated: section:notes

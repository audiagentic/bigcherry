---
id: PRBE105
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-12T20:32:09.189511+00:00'
breadth: ''
skill: ''
created-by: agent
---

# run_rd13_ppl_check() build directories collide across architectures

## Description

run_rd13_ppl_check() (tools/bigcherry/patch/validation_campaign.py) names its build directories 'rd13-ppl-subject'/'rd13-ppl-control' unconditionally, not namespaced by target architecture. When the same build_root is reused across a multi-arch validation run (e.g. gfx1100 then gfx1201 then gfx1030, as the new standardized multi-arch test criteria requires), the second/third architecture's configure step fails with a real CMake error: 'The source ... does not match the source ... used to generate cache' -- the stale cache from the first architecture's build is still present. Found and worked around 2026-09-13 during RD13's real multi-arch validation run by passing a distinct build_root per architecture from the caller; the function itself was not changed.

## Steps

1. In run_rd13_ppl_check(), include amdgpu_targets in the build subdirectory names (e.g. f'rd13-ppl-subject-{amdgpu_targets}') so repeated calls with different architectures against the same build_root do not collide.
2. Check whether the same pattern exists in other per-patch producers that share this naming convention (run_rd08_contract_correctness, run_rd26_ppl_check, run_rd43_ppl_check, run_rd58_state_restore_evidence) -- likely the same bug, just not yet hit for those because they haven't been run multi-arch against a shared build_root yet.
3. Add a regression test exercising two calls with different amdgpu_targets against the same build_root.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Low severity (workaround is trivial: pass distinct build_root per arch), but will recur for every future multi-arch validation run using these producers unless fixed at the source.

## Change Log

- 2026-09-12T20:32:09.189511+00:00 (created-by): Created by agent

## Ledger-events

- chg_20260912_203230_patch-1206-rd13-now-has-real_7484
- 2026-09-12T20:32:30.630645+00:00 (updated-by): Updated: section:ledger-events

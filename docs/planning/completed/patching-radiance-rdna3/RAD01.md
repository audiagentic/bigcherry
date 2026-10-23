---
id: RAD01
order: 0
plan: patching-radiance-rdna3
state: completed
created-at: '2026-10-12T21:08:01.597076+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Bring Radiance engine patches and libr3 switches under the patch-set, profile and campaign system

## Description

Owner question 2026-10-13: 'Are we not using the campaigns and profiles for patch sets like we do for llamacpp'. Today the Radiance work on branch feat/rad11-gfx1100-bootstrap runs outside the BigCherry system: engine patches are numbered diffs in engines/radiance/engine-patches/ applied all-or-nothing by tools/lab/radiance/engine-patched-build.sh; libr3 native forms are .rw rewrite files gated by R3_* environment variables; configurations are environment variables and MORE= strings in ad-hoc lab scripts; evidence lives only in engines/radiance/kernels/README.md. There are no patch packages (patch.toml, tests, SUMMARY), no recipes selecting a patch set, no runtime profiles holding the switches, no campaign/ABBA harness or evidence records, and none of it is on main.

## Steps

1. Inventory: engine patches 0001-0003; libr3 forms (attn native default on, gdn native shelved, moe/hc native, wire4, ar-pipe, ar-timing, ngram helper); the Flash-Next configuration of record.
2. Model each engine patch and each libr3 form as a patch package (id, state, requires/conflicts, mechanics tests, SUMMARY with the measured evidence already in the README).
3. A Radiance recipe that selects the patch set and a runtime profile per model (Flash-Next two-XTX, 27B two-XTX) holding only the flags that differ.
4. Wire the serve/KL lab scripts into the campaign harness (balanced arms, activation evidence, evidence records) so promotion uses the lightweight tier as for llama.cpp.
5. Merge to main (shelved forms kept as evaluated, default off).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Patch-lint and mechanics tests pass for every Radiance package; the recipe builds the patched engine and libr3 from main; a profile run reproduces the configuration-of-record numbers within drift; KL gate runs from the campaign harness.

## Effort & Risk



## Standards



## Acceptance Criteria

- Radiance engine patches are packages under engines/radiance/patches and pass patch-lint (met: #210, #216).
- Each engine declares its recipes; radiance has sources, patch sets, experiments and runtime profiles (met: #219).
- bigcherry build builds a radiance lane on dev from the pin plus patch sets, including radiance's runtime and the libr3 component, with the runtime bound into build identity (met: #220, #221, #224, #225; verified build b4aeaa56).
- Profiles carry an environment checked against the component's declared settings (met: #222).
- Activation and KL quality evidence are engine-neutral with a radiance adapter (met: #223).
- A radiance A/B runs through ab-benchmark (met: #226; runs/rad-ab2, engine fix 0001 speed-neutral).
- Runbook (met: #227).
- Every PR merged with all checks green and the full suite on dev matching main's pre-existing failures.

## Notes

Related: run-multi-engine (MEN*). Rules that apply: shelved work kept on main; bug fixes inside their patch; universal patches default on; lightweight promotion tier; the R9700 is compute, not storage.

## Change Log

- 2026-10-12T21:08:01.597076+00:00 (created-by): Created by agent

## Ledger-events

- chg_20261013_030412_radiance-is-now-built-patched_5466
- 2026-10-13T03:04:20.242361+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-13T03:04:40.209258+00:00 (updated-by): Updated: section:acceptance_criteria
- 2026-10-13T03:04:47.177533+00:00 (state-transition): State: pending → completed

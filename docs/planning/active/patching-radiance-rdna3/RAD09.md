---
id: RAD09
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-10-14T22:21:18.124989+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Reconcile feat/rad03-gpu-lanes with main: one rad0002 and one hook

## Description

feat/rad03-gpu-lanes (GPU mailbox protocol v11, libr3moe plugin, two helpers, SKIP_OWNED placement, heat ownership, decode GEMV) branched before main's #230 (slice 2a: host-copy hook in libr3, rad0002 publishing through mover.h/ctx.h/residency.h) and #232 (rad prefix). A merge of origin/main conflicts in CMakeLists.txt, component.toml, moe_remote_hook.cpp and the radiance patch tests: the two branches carry different implementations of the same patch and hook. Reconcile into one: the branch's GPU-mailbox design is the measured one (2,760 prefill / 108.5 decode); decide per piece whether main's mover/residency edits are still needed, migrate everything to rad0002 naming, one hook (in libr3moe), and repin the patch-test PIN_BLOBS.

## Steps

1. Diff main's rad0002 patch.py against the branch's 0002 edit by edit.
2. Keep one implementation; fold any needed main-only edit (mover publish changed_shape) into it inside the patch (bug fixes live in the patch).
3. Merge main into the branch (no rebase: the branch has another session's commit; no force push).
4. Offline tests, patch-lint, build on dev, --compare, one balanced A/B against the 2,760/108.5 baseline.
5. Open the PR.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

patch tests and patch-lint pass; rad0002 dry-run applies at the pin; helper layout reproduces 2,755-2,764 prefill and ~108.5 decode with KLD in the noise floor.

## Effort & Risk



## Standards



## Acceptance Criteria

Met 2026-10-15: one rad0002 and one hook (libr3moe) on main via #236 (63cc952f); patch-lint and patch tests pass; radiance-lab build ok on dev; helper vs engine rel_l2 0.0011; prefill 2,760/2,761 tok/s, decode 108.4/108.9, identical completion hash.



## Notes

Blocks merging any of RAD02-RAD06's code to main.

## Change Log

- 2026-10-14T22:21:18.124989+00:00 (created-by): Created by agent

## Ledger-events

- chg_20261014_233548_radiance-builds-that-apply-pat_7437
- 2026-10-14T23:35:56.607672+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-15T00:20:15.841891+00:00 (updated-by): Updated: section:acceptance_criteria
- 2026-10-15T00:20:22.319899+00:00 (state-transition): State: pending → completed

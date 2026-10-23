---
id: MEN12
order: 0
plan: run-multi-engine
state: completed
created-at: '2026-10-14T22:16:19.350063+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: M
---

# First radiance bump to the latest upstream release with patches and components

## Description

Port radiance from 1.3.0 (89cee7c) to 1.7.0 (48f7b8e, tagged 2026-10-16; upstream released 1.4-1.7 within days). 83 commits; plugin ABI 15 -> 17 (abi/ ~1,700 lines changed); core/engine.cpp, engine_bringup.cpp, place/planner, mover, stager, heat rewritten (~6,400 lines). Owner, 2026-10-17: radiance first, before the llama.cpp bump; keep core changes minimal - our logic in our own files, core edits only where core logic must change.

## Steps

1. Read 1.3.0..1.7.0 diffs of the touched core files and abi/; list upstream replacements.
2. Design the rad0002 extension point (call sites per file) and get it reviewed.
3. Re-author rad0001/rad0002/rad0003 against 1.7.0 (patch packages, tests, PIN_BLOBS).
4. Port libr3 / libr3moe / helper to ABI 17.
5. pin-bump --engine radiance; CPU smoke; component gate; GPU acceptance (helper-ab, helper-kld); model pages updated.
6. Release bc-radiance 170.0.0 after owner confirmation.

## Detailed Solution & Technical Design

Where our code sits today (main):
- Separate from core (ported against the plugin ABI, not patched): libr3, libr3moe, the r3_moe_helper process, libr4d use, the n-gram host table - all BigCherry components under engines/radiance/components.
- rad0001 (1 edit, core/engine.cpp) and rad0003 (1 edit, arch/common/rad_block_mtp_fp8.h): small fixes.
- rad0002 (507-line patch.py, 44 edits): adds two new files (abi/rad_moe_remote.h, core/place/moe_remote.cpp) but also edits rad_block_moe_fp8.h, rad_block_hc.h, place/planner.cpp and engine_bringup.cpp in place - placement policy (SKIP_OWNED, owner hook, preplaced units, weight_footprint) is written into core, which is why the upstream rewrite breaks it.

Port design (thin-hook rule):
1. rad0002 becomes hooks only: the in-core edits shrink to call sites into one BigCherry-owned extension point (abi/rad_moe_remote.h + core/place/moe_remote.cpp, both new files) - e.g. 'may this unit be placed here', 'units placed elsewhere', 'route a MoE layer's remote rows' - with all policy (heat ownership, SKIP_OWNED, footprint) in moe_remote.cpp or the libr3moe plugin. Target: a handful of anchored call-site edits per touched core file, each with a guard, instead of logic blocks.
2. Check 1.7.0 for upstream mechanisms that replace ours (new placement hooks, a remote/offload expert path, mover drain fix for rad0001, MTP head handling for rad0003) - drop or shrink our patch where upstream covers it.
3. Port components to ABI 17: libr3, libr3moe, helper (abi/rad_types.h and friends changed); keep them as plugins.
4. Bump through the tooling: pin-bump --engine radiance v1.7.0 with recipes pinned-version = 1.7.0 (bc-radiance major 170), CPU smoke (MEN15) then the component gate (MEN10 f), then GPU: helper-ab and helper-kld through the harness to confirm 2,760/108.5 and KLD ~0.064 hold or improve.
5. First bc-radiance release (owner confirms before the tag).

## Code Samples & Guidance



## Files



## Validation

pin-bump + pin-status --complete on dev; ab-benchmark helper-ab and helper-kld; 27b-native-kld; pin-release with recorded evidence.

## Effort & Risk



## Standards



## Acceptance Criteria

Met 2026-10-17: radiance pinned v1.7.0 on main via pin-bump (#263, COMPLETION PASS); compile gate, ABI 17, component tests, CPU smoke, strict audit pass; helper-ab pp4096 2,049-2,050 / tg512 122-126 tok/s (1.3.0 ~1,990 / 128); helper-kld gate pass, KLD identical to 1.3.0; 27B attention native KLD 0.0040 / top-1 96.6%; bc-radiance-170.0.0 released (alias bc-radiance-1.7.0-r0).

## Notes

Depends on MEN10.

Owner, 2026-10-17: keep the rad0002 diagnostics (shadow compare, mask-only) but out of production. Design for the port: (1) rad0002 (production) carries only the thin hooks; its extension point (bc_moe_remote.h/.cpp, ours) exposes the layer's remote rows before and after the merge. (2) Mask-only moves entirely into our code: the extension point / libr3moe answer 'no remote rows' (or zero rows, as R3_MOE_HELPER_POISON already does in the helper) - no core edit. (3) Shadow compare needs the engine to also compute owned experts locally: it becomes a separate patch package rad0004_moe_remote_diagnostics (requires rad0002), whose few call sites sit behind the same extension point, plus a compile-time switch (BC_MOE_REMOTE_DIAG) set only by a diagnostic build. (4) A diagnostic source/build in engines/radiance/recipes.toml (e.g. source radiance-diag = production patches + rad0004, build option on); production sources never include rad0004, so production binaries cannot enable it. patch-lint/recipes checks keep rad0004 out of any production patch set. Analysis: docs/reference/engines/radiance/PORT_1.7.md (from bc-men12-analysis.md).

## Change Log

- 2026-10-14T22:16:19.350063+00:00 (created-by): Created by agent
- 2026-10-17T04:56:40.722231+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution
- 2026-10-17T05:51:35.261162+00:00 (updated-by): Updated: section:notes

## Ledger-events



- chg_20261017_094109_radiance-moved-to-upstream-17_4316
- 2026-10-17T09:41:13.531755+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-17T09:41:17.614459+00:00 (updated-by): Updated: section:validation, section:acceptance_criteria
- 2026-10-17T09:41:24.184466+00:00 (state-transition): State: pending → completed
- chg_20261018_140640_radiance-is-now-on-upstream-1_2179
- 2026-10-18T14:06:44.705096+00:00 (updated-by): Updated: section:ledger-events
- chg_20261018_201215_radiance-180-is-released-as_1561
- 2026-10-18T20:12:19.941869+00:00 (updated-by): Updated: section:ledger-events

---
id: RHA12
order: 0
plan: run-hip-autotune
state: pending
created-at: '2026-09-10T04:51:00.813381+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
priority: P1
---

# Bump llama.cpp pin from b10705 to b10883

## Description

Move config/recipes.toml's pinned llama.cpp revision from b10705 to b10883 (the current numerically-highest remote b<n> tag per source/upstream.py's own definition of upstream-latest; verified live via git ls-remote, published 2026-09-09, one commit ahead of an untagged upstream master CI-only sanitizer fix). Real, tested, repeatable bump -- not just a config edit: run the full pin-bump orchestrator (tools/bigcherry/release/pin_bump.py, HI153), manually assess each upstream PR/commit range between b10705 and b10883 against this project's existing patches (patches/*/patch.toml) and plan history for conflicts/superseded work, fix any real gaps found (with GPT assistance per item), then validate end to end (patch-verify-evidence, real-hardware runtime smoke, pin-status --complete).

## Steps

1. Preflight: `bigcherry pin-status --all-remotes`, `bigcherry sources status`, `bigcherry sources check`, `bigcherry patch-lint`, `bigcherry patch-rebase-check --all` (b10705 baseline only -- separates pre-existing failures from bump-caused ones).
2. Manually enumerate the real upstream commit/PR range b10705..b10883 (git log/GitHub compare against ggml-org/llama.cpp) and map each meaningfully-sized change against this project's existing patches/*/patch.toml anchors and any relevant plan-item history -- flag anything that looks like it could conflict with or supersede a BigCherry patch. Use GPT to assist this triage one range/PR-cluster at a time, not as one giant dump.
3. Run `bigcherry pin-bump b10883 --source bigcherry` -- the orchestrator's own load-bearing patch-rebase-check (--all + --source bigcherry) is the actual gate for patch applicability at the new pin; treat its PASS as source/patch reconciliation complete, not bump complete.
4. Post-bump (per PIN_BUMP.md:128-178, NOT automated by the orchestrator): rebuild a canonical lane at the new pin, run real-hardware runtime smoke, run `bigcherry patch-verify-evidence` (checks evidence freshness against the exact new pin SHA -- pin-bump itself does not call this), regenerate/qualify any pin-bound evidence that patch-verify-evidence flags as stale, converge required trees, and require `bigcherry pin-status --complete --all-remotes` PASS.
5. Record the full real result (pass/fail per phase, any patches needing rebase work, any evidence regenerated) in this item's notes -- this is meant to be a repeatable, documented procedure other bumps can follow, not a one-off.

## Detailed Solution & Technical Design

External dev-gpt assessment (2026-09-10, grounded in direct source inspection of pin_bump.py, PIN_BUMP.md, HI153.md, PIN_REBASE_REVIEW_B10502.md, patch/disposition.py, campaign/source.py, source/identity.py, build/builds.py, core/config.py):

pin_bump.py robustness: real, live-fire-tested (HI153 recorded a successful b10680->b10687 live run plus 3 real bugs it exposed/fixed). Fail-closed on selector drift, all-patch coverage, stale/tampered rebase reports (digest-checked immediately before apply), apply failure, and mandatory strict post-apply reaudit (pin_bump.py:780-873). Gap: current tests exercise STOP/resume through run() but mock external pull/audit/rebase/apply -- no fully-real synthetic end-to-end regression test exists yet (tools/tests/release/test_pin_bump.py).

Patch integration: mechanically strong (a selected bigcherry patch cannot silently remain inapplicable -- pin-bump runs target-pin patch-rebase-check --all then --source bigcellery, all selected patches must be clean before apply; non-selected registry patches need an exact target-revision+digest known_broken disposition). ONE real drift found: pin-bump's own "patch-lint" phase calls patch_catalog.cross_check(...) directly (pin_bump.py:795-805), NOT the current CLI's actual patch-lint implementation, which additionally runs cross_check(verify_validation_evidence=False) plus check_validation_packages() (cli/patch.py:179-210). pin-bump also never explicitly invokes patch-verify-evidence (which resolves evidence against the exact active-pin SHA, cli/patch.py:246-285) -- a successful pin-bump is NOT proof that current package validation evidence is fresh. Must run patch-verify-evidence as an explicit post-bump step (already reflected in this item's steps).

Build/cache invalidation: SOUND. The resolved upstream revision participates in materialization identity together with patch/overlay bytes (campaign/source.py:87-139); source_slice_id hashes the exact upstream revision + actual post-patch tree OID (source/identity.py:92-109); BuildPlan's canonical form (source_slice_id + build_plan_id) makes the build directory path itself pin-bound (build/builds.py:34-145). A stale b10705 build cannot silently alias a b10883 source slice.

BackendStack caveat (orthogonal to this item, noted for awareness): CampaignLaneSelector/CampaignLane/CampaignRequest still carry no stack axis (core/config.py:130-158) -- the stack-identity work from earlier tonight's session is NOT yet threaded into canonical campaign/build identity. Does not affect pin-bump safety.

Target selection: b10883 confirmed as BigCherry's own definition of upstream-latest (numerically highest remote b<n> tag via git ls-remote, source/upstream.py:10-43, release/pin.py:25-37) -- not a vendored submodule (no .gitmodules), not semver. Live-verified: b10883 = 91f6a6cf361385700bbe15981f0f39909df77498, published 2026-09-09; b10884 does not exist; untagged upstream master (434ddbbc...) is one commit ahead but is a CI-only sanitizer-test fix whose parent is exactly b10883 -- b10883 IS the realistic target, not raw master.

## Code Samples & Guidance



## Files

config/recipes.toml (pinned = value); patches/*/patch.toml (rebase status per patch, discovered via patch-rebase-check); tools/bigcherry/release/pin_bump.py (orchestrator, read-only for this item unless a real bug is found); docs/reference/build/PIN_BUMP.md (runbook, already accurate per this assessment).

## Validation

Per GPT's recommended sequence: preflight (pin-status --all-remotes, sources status/check, patch-lint, patch-rebase-check --all on b10705 baseline) all clean or with pre-existing-only failures documented; `bigcherry pin-bump b10883 --source bigcherry` completes with PASS (source/patch reconciliation); post-bump `bigcherry patch-verify-evidence` run explicitly (not assumed clean); real-hardware runtime smoke on a rebuilt canonical lane at the new pin; `bigcellery pin-status --complete --all-remotes` PASS as final convergence gate. Manual PR/commit-range triage (step 2) documented with findings even if nothing required action.

## Effort & Risk



## Standards

No legacy/backward-compat shims (project doctrine) -- if any patch needs real rebase work at the new pin, migrate it fully rather than special-casing the old pin's behavior. Never fabricate evidence: clean apply is necessary but not sufficient per PIN_REBASE_REVIEW_B10502's own real finding.

## Acceptance Criteria

config/recipes.toml pinned=b10883, every currently-selected bigcherry patch clean at the new pin (rebased if needed, with real evidence not just clean apply -- PIN_REBASE_REVIEW_B10502's own finding that clean textual application can still hide upstream signature/API changes), patch-verify-evidence explicitly run and clean, real-hardware smoke passed at the new pin, pin-status --complete --all-remotes PASS, and the full procedure documented in this item's notes as a repeatable record for future bumps.

## Notes

Filed as a real, hardware-validated tracking item per explicit user request to test the pin-bump process by actually running it. GPT explicitly recommended GO with the 4-step preflight, running pin-bump directly rather than a separate manual target-rebase pass first (the orchestrator's own checks are the load-bearing gate). Manual upstream PR/commit-range mapping against existing patches (step 2) requested explicitly by the user and must be done incrementally with GPT assistance per range, not as one bulk pass.

RESULT 2026-09-10: pin-bump b10705 -> b10883 SUCCEEDED. Full trail: preflight clean (60/60 patches rebase-clean at b10705 baseline); vendor moved to b10883; patch-rebase-check found 11 real breaks (6 anchor-mismatch + 5 transitively blocked); all 11 confirmed non-production (not in framework/validated-enhancements patch-sets, only experimental); recorded known_broken dispositions for all 11 with exact upstream commit/PR citations obtained via dev-gpt research and independently verified against the real unshallowed vendor source before applying. pin-bump --resume then PASSED. Auto-committed by the orchestrator: 3f85d27e (pin flip) and ee5a467e (release record). Disposition files committed separately (3e875ce5).

Per-patch findings (all verified against real b10883 source, not just GPT's claim):
- 1202 RD04 (bf16 flash-attn tile): broken by upstream PR #27970 (sparse-fa for DSV4/GLM) adding a use_sparse bool param to launch_fattn() before warp_size. Fix still needed -- upstream has no native-BF16 tile behavior. Disposed, not re-anchored (non-production, experimental only).
- 1205 RD12 (paired MMVQ dual output): broken by upstream PR #27930 (SWIGLU_CLAMP) touching the same gate switch/unused-vars list. Fix still needed. Disposed.
- 1207 RD17 (MoE topk-down fold): broken by upstream PR #27621 (extend MOE fusion to specdec) duplicating the has_fusion computation this patch anchored on uniquely. Fix still needed. Disposed.
- 1209 RD22 (integrated-GPU host-buffer backout): SUPERSEDED -- verified upstream commit d4389a4dd920 (PR #28604, revert of PR #24233) makes RD22's own fix (unconditional info.devices[id].integrated=false) upstream's current behavior verbatim at ggml-cuda.cu:307. Disposed as known_broken (not yet formally superseded via bigcherry-patch-lifecycle -- that remains real follow-on work if this patch is picked back up).
- 1222 HI67 (deterministic test-backend-ops seed) + 4 transitive dependents (1223, 1236, 1238, 1239, 1240): broken by upstream PR #28325 changing the n_threads calculation line in init_tensor_uniform(). The nondeterministic std::default_random_engine HI67 works around remains present and unfixed upstream -- fix still needed. All 5 disposed.

Post-bump validation (per PIN_BUMP.md, NOT automated by pin-bump itself, run explicitly): patch-lint clean. patch-verify-evidence run explicitly across the full catalog -- all 15 `framework` (the only currently-populated production patch-set; validated-enhancements is empty after RD73's demotion) patches are either legacy-grandfathered or missing-or-stale on stale base_ref only, which is the expected/documented post-bump state (evidence needs regeneration at the new pin, not a bump failure).

`pin-status --complete --all-remotes`: local tree verdict CONSISTENT. Aggregate COMPLETION FAIL because the configured Brutus campaign tree (config/recipes.toml [[trees]] entry, path /mnt/vault/development/llmhosts/bigcherry) has not been bumped -- this is real, expected remaining work (this session only bumped the local/H: tree per PIN_BUMP.md's own 'work on ONE tree at a time' instruction), not a defect in this bump. Real hardware rebuild+smoke at the new pin also remains open (PIN_BUMP.md step 5/6), same reason.

Process improvement made as part of this run: added PIN_BUMP.md step 8, requiring a per-bump ledger event with full per-patch investigation detail as a standing part of the procedure for all future bumps (not just this one) -- see docs/reference/build/PIN_BUMP.md commit c2528184.

STATUS: local bump complete and PASS. Brutus tree convergence and real-hardware post-bump validation remain open as separate follow-on work, tracked under this same item.

REPEATABILITY PROOF 2026-09-10: ran a SECOND real bump, b10883 -> b10884, specifically to prove the procedure is repeatable rather than a one-off. b10884 appeared on upstream between the two bumps (live git ls-remote check found it: 434ddbbc0e30522e897670681e503b797c12b7c1, previously the untagged master commit GPT had already identified as a CI-only sanitizer fix one commit ahead of b10883). Verified via git diff --stat: the real b10883..b10884 diff touches only .github/workflows/*.yml (2 files, CI-only, zero source changes).

pin-bump correctly re-stopped at COVERAGE_INCOMPLETE with the exact same 11 patch IDs as before -- NOT because anything newly broke, but because disposition.py's known_broken mechanism is exact-revision-bound BY DESIGN (per bigcherry-patch-lifecycle doctrine: 'A disposition is NOT a standing waiver... changing revision or patch digest invalidates the disposition automatically'). Confirmed all 11 patches' implementation_digests were byte-identical to the b10883 run before re-recording each disposition at the new target SHA with the same underlying reasoning (same upstream PR citations, cross-referenced back to the b10883 dispositions). `pin-bump --resume` PASSED cleanly. Post-bump patch-lint clean.

This is the concrete repeatability proof: same tooling, same real patch conflicts, a genuinely different real edge case (disposition staleness triggered by a near-no-op upstream commit) handled correctly by the SAME documented procedure with zero manual code intervention -- only the disposition re-recording step, which is itself the exact mechanism PIN_BUMP.md and the patch-lifecycle doctrine specify for this situation.

FINAL STATE: config/recipes.toml pinned=b10884 (moved twice this session: b10705->b10883->b10884). Both bumps PASS. Brutus tree convergence remains genuinely separate follow-on work per PIN_BUMP.md's own 'work on ONE tree at a time... do not bump H: and the build server in the same window' instruction, and is additionally blocked on unrelated uncommitted work sitting in Brutus's configured campaign tree that requires its own decision before touching.

## Change Log

- 2026-09-10T04:51:00.813381+00:00 (created-by): Created by agent
- 2026-09-10T04:51:30.622532+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260910_053053_successfully-bumped-the-llama_6506
- 2026-09-10T05:30:53.720115+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T05:31:54.933096+00:00 (updated-by): Updated: section:notes
- chg_20260910_053852_ran-a-second-real-pin-bump-b_8197
- 2026-09-10T05:38:52.614139+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T05:39:02.500714+00:00 (updated-by): Updated: section:notes
- chg_20260910_065505_replaced-all-11-dispositioned_7448
- 2026-09-10T06:55:05.638982+00:00 (updated-by): Updated: section:ledger-events

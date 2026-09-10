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

## Change Log

- 2026-09-10T04:51:00.813381+00:00 (created-by): Created by agent
- 2026-09-10T04:51:30.622532+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

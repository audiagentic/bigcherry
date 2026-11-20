# llama.cpp patch-disposition records

**Owner:** llama.cpp patch maintenance and pin-bump qualification (HI152). These are tracked, reviewable coverage decisions, not runtime configuration, tune winners, successful builds, or permanent exemptions. Keep them next to this engine's patch packages rather than at repository root.

Each JSON record is keyed by a patch ID, exact upstream `target_revision`, and `patch_digest`. Only `known_broken` is currently supported, and only for **non-recipe** patches. A recipe-selected patch cannot be waived; fix or remove its selection through review. The coverage gate fails closed when the upstream revision or patch bytes change. The record's `reason`, `failure_status`, `owner`, and `tracking_item` must be actionable.

## Lifecycle

1. During an upstream pin bump, run `bigcherry --engine llamacpp patch-rebase-check --all` and identify exactly which non-selected patch fails and why. Do not infer failure from age.
2. Create or update a record using `bigcherry --engine llamacpp patch-disposition set` with the *actual* target commit SHA, current implementation digest, failure status, precise reason, owner and tracking item. Commit and review the JSON along with the associated plan decision.
3. Before each new pin bump or recipe change, run `bigcherry --engine llamacpp patch-disposition list`. Check every record against the candidate upstream revision, its live package digest, current catalog state and recipe selection. If any input differs, the disposition is **stale and inapplicable**; rerun the rebase check, then explicitly renew it with new evidence or clear it.
4. Use `bigcherry --engine llamacpp patch-disposition clear --patch-id <patch-id>` when the patch is repaired, rejected, superseded, or the historical waiver no longer helps the new pin. Preserve decision-grade failure evidence in its patch SUMMARY or `docs/evidence/` before clearing. Never silently delete a record during an update.
5. Do not use these records for Radiance patches, hardware acceptance, performance comparisons, or general archival notes. Rejected/superseded patch identities are handled by the patch catalog.

## Checkpoint

The six active records target upstream b11552. The older 1275 record bound to b11474 is in `docs/evidence/2026-11-20-branch-retirement/1275_ar_small_latency.stale-disposition.json` as historical failure evidence only, not as a live waiver. Each remaining record names a tracking item and retains its exact implementation digest. Age alone is not a safe basis for discarding a current-pin decision.

The authoritative CLI and implementation are `tools/bigcherry/cli/patch.py`, `tools/bigcherry/patch/disposition.py`, and `tools/bigcherry/release/pin_bump.py`. See `docs/reference/patches/PATCH_REFACTOR_RUNBOOK.md`. Generated reports belong in ignored artifacts, not in this source directory.

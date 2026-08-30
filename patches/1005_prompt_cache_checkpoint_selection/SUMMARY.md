# 1005_prompt_cache_checkpoint_selection: Upstream backport: fix hybrid/recurrent memory checkpoint and prompt-cache entry selection

**Status:** untested
**Group:** upstream-fixes
**Plan item:** none

## What it does

Squashes a 4-commit upstream stack into final-state edits: adds ctx_tgt_state_exact() and gates hybrid/recurrent checkpoint creation/selection/invalidation on it (since that memory is only valid at its exact final position, unlike range-valid SWA memory), and fixes prompt-cache entry selection to stop rejecting salvageable entries below a 25% keep-ratio floor and to compare candidates against the best available entry rather than the current slot.

## Why

Pre-fix code treated every checkpoint as range-valid, so hybrid/recurrent checkpoints (e.g. GatedDeltaNet/Mamba-style models) could be selected and restored incorrectly, and prompt-cache entry selection could discard a large reusable prefix in favor of a trivially-better current slot.

## Upstream / provenance

Cherry-picked and squashed from a 4-commit upstream stack (upstream PR #24055 plus three follow-ups), re-verified against this project's current base rather than trusted from an older, since-superseded local patch pack.

# 1268_prbe52_adaptive_mtp_wiring

**Status:** rejected
**Plan item:** PRBE52

## What it does

Wires patch 1255's pure adaptive MTP controller into per-sequence MTP drafting behind the explicit `--spec-draft-n-min-adaptive`/`LLAMA_ARG_SPEC_DRAFT_N_MIN_ADAPTIVE` opt-in.

## Safety

Default value 0 preserves fixed-depth MTP. State resets in `begin()`, draft accounting is per sequence, and controller updates consume the runtime `n_accepted` value rather than recomputing acceptance.

## Validation

Greedy token identity is the correctness gate because accepted MTP draft tokens do not provide complete full-vocabulary rows. Performance uses batched MTP server requests; ordinary decode is the control lane.

## Reconciliation to b11402 (2026-10-06)

Upstream #27694 (probabilistic MTP) changed three of this patch's sites: `begin()` now resets the per-sequence sampler
first, the draft start chooses between greedy and rejection-sampling drafts before any sampler reset, and the depth
cap follows the optional candidate capture. The three edits (`prbe52-begin-reset`, `prbe52-draft-reset`,
`prbe52-depth-limit`) are re-anchored on that shape and now insert only their own lines. Behaviour is unchanged:
floor 0 keeps fixed-depth MTP. The earlier `known_broken` disposition (bound to pin 0504396 and the old digest) is
cleared. Evidence from before the reconciliation does not carry over; the hardware gate is rerun on this pin
(`tools/lab/flash-next/queue-adaptive-mtp.sh`).


## Retirement (2026-10-08)

Rejected under BPB01 at b11474. The package no longer applies at the current pin, depends on the separately broken
1210 decode/verify patch, is not selected by production, and the owner chose not to carry this adaptive-depth mechanism
forward. PA44's stale-branch review records 1268 as retired. Historical hardware/evaluation evidence above is preserved;
this is a lifecycle retirement, not a claim that those earlier measurements were invalid.

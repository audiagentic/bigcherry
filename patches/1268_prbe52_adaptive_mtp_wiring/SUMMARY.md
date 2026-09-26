# 1268_prbe52_adaptive_mtp_wiring

**Status:** untested
**Plan item:** PRBE52

## What it does

Wires patch 1255's pure adaptive MTP controller into per-sequence MTP drafting behind the explicit `--spec-draft-n-min-adaptive`/`LLAMA_ARG_SPEC_DRAFT_N_MIN_ADAPTIVE` opt-in.

## Safety

Default value 0 preserves fixed-depth MTP. State resets in `begin()`, draft accounting is per sequence, and controller updates consume the runtime `n_accepted` value rather than recomputing acceptance.

## Validation

Greedy token identity is the correctness gate because accepted MTP draft tokens do not provide complete full-vocabulary rows. Performance uses batched MTP server requests; ordinary decode is the control lane.

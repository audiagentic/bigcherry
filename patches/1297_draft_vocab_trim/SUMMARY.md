# 1297_draft_vocab_trim

**Status:** untested
**Plan item:** QFN01

## What it does

Opt-in (`BIGCHERRY_DRAFT_VOCAB_N=N`): the Qwen4Exp MTP draft computes logits over output rows [0, N) plus all
control / user-defined / end-of-generation tokens, scattered into -inf full-vocabulary logits.

## Why

The MTP draft's own output.weight (Q8_0, 2560 x 248320, ~675 MB) is read for every draft token: ~31% of the
6900's draft kernel time per MTP step at 80K. The target verifies every draft token, so output cannot change;
only acceptance can drop.

Validation: decode ms/step and acceptance A/B for N = 16384 / 32768 / 65536 vs unset at 10K and 80K; greedy
parity is guaranteed by verification but checked without MTP-irrelevant changes (draft only).

# PRBE20/RD26 decode-vs-verify divergence: ubatch bisection

## Question

Patch 1210 (RD26-DECODE-VERIFY-BIT-IDENTITY) fails its own bit-identity
contract: subject decode (ubatch=1) vs verify-shaped (ubatch=5) raw F32
logits differ from the very first token's logit row (confirmed via the
rd26-decode-verify-diagnostic.json artifact added 2026-09-28, PRBE20).
Two source-verified fix attempts (Wave1+Wave2 fork port, then a second
MMVF fusion-gate call site) both left the identical byte-480 mismatch.

Does the divergence appear between ANY two different ubatch sizes
(implicating a general batch-width-dependent kernel somewhere, not
specific to RD26's targeted kernels), or specifically only between
ubatch=1 and ubatch>=2 (implicating something specific to the
single-token decode path RD26 targets)?

## Inputs

Reuses the SUBJECT `llama-results` binary already built by campaign
session t-1210diag-gfx1100-s1 (patch-refactor commit 00688dee), cached at
`work/worktrees/builds/1210_rd26_bitidentical_decode_verify_standalone-subject-gfx1100+gfx1201+gfx1030/bin/llama-results`
on Brutus. No new build required -- this reruns the existing binary with
different `--ubatch-size` values and diffs the raw GGUF outputs directly,
the same mechanism 1210's own producer.py uses internally.

## Outputs

Pairwise bit-identity (or first-mismatch offset) between ubatch in
{1, 2, 3, 4, 5}, run against the same RD26 probe prompt and gfx1100
device the real contract uses.

## Runtime

Single GPU (gfx1100, device 0), a few minutes -- 5 short single-prompt
generations, no build.

## Safety

Read-only against the existing binary; writes only to a scratch dir under
this lab topic's own workdir. Does not touch the live queue or any shared
build root.

## Disposition

Diagnostic only, not a production tool. Delete or archive once PRBE20's
root cause is found.

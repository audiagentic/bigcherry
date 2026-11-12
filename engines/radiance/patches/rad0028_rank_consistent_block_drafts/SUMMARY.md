# rad0028_rank_consistent_block_drafts

**Status:** untested
**Plan item:** VTP31

Scope: `qwen35_fp8` and `qwen35_w8a8` with the DFlash2 block drafter at two or more ranks (chain proposals).

## Summary

Every rank verifies rank 0's draft. The selected draft tokens are all-gathered after the selector, and the engine
proposes from rank 0's slice of the gathered buffer. Before this, a rank of another card architecture could select
another token on a near tie, accept a different count, and drift: wrong partial sums in the step, and a GPU memory fault
when the drifted rank's attention read past the host's block table.

## Outcome

Untested (not promoted). Built for VTP31: int8 TP3 on two gfx1100 and one gfx1201 faulted in about half of its stress
runs.

## Detail

See README.md.

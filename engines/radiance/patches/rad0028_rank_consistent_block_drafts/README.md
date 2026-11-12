# rad0028_rank_consistent_block_drafts

## Summary

Under tensor parallelism every rank runs the DFlash2 block drafter's selector (`dflash_select`) itself, then verifies
and accepts its own draft. The selector's candidates and their scores are merged from an all-gather, so they agree on
every rank. Its transition scores (`hp`, a GEMM over the drafter's hidden rows) are computed by each rank from its own
drafter state, and cards of different architectures round that differently. On a near tie, one rank selects another
token.

The ranks then verify different drafts and accept different counts. The step's all-reduces mix partial sums of
different tokens, and the drifted rank's context runs ahead of the host's. The decode attention on that rank then reads
past the end of the block table the host staged, into its -1 padding: a GPU memory fault at the KV pool base minus one
block.

This patch all-gathers the selected tokens after the selector and points the engine's proposal at the gathered buffer,
whose first slice is rank 0's. Every rank verifies the same draft.

## Outcome

Untested (not promoted).

VTP31, TP3 on two RX 7900 XTX (gfx1100) and an R9700 (gfx1201), Qwen3.8-27B w8a8 with DFlash2, three speculative
tokens:

- Before: 11 of 22 stress runs faulted on the R9700 (`r4d_attn_decode_kernel` read block -1). With speculation off, 0 of
  10 did.
- A diagnostic comparing every rank's step slots with rank 0's found 166 rows where rank 2 differed (first in a drafted
  token, then in acceptance and context); rank 1 never differed.

## Detail

| part | change |
|---|---|
| `rad_block_dflash2.h` Wire | `tokg` (every rank's tokens, gathered) and `tok_rows` |
| `rad_block_dflash2.h` declare | `all_gather` of the whole tokens buffer into `tokg` when declared and the proposal is a chain |
| `rad_block_dflash2.h` issue | the gather after the selector and the coupled walk |
| `qwen35_fp8.cpp` | above one rank: tokens rows rounded up to whole 16-byte words, `dflash_tokens_ranks` declared, and the drafter's proposal set to it |

The gather moves the tokens' bytes as 32-bit words. It is a plain concatenation, so rank 0's tokens sit at the start of
`tokg` at the same pitch, and the engine's read (`proposal`, `proposal_pitch`) is unchanged. A tree proposal is left as
it was, since the engine reads its parents from another buffer.

The cost is one all-gather of `world * tok_rows * steps` 32-bit words per draft pass (under 2 KiB at eight sequences and
fifteen steps).

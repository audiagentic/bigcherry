# rad0027_mixed_one_pipelined_prefill

## Summary

rad0026 pipelines a mixed step with two or more prefill sequences. A mixed step with one prefill sequence stays
serial. That is the step a long prompt makes beside streams that are generating: every chunk after its first, and the
usual shape when requests arrive one at a time. In VTP31 experiment 10, four 4096-token prompts at once reach their
first token 15% later at TP3 than at TP2.

This patch cuts such a step inside its one prefill sequence, as rad0020 cuts a one-sequence prefill step, with the
decode rows in the first piece.

## Outcome

Untested (not promoted).

## Detail

| part | change |
|---|---|
| `qwen35_fp8.cpp` `pipe_ready` | takes a mixed step with one prefill sequence when its decode rows are under 64 and both pieces have at least 64 rows; the first piece is the decode rows and the sequence's first `pipe_rows(T - DT)` rows |
| `qwen35_fp8.cpp` bounds | {DT, R, T} from a pinned table per decode height, made on first use and freed with the model, so a recorded step replays the same copy |
| attention block | the one-sequence piece path takes the step; the first piece issues the serial path's decode launch over `[0, D)`, then the prefill piece over sequence D's block-table row and seqused, at its own row count |
| delta-net block | the one-sequence piece path takes the step after rad0026's decode half; the prefill pieces use sequence D's convolution and state indices, and the first piece starts at row DT |

The prefill pieces are rad0020's one-sequence pieces at an offset row, and the decode launches are the serial path's.

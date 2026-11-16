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

### Promotion record

Model: Qwen3.8-27B MXFP4 (.rad), TP3 over two RX 7900 XTX and an R9700
Mechanics: the patch's tests in tools/tests/patch pass and patch-lint is clean on the promotion PR; the lane builds carrying it (VTP logs named below) build and serve.
Activation: the scheduler's "runs serially" note is gone in every arm that carries rad0025; rad0026 and rad0027 are the mixed-step shapes (decode beside two or more prefills, decode beside one cut prefill), each exercised by the workload that makes that shape (VTP31.md experiments 8, 9, 11).
Identity: pipelined steps compute the serial step's bytes; greedy texts differ only where the base build's own runs differ (VTP31.md: rad0025-0027 diverge only where the base build's runs diverge); rad0020 output unchanged.
A/B: 2x ABBA, one binary per arm, same settings. rad0020 4096-token TTFT -5% (VTP14.md); rad0025 four 4096-token prompts -7.0% slowest TTFT, 4 streams +4.0%, 1/8 streams 0; rad0026 4/6/8 streams +7.5/+10.0/+10.6%, four-prompt TTFT -6.7%; rad0027 8192-token prompt beside 4 streams TTFT -19.2%, standard bench unchanged. Logs e8-1223, e9-1328, e11-1421 (VTP31.md).

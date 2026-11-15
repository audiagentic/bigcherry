# rad0026_mixed_pipelined_prefill

## Summary

rad0025 pipelines a pure prefill step of several sequences, cut between them. A mixed step, with decode rows beside
the prefill, stays serial. In a multi-stream run, once the first streams are generating, the later prompts' prefill
lands in mixed steps, so at 8 streams most of the prefill is serial and rad0025 changed nothing there (VTP31
experiment 8).

Each block's serial path already splits a mixed step into a decode prefix (sequences `[0, D)`, rows `[0, DT)`, with the
decode and verify launches) and a prefill tail (`[D, S)`, the chunked launches over the step's absolute bounds). This
patch keeps that split and cuts the tail between two prefill sequences. The first piece is the decode prefix plus the
prefill sequences `[D, seq_cut)`, and the second piece is the rest.

## Outcome

Untested (not promoted).

## Detail

| part | change |
|---|---|
| `qwen35_fp8.cpp` `pipe_ready` | takes a mixed step with two or more prefill sequences, cut at the prefill boundary nearest half the rows (both pieces at least 64 rows); a mixed step with one prefill sequence stays serial, without a note |
| attention block | the first piece issues the serial path's decode launch over `[0, D)` at the decode band, then its prefill sequences; where the serial mixed path launches its prefill half at the top band (`g.max_tok`), both pieces do |
| delta-net block | the first piece issues the serial path's decode-half ops over `[0, D)` (the fused or split recurrent update, which gates and quantises its rows), then the chunked ops over its prefill sequences; the gated norm and quantiser cover the prefill rows only |
| `core/runtime/ctx.cpp` `pass_key` | a mixed pass of several sequences is keyed by the host bounds too (rad0025 keys a prefill pass) |

Every launch is one the serial step issues, over the same sequences at the same band, and the per-row ops cover the
same rows, so the output is the same bits.

### Promotion record

Model: Qwen3.8-27B MXFP4 (.rad), TP3 over two RX 7900 XTX and an R9700
Mechanics: the patch's tests in tools/tests/patch pass and patch-lint is clean on the promotion PR; the lane builds carrying it (VTP logs named below) build and serve.
Activation: the scheduler's "runs serially" note is gone in every arm that carries rad0025; rad0026 and rad0027 are the mixed-step shapes (decode beside two or more prefills, decode beside one cut prefill), each exercised by the workload that makes that shape (VTP31.md experiments 8, 9, 11).
Identity: pipelined steps compute the serial step's bytes; greedy texts differ only where the base build's own runs differ (VTP31.md: rad0025-0027 diverge only where the base build's runs diverge); rad0020 output unchanged.
A/B: 2x ABBA, one binary per arm, same settings. rad0020 4096-token TTFT -5% (VTP14.md); rad0025 four 4096-token prompts -7.0% slowest TTFT, 4 streams +4.0%, 1/8 streams 0; rad0026 4/6/8 streams +7.5/+10.0/+10.6%, four-prompt TTFT -6.7%; rad0027 8192-token prompt beside 4 streams TTFT -19.2%, standard bench unchanged. Logs e8-1223, e9-1328, e11-1421 (VTP31.md).

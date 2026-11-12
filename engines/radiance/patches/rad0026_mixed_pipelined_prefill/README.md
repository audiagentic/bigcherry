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

# rad0025_multiseq_pipelined_prefill

## Summary

The pipelined prefill (rad0020, upstream `step_pipelined`) overlaps each block's all-reduce for one half of a prefill
step with the other half's compute. It takes a step of one sequence only, cut inside it. When several prompts arrive
together the scheduler packs their prefill into one step, which then runs serially ("runs serially: more than one
sequence"); at three ranks such a step spent about 70% of its profiled device time in collectives (VTP31), and the 4-8
stream bench's TP3 margin over TP2 is a third of the decode-only margin.

A step of several whole sequences is cut between them. The pieces share no attention keys and no recurrent state, and
the blocks' serial paths already run over a contiguous run of sequences by offsetting the per-sequence arrays over the
step's absolute row bounds, so each piece takes the same form.

## Detail

| part | change |
|---|---|
| `abi/rad_runtime.h` | `RadBatch::cu_seqlens_h`, trailing: the step's bounds on the host, or null |
| `core/sched/batch.cpp` | the scheduler's batch sets it from its host staging (`h_cu`), beside `positions_h` |
| `core/runtime/ctx.cpp` | the RadBatch size check moves to 480 and `pass_key` mixes the host bounds of a prefill pass of several sequences, so a recorded pass never replays over a different cut |
| `arch/common/rad_fp8.h` | `ArPieces::seq_cut`: the first piece is sequences `[0, seq_cut)` |
| `qwen35_fp8.cpp` | `pipe_ready` takes a pure prefill step of two or more sequences, cut at the boundary nearest half the rows, both pieces at least 64 rows; the first piece's rows go through `step_pipelined` and the hand-off instead of `pipe_rows(T)` |
| attention and delta-net blocks | each piece over its slice of the per-sequence arrays: query bounds, seqused and block-table rows; convolution and state indices and context lengths; launched as the serial path's prefill launch |

A step with decode rows, media rows, or without host bounds (drivers other than the scheduler) stays serial, as does
one whose best boundary leaves a piece under 64 rows. The one-sequence cut is unchanged.

Every row's arithmetic is the serial step's: the per-sequence kernels see the same sequences over the same bounds, the
per-row ops the same rows in the band of the whole step, and the all-reduce reduces within a row. The output is the same
bits; the pipelined prefill's existing contract.

Source-pinned to radiance v1.17.0 (19d4e8fe), after rad0020.

## Outcome

Untested. VTP31 measures it against the rad0024 build: standard bench at 1/4/6/8 streams, four 4096-token prompts at
once, output identity, and a profile arm.

### Promotion record

Model: Qwen3.8-27B MXFP4 (.rad), TP3 over two RX 7900 XTX and an R9700
Mechanics: the patch's tests in tools/tests/patch pass and patch-lint is clean on the promotion PR; the lane builds carrying it (VTP logs named below) build and serve.
Activation: the scheduler's "runs serially" note is gone in every arm that carries rad0025; rad0026 and rad0027 are the mixed-step shapes (decode beside two or more prefills, decode beside one cut prefill), each exercised by the workload that makes that shape (VTP31.md experiments 8, 9, 11).
Identity: pipelined steps compute the serial step's bytes; greedy texts differ only where the base build's own runs differ (VTP31.md: rad0025-0027 diverge only where the base build's runs diverge); rad0020 output unchanged.
A/B: 2x ABBA, one binary per arm, same settings. rad0020 4096-token TTFT -5% (VTP14.md); rad0025 four 4096-token prompts -7.0% slowest TTFT, 4 streams +4.0%, 1/8 streams 0; rad0026 4/6/8 streams +7.5/+10.0/+10.6%, four-prompt TTFT -6.7%; rad0027 8192-token prompt beside 4 streams TTFT -19.2%, standard bench unchanged. Logs e8-1223, e9-1328, e11-1421 (VTP31.md).

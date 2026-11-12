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

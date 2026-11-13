# rad0033_ts3_once

## Summary

In the exact two-shot's second hop a rank stores every reduced word of its shard once per peer, into its slot in that
peer's scratch. Over peer access those are different cards' memories. Through the host mailbox every rank's scratch is
pinned host memory that every card maps, so the copies are the same bytes written twice across the writer's own
link: at three ranks a rank that owns half of the message sends all of it in the second hop where half would do.

Under the axis `ts_once` (0, 1; default 0) of the all-reduce rows, on a host-mailbox instance:

- a rank writes its reduced shard once, into region B of its own slot in its own scratch. The wide family never uses
  that slot otherwise: a rank's own contribution comes from its input;
- each peer assembles that shard from the owner's scratch instead of its own, in the plain assemble and in rad0032's
  fused step.

The handshake is unchanged: the owner publishes the step's flag after its stores have drained, and peers read only
after seeing it. The slot's parity alternates per launch as before, and a rank reaches launch k + 2, the next writer
of the same parity, only after every rank has finished launch k. The reduced words are the same, so the bits are too.

Selected per run with `--tune-pin` (rad0022), e.g. `all_reduce:nblocks=4,nthreads=256,ts_once=1`. Over peer access
the axis is ignored.

## Outcome

Untested (not promoted). VTP36, XTX + XTX + R9700 without peer access; numbers and the ABBA of the combined setting
are in rad0032's README and VTP36.md. Alone it takes the XTX ranks' 32-row decode call from 81 to 72 us; with the
duplex weights (rad0032) to 65 us. In a pipelined prefill it is worth about 10% of a 1024-row call beside four fused
pieces with the settled weights, and nothing measurable once the duplex weights are on.

## Detail

- `r4d_ar_twoshot_Nrank_exact.hip`: the kernel takes `ts_self` (this rank's own scratch, null for upstream's copy per
  peer); `reduce2_once`, the fold with one store, as a loop of its own so that the upstream fold's loop is untouched;
  the two readers of a peer's region B take the owner's scratch when `ts_self` is set.
- A first form derived the pointer from the kernel's `restrict` scratch pointer and chose the store inside the fold's
  loop. That made the upstream fold 13 times slower on gfx1100 with the option off, which is why the once form has its
  own argument and its own loop.
- `r4d_ar_entry.hip`, `r4d_rows.cpp`, `r4d_args.h`, `r4d.h`: the axis on every all-reduce tunable table, its reader,
  and the three-rank launch's `once` argument (host mailbox only).
- Source-pinned to radiance v1.17.0 (19d4e8fe); after rad0032.

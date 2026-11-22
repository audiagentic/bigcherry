# LTX issued block timing: one card, two cards, three cards

Measured on the native lane (`tools/lab/radiance/ltx-tp3/native-c01-gate`, modes `--ltx03-block-time-aql` and
`--ltx03-wire-error-aql`) with the IQ8R checkpoint at block 0: a warm step of one issued block, median of 20,
slowest rank per iteration, under short per-arm GPU leases. TP1 runs on one gfx1100 card; two cards are the two
gfx1100s; three cards add the gfx1201 card behind the narrower link. Everything uses the engine's exact all-reduce
unless a file says otherwise. No machine names appear here: card names are architectures.

## Findings (2048 tokens, video)

| arrangement | FFN | attention + FFN block |
|---|---|---|
| one card (TP1) | 6.7 ms | 13.1 ms |
| three cards, default all-reduce | 8.1 ms (0.83x) | 30.8 ms (0.42x) |
| three cards, ts_pieces=4 + ts_fuse + ts_duplex pinned | 6.2 ms (1.09x) | 24.6 ms (0.54x) |
| three cards, the same plus the FFN in two row pieces, each reduction on the second lane | 7.0 ms | 21.3 ms (0.61x) |
| two cards, exact | 6.5 ms (1.04x) | 19.7 ms (0.67x) |
| two cards, exact, FFN pipelined | 6.6 ms | 19.9 ms |
| two cards, lossy WHT6 wire | 6.3 ms (1.09x) | 13.3 ms (0.99x) |

* The cost is the reductions, not the balance: moving all work off the card behind the narrow link saves about 12%
  of a three-card step (`timing-split-sweep-2048.txt`). Each exact reduction of the 16.8 MB activation costs about
  3 ms on two cards and 5 ms or more on three, the same link-bound figure the Qwen 27B exact wire shows.
* Three cards need the three-rank all-reduce axes pinned together; each alone does nothing
  (`timing-allreduce-axes-pinned.txt`). Four pieces is the best FFN setting.
* The attention block issues two further full-width reductions for the all-head Q and K norms, which is why it
  stays at or below one card. Pipelining the attention output projection measured slower than leaving it serial
  (24.2 against 21.3 ms) and was not kept.
* WHT6 is level with one card for the block but its error against the TP1 issue is 5-7 times the exact wire's
  (relative RMS 1.4-2.5% against 0.27-0.73% per block), so the adapter stays exact-only
  (`timing-two-card-and-wire.txt`).
* One card alone differs by architecture (`timing-one-card-per-architecture.txt`, `LTX03_TIME_TP1=1` with one visible
  device): the gfx1201 card runs the video block in 6.7 ms and the FFN in 3.7 ms, 1.95 times faster than a gfx1100
  card (13.0 and 6.7 ms) on the int8 GEMMs. That one card beats every multi-card tensor-parallel arm above.
* Why: a plain rocBLAS int8 GEMM at the FFN's shape is the other way round, 127 TOPS on the gfx1100 card against
  81 on the gfx1201 one (`i8-gemm-reference/`). The engine's FFN reaches 132 TOPS on gfx1201, above that
  reference, but only 72 TOPS on gfx1100, so the 2x is mostly the gfx1100 kernels running well below what the card
  does in a generic GEMM (the engine figure also holds the activation quantisation, GELU and bias).
* Conclusion: with exact reductions over this host's links, tensor parallelism makes an LTX block no faster than
  one card. The Q/K statistics reduction (per-token sums of squares instead of the full-width tensor) is the one
  change expected to remove two of the large reductions; independent guidance passes on separate cards need no
  reductions at all.

## Files

* `timing-3card-and-1card.txt`: FFN and block at 64, 512 and 2048 tokens, one card and the two three-card splits.
* `timing-split-sweep-2048.txt`: five head splits with per-rank step times.
* `timing-allreduce-axes-pinned.txt`: the pinned all-reduce axis arms.
* `timing-two-card-and-wire.txt`: two cards with and without the FFN pipelined, and exact against WHT6 (time and error).

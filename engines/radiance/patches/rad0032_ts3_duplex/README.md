# rad0032_ts3_duplex

## Summary

A three-rank world without peer access exchanges every all-reduce through pinned host memory, and with one card
behind a narrower link the exact two-shot (rad0021, rad0024) is bound by that card's link. rad0031's timeline shows
how on XTX + XTX + R9700 (Gen4 x8, x8, x4), for the 1024-row message of a pipelined 2048-token prefill: the R9700
sends for 1.4 ms, waits 1.1 ms while the other two reduce, then receives for 1.3 ms. Its link carries one direction
at a time, and the second lane of a pipelined prefill is nothing but these calls, so their length is the prompt's
time to first token.

Upstream has two remedies at eight ranks only, and a third is new. This patch offers all three to the three-rank
two-shot as axes of the all-reduce rows, default off, pinned per run with `--tune-pin` (rad0022):

| axis | values (default) | effect |
|---|---|---|
| `ts_pieces` | 1, 2, 4, 8, 16 (1) | each block's words run as a pipeline of pieces (upstream's PIECED kernel): a rank sends piece j + 1 and assembles piece j - 1 while its peers reduce piece j, instead of idling |
| `ts_pieces_min` | 0, 524288 ... 8388608 (0) | elements from which `ts_pieces` applies; 0 is the instance's floor (upstream's 262144) |
| `ts_fuse` | 0, 1 (0) | in a step that both sends and assembles (three pieces or more), the two run as one loop: each pass loads one word from the exchange and stores one to it |
| `ts_duplex` | 0, 1 (0) | where the settled link weights apply, upstream's duplex table (`r4d_ar_weights_duplex`) instead, if its shard pair fits the slot: here 64,64,0 for 4,4,1, so the card behind the narrow link owns no shard and does no fold |

Why the fused loop: a card's stores and loads to host memory do not overlap when they are issued as separate loops
or separate kernels. On the R9700 (32 MiB, `docs/evidence/2026-11-13-vtp36-allreduce/`): 7.0 GB/s out or 6.7 GB/s in
alone, 7.6 GB/s in total as two kernels at once, 11.9 GB/s in total as one loop that loads and stores each pass (an
XTX: 14.2, 12.9, 23.6 and 25.3).

Which rank owns a word, how a block's words are cut into pieces and whether two copies share a loop change only when
a word crosses which link, never its fold (fp32, ascending rank), so every setting gives the bits of the default.
With every axis at its default the launch is the one rad0021 and rad0024 make.

## Outcome

Untested (not promoted). VTP36, w8a8 and MXFP4 Qwen3.8-27B on XTX + XTX + R9700, with rad0033's `ts_once`, as
`all_reduce:nblocks=4,nthreads=256,ts_pieces=4,ts_fuse=1,ts_once=1,ts_duplex=1` against the layouts' 4 x 256 pin:

| | w8a8 | MXFP4 |
|---|---|---|
| 1 / 4 / 6 / 8 streams, tok/s | +1.0 / +7.6 / +10.9 / +13.9% | +4.8 / +13.6 / +13.5 / +16.7% |
| one prompt, time to first token at 522 / 2021 / 6059 tokens | -13.0 / -24.4 / -21.6% | -21.2 / -27.2 / -25.1% |
| four 4096-token prompts, slowest time to first token | -14.0% | -22.9% |
| 6144-token prompt beside 4 generating streams | -21.4% | -25.1% |

2x ABBA on one build, every server alive at the end of its arm. First-token top-20 log-probabilities at eight prompt
lengths are bit-identical to the layout pin's on both formats, and one-stream greedy texts are identical at seven
lengths. The int8 12-run stress is in VTP36.md.

## Detail

- Timeline (rad0031), 1024-row call, the R9700's total: 3.8-4.1 ms as is; 3.3 ms in four pieces; 3.5 ms fused; 3.1 ms
  fused with `ts_once` or with `ts_duplex`; 2.9-3.0 ms with all four. Eight and sixteen pieces measure 3.0 ms: smaller
  pieces fuse less well and add handshakes. A wider launch grid (8 x 256, 4 x 512, 8 x 512, 16 x 256) is no better.
- Tried and removed: odd blocks assembling before they push, so that half the blocks send while half receive. No
  gain (3.3 ms), because separate loops do not overlap on the link.
- `r4d_ar_twoshot_Nrank_exact.hip`: `r4d_ts_push_assemble` (the fused loop), `push_assemble` in the kernel body, the
  kernel argument `ts_fuse`, and `r4d_ar_twoshot_3rank_exact_wp` (shard table, pieces, fuse). The split and ring forms
  and the other world sizes are untouched.
- `r4d_ar_entry.hip`: the duplex table settled beside the link weights at three ranks; the launch reads the axes.
  `ts_pieces` applies from the piece floor, `ts_fuse` from three pieces.
- `r4d_rows.cpp`, `r4d_args.h`: the axes on every all-reduce tunable table (as `ts_min`), so one pin is legal for every
  row the op resolves to.
- Source-pinned to radiance v1.17.0 (19d4e8fe); after rad0021, rad0024 and rad0031.

### Promotion record

Model: Qwen3.8-27B MXFP4 and w8a8 (.rad), TP3 over two RX 7900 XTX and an R9700
Mechanics: the patch's tests in tools/tests/patch pass and patch-lint is clean on the promotion PR; the lane builds carrying it (VTP logs named below) build and serve.
Activation: rad0031 (diagnostic, on only with its timeline env) emitted the push, peer-wait, reduce and assemble stamps that VTP36.md used to split the three-rank two-shot, and its stamps are the activation proof for rad0032/rad0033; rad0021 weights 4,4,1 and rad0032 duplex weights are reported in the all-reduce setup; rad0024 hands decode-sized messages to the host-mailbox one-shot above ts_min=32768 elements; rad0033 ts_once writes the reduced shard once (XTX 32-row decode call 81 to 65 us); rad0022 pins all_reduce:nblocks=4,nthreads=256 through the tune pin the engine echoes at start (VTP14.md, VTP31.md experiments 1 and 3, VTP36.md).
Identity: rad0031 is read-only timing (unset, the kernel takes no stamp and does the same work); output byte-identical in every arm: rad0021 byte-identical, rad0024 greedy hashes identical, rad0022 greedy identical, rad0032 bit-identical (VTP14.md, VTP31.md, VTP36.md).
A/B: 2x ABBA on one binary. rad0021 4096-token TTFT -11% at one stream, decode unchanged; rad0024 4/6/8 streams +5.0/+3.5/+6.1% vs ts_min=0, one stream and TTFT unchanged; rad0022 +1.7/+6.7/+3.5/+5.7% at 1/4/6/8 streams; rad0032 with rad0033 2048-token TTFT -24% (w8a8) and -27% (MXFP4), 8 streams +14% and +17%.

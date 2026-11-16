# rad0021_vtp14_weighted_twoshot_3rank

## Summary

libr4d's exact two-shot all-reduce takes a shard table: each rank reduces its own range of the message. At eight ranks
the engine weights the table by each card's slowest PCIe hop (`ar8_settle`, `r4d_ar_weights_from_rates`), so a rank
behind a slow link owns a smaller shard, or none, and moves fewer bytes over that link. Who owns a word changes only the
link it crosses, never its fold (fp32, ascending rank): every table gives the same bits.

At three ranks the same kernel ran with the even split. On two RX 7900 XTX (x16) and an R9700 behind a Gen4 x4 root
port, a third of every prefill all-reduce crossed the x4 link twice. This patch settles the weights at three ranks as
at eight and passes them to the three-rank launch. With the rule `round(4 (rate / fastest)^2)` the x4 card owns no
shard: it only pushes its input and gathers the result, N each way instead of 4/3 N.

- Equal links settle to the even split, exactly as before.
- Messages under the weights' floor (262144 elements) stay even, so decode and the one-shot are unchanged.
- libr3 is generated from this libr4d source, so a mixed world's two libraries derive one table, from the shared
  rendezvous (rad0017) after a barrier. The engine logs the links and weights once.

## Outcome

Untested (not promoted). On cards (MXFP4 6/6/12 TP3, exact wire, ABBA against the same build without it): the links
settle to 4,4,1 (XTX at Gen4 x8, R9700 at x4). A 4096-token prompt's time to first token falls 2516 -> 2241 ms (-11%,
one stream) and 11.4 -> 11.2 s (-2.3%, four streams); 512-token prompts and decode are within noise; greedy output is
byte-identical (VTP14.md).

## Detail

- `r4d_ar_entry.hip`: `ar8_settle` also at world 3; the weighted scratch bound uses the world, not 8; the three-rank
  launch takes `ar8_weights_for` and the shard table; the weights log line names the world.
- `r4d_ar_twoshot_Nrank_exact.hip`, `r4d.h`: `r4d_ar_twoshot_3rank_exact_w`, the three-rank launch with a shard table
  (the existing `r4d_ar_twoshot_3rank_exact` stays, with the even split).
- Source-pinned to radiance v1.17.0 (19d4e8fe), applied after rad0017.

### Promotion record

Model: Qwen3.8-27B MXFP4 and w8a8 (.rad), TP3 over two RX 7900 XTX and an R9700
Mechanics: the patch's tests in tools/tests/patch pass and patch-lint is clean on the promotion PR; the lane builds carrying it (VTP logs named below) build and serve.
Activation: rad0031 (diagnostic, on only with its timeline env) emitted the push, peer-wait, reduce and assemble stamps that VTP36.md used to split the three-rank two-shot, and its stamps are the activation proof for rad0032/rad0033; rad0021 weights 4,4,1 and rad0032 duplex weights are reported in the all-reduce setup; rad0024 hands decode-sized messages to the host-mailbox one-shot above ts_min=32768 elements; rad0033 ts_once writes the reduced shard once (XTX 32-row decode call 81 to 65 us); rad0022 pins all_reduce:nblocks=4,nthreads=256 through the tune pin the engine echoes at start (VTP14.md, VTP31.md experiments 1 and 3, VTP36.md).
Identity: rad0031 is read-only timing (unset, the kernel takes no stamp and does the same work); output byte-identical in every arm: rad0021 byte-identical, rad0024 greedy hashes identical, rad0022 greedy identical, rad0032 bit-identical (VTP14.md, VTP31.md, VTP36.md).
A/B: 2x ABBA on one binary. rad0021 4096-token TTFT -11% at one stream, decode unchanged; rad0024 4/6/8 streams +5.0/+3.5/+6.1% vs ts_min=0, one stream and TTFT unchanged; rad0022 +1.7/+6.7/+3.5/+5.7% at 1/4/6/8 streams; rad0032 with rad0033 2048-token TTFT -24% (w8a8) and -27% (MXFP4), 8 streams +14% and +17%.

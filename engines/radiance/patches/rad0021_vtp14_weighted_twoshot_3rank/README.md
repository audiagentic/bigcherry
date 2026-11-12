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

Untested (not promoted). To be measured: prefill (TTFT) and 8-stream throughput at TP3, against the even split.

## Detail

- `r4d_ar_entry.hip`: `ar8_settle` also at world 3; the weighted scratch bound uses the world, not 8; the three-rank
  launch takes `ar8_weights_for` and the shard table; the weights log line names the world.
- `r4d_ar_twoshot_Nrank_exact.hip`, `r4d.h`: `r4d_ar_twoshot_3rank_exact_w`, the three-rank launch with a shard table
  (the existing `r4d_ar_twoshot_3rank_exact` stays, with the even split).
- Source-pinned to radiance v1.17.0 (19d4e8fe), applied after rad0017.

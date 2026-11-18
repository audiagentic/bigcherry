# Flash-Next layout smoke with the packed KV types (VTP44)

## Summary
Every Flash-Next layout the launcher can drive starts, passes its canary and serves with bf16_fp8 and bf16_i8 on this build, beside the
fp8 cache the layouts ship with and bf16. Nothing regressed at the level a short test can see. One layout, `flashnext-4gpu`, does not fit
the packed types (or bf16) in the R9700's VRAM at its default session sizing and fits all of them at four sessions. The helper-free
`flashnext-r9700` layout cannot be driven by the launcher, and its engine-only start needs io_uring, which this container refuses.

## Outcome
The build with the packed KV patches (rad0037, 0038, 0040, 0041, 0042), the real launcher (`launch-flashnext-best.sh start`) with each
layout as its server-args define it and only `--kv-dtype` changed, the repository heat file, n-gram placement on disk. Each configuration:
the launcher measures a plan, starts, passes its canary, then a short client sends an 8k-token prompt (time to first token), decodes 256
tokens on one stream and answers two needle questions in 12k-token contexts, then stops.

| layout | KV type | TTFT 8k (s) | decode, 1 stream (chunks/s) | needle probes |
|---|---|---|---|---|
| flashnext-2xtx | fp8 | 3.82 | 29.7 | 2/2 |
| flashnext-2xtx | bf16 | 3.91 | 28.1 | 2/2 |
| flashnext-2xtx | bf16_fp8 | 3.52 | 29.2 | 2/2 |
| flashnext-2xtx | bf16_i8 | 3.66 | 28.7 | 2/2 |
| flashnext-2xtx-combine | fp8 | 3.37 | 29.8 | 2/2 |
| flashnext-2xtx-combine | bf16 | 3.45 | 29.0 | 2/2 |
| flashnext-2xtx-combine | bf16_fp8 | 3.37 | 28.5 | 1/2 |
| flashnext-2xtx-combine | bf16_i8 | 3.39 | 28.6 | 2/2 |
| flashnext-2xtx-rdna2-head | fp8 | 4.18 | 26.9 | 1/2 |
| flashnext-2xtx-rdna2-head | bf16 | 4.29 | 27.3 | 2/2 |
| flashnext-2xtx-rdna2-head | bf16_fp8 | 4.36 | 28.5 | 2/2 |
| flashnext-2xtx-rdna2-head | bf16_i8 | 9.16 | 23.1 | 2/2 |
| flashnext-2xtx-rdna2-tail | fp8 | 3.86 | 29.8 | 2/2 |
| flashnext-2xtx-rdna2-tail | bf16 | 10.00 | 23.0 | 2/2 |
| flashnext-2xtx-rdna2-tail | bf16_fp8 | 7.40 | 16.5 | 2/2 |
| flashnext-2xtx-rdna2-tail | bf16_i8 | 3.82 | 28.1 | 2/2 |
| flashnext-2xtx-rdna2-tail-combine | fp8 | 3.89 | 28.0 | 2/2 |
| flashnext-2xtx-rdna2-tail-combine | bf16 | 8.75 | 12.0 | 2/2 |
| flashnext-2xtx-rdna2-tail-combine | bf16_fp8 | 3.45 | 28.4 | 2/2 |
| flashnext-2xtx-rdna2-tail-combine | bf16_i8 | 3.38 | 28.0 | 2/2 |
| flashnext-4gpu | fp8 | 3.23 | 29.5 | 2/2 |
| flashnext-4gpu | fp8@seqs4 | 3.42 | 29.2 | 2/2 |
| flashnext-4gpu | bf16@seqs4 | 3.33 | 28.7 | 2/2 |
| flashnext-4gpu | bf16_fp8@seqs4 | 3.38 | 29.5 | 2/2 |
| flashnext-4gpu | bf16_i8@seqs4 | 3.44 | 26.1 | 2/2 |
| flashnext-tp2xtx | fp8 | 4.64 | 25.2 | 0/2 |
| flashnext-tp2xtx | bf16 | 4.46 | 24.3 | 2/2 |
| flashnext-tp2xtx | bf16_fp8 | 7.15 | 24.1 | 2/2 |
| flashnext-tp2xtx | bf16_i8 | 4.26 | 25.6 | 2/2 |

- Eight layouts were queued: seven started every KV type tried; `flashnext-r9700` was refused by the launcher (below).
- Within a layout the four KV types sit together: decode within about 5% of each other apart from the cases noted, time to first token
  within noise. These are single short runs; they show nothing broke, not a ranking. KLD is the quality comparison (FLASHNEXT-REPLAY-11DOCS.md).
- `flashnext-4gpu` (R9700 engine, two 7900 XTX and the RX 6900 XT as helpers): at the layout's default sizing the engine card (32.2 GB free)
  refuses bf16_fp8 (short 4.9 GB), bf16_i8 (short 5.5 GB) and bf16 (short 13.3 GB) before loading weights with `gpu_room`; fp8 fits. The
  default sizing is the launcher's 8 sessions of up to 262,144 tokens (the worst-case KV is about 35 GB at bf16). With `--max-num-seqs 4` all
  four types start (rows marked `@seqs4`, fp8 also run at four sessions for comparison). The packed types therefore need fewer or shorter
  sessions than fp8 on this card at that sizing; the capacity gain over bf16 is 1.28x (bf16_i8) and 1.31x (bf16_fp8), against 1.89x for fp8.
- Needle misses are the model declining ("I cannot provide the vault passphrase."), not failing to find the text, and they fall on the shipped
  fp8 cache (`flashnext-tp2xtx`, 0 of 2) as well as bf16_fp8 and at random, so with two probes they say nothing about the cache types.
- Timings that stand out (rdna2-tail bf16 and bf16_fp8, rdna2-head bf16_i8, rdna2-tail-combine bf16: first token 7 to 10 s or decode 12 to 23
  chunks/s) were measured while builds and analyses ran on the same host, which has few spare cores; treat all timings here as unreliable
  beyond "it ran".
- `flashnext-r9700` (helper-free): the launcher refuses `--kv-dtype`, `--heat` and `--ngram-placement` for a helper-free layout (its options
  live in the layout's server-args), so the packed types cannot be selected through it. Run directly, the engine on that card reads the model's
  47.7 GiB n-gram table from disk through io_uring, which this container's seccomp profile refuses; the table does not fit in RAM beside the
  expert pool. The helper layouts do not have this limit (the helper serves the table).

## Limits
Short single runs on a host that was also building; one prompt length; two needle probes; the R9700-only and 6900-XT-only paths not
separated; no comparison against the earlier 9700 / 7900 XTX / 6900 XT numbers in the same layouts (a matched run would be needed).

## Files
`layout-smoke/`: a JSON per configuration, a receipt per layout, `table.md`, the first refusal log for `flashnext-4gpu`, the runner scripts
(`smoke_layout.sh`, `smoke_all.sh`, `smoke_pass2.sh`), `fnsmoke.py` (the client) and `queue.log`. Absolute paths are replaced by placeholders.

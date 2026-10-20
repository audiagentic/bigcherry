# Qwen3.6-35B-A3B FP8 (MoE) on radiance (Brutus)

Current-state page. Rule: every measurement or decision updates this page in the same change. Overview:
[../README.md](../README.md). Plan items: RAD13 (FP8 worker), RAD14 (shape-aware geometry and the MTP bisection).

## Container

`/mnt/data/llm-models/radiance/StillDeadcode/qwen3.6-35b-a3b-fp8/qwen3.6-35b-a3b-fp8.rad`: 34.6 GiB, block fp8 (E4M3 weights,
bf16 128x128 scales), E4M3 activations with an f32 scale per 128 of K. 256 experts, top-8 (read from the container's
`n_expert_used`). Hybrid: gated delta-net (linear state) plus full attention; an MTP head is in the checkpoint.

## MTP acceptance collapse with the prefix cache on (open radiance bug, 2026-10-20)

**Symptom.** Engine alone, `--num-speculative-tokens 3`, one R9700 (`--tp 1`), prefix cache at its default (on): draft
acceptance is 0.61 / 0.64 at 1 / 2 streams and 0.02-0.07 at 4 and 6 streams. The same at 1 and 2 draft tokens
(MTP1 0.16 / 0.14, MTP2 0.10 / 0.04 at 4 / 6 streams). Engine decode falls to 105-217 tok/s at 4-6 streams.

**Workaround.** `--no-prefix-cache`: acceptance 0.55-0.62 at 4 streams and 0.55-0.60 at 6 streams (second rep); engine decode
382-492 tok/s at 4 and 263-649 at 6. Use it in every MTP profile for this model until the cause is found. The cost is
no prefix reuse across requests.

**Where it reproduces** (all 3 draft tokens, 512-token prompts, 128 new tokens, temperature 0):

| Run | Acceptance 4 / 6 streams, cache on | cache off |
|---|---|---|
| 35B, our build (rad0002 hand-off compiled in, no worker) | 0.02-0.07 / 0.02-0.04 | 0.58-0.62 / 0.55-0.60 |
| 35B, near-native (lab source: rad0005 compile fix only; the libr3 component still built, whether it is loaded by default is unchecked) | 0.02-0.07 / 0.02-0.04 | 0.55-0.62 / 0.55-0.60 |
| Qwen3.8-27B MXFP4 dense, one R9700 | 0.70-0.75 / 0.69-0.73 | 0.70-0.75 / 0.69-0.73 |
| Qwen3.8-Flash-Next, two XTX (tp 2) | 0.33-0.81 / 0.36-0.45, mild decline, no collapse | not measured (server died on first request, no error logged; host was near its memory limit) |

Not yet tried: the 35B at `--tp 2` on the two XTX with the cache on (separates the model from the single-card gfx1201 path).

**Ruled out.**
- The expert mover: promotions frozen after 400 heat steps (rad0006, `RAD_DIAG_FREEZE_PROMOTION_AFTER`) did not help.
  `--deterministic` cannot start on this card (the pinned plan leaves the KV cache short by 263 MiB at 6 sequences and
  281 MiB at 4, with 32 MiB headroom and smaller batched tokens).
- Our workers and the hand-off (collapse is engine-alone, and with the workers computing every expert it did not occur).
- Checkpoint-slot shortage: 36 slots for 6 sequences (6 each, interval 4096), no warning.
- Speculation depth (same collapse at 1, 2, 3 tokens).
- Visible stream mixing: the first 400 characters of every reply are coherent and on-topic at every stream count.

**Other observations.**
- Speculation off: per-stream reply hashes are identical across repeats at 1, 2 and 4 streams and differ at 6; with 3 draft
  tokens they differ from 4 streams. The batched engine is not bit-reproducible above a batch threshold (spec §19 says as much).
- Isolation probe (5 prompts solo, then in groups of 2-5, forward and reversed): solo replies repeat exactly; every batched
  reply diverges from its solo reply at a fixed character position per prompt (about 90-390 characters) from group size 2,
  with the cache on or off. That is ordinary batch numerics, not the collapse: with the cache off acceptance stays
  0.57-0.67 in every group. With the cache on, acceptance in the probe was 0.02-0.11 in most groups and depended on the
  order of the streams (0.40-0.50 in some reversed groups).
- The prompts are 512 tokens against a 4096-token checkpoint interval, so no linear-state snapshot can be reused, yet the
  cache still hurts. The mechanism is unknown; the attention-block half of the prefix cache is the suspect.
- The first 6-stream round with the cache off reads 0.22-0.24 in every run (warm-up), then 0.55-0.60.
- Docs: the radiance README and GUIDE state no concurrency limit (every model serves `--max-num-seqs` up to 32; the README
  lists `--max-num-seqs 32` for Qwen3.6-35B-A3B at `--tp 2`). No upstream issue found; none filed (owner decides).

**Run specifics.** Engine alone: `--kernels libr4d,libr3moe,libref --placement expert_tiered --max-num-seqs 6
--max-num-batched-tokens 4096 --checkpoint-interval 4096 --kv-cache-dtype fp8 --num-speculative-tokens 3 --tp 1
--host-pool-mib 30720 --max-model-len 32768`, `HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=2` (the R9700, BDF 0000:09:00.0).
Each config is two identical arms x 2 pairs (ab-benchmark needs complete permutation blocks) x 2 reps per stream count.
Configs on branch `feat/rad14-mtp-bisect`, `tools/lab/radiance/`: `qwen35b-engine-mtp{1,2,3-det,3-freeze,0,3-nocache}.json`,
`qwen35b-engine-isolation{,-stagger,-nocache}.json`, `qwen35b-native-mtp3-{cache,nocache}.json`,
`qwen27b-engine-mtp3-{cache,nocache}.json`, `flashnext-engine-mtp3-{cache,nocache}.json`. Probe: `bigcherry/bench/stream_isolation.py`
(`stream_bench.isolation` in the config; metrics `iso_*`). The staggered probe (streams started 1.5 s apart) is written but unrun.
Builds on Brutus: diagnostic `radiance-diag:moe-diag:linux-radiance` (rad0006 included, plan 058e639b747b85e5177bdab8b38f6ba3),
near-native `radiance-lab:stock:linux-radiance` (plan 89ad5d838751abe75bee0dce4dad3ca0); a stock radiance 1.8.0 build fails on
Brutus (FFmpeg 6, `core/mm/mp4.cpp`, fixed by rad0005) and clang segfaulted once on a libr3 kernel (clean on retry).
Run record dirs `q35h-1020-*` under the work root. Flash-Next's engine-alone start showed one GPU hang (`aql_fill1` never ran,
host at 86 of 91 GiB) after a forced kill; a clean stop and retry loaded fine.

## Worker (FP8 expert helper)

See RAD13 and the worker notes in the lab log: the FP8 worker is accurate, hand-off cost is small with partial ownership,
prefill belongs on the R9700, and workers beat the engine alone only at 1 stream with MTP on.

# Qwen3.6-35B-A3B FP8 (MoE) on radiance (dev)

Current-state page. Rule: every measurement or decision updates this page in the same change. Overview:
[../README.md](../README.md). Plan items: RAD13 (FP8 worker), RAD14 (shape-aware geometry and the MTP bisection).

## Container

`${BC_MODEL_ROOT}/radiance/StillDeadcode/qwen3.6-35b-a3b-fp8/qwen3.6-35b-a3b-fp8.rad`: 34.6 GiB, block fp8 (E4M3 weights,
bf16 128x128 scales), E4M3 activations with an f32 scale per 128 of K. 256 experts, top-8 (read from the container's
`n_expert_used`). Hybrid: gated delta-net (linear state) plus full attention; an MTP head is in the checkpoint.

## radiance 1.13.0 sanity (RAD32, PR #451, 2026-10-29)

R9700 alone, `--kernels libr4d,libr3moe,libref --placement expert_tiered --host-pool-mib 30720 --tp 1`, fp8 KV, one
stream, no drafter, engine-bench 7.2K prompt + 128 decode tokens, three repeats: loads, serves and stops cleanly; prefill
3,571 / 5,162 / 6,674 tok/s, decode 54.3 / 79.1 / 84.8 tok/s, rising over the repeats (warm-up, and a prefix-cache share
of the repeated prompt is likely). No v1.9.0 run of this setup exists, so it is a sanity check, not a comparison
(`rad32-sanity-10290400`). Without `--host-pool-mib` the 34.6 GiB container does not fit the R9700's 32 GiB.

## radiance 1.14.0 (RAD37, 2026-10-31): 1.13.0 cannot serve MTP depth 3 on this model, 1.14.0 can

R9700 alone, `--kernels libr4d,libr3moe,libref --placement expert_tiered --ngram-placement ram --host-pool-mib 30720 --tp 1`, fp8 KV,
`--max-num-seqs 6 --max-num-batched-tokens 4096 --num-speculative-tokens 3 --max-model-len 32768` (the `qwen35b-engine-ab.json`
settings). On 1.13.0 the first request fails with HTTP 500: a decode attention kernel (`attn_decode_h256_gqa8_fp8kv`) refuses the
step at 4 query tokens with "scratch buffer absent or too small" and the engine stops serving, so the benchmark harness reported a
stream that "produced no content". On 1.14.0 the same flags serve: decode, median of 3 requests of 384 tokens at temperature 0
(`ignore_eos`), two runs: count 218.3 / 218.5 tok/s, story 170.9 / 170.8, code 212.1 / 212.2 (`/work/bigcherry/runs/rad37-g35-*`,
direct script in the RAD37 notes). The cause on 1.13.0 was not isolated (the 1.13.0 sanity above ran without a drafter, so it never
reached this path); the 1.14.0 release changed the decode attention scratch handling.

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
Builds on dev: diagnostic `radiance-diag:moe-diag:linux-radiance` (rad0006 included, plan 058e639b747b85e5177bdab8b38f6ba3),
near-native `radiance-lab:stock:linux-radiance` (plan 89ad5d838751abe75bee0dce4dad3ca0); a stock radiance 1.8.0 build fails on
dev (FFmpeg 6, `core/mm/mp4.cpp`, fixed by rad0005) and clang segfaulted once on a libr3 kernel (clean on retry).
Run record dirs `q35h-1020-*` under the work root. Flash-Next's engine-alone start showed one GPU hang (`aql_fill1` never ran,
host at 86 of 91 GiB) after a forced kill; a clean stop and retry loaded fine.

## Worker decode kernel (RAD14, 2026-10-20)

**Real decode requests.** 120 captured worker requests (4 streams, MTP 3, `R3_MOE_WORKER_CAPTURE`, config `qwen35b-helper-capture-decode.json`,
captures on dev in `${BC_WORK_ROOT}/caps-q35-decode`): every request is 16 rows (4 streams x 4 verify rows); the worker owns about 36 of
the 128 routes, on about 12 distinct experts: **5.1 routes per touched expert** (1 route 12%, 2 routes 8%, 3 routes 10%, 4 routes 36%, 8 routes 11%).
The 4-row groups are the verify positions of one stream, which route alike. Flash-Next decode requests, by contrast, are 1.0 route per touched expert at
every size (4 to 24 rows), so multi-row reuse matters for this model and not for Flash-Next.

**Kernel time per request** (`tools/lab/radiance/replay-kernel-time.sh`, one worker process replaying the captures under rocprofv3, device kernels only;
SMALL_ROWS=8, so 16-row requests take the large lane):

| Card | Kernel | gate/up | down |
|---|---|---|---|
| RX 7900 XTX (gfx1100) | `lane_moe_gemv<FP8>` (default) | 301 us | 149 us |
| RX 7900 XTX | `lane_moe_rows<FP8>` (`R3_MOE_WORKER_ROWS=1`, new) | 286 us | 183 us |
| R9700 (gfx1201, lower bandwidth) | `lane_moe_gemv<FP8>` | 235 us | 131 us |

The other lane kernels total about 50 us a request; live worker compute is 560 us a request (7,273 requests in a 4-stream run), which matches the
kernel time, so orchestration adds little. About 36 MiB of weights are touched a request: 45 us at 800 GB/s. The kernels are 8 to 10 times off that,
about 85 GB/s effective, and the R9700 (less bandwidth) is faster than the XTX, so the limit is latency and synchronisation, not bandwidth or
RDNA3-specific overhead. Porting the W4 few-rows structure to FP8 (decode each weight once, apply to every row of the expert) did not help.
Redesign question with these numbers sent to GPT (session "Kernel redesign").

## Worker (FP8 expert helper)

See RAD13 and the worker notes in the lab log: the FP8 worker is accurate, hand-off cost is small with partial ownership,
prefill belongs on the R9700, and workers beat the engine alone only at 1 stream with MTP on.

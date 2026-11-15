# VTP41: prefill order by remaining work and an adjustable long-prefill threshold (rad0035)

Hardware A/B, `qwen27b-tp3-int8-556-long-bf16` (624k-token bf16 cache), served argv with every arm on the shipped
`-long` scheduling (`--max-prefill-seqs 2 --prefill-interleave 2`); one fresh engine start per run, order SRTUUTRS,
two starts an arm.

Workload (`mixed.py`): eight distinct ~100k-token prompts sent together (799,760 tokens), then two waves, 30 s and
150 s in, each of one ~450, ~4k, ~8k and ~16k-token request (32 tokens out). Every request returned 200 in every run;
no preemptions in any run.

## First-token time

| arm | ~450 | ~4k | ~8k | ~16k | 100k mean | first / last 100k |
|---|---|---|---|---|---|---|
| S shipped | 5.9-6.4 s | 317-388 s | 323-402 s | 327-433 s | 270 s | 81 / 466 s |
| R `--prefill-order remaining` | 4.2-5.2 s | 53-54 s | 60-62 s | 70-73 s | 288 s (+6%) | 82 / 471 s |
| T `--long-prefill-tokens 16384` | 5.2-6.1 s | 17-18 s | 31-39 s | 68-76 s | 289 s (+7%) | 94 / 466 s |
| U both | 2.3-5.4 s | 4.3-7.6 s | 11-15 s | 20-27 s | 298 s (+10%) | 100 / 477 s |

Ranges are both waves over both starts; repeat starts agree within 1 s.

## Reading

- **Shipped:** a 4k-16k request counts as long (more than one 2048-token step) and queues in arrival order behind all
  eight 100k prompts: five to seven minutes. VTP39's short-request probes were ~450 tokens and missed this.
- **Remaining order:** mid-size requests are admitted first when a cap slot opens, so they wait about one long
  prefill (~50-70 s).
- **Threshold 16k:** requests up to 16k bypass the cap and start at once, but in arrival order they are walked behind
  the running long prefills and get a share of each step; a 16k request still takes ~70 s.
- **Both:** bypass plus least-left-first walking; a 16k request reaches its first token in 20-27 s, a 4k in 4-8 s.
  The 100k prompts pay 10% on the mean (first +19 s, last +2%) for the ~60k tokens of mid-size work done ahead of them.
- Not measured here: `--prefill-age-ms` (card-free test only), fp8 `-long`, decode beside the mix.

## Age limit on the card (2026-11-16)

Same workload and engine; order VUUV, U = both options as above, V = U + `--prefill-age-ms 120000`.

| arm | ~450 | ~4k | ~8k | ~16k | 100k mean |
|---|---|---|---|---|---|
| U (30 s / 150 s wave) | 5 / 2-5 s | 7 / 4-7 s | 14 / 11-13 s | 27-28 / 20-23 s | 299-300 s |
| V (30 s / 150 s wave) | 4-5 / 3-10 s | 6-7 / 43-69 s | 13-14 / 128-151 s | 26-27 / 139-160 s | 296-309 s |

Under rad0035 the age applied to running prefills as well: by 150 s every 100k prompt was older than 120 s, went back
to arrival order and was walked ahead of the second wave, which then waited one to two and a half minutes. The 100k
prompts gained nothing. A running prefill already keeps a share of every step and cannot starve, so rad0036 (VTP42)
ages only prompts still waiting to start; see `engines/radiance/patches/rad0036_prefill_governor/README.md`.

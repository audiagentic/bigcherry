# 1337_moe_expert_caching

**Status:** rejected
**Plan item:** MET01

Kind: upstream backport (llama.cpp PR #29887, commit 6b7b03aab, not merged at the b11402 pin). Requires 1336.

With `--moe-cache-mib N` (`LLAMA_ARG_MOE_CACHE_MIB`), MUL_MAT_ID ops whose expert weights are kept in host memory
(`--n-cpu-moe`) run on the GPU against a persistent device buffer: the experts a batch selects are uploaded on a miss
into least-recently-used cache slots, and the op reads a remapped ids tensor naming the slots. Without the flag
nothing changes. Upstream limits apply: one device, no pipeline parallelism.

The patch is the upstream commit hunk for hunk, with the seven hunks that overlap 1336 anchored on 1336's form, and
creates `src/llama-moe-cache.cpp` and `.h`.

The published PR does not sit on the #29943 copy callback: it adds its own scheduler hooks and rewrites the graph at
split time, which a copy callback cannot do.

Superseded when the pin reaches a llama.cpp release that contains #29887.

## Rejection evidence

Brutus, 2026-10-08, b11474, build `moe-cache-profile`, `queue-moe-cache.sh mig`, NCMOE=41, one R9700 target + MTP sidecar:

| arm | short decode t/s (3 requests) | long prefill t/s | long decode t/s | greedy md5 short / long |
|---|---|---|---|---|
| C0 (no cache) | 23.66, 23.90, 24.05 | 89.2 | 18.86 | 5e3cfde77817 / 50154e2cebe4 |
| C4096 (`--moe-cache-mib 4096`) | 11.01, 11.01, 10.96 | 89.3 | 7.91 | 4a8ba04d28e8 / 00758459b1cb |
| R (profile record) | 11.03, 11.01, 11.05 | 92.1 | 8.15 | 4a8ba04d28e8 / 4f222b4dfef1 |
| P4096 (cache + profile) | 8.57, 8.60, 8.75 | 95.9 | 5.27 | 4a8ba04d28e8 / 4f222b4dfef1 |
| C0b (no cache, repeat) | 23.75, 24.05, 24.01 | 89.6 | 18.75 | 5e3cfde77817 / 50154e2cebe4 |

**Rejected:** 4096 MiB cache cuts short decode from ~24.0 to ~11.0 t/s; profile mode falls to ~8.6 t/s. Prefill improves at most ~7%, generated text changes, and C0/C0b controls agree. This path is not eligible for promotion on the qualified pin/topology.

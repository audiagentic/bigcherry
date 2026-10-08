# 1338_moe_cache_profile

**Status:** rejected
**Plan item:** MET01/MET07

Kind: enhancement, variables `BIGCHERRY_MOE_CACHE_PROFILE`, `BIGCHERRY_MOE_CACHE_PIN_PCT` (85),
`BIGCHERRY_MOE_CACHE_LARGE` (1), `BIGCHERRY_MOE_CACHE_PROFILE_OUT`. Requires 1337 (`--moe-cache-mib`).

Frequency residency for host-resident MoE experts, borrowed from Strata. Routing is strongly skewed: a frequency
ranking with half of all (layer, expert) pairs resident covers 91-97% of the routed pairs of an unseen request,
where whole layers (`--n-cpu-moe`) cover 50%. 1337's cache is a pure LRU and serves only ubatches of up to 32
tokens, so prefill still uploads every selected expert of every host layer per micro-batch.

With a profile (Strata's `STRP` format: pairs ranked by routing frequency) the top pairs are uploaded at start-up
into pinned slots that are never evicted; the remaining slots stay an LRU. Because a prefill sweep can no longer
evict the hot set, large batches use the cache as well: hits are computed from resident experts and only the
misses are uploaded, into the LRU tail. A batch is only given to the cache when its worst case fits the unpinned slots of
its layer group; otherwise it takes 1336's selective upload as before. `BIGCHERRY_MOE_CACHE_PROFILE_OUT` writes a profile from the routing the cache sees.

Nothing changes without the variables.

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

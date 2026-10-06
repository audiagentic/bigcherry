# 1338_moe_cache_profile

**Status:** untested
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

## Evidence

- Offline mechanics test: `tools/tests/patch/test_1338_moe_cache_profile.py`. Hardware: pending.

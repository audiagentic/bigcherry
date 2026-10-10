# 1338_moe_cache_profile

**Status:** untested
**Plan item:** MET01/MET07

Kind: local enhancement on 1337. Variables: `BIGCHERRY_MOE_CACHE_PROFILE`,
`BIGCHERRY_MOE_CACHE_PIN_PCT` (default 85), `BIGCHERRY_MOE_CACHE_LARGE`,
`BIGCHERRY_MOE_CACHE_PROFILE_OUT`.

A Strata-format profile pins hot (layer, expert) pairs into slots that are never evicted; the remaining slots stay
LRU. With a pinned set, large batches may use the cache when their non-pinned working set fits the remaining slots.
Without the profile variables, 1338 is inert and 1337 keeps its <=32-token eligibility rule.

The earlier NCMOE=41 / 4096 MiB result is retained as historical evidence, not a rejection. Its output divergence
must first be separated into (a) the compact cache-bank/slot-ID execution path, including fusion/kernel eligibility,
from (b) an actual mapping/copy defect. Its throughput also needs hit/miss/upload evidence rather than inference from
one cache size.

Use the restored `moe-cache-profile` experiment for 1337+1338 and `moe-expert-cache` for 1337 alone. Hardware
qualification must respect upstream's one-target-device/no-pipeline limits.

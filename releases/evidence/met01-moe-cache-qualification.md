# MET01 MoE cache qualification — rejection

Brutus, 2026-10-08, b11474, build `moe-cache-profile`, `queue-moe-cache.sh mig`, NCMOE=41, one R9700 target + MTP sidecar.

| arm | short decode t/s (3 requests) | long prefill t/s | long decode t/s | greedy md5 short / long |
|---|---|---|---|---|
| C0 (no cache) | 23.66, 23.90, 24.05 | 89.2 | 18.86 | 5e3cfde77817 / 50154e2cebe4 |
| C4096 (`--moe-cache-mib 4096`) | 11.01, 11.01, 10.96 | 89.3 | 7.91 | 4a8ba04d28e8 / 00758459b1cb |
| R (profile record) | 11.03, 11.01, 11.05 | 92.1 | 8.15 | 4a8ba04d28e8 / 4f222b4dfef1 |
| P4096 (cache + profile) | 8.57, 8.60, 8.75 | 95.9 | 5.27 | 4a8ba04d28e8 / 4f222b4dfef1 |
| C0b (no cache, repeat) | 23.75, 24.05, 24.01 | 89.6 | 18.75 | 5e3cfde77817 / 50154e2cebe4 |

## Verdict

Reject `1337_moe_expert_caching` and `1338_moe_cache_profile` on this pin/topology. The 4096 MiB cache halves short decode (~24.0 -> ~11.0 t/s); profile mode is worse (~8.6 t/s). Prefill gains at most ~7%, generated text changes, and C0/C0b controls agree.

## Verified / not verified

Verified from owner-supplied hardware results: build/launch, activation of cache/profile arms, repeated controls, throughput table, and greedy md5 divergence.

Not verified here: no local ROCm hardware execution was available in the connector environment; this commit records the supplied Brutus result and lifecycle decision.

# Flash-Next MoE cache recheck

Plan item: MET01
Status: active
Owner: BigCherry
Question state: open

## Question

On one target GPU with host-resident routed experts, is 1337 byte-identical to the no-cache path when fusion is
disabled, and how do hit rate / uploaded bytes / throughput change with host-layer count and cache size? After 1337
is correct, 1338's profile policy can be evaluated separately.

## Primary matrix

`tools/lab/flash-next/queue-moe-cache.sh <tag>` builds `moe-cache-profile` once and runs:

- `NCMOE={8,20,41}`
- `--moe-cache-mib={0,2048,8192}`
- 3 repeats
- short decode request and long-prompt request
- `GGML_CUDA_DISABLE_FUSION=1` on every comparison arm
- one request per server process, so the trace-gated 1337 shutdown counters are request-scoped

The binary includes 1337+1338; no 1338 profile variable is set in the primary matrix. Use the existing
`ARMS=profile` / `ARMS=record` lanes in `moe-copy-ab.sh` for the profile follow-up once 1337 identity is clean.

## Outputs / gates

Generated outputs are under `/mnt/data/bigcherry-work/runs/moe-cache-<tag>/ncmoe-<N>/`.
For each request the runner prints greedy md5, prompt/decode t/s, MTP acceptance and the
`BIGCHERRY_PATCH_HIT patch=1337_moe_expert_caching ...` hit/miss/upload marker.

Correctness gate: for a fixed request and NCMOE, all cache sizes/repeats must have the same greedy md5 with fusion
disabled. Do not interpret speed until this passes. Performance diagnosis uses decode t/s together with hit rate and
uploaded MiB, not cache size alone.

## Limits

One target device; no pipeline parallelism. The optional MTP sidecar is a separate context/device and the cache is
disabled for the draft context. No canonical-state mutation.

# Flash-Next MoE cache qualification

Plan item: MET01
Status: active
Owner: BigCherry
Question state: open

## Question

On the production Flash-Next workload with routed experts kept in host memory, does a 4096 MiB expert cache improve end-to-end prefill/decode without changing greedy output, and does a frequency profile improve the same cache while preserving MTP acceptance?

## Inputs

- `tools/lab/flash-next/queue-moe-cache.sh <tag>`
- Experiment: `moe-cache-profile` (1337 + 1338 on top of production).
- Target: R9700 by default (`GPU=2`) with the MTP sidecar enabled.
- `NCMOE=N` controls `--n-cpu-moe N` (default 41).
- Fixed cache comparison: 0 MiB versus 4096 MiB.
- `PROFILE=/path/to/profile.bin` supplies a held-out STRP profile. If omitted, the R arm records `profile.bin` from the same request sequence before P4096; label that result in-sample.

Arms: C0 (no cache), C4096 (LRU cache), R (profile recorder only when needed), P4096 (4096 MiB + `BIGCHERRY_MOE_CACHE_PROFILE`), C0b (repeat control).

## Outputs

Generated outputs go under `/mnt/data/bigcherry-work/runs/moe-cache-<tag>/` on Brutus. The queue log reports per request/arm prefill t/s, decode t/s, `accepted/draft`, and greedy md5. Server logs retain 1337/1338 cache/profile activation counters.

## Runtime

GPU required: yes
Real compilation required: yes, unless `RUN_OVERRIDE` names an existing `moe-cache-profile` build
Mutates canonical BigCherry state: no

## Safety

- Canonical-state mutation: none.
- C0/C0b must bracket the cache/profile arms; treat drift or md5 mismatch as a failed correctness gate, not a speed result.
- A generated in-sample profile is diagnostic only. Promotion evidence requires a held-out profile representative of both prefill and decode.
- Stop on `SERVER_FAILED` / `BUILD_FAILED`; the queue serializes GPU use and the runner shuts each server down before the next arm.

## Disposition

When complete, record compact decision-grade evidence under `releases/evidence/` or the owning patch/plan and either delete this active driver when MET01 closes or retain only if it becomes a reusable qualification harness. Promotion is a separate slice.

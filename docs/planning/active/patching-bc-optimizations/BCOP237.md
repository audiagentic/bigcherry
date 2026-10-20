---
id: BCOP237
order: 237
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-15T17:08:00Z'
created-by: agent
priority: P2
work: S
---

# QFP28/1295: executed QSA gather witness before performance promotion

**Discovery:** 1295's static `BIGCHERRY_PATCH_HIT` is emitted before its `plain` K/V predicate: it can mark a declined route and miss later true routes. The pinned QSA `n_kv` sentinel is explicitly clamped and `-inf`-masked, so external DEVGATHER's uninitialized destination is not established here. `determinism.sh` prints repeated-text differences but does not fail on them. No new GPU correctness failure or speedup was observed.

**Owners/consolidation:** QFP28 owns the same-process/multi-ubatch/cache/abort integrity gate; QFP04/1295 owns route-witness correction and typed-gather decisions; QFP17/1332 and QFP25/1334 retain separate QSA work. MET01/MET07 retain host-expert cache policy. Reuse `long-ctx-profile.sh`, `determinism.sh`, `kqv_out_qsa_gather` and existing HIP/graph profiling. No duplicate gather, cache, scheduler, allocator, dispatcher or telemetry.

**Eligibility/novelty:** Independent selected `main` HEAD `08441aa4a69e` (MEN10/RAD09, 2026-10-15 05:21 UTC). QFP28 last independent `6c008f29cae9` (Oct 5); QFP04 last `63cc952f5d` (Oct 15 00:20 UTC); 1295 patch last relocated Oct 9. No relevant independent work, PR or queued experiment within 12h. MEN10/RAD09/RHP work protected. BCOP227-236 address other capabilities; BCOP234's SSM marker is a different implementation seam.

**Upstream/fork:** llama.cpp #29887 merged and pinned, with a zeroed `n_slots+1` cache bank; BigCherry 1337 is superseded. llama.cpp #29949 / lich-77 `c81901a57` adds graph-resident LRU_DECIDE/LRU_FILL for pinned host experts; vLLM #38256/#57794 discuss expert mapping/cache policy. None is a current all-GPU QSA port. External gains are not AMD BigCherry measurements.

**Action:** Host-prove false marker and sentinel/pad behavior; correlate selected 1295 graphs with actual per-request GET_ROWS+FA execution using existing tracing, and add a fail-closed repeated-request comparator. Reject injected response-2 corruption. Only then current-pin 8K/80K/160K f16-KV no-MTP/MTP A/B on gfx1100/gfx1201. Reuse QFP04's >=5% GPU-share or >=0.5 ms/token admission; promote only with numerical/multi-request correctness, >=3% exposed end-to-end gain and <=1% control regression.

**Terminal:** `MARKER_ONLY_UNTRUSTED`, `ROUTE_UNPROVEN`, `MULTIREQUEST_FAIL`, `NUMERIC_FAIL`, `NO_EXPOSED_GAIN`, or `PROMOTABLE`. Do not queue hardware on a failed host/route gate.

**Evidence/validation:** Historical 1295 +14% decode at 98K and +13% at 196K (prior pin); QFP04 f16-KV 80K neutral and 160K MTP acceptance-confounded. This run: 14/14 source predicates, 10/10 host-model tests, Python compilation passed. No repository pytest, HIP build, GPU execution or new benchmark.

---
id: BCOP213
order: 213
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-14T16:13:25Z'
created-by: agent
priority: P2
work: S
---

# QFP15: trace pre-draft target seed before assigning MTP nondeterminism

**Finding:** 1315 logs a draft step without its pre-draft target seed and records target pending_h only after accept-row selection. First-step divergence is causally ambiguous. Its FNV offset is nonstandard. The historical "stock-none" control retained BigCherry overlays. No reproduced GPU fault or new speedup.

**Owners:** QFP15 is authoritative for provenance and benchmark hygiene; 1315 owns the existing optional trace. 1291/1294/1303/1321/1359, QFP07/QFP13 and FMTP retain their implementation paths. No new sampler, scheduler, cache, profiler or hardware queue. Active RAD02/RAD03/RAD04 work untouched.

**12-hour/novelty:** selected feat/rad03-gpu-lanes from independent 2e64d47eba0965cadd966cc2277f1fe1cc657cb8 (2026-10-14 15:52 UTC). Last independent QFP15 change ff6c1e0a2a58e47d4d6b07171631562039299d2f (2026-10-12 03:41 UTC), 1315 last relocation 2026-10-09. No related independent PR/queued lane in 12 hours. Recent BCOP203-212 cover other capabilities.

**Upstream:** llama.cpp #30257 already provides MTP state propagation at b11552; #30459 (merged 14 October) adds independent draft n_ubatch, not a determinism fix. vLLM per-position acceptance accounting is a measurement reference only.

**Action:** receipt-first identity check, then only if needed extend 1315 with pre-draft seed, selected accept-row index, request/round and physical draft GPU identity. Compare gfx1030 vs gfx1100 only with fixed target and idle hardware. Do not implement kernel changes without first divergent producer.

**Terminal:** UNATTRIBUTABLE, TARGET_SEED_DIVERGES, STABLE_SEED_DRAFT_DIVERGES, ACCEPT_ROW_SELECTION, UPSTREAM_REPRO or BENCHMARK_NOISE_ONLY.

**Validation:** 4/4 host-only synthetic assertions and Python compilation passed. No repository pytest, HIP build, GPU test or new performance measurement.

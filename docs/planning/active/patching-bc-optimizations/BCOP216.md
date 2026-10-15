---
id: BCOP216
order: 216
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-15T00:00:00Z'
created-by: agent
priority: P2
work: S
---

# MET12/MET13: retire duplicate host-expert prefetch; require exposed-copy proof

**Discovery/change:** MET13 independently requests the same Strata-style layer-ahead expert staging, ring and copy stream that MET12 already owns. Move MET13 to completed (`SUPERSEDED_DUPLICATE`) and keep the technical gate only in MET12. Pinned b11552 `ggml_backend_sched_copy_input` / `llama_context::sched_copy_experts` already implement exact **post-router** host-weight copies; historical 1336 is superseded. Exact same-layer IDs cannot be pre-staged before routing. Strata's >=1024-token stream-all mode copies unselected expert payload; MET10's historical unique coverage implies idealized +63.9% bytes at ub512 and +26.6% at ub2048 versus exact selection. These are theoretical transfer ratios, not new GPU measurements.

**Authoritative owners:** MET12 pre-router conditional prefetch; MET06 native scheduler/#29963 qualification; MET07 cache/profile/#30112; MET10 host-tail and expert-ID mapping; MET05 transport. No new scheduler, allocator, cache, copy callback, ring or hardware queue. Do not touch active RAD02–RAD05 helper/GEMV, RAD04 prefill or their patches/benchmarks.

**External mechanisms:** Strata `src/prefill/prefill.cpp` @ `296b59d2` implements event-fenced routed-only vs stream-all modes; llama.cpp #29963 is open (head `273cd26a`, 2026-10-13) and layer-split only; #30112 merged 2026-10-08 and supplies native MoE cache. SGLang OffloaderV2 is a lifetime reference. No external gain is transferred to AMD.

**12-hour/novelty:** selected `feat/rad03-gpu-lanes` from newest independent Radiance helper work; MET12 last changed 2026-10-11 11:17 UTC, MET13 2026-10-13 03:06 UTC, 1336 last relocated 2026-10-09. No independent same-capability commits, PR/submission, evidence or queued experiments were identified in the preceding 12h. Excluded recent RAD02/03/05 helper routing/mailbox/GEMV and RAD04 prefill, plus recent BCOP207–215 capabilities; no recent scheduled-audit duplication of MET12/13.

**Unresolved action:** use terminal current-pin host-expert receipts first; admit only measured >=3% removable exposed H2D wait on a genuine over-VRAM lane with positive host bytes, adequate PCIe/ring budget and immutable profile identity. Require event/generation fences, exact misses, all work accounted, repeated-request and full logits/MTP parity. Otherwise close `NO_HOST_RESIDENCY`, `NO_EXPOSED_WAIT`, `BELOW_3PCT_CEILING` or `LIFETIME_UNPROVEN`. If qualified, a single MET12 owner may implement the smallest opt-in mechanism, subject to >=3% CI95-low end-to-end PP and <=1% decode/control regression.

**Validation:** 10/10 host-only unittest cases passed for routing, miss, ring, generation, memory and exposed-wall admission. No repository pytest, patch application, HIP/C++ build, GPU execution or new throughput benchmark. No new measured inference gain.

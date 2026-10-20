---
id: BCOP223
order: 223
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-15T03:13:11Z'
created-by: agent
priority: P2
work: S
---

# FMTP07: qualify post-pin independent draft ubatch before memory or overlap claims

**Discovery/change:** llama.cpp [#30459](https://github.com/ggml-org/llama.cpp/pull/30459) merged 2026-10-14 10:58 UTC **after** BigCherry's b11552 pin. It introduces draft-only `--spec-draft-n-ubatch` (0 inherits target, positive overrides, negative rejected). b11552 inherits target ubatch for the draft context. This may reduce draft graph/workspace peak VRAM, **not** target ubatch, draft model weights or full-context KV state; no BigCherry gain is measured. The upstream token-equality smoke does not qualify native Flash-Next MTP rollback, multi-ubatch, gfx1030 or 1322/1348/1359 interactions.

**Owner/consolidation:** FMTP07 is the sole draft-ubatch admission/benchmark owner. Native upstream speculative params/context construction is the implementation; FMTP03/1322 owns overlap, QFP16 target submit, QFP50/1359 NextN prefill, QFP17 target/QSA ubatch, 1317/1318/1346 existing timing. No new patch, allocator, scheduler, placement selector, telemetry, experiment queue or neighbouring plan changes. Current RHP/MEN/RAD/RWQ/PGC15 work protected by the 12-hour exclusion remains untouched. FMTP07 last independent edit 2026-10-04; no same-path independent commit, PR, plan update, evidence or newly queued run found in the preceding 12 hours. Selected main at independent a1d5eeaffe5bb6e152629e7c52a85178c6972c25.

**Unresolved bounded action:** first inspect deployed pin, effective `llama_n_ubatch(ctx_dft)`, draft allocation/workspace peak and existing jobs. Only if memory pressure is real and #30459 is in an approved same-pin build, compare inherited draft ubatch with 32/64/128 on identical binaries and fixed target `-ub`, then qualify MTP state/graph/NextN/acceptance. Terminal `PIN_MISSING`, `NO_EXPOSED_DRAFT_HEADROOM`, `UNSAFE_DRAFT_CHUNK`, `STATE_OR_NUMERIC_FAIL`, `REGRESSION`, `MEMORY_FIT_OPT_IN`, or `QUALIFIED_OPT_IN`. Memory acceptance: OOM->fit or >=256 MiB measured draft peak saving with <=1% E2E regression; speed acceptance CI95-low >=3% E2E. Fallback: upstream inherited draft ubatch. Exact source, tests, architecture matrix and admission details remain in FMTP07.

**Validation:** 11/11 source assertions and 10/10 host-only override/chunk-partition model tests passed. No repository suite, C++/HIP build, GPU run, physical memory or performance measurement.

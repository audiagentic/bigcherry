---
id: BCOP225
order: 225
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-15T05:12:45Z'
created-by: agent
priority: P1
work: S
---

# PNRO04: reconcile validated 1253 GDN with post-pin upstream and unsupported GQA abort

**Discovery/change:** Existing 1253 BF16 chunked GDN is implemented, default-on for its eligible K=1 route and `validated` in `patch.toml` on b11126; PNRO04 and package README/TESTING retained contradictory predicate-only/untested/1221-requires claims. `patch.py` selects BF16 without prechecking GQA ratio; its KKT switch accepts {1,2,3,4,6,8} and aborts for other divisible ratios. Conditional source-level correctness risk, **not a reproduced production crash**. New upstream llama.cpp [#29353](https://github.com/ggml-org/llama.cpp/pull/29353) merged 2026-10-14 14:00Z, post-b11552: different single-block BF16 high/low MMA, HIP restricted to gfx115x RDNA3.5 and >=2048 tokens, not gfx1100/gfx1201. Its K>1 snapshot contract is weaker than untested 1254's exact tail/snapshot promise.

**Owners/consolidation:** PNRO04/1253 owns BF16 K=1 shape gate, dispatch and qualification. PNRO05/1254 owns K>1 prefix/tail/snapshot separately. RD50/1221 is conflicting, not a dependency; no duplicate GDN kernel, scheduler, allocator, scratch pool, telemetry or new patch. RHP/Radiance, MEN10, RWQ and PGC15 active work untouched.

**Selection/eligibility:** Selected `main` from independent `a1d5eeaffe5bb6e152629e7c52a85178c6972c25` (2026-10-15 02:58Z RHP planning). PNRO04 substantive plan last 2026-09-28; 1253 implementation last 2026-10-09 mechanical move. No independent GDN code/plan/PR/benchmark/queue activity found in prior 12h; recent BCOP219-224 address other mechanisms. Recheck immediately before publication.

**Evidence:** Historical b11126 final validated 1253 receipts: gfx1100 pp512 +9.31884% (CI95-low 8.4784%), gfx1201 +10.99672% (CI95-low 10.8858%), near-neutral tg128, CPU-reference subject 46/46 and control 36/36. This run: 13/13 source assertions and 20/20 host-only selector checks. No BigCherry repository suite, HIP build, GPU test or new benchmark.

**Unresolved action/terminal:** Existing-owner 1253 pre-launch ratio guard and b11552 composition test; reject unsupported shapes to sequential before allocation. At future upstream pin, compare #29353/stock/1253 with GDN K=1 state/logits/graph/VRAM and E2E, keep 1254 K>1 snapshots isolated. Terminal `KEEP_1253`, `UPSTREAM_SUPERSEDES`, `SOURCE_CONFLICT`, `CORRECTNESS_FAIL`, `NO_GAIN`, `HARDWARE_NOT_QUALIFIED`; promotion requires 4 sessions/arch, >=10 pairs/session, CI95-low >=3% pp and <=1% control regression. Technical algorithm and tests in PNRO04 and package TESTING.

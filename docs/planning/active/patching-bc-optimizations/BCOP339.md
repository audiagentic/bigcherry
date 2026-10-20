---
id: BCOP339
order: 339
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-20'
created-by: agent
priority: P3
work: S
---

# QFP10: reject direct gfx1100 Q4_K MoE VDR4 port without source/quality/critical-path proof

**Discovery:** [RDNA fork #167](https://github.com/stew675/llama-cpp-rdna-boosts/pull/167) (merged 2026-10-17) reuses [#162](https://github.com/stew675/llama-cpp-rdna-boosts/pull/162)'s 32-element Q4_K VDR4 helper for MoE. External gfx1100 Q4_K expert kernels fall 10%/18%, but total GPU request time falls only 1.92%, decode rises ~1.4%, and greedy text changes. Gfx1201 opt-in is +0.7% and also changes text. BigCherry b11552 has neither the helper nor the fork's `mmvq-vecdot.cuh`; a one-line transplant is invalid.

**Owner/consolidation:** QFP10 is the authoritative typed MoE expert-service gate; PRBE20 owns cross-width bit-identity, PRBE111/1273 IQ, PRBE47 Q6_K, PRBE21 already-native MoE rpb2, QFP44 MMQ and QFP37 Stream-K are untouched. No new technical plan, kernel, dispatcher, allocator, cache, telemetry, flag, GPU queue or duplicate owner.

**Eligibility:** `feat/rad14-mtp-bisect` selected from independent Audumla `c4061b723042` (2026-10-20 02:08:32 UTC), following earlier independent `6ec837364822` (01:53:34 UTC); active RAD14 MTP isolation/bisect, RAD13 FP8 worker/GEMM and RCS09/RCS05 helper geometry are protected. QFP10 last independent edit 2026-10-05 04:56:52 UTC; no same-path Q4_K MoE VDR4 commit, PR, patch, plan edit or queued experiment found in preceding 12 h in inspected repository. Off-repository queues unverified. This is a distinct Q4_K MoE integer-accumulation mechanism, not BCOP332 MMQ k01, BCOP335 Q6_K MMVQ or recent IQ MMVQ review.

**Completed:** inspected pinned b11552 `mmvq.cu`/ `vecdotq.cuh` Q4_K selector, Q8_1 conversion, dedicated MoE launch/width cap, existing 0600/1273 patch scope, #162/#167 diffs and measured controls, vLLM/SGLang GGUF mechanism boundaries. Host-only 50,000 signed-dot equivalence cases passed; 9,198 F32 grouping differences observed. No HIP build, GPU run, benchmark or patch created.

**Unresolved action/terminal:** `HOLD_NO_PORT` until a repository-attested gfx1100 Q4_K MoE width 2–4 route, >=2.9126% removable *exposed* E2E wall and strict numerical contract are all proven without touching active RAD14 lanes. Otherwise close `NO_Q4K_MOE_MMVQ_ROUTE`, `BELOW_CEILING` or `NUMERIC_CONTRACT_FAIL`. A surviving pinned-source prototype requires complete K/work/alias/graph proof, ISA/VGPR/spill qualification and CI95-low >=3% E2E with <=1% controls; else `NO_GAIN`/`REGRESSION`. Technical detail only in QFP10.

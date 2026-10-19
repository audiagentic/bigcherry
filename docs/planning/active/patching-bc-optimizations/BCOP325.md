---
id: BCOP325
order: 325
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-19'
created-by: agent
priority: P1
work: S
---

# RNX02: RDNA4 Q4_0 GQA vector-FA reuse admission

**Discovery (2026-10-19):** New [llama.cpp #30803](https://github.com/ggml-org/llama.cpp/pull/30803) assigns a packed Q4_0 K block to each lane and processes 2/4/6/8 query heads per KV head in one HIP RDNA4 VEC block. External R9700 D256 Q4_0 decode at 65K reports 22.03 -> 26.76 t/s (+21.5%), and at 131K 17.81 -> 25.17 (+41.3%). This is not measured BigCherry gain. It differs materially from BCOP320 Q8 native TILE and the rejected 1298 multi-query selector override.

**Authoritative owner:** RNX02 (D256 quantized KV attention); 1300 is a default-off Q8-only selector and does not implement Q4 GQA sharing. No second kernel, dispatcher, allocator, config or telemetry owner. QFP04 sparse QSA, PRBE02 WMMA, QFP07 placement and BCOP320 Q8 TILE stay separate.

**Eligibility:** Selected independent branch feat/rcs09-07-width, Audumla head 985a6a60 (2026-10-19 11:09:31 UTC). Repository-visible 12-hour check and novelty gate passed for Q4 GQA VEC; no matching independent BigCherry work or queued evidence in the window. Protected RCS09/RCS05, RAD11/12/10, WSL and release work excluded. Off-repository queues not verifiable.

**Cheap test:** Host-only Q4 signed-nibble dot 65,536/65,536 integer-exact; 4,320 GQA head tuples unique/complete; 15 route/fallback assertions; Python compilation passed. Host F32 grouping produced 41,211 bit-different samples (max abs 0.00292969), so do not claim GPU bit identity. No HIP build or hardware benchmark.

**Unresolved action:** Gate 1: prove a real gfx1201 Q4_0 K+V D256 GQA VEC route on a physical rank and >=2.9126% removable exposed E2E latency. If absent, close NO_RDNA4_KV_ROUTE / BELOW_CEILING. Gate 2: prefer #30803 upstream merge/pin absorption, no premature local port. Gate 3: full head/KV/output work, logits/KLD/MTP, graph replay, ISA VGPR/spill/occupancy. Gate 4: paired CI95-low >=3% E2E improvement without >1% control regression. Otherwise close NO_GAIN / REGRESSION / NUMERIC_FAIL. Technical design and matrix are in RNX02, not duplicated here.

**No queued work or implementation changes.** Preserve the current recurring audit and all active developer lanes.

---
id: BCOP338
order: 338
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-20'
created-by: agent
priority: P2
work: S
---

# QFP25: upstream HIP k-pool mask does not fix sparse-FA group-union capacity

**Discovery:** New open [llama.cpp #30815](https://github.com/ggml-org/llama.cpp/pull/30815) (updated 2026-10-20 00:33 UTC) adds a 512-thread-per-token `GGML_OP_KPOOL_MASK` replacing Qwen4Exp mask-building with one dense-output scatter. It retains `O(n_kv*n_query)` output traffic and passes the same per-query `n_sel` hint to FA. BigCherry 1334's eight-query union compactor still clips to that hint: this upstream prototype does not repair BCOP199's correctness boundary. Synthetic overlapping pool/tail fixtures reveal duplicate destination stores, not an observed production race.

**Authoritative owner/consolidation:** QFP25/1334 alone owns HIP sparse-QSA admission, compaction, graph and patch decisions. QFP17 consumes prefill cost; QFP28 owns broader fidelity; QFP04 decode gather. No new QSA op, dispatcher, allocator, cache, telemetry, patch, technical plan or hardware queue. SGLang/vLLM per-query index paths are semantic comparators only.

**Eligibility:** selected `feat/rad14-mtp-bisect` from independent Audumla `53ddf5da` (2026-10-20 01:07:54 UTC). Repository-visible prior 12h RAD14 MTP/FP8 bisect and RAD13 FP8 grouped GEMM protected; no QFP25/1334 independent work or queued experiment found. Off-repository queues unverified. #30815 is material new upstream source since BCOP199, not a repeat audit.

**Completed:** inspected pinned b11552 Qwen4Exp/FA dispatch, BigCherry 1334 HIP compactor and historical b11402 +13%/+28% prefill evidence, upstream #30815 changed source/tests, and SGLang/vLLM per-query mechanisms. Host-only oracle: **22,400** set/reference cases passed, **1,238** synthetic union-overflow groups, **4,605** synthetic duplicate-store rows, 8x512 => **3,584** lost; Python compilation passed. No GPU build, hardware benchmark or repository tests.

**Action/terminal:** first prove zero missing finite cells in every actual eight-query 1334 group at 24K/99K/202K, otherwise disable affected sparse route (`UNION_OVERFLOW_BLOCKED`) and correct before any speed claim. Only if correct and >=2.9126% removable exposed E2E latency exists, compare existing 1334 to #30815 and minimal direct-index adaptation with full graph/alias/physical-work, wave32/64 ISA/VGPR/spill, logits/KLD/MTP and paired CI95-low >=3% incremental E2E. Else `PIN_ABSORBED`, `NO_KPOOL_ROUTE`, `RACE_OR_NUMERIC_FAIL`, `BELOW_CEILING`, `NO_INCREMENTAL_GAIN` or `QUALIFIED_EXISTING_1334`. Do not race RAD14/RAD13 hardware work. Technical detail only in QFP25.

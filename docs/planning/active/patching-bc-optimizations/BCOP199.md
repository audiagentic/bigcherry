---
id: BCOP199
order: 199
plan: patching-bc-optimizations
state: done
created-at: '2026-10-14T02:11:59Z'
created-by: agent
priority: P1
work: S
---

# QFP25/1334: grouped sparse-FA union-capacity admission

**Discovery/change:** 1334 is already validated/default-on on historical b11402; a second HIP sparse op proposed in older QFP25 prose would duplicate it. Pinned b11552 Qwen4Exp supplies per-query `n_sel`, while the HIP 8-query mask compactor caps the union to that hint. This is a source-confirmed possible loss of finite-mask cells, not a reproduced GPU or real-model error. Five host-only cases proved the abstract truncation boundary; historical 18/18 backend-op tests and activation marker do not measure unclamped group cardinality.

**Owner/consolidation:** QFP25/1334 owns sparse prefill and the remaining correctness/current-pin gate. QFP04 owns decode gather, QFP17 consumes prefill performance, QFP28 fidelity. No new op, scheduler, cache, allocator or telemetry registry. The selected capability last received independent QFP25 work 2026-10-05 and 1334 code work 2026-10-12T01:49Z; no related independent work, submission or queued hardware run was visible in the preceding 12h. Active Radiance GPU mailbox/helper and engine patch-prefix/release work was excluded and untouched. Recent BCOP189-198 address other capabilities.

**Existing evidence:** Historical b11402 sparse prefill +~13% at 99K and +~28% at 202K, decode neutral; no new GPU measurement. Host-only discriminator 5/5: random 8-query/2051 cap union 13274, 11223 dropped, synthetic softmax output difference 0.273 maximum absolute. No C++ patch application, HIP build, GPU, logits or current-pin benchmark.

**External mechanism:** llama.cpp #29639/#30003 Vulkan per-row compaction and sparse quantized/cooperative-matrix gates; SGLang QSA and vLLM AMD sparse attention consume per-query index slots. Reference semantics only; no claimed transferable speedup.

**Unresolved action / terminal disposition:** QFP25 now specifies unclamped group-union counters, exact production-route receipts, per-query finite-set parity, adversarial fixtures, b11552 composition, multi-request/multi-ubatch/long-context GPU correctness and paired performance. On overflow, fail closed to dense before claiming speed and qualify one bounded correction under 1334; if none and full reference passes, close without a new implementation. Preserve validated historical status until evidence supports changing it.

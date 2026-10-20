---
id: BCOP349
order: 349
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-21'
created-by: agent
priority: P2
work: S
---

# QFP43: Qwen4Exp HC-contracted DFlash tap

**Discovery:** Pinned llama.cpp b11552 and current master expose raw [2560,4,T] residual as Qwen4Exp layer input before PLE and HC mix; extract_layer_inputs requires 2560 floats/token, matching the earlier PixelML Flash-Next DSpark assertion. The already computed contracted HC read is [2560,T], but layer-boundary semantic parity is not established. Five F32 taps cost 51,200 logical bytes/token per transfer direction, not measured traffic.

**Owner:** QFP43 target-feature semantics; reuse existing HC read, 1340 OUTPUT-set arena identity, and 1286 hash trace. QFP42/1348 retains MTP deferred lifetime; active RAD15 gfx1201 hc_read/MoE GEMV untouched. No duplicate kernel, scheduler, allocator, telemetry or experiment.

**Eligibility:** Independent branch feat/bc-gfx12-hc-read e5dafe5bf2f8 (2026-10-20 21:08:57 UTC); QFP43 last independent edit a200912123b2 (2026-10-10 01:11:40 UTC). Protected 12-hour work: RAD15/16, RAD06, RAD14, MEN24/release. No matching repository change or new QFP43 queue observed; off-repository queues unverified.

**Validation:** Pinned/current source-order checks passed; 300 host-only shape/PLE/row-packing cases and Python compile passed. No reference checkpoint, C++/HIP build, GPU benchmark or new queue.

**Disposition:** HOLD_SEMANTIC_ORACLE: prove five reference HC tap values and layer labels with a 1–2-token fixture before a Qwen4Exp-only graph-output correction. Then require multi-request, multi-ubatch, logits/KLD/greedy and graph lifetime parity. Separate deferred transfer optimisation only if exposed E2E ceiling >=2.9126% and paired CI95-low >=3%; else NO_SEMANTIC_ORACLE, TAP_MISMATCH, BELOW_CEILING or NO_GAIN. Technical design belongs in QFP43.

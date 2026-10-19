---
id: BCOP331
order: 331
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-20'
created-by: agent
priority: P3
work: S
---

# RNX08: HIP Q/K norm+RoPE to FWHT fusion gate

**Discovery:** b11552 already fuses RMS_NORM→MUL→ROPE. [RDNA fork #172](https://github.com/stew675/llama-cpp-rdna-boosts/pull/172) adds 64/128/256-wide FWHT in the same 256-thread CTA, removing a separate FWHT launch and F32 intermediate global write/read. Accepted downstream r78 on 17 Oct, not upstream ggml-org. Maintainer's combined two-patch gfx1201 Gemma4 gain +1.5%, not BigCherry or isolated. **Qwen3.5/3.6/3.8 IMRoPE cannot activate this NORMAL/NEOX matcher.**

**Authoritative owner:** RNX08 norm+RoPE specialization. Rejected 1004 remains provenance; PRBE15 owns IMRoPE, PRBE05/1307 Q8_1 cache and fork patch 0002, PRVP06 Vulkan wide FWHT, QFP13 launch census. No duplicate implementation, allocator, dispatcher, telemetry, config or queue.

**Eligibility/novelty:** Selected feat/rad13-fp8-worker from independent Audumla b0973b1187b2 (19 Oct 17:08 UTC), newer than main f792f51e. No independent same-capability BigCherry commit, PR, plan, benchmark or recorded queue in previous 12h. BCOP327/329 are scheduled audits of distinct paths. Protected RAD13/PR #330, RCS09/RCS05, RAD11/12/10, WSL and release untouched; off-repository queues unverified.

**Completed:** Inspected pinned ggml-cuda.cu/rope.cu/fwht.cu, Qwen4Exp graph, fork #172 patch and 17 Oct maintainer disposition, 1004/1307 ownership, SGLang comparison. Disposable host-only exact-integer butterfly/warp/alias/selector oracle **842 passed**, Python compilation passed. No HIP build, GPU test or first-party benchmark.

**Next/terminal:** RNX08 read-only five-node route and per-rank exposed-wall census. NO_ELIGIBLE_FWHT_CHAIN, BELOW_CEILING (<2.9126% optimistic E2E wall), or PIN_ABSORBED → close without patch. Only if admitted, one default-off native-fusion-path prototype with exact-alias/full-window lifetime proof, gfx1100/gfx1201 resource/physical-work and logits/KLD/MTP/graph tests; promote only CI95-low ≥3% E2E with no >1% control regression. Otherwise ALIAS_UNSAFE, NUMERIC_FAIL, REGRESSION or NO_GAIN.

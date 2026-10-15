---
id: BCOP192
order: 192
plan: patching-bc-optimizations
state: done
created-at: '2026-10-13T19:13:24Z'
created-by: agent
priority: P3
work: S
---

# PNRO11: retire target-wide lm_head mirroring for DFlash tensor split

**Discovery/change:** llama.cpp b11552 sharded `output.weight` yields axis-0 logits; DFlash2's in-graph `TOP_K` cannot consume them under a TP draft. The original PNRO11 proposed mirroring the **target** head, but existing BigCherry 1286 already makes one draft-local copy for the observed single-device DFlash/DSpark + TP target route without altering native MTP. Upstream #28777 reports external NVIDIA target-wide mirroring regression (+455 MiB/rank and +7–10% decode step time); open #27858 implements a distinct CPU-side selector for TP **drafts**. No AMD improvement inferred.

**Owner/12-hour exclusion:** PNRO11 is `done` for its original global-mirror scope; QFP43/1286 remain sole implementation/qualification owners. QFP43's older mirror proposal is explicitly marked historical. PNRO11 last independent change 2026-10-04; 1286/QFP43 last changed 2026-10-10; no independent PNRO11/shared-head implementation, PR, benchmark or queued run found in the preceding 12 hours. Current Radiance GPU-mailbox/expert-helper, RAD01 patch-prefix and release work excluded. Recent BCOP181–191 cover different mechanisms.

**Unresolved action/terminal disposition:** None under PNRO11. Retain 1286 untested until QFP43 qualifies source/destination bytes, VRAM, graph/request lifetime, logits/acceptance and matched MTP controls. TP-draft in-graph `TOP_K` and foreign-device layer-split are **different** unsupported routes: use an early explicit error; only consider #27858's host selector after a real draft and >=3% exposed end-to-end opportunity. Never add a global target-head mirror or duplicate allocator/selector/dispatch system.

**Evidence:** `src/llama-model.cpp`, `ggml-backend-meta.cpp`, `src/models/dflash.cpp` at b11552; BigCherry `1286/patch.py`, QFP43 historical 27B MTP/DFlash results; [upstream #28777](https://github.com/ggml-org/llama.cpp/issues/28777), [#27858](https://github.com/ggml-org/llama.cpp/pull/27858), [#26636](https://github.com/ggml-org/llama.cpp/pull/26636). Source assertions 12/12, PR-head assertions 6/6, host route assertions 9/9, Python compile passed. No GPU, HIP build, repository pytest or new throughput result.

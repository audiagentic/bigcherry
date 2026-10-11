---
id: BCOP140
order: 140
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-11T14:04:39+00:00'
created-by: agent
priority: P3
work: S
---

# PRBE62: prevent Vulkan graphics-queue A/B misclassification across pin and env policy

## Discovery / disposition

Pinned b11552 treats `GGML_VK_ALLOW_GRAPHICS_QUEUE` and `GGML_VK_ASYNC_USE_TRANSFER_QUEUE` as **presence** flags; `=0` still enables them. The graphics flag also changes transfer-family admission and AMD dGPU async preference. A control arm using `=0`, or a mixed-pin A/B, cannot isolate graphics compute. Merged upstream [#29998](https://github.com/ggml-org/llama.cpp/pull/29998) (2026-10-11 12:02 UTC) changed Vulkan `MUL_MAT_ID` row-tile prepass **after** b11552; do not attribute its workgroup changes to graphics queue. No new queue patch.

## Owners, novelty and protected work

PRBE62 is the authoritative owner of queue-policy qualification; upstream owns queue selection and `MUL_MAT_ID` implementation. BCOP53 is historical, not a second design. PRBE55/61 own MMVQ/`rm_kq` routing. Last independent PRBE62 plan change 2026-10-08 06:47 UTC; no independent graphics-queue patch/PR/benchmark/queue was identified in the preceding 12 hours. Protected active PA49/b11552 patch reconciliation, QFP41/1356, QFP48, QFP49/1360/1361, MTP/1359, and PGC17 transport were not modified.

## Evidence, next action, terminal gate

Nine source-static assertions passed after a corrected fetch range; 42,810 exhaustive + 10,000 seeded host row-tile-bound checks passed; presence semantics confirmed. No hardware/build/benchmark ran; no new BigCherry gain measured. **Gate 0:** one pinned binary, separate processes, env *unset* vs *present*, exact driver/device/queue-family/async receipt; no actual family switch => terminal no applicable lane. **Gate 1:** same-pin matched MoE/dense controls, expert-work, transfer/sync, logits/KLD, multi-request and MTP correctness; never compare b11552 to post-#29998. **Terminal:** <3% MoE CI95-low or correctness failure => no change; positive MoE with dense/prefill regression => process opt-in only. No second Vulkan scheduler. See PRBE62 for exact technical gates.

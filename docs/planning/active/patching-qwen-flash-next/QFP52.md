---
id: QFP52
order: 52
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-10T21:23:48.693932+00:00'
breadth: ''
skill: advanced
created-by: claude
priority: P3
work: M
---

# Where the MTP hidden-state (NextN) output lives: owner device, copies and a placement ablation

## Description

Split out of QFP42 (closed 2026-10-11 as done by 1359), where it had been folded in as the PRBE57 follow-up. Not done, and not part of the barrier removal.

The target hands the drafter one row of hidden state a token (`t_h_nextn`, n_embd_out = 10240 wide on Flash-Next: about 21 MB a 512-token batch). With the tensor split it is not established which device owns `NEXTN_PROJ_PRE/POST` and `t_h_nextn`, how many device-to-host and host-to-device copies a batch costs, or whether a different owner would be cheaper. Since 1359 the rows are copied out twice a batch (llama's own output buffer and the fenced pinned buffer).

What is known: the same prefill with no drafter at all is +14.6% (run nomtp1) and 1359 recovers +7 to +13% of that by reordering; part of the rest is the hidden-state output itself. The external reference (fork commit 1fcc05da) changes Vulkan NextN placement and is not a HIP implementation; 41a8ca78 is a separate backend-resident hand-off (Vulkan, parked as PRBE58).

## Steps

1. Measure in the production HIP / Meta layout: the owner device of the NextN tensors, the real D2H copies and their sizes and stream, and the drafter-side upload.
2. Remove the duplicate copy if 1359's fenced buffer can serve llama's own readers (decode-phase callers read the masked output through the original buffer today).
3. Only if step 1 shows a copy or ownership bottleneck: a default-off placement ablation `BIGCHERRY_MTP_TOPOLOGY=off|auto|device:N`, costed as copy-in + copy-out + synchronise + host staging.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Byte-identical NextN rows, greedy identity, accepted counts, copies a token and no extra VRAM pressure; ABBA on one binary. Placement is not to be conflated with the asynchronous hand-off (1359).

## Effort & Risk



## Standards



## Acceptance Criteria

- Owner device, copy count and bytes a batch are recorded for the production layout.
- The duplicate copy is removed or shown to be needed.
- The placement ablation is run only on the evidence of step 1, and promoted or rejected with its measurement.

## Notes

Origin: PRBE57, folded into QFP42 on 2026-10-08, split out again 2026-10-11 on the owner's instruction. The third thing QFP42 carried (PRBE07, the BridgeSpec review) was closed as inspiration only; its one live idea, kernels for verify widths 2..8, belongs to QFP38 and is noted there.

## Change Log

- 2026-10-10T21:23:48.693932+00:00 (created-by): Created by claude

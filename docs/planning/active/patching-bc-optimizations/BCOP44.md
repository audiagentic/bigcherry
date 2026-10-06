---
id: BCOP44
order: 44
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-07T01:07:00+11:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: S
---

# Gate AMD MMQ GLU prefill fusion

## Disposition

Authoritative owner: PRBE37. QFP13 remains decode launch-ranking only; PRBE38 remains literal UNARY-to-MUL ownership.

This audit found upstream draft llama.cpp #29948 providing the concrete dense FFN MMQ prefill mechanism PRBE37 previously lacked: interleave gate/up weight rows, retain one accumulator, and apply canonical GLU in MMQ write-back. The upstream matcher is explicitly NVIDIA-only because its write-back depends on the NVIDIA MMA accumulator layout. Published Qwen3.8-27B pp16384 gains are therefore mechanism evidence, not RDNA performance evidence.

No subsequent BigCherry work has implemented this MMQ-prefill fusion. First attribute the exact canonical MUL_MAT(up)+MUL_MAT(gate)+GGML_OP_GLU MMQ topology on gfx1100/gfx1201. Continue only if removable GLU/materialization cost is >=3% of representative prefill wall time or >=5% of MMQ+GLU critical-path time. If it passes, reuse #29948's interleaved-loader/write-back mechanism under PRBE37 with an AMD accumulator lane-mapping fixture; if it fails, park and wait for upstream AMD support. Do not create another matcher, fusion descriptor, scheduler, or dispatch table.

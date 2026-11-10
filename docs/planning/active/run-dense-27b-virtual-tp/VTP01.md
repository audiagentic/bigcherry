---
id: VTP01
order: 0
plan: run-dense-27b-virtual-tp
state: active
created-at: '2026-11-10T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Virtual TP4: logical four shards, only three physical GPU executors

## Scope and premise

Investigate weighted logical slices **30/30/20/20** mapped to physical
{XTX0:[rank0], XTX1:[rank1], R9700:[rank2,rank3]}. The R9700's two logical
ranks must share a single **32 GiB** VRAM ledger and physical GPU bandwidth.
This is not two new GPUs or a way to claim 4× independent PCIe endpoints.

Keep separate the comparison against (a) existing XTX TP2 at identical Q8/Q6
and KV configuration, and (b) XTX TP2 + an R9700 contiguous-layer stage
(XDP02/HD2701). Do not reuse Flash-Next MoE helper protocols for dense TP.

## Source constraints (source audit, no implementation)

- BigCherry `radiance_layout.py` currently demands one architecture/plugin for
  every engine rank and unique selected devices. The pinned Radiance engine
  treats one rank as owning one device, one rank thread, graph, buffers,
  checkpoint and KV allocation. Mixed gfx1100/gfx1201 engine ranks and
  duplicate-device rank aliases are NOT presently supported.
- `libr4d/r4d_ar_entry.hip` supports multi-rank operations but its rendezvous,
  scratch/flags and peer assumptions do not certify two ranks on one device.
  GPU-side spin waits between co-resident ranks can deadlock or starve.
- Qwen3.8-27B proxy has 64 layers, 16 full attention, 4 KV heads (HD=256),
  24 query heads and GDN state; nominal 262144-token model context. Four
  logical KV ranks can own one KV head each, giving physical 25/25/50% KV
  ownership, **not 30/30/40%**. Head/FFN/quant tile splits require explicit
  integer feasibility per operator: fractional 30% tensor dimensions are
  not executable shards.

## Proposed seam (needs review first)

Declare 4 **logical** weight/output shards but only 3 **physical** executors
and 3 participation endpoints. The R9700 schedules two compute slices in
one physical device context, optionally on two nonblocking HIP streams; its
local partials sum on-device **before** the physical three-way exact reduction.
Use one rank's visible output per physical GPU; no second host transfer
for the R9700 virtual rank. Retain mandatory numerical order and matching
fused residual/norm/quant semantics.

One physical-device resource ledger covers BOTH R9700 slices: weights,
replicated tensors, GDN/conv, KV heads, scratch, graph arenas, pinned-host
buffers, queues, checkpoint/drafter and safety reserve. No duplicate VRAM
claims and no in-serve repartitioning. Model admission fails closed on actual
per-physical-device capacity, missing gfx11/gfx12 kernels or illegal layout.

## Explicit correctness / deadlock hazards

- Map immutable logical id -> physical uuid, device arch/plugin, queue,
  tensor offsets and KV-head ownership. Prevent non-unique physical-id
  bypass of current launcher/engine guards before a dedicated runtime exists.
- Initialisation and teardown: logical shards must not cross-address GPU
  allocations, double-free buffers or reuse global collective flags.
- No GPU busy-spin wait for a sibling logical rank on the same GPU.
  Each physical contributor publishes once, after both local sub-shards,
  with HIP events, graph replay epochs and cancellation-safe state.
- Verify all-reduce across gfx11/gfx12 without P2P, using mapped pinned
  host transport and staged physical-device operations. Concurrency and
  fallback must not silently change wire precision or accumulation order.
- Four-rank attention and GDN head split is **not** equivalent to arbitrary
  30/30/20/20 capacity: shape and/or partial replication may dictate
  additional work, requiring a complete per-op mapping table.

## Competing architecture: unequal *physical* TP3 (VTP12/VTP13)

Do not confuse "two KV heads on R9700" with "two execution ranks on
R9700." A single gfx1201 executor can own and process both heads
**concurrently** within one kernel/grid (or proven overlap-safe launches),
without serial per-layer pipeline stages or co-resident logical ranks.
One XTX rank each owns a KV head; the R9700 owns two (1/1/2). For
full attention the natural Q-head mapping is 6/6/12 of 24; GDN key
4/4/8 of 16 and value 12/12/24 of 48 are candidate head layouts,
not proof of legal fused GDN/conv/state geometry. Independently split
FFN tiles and preserve a single three-physical-endpoint collective.

Asymmetric TP3 needs new per-op offsets, nonuniform rank dimensions,
arch-specific kernel dispatch, attention/GDN state ownership and
per-GPU allocator accounting. Existing `--tp 3` is NOT supported for
this model: equal division of four KV heads fails. It is not a
shortcut around the present uniform-world-size and mixed-architecture
checks. The alternative must compete with TP4 and identical dual-XTX
control at real concurrent serving speed and resident KV capacity.

[VTP12](VTP12.md) owns the native INT8/FP8 and mixed-precision
operator proof; [VTP13](VTP13.md) owns the asymmetric physical TP3
runtime and full A/B. [XDP10](../run-xtx27-prefill/XDP10.md) owns
load-time model variant selection. Do not choose TP3 or virtual TP4
from bandwidth-only theoretical splits.

## Deliverable / acceptance

Card-free table of logical-to-physical ownership for *every* affected
projection, KV head, GDN state, output/embedding and collective; derive
legal rounded sizes. An ABI/graph and memory-lifecycle review covering
same-device co-resident rank deadlock, exact accumulation and failure
recovery. NO virtual-TP4 production code before VTP02 fit evidence and
VTP03 physical three-way microbench lower-bound gate.

Implementation is a **new opt-in engine architecture feature**, not a
fake GPU inventory or just passing `--tp 4`. Reuse pinned managed patches,
never edit `vendor/radiance` directly.

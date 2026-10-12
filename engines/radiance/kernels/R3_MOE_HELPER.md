# R3_MOE_HELPER: design result against Radiance 1.3.0

Pinned source: `89cee7ce414486c717aa2085c99e1f522e4c0ea6`.

## Result

Do not implement `R3_MOE_HELPER` at the `moe_gemm_q` plugin seam. It cannot meet the
one-round-trip-per-routed-layer requirement in Radiance 1.3.0. The next use of the R9700 is the
n-gram device gather described by `R3_NGRAM_DEV`.

## a. Weight ownership and why a kernel plugin cannot intercept a pooled expert

`moe_gemm_q` reads a `RAD_OPD_WTAB`. `Ctx::WeightTable` is a device array of resolved
pointers. `runtime/issue.cpp::resolve_weight_table` asks `ResidencyTable::lookup`, waits for the
published generation, and writes only `WeightSlot::ptr` into that table. The kernel then does
`Wtab[e]`; it receives no movement-unit id, tier, site, slab slot, heat, or pooled/resident bit.

The owners are:

- `Planner`: static per-rank movement units, initial tier/site and budgets.
- `HeatEngine`: routing heat and promotion/demotion choice.
- `Mover`: slabs and host pools, transfers, and publication into `ResidencyTable`.
- `ResidencyTable`: public runtime seam containing only pointer, ready event and generation.
- `Ctx::WeightTable`: the final pointer-only table consumed by `moe_gemm_q`.

A plugin therefore cannot safely infer "pooled" from the pointer. The smallest engine extension
would add movement-unit identity/site to the residency interface and expose a routed-block helper
decision before the two expert GEMMs. A `.rw` against `libr4d` is insufficient; this is a core
plus architecture change and must be carried as a source patch against the pinned engine.

Intercepting only `moe_gemm_q` is also the wrong execution boundary. A routed expert is
gate/up GEMM -> gated activation/quantisation -> down GEMM -> gather. A GEMM-level helper would
send intermediate activations back after gate/up and send them to the helper again for down.
That is two helper transactions per layer, not one.

## b. Arithmetic and injection point for a viable engine implementation

A viable helper owns both TP shards for each helper-resident expert but computes the two rank
partials separately:

1. consume the same E4M3 activation used by the routed gate/up;
2. decode the stored `w4nl64a8h` nonlinear nibble through its E4M3 codebook;
3. apply the per-(row,64 K) E4M3 scale and fixed second-level `2^-13` fold;
4. run the same bf16-WMMA/f32-accumulate gate/up arithmetic;
5. run the same gated activation and activation quantisation;
6. run each rank's down-projection shard independently;
7. multiply by the routing weight and return two BF16 partial-output planes.

Rank 0 receives helper partial 0 and rank 1 helper partial 1 at the same point their local
`moe_gather` contribution is formed, before the existing all-reduce. The normal all-reduce then
still sums one partial per TP rank and the helper cannot be counted twice.

Do not compute one whole expert and inject it on both ranks. Injecting a whole result on one rank
would also change the per-rank distribution seen by the lossy `wht6` wire.

Cross-architecture results cannot be promised bit-identical. Radiance already documents the TP
slice as agreement-tested rather than byte-identical because the split changes rounding. A future
helper selftest should require, for final BF16 routed-block output, `rel_l2 <= 2e-3` and
`max_abs <= 2e-2`, then tighten those bounds from hardware data. The existing same-gfx11
`R3_MOE_COMPARE` byte comparison remains unchanged.

## c. Device ownership

The engine enumerates all visible devices, then takes discrete devices matching the first chosen
architecture. With HIP 0,1 = gfx1100 and HIP 2 = gfx1201, TP=2 uses 0,1 and ignores 2 as a rank.

That does not provide a second libr4d execution context. Plugin loading and kernel selection are
bound to the rank device, and `RadCtx` owns one device. Loading upstream libr4d and calling its
`moe_gemm_q` as a second gfx1201 context inside the rank process is therefore not a supported
1.3.0 operation.

The safe whole-block design is a helper process owning only the R9700
(`HIP_VISIBLE_DEVICES=2`). Shared pinned host memory carries activation and result slabs; a
sequence counter plus futex/eventfd publishes one request and one completion per routed layer.
No P2P is required.

For a 2048-token chunk, ~5 MiB each direction is small relative to hundreds of MiB of cold expert
weights, but without P2P the bytes still cross PCIe on both sides of the pinned buffer. At the
observed 3-5 GB/s link rate, 5 MiB is ~1.0-1.7 ms per PCIe traversal. A complete request/response
contains four GPU<->host/device traversals unless the helper directly consumes/produces mapped
host pages, so the design must be measured rather than assuming a ~2 ms round trip.

## d. Batching, decode break-even and heat

One request must contain all helper-routed experts for one layer and one chunk. Never signal per
expert.

The admission rule must be measured, not static:

`T_helper(M) + 20% < T_current_pooled(M)`.

Until the M=1 helper measurement satisfies that inequality, decode stays on the existing path.
That is the only safe interpretation of "decode must not get slower". At M=1 the current down
projection is ~78 us resident and ~466 us pooled, leaving ~388 us of miss penalty per call; a
helper with a fixed transaction cost larger than that is not a decode win. Prefill has much more
room: at M=2048 the measured all-pooled down path is ~47.1 ms versus ~5.13 ms resident.

Helper-owned units must be a fixed non-movable tier for the first implementation. Exclude them
from HeatEngine swap candidates, so the long tail has a home and the 20k+ promotion/demotion
thrash stops. Rebalancing XTX-hot versus helper-cold belongs at a coarse epoch after the helper is
proven, not in the per-step mover.

## e. Memory plan

Do not target the RX 6900 XT for expert GEMM. gfx1030 has no matching WMMA path and cannot execute
the gfx1201 libr4d code object or the gfx1100 libr3 compatibility path without another port.

If the routed-block helper is later implemented, reserve roughly 2-3 GiB on the 32 GiB R9700 for
runtime/scratch and use ~28-30 GiB for both TP halves of helper experts. Leave the remaining tail
in the existing host pool.

The n-gram table is 47.68 GiB, so every GiB given to it is ~2.10% of the rows. Because
`embed_lookup_q` is 9.2-10.1 ms/call while cold MoE can add tens of milliseconds at M=2048,
experts have first claim on R9700 memory. Decide with an end-to-end prefill sweep of 30/0, 28/2 and
24/6 GiB (expert/ngram), recording total `moe_gemm_q` and `embed_lookup_q` time. A tiny n-gram
shard only pays if measured row locality is substantially higher than its capacity fraction.

Since the requested helper does not survive (a)-(d) on the 1.3.0 plugin ABI, the immediate best
use of the R9700 is the n-gram gather. With ~30 GiB available it can hold ~63% of the 47.68 GiB
table; unlike expert offload, that can be implemented as a host-kernel replacement without
changing the routed-MoE execution graph.

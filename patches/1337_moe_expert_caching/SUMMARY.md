# 1337_moe_expert_caching

**Status:** untested
**Plan item:** MET01

Kind: upstream backport of llama.cpp #29887, originally taken from commit `6b7b03aab` and rebased to b11474.
Requires the native selective-copy callback already present at this pin. With `--moe-cache-mib N`, host-resident
MoE expert weights get a persistent GPU LRU cache. Upstream limits remain: one target device, no pipeline parallelism.

## Mechanics at b11474

This rebased revision uses the pre-final #29887 mechanism: a cached `MUL_MAT_ID` reads a compact cache-bank tensor
instead of the host expert tensor, and the scheduler replaces its expert IDs with cache-slot IDs in an `ids_copy`
tensor. `prepare()` plans the LRU, uploads every missed expert projection with
`ggml_backend_tensor_set_async`, and writes the remapped slot IDs before the split executes.

That is mathematically an exact weight substitution when the slot map is correct, but it is not the same
`MUL_MAT_ID` operand shape/ID stream as the no-cache path: `src0.ne[2]` is the cache slot count rather than the
model's expert count and the IDs are slot numbers rather than model expert IDs. Kernel/fusion selection can therefore
differ even when copied weight bytes are exact.

The merged upstream #29887 was subsequently rewritten on top of #29943: it uses the native scheduler copy callback
and a per-layer slot-map lookup in the graph instead of 1337's scheduler-owned `ids_copy` rewrite. The b11474 port
therefore needs fresh correctness evidence; the prior greedy divergence is not explained by a proven bad LRU mapping
in source.

## Performance mechanics

The plain cache is eligible only when `n_tokens <= 32` and the selected experts fit its slots. Decode/MTP-sized
batches are therefore cache-eligible; the 32-token gate primarily bypasses large prefill. On each miss, all expert
projections bound to the layer are uploaded. A small budget spread across many host-resident layers can therefore
thrash and become PCIe-bound.

With `BIGCHERRY_PATCH_TRACE=1`, shutdown emits a machine-readable 1337 marker with small/large hit, miss and
uploaded-MiB totals. The recheck runner uses one request per traced process so those totals are request-scoped.

## Prior evidence

The 2026-10-08 NCMOE=41 / 4096 MiB single-point run remains historical evidence in
`releases/evidence/met01-moe-cache-qualification.md`; it is not a final disposition. Recheck fusion-off greedy
identity first, then throughput/hit-rate across the restored matrix.

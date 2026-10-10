---
id: PRVP03
order: 0
plan: patching-rocm-vulkan-provider
state: pending
created-at: '2026-10-02T12:35:07.968893+00:00'
breadth: ''
skill: ''
created-by: agent
priority: P2
work: L
---

# Vulkan meta-backend AllReduce provider

## Description

Authoritative owner for BigCherry Vulkan tensor-parallel communication-provider work. Expose the meta communication SPI (comm_init / comm_allreduce_tensor / comm_free) from ggml-vulkan via get_proc_address. Phase 0 (patch 1290) is the evaluated, default-off F32 host-reduction reference provider proving API wiring/correctness. Phase 1 may replace only the data path while preserving this SPI and fail-closed fallback.

PRBE09/RD104 is now terminally absorbed here. Do not create a second Vulkan AllReduce/provider implementation from that historical upstream-tracking item.

## Steps

1. Keep 1290 as the reference provider and stock-fallback control.
2. Before phase-1 implementation, verify the current Vulkan buffer/command/descriptor lifetime and the exact meta communication call ordering on the pinned source.
3. Prototype the smallest mapped-host discriminator first: dual-XTX RADV, exact F32, 32-256 KiB, one aligned host region imported into both devices. Prove import/coherency and synchronization semantics before chunking or double buffering.
4. Only after the single-buffer path is correct, compare external-timeline synchronization and root-device reduce-shader timing against stock meta fallback.
5. Add chunking/double buffering only if profiling shows transfer/reduce serialization is material; do not assume overlap helps on PCIe without P2P.
6. Unsupported types, heterogeneous registries, failed imports, unsupported extensions, or unsafe synchronization must return false/use stock meta fallback rather than create a second fallback stack.

## Detailed Solution & Technical Design

The provider boundary already exists in 1290: Vulkan exposes `ggml_backend_comm_init`, `ggml_backend_comm_allreduce_tensor`, and `ggml_backend_comm_free` through registry `get_proc_address`. The meta backend owns collective invocation and fallback. Phase 1 therefore changes the provider's transport only; it does not add a scheduler, placement solver, second collective API, or another dispatch registry.

The proposed mapped-host path is a hypothesis until hardware-qualified. BigCherry's topology has no normal GPU P2P, so a Vulkan design must be evaluated as host-visible PCIe transport. Correctness gates precede throughput: fixed work count, exact/declared numerical reduction semantics, repeated same-process requests, multi-ubatch, long-context, and clean fallback after provider rejection.

## Files

- `patches/1290_vulkan_allreduce_host_f32/{patch.toml,patch.py,SUMMARY.md}` — reference provider.
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp` — provider registry/data-path seam in the patched source.
- Meta backend communication SPI/fallback — invocation owner; do not duplicate.
- `tools/lab/vulkan/queue-vk-ar.sh` and the canonical Vulkan probe lane — qualification owner.

## Validation

Phase 0 measured evidence (2026-10-02, first-party): 27B Q8_0 dual XTX RADV, same binary, three requests. Provider OFF / stock fallback: tensor 496 pp / 24.6 tg, MTP5 50.4. Provider ON / synchronous host-F32: 259 pp / 20.1 tg, MTP5 40.7. Layer split was 947-951 pp / 20.8 tg. The provider marker fired 59,136 times and greedy output matched the layer-split reference in both arms. This proves SPI activation/correctness for the tested lane and rejects synchronous CPU reduction as a performance path.

Phase-1 gate:
- cheapest discriminator: exact-F32 32-256 KiB dual-XTX mapped-host micro/fixture before model benchmarking;
- reject on import/coherency/synchronization ambiguity, missing required extension support, output/work mismatch, or inability to fall back cleanly;
- if correct, run stock-fallback vs provider paired model lanes with route markers and collective bytes/counts;
- promotion requires CI95-low-positive >=3% end-to-end decode/MTP throughput on a production tensor-split lane, <=1% unaffected-control regression, and no correctness failure;
- gfx1201 and gfx1030 are separate qualification lanes, not assumed from gfx1100.

## Effort & Risk

Phase 0 evaluated. Phase 1 L/high: no normal P2P; Vulkan external-memory/coherency and synchronization rules are driver/device dependent. The existing stock fallback is already faster than phase 0, so phase 1 must beat a real control rather than merely beat the reference provider.

## Standards

One provider SPI; fail closed; measured work/bytes; no transferred CUDA/HIP performance claims; correctness before speed; no new runtime policy surface unless a measured stable selector is required.

## Acceptance Criteria

Phase 1 is retained only if the mapped-host discriminator is correct and a production lane clears the >=3% CI95-low E2E gate. Otherwise 1290 remains an evaluated reference/fallback-boundary proof and optimized Vulkan AllReduce work is rejected/deferred until a materially different mechanism or upstream provider appears.

## Notes

2026-10-02 phase 0 (1290 host-f32) on hardware, 27B Q8_0 dual XTX RADV, same binary b-vk-ar-host, 3 requests: provider OFF (stock meta fallback) -sm tensor 496 pp / 24.6 tg, MTP5 50.4; provider ON 259 / 20.1, MTP5 40.7; -sm layer unaffected (947-951 / 20.8). Marker fired 59,136 times; greedy identical to -sm layer in both arms. SPI wiring and correctness proven; the synchronous CPU reduce is slower than the stock fallback, as expected.

2026-10-08 audit: PRBE09/RD104 was found stale because PRVP03/1290 had already implemented and hardware-screened the capability it was still tracking as hypothetical. PRBE09 is closed/absorbed here. Current upstream has CUDA/HIP backend-local AllReduce but no Vulkan-local equivalent; external ROCm collective mechanisms are mechanism references only.

## 2026-10-11 safety / upstream disposition (TRVP16, BCOP120)

**Source-level correctness gate, not a reproduced GPU failure.** In pinned b11474, `ggml-backend-meta.cpp::allreduce_fallback` fills ranks lacking `GGML_TENSOR_FLAG_COMPUTE` with zero before reducing (upstream merged #29793). Phase-0 `1290_vulkan_allreduce_host_f32::ggml_backend_vk_comm_allreduce_tensor` reads every rank unconditionally. A disabled/empty split rank with stale or NaN tensor contents can therefore violate the generic fallback's semantics. **PRVP03 owns** the minimal 1290 fix: return false **before any reads/writes** if any rank lacks COMPUTE, letting meta perform its existing FILL fallback. Do not modify the provider in this documentation-only audit. Host model: six passing unittest methods, 10/10 source checks; no GPU failure proven.

The meta caller invokes generic fallback whenever a provider returns false; **never return false after writing or submitting a partial reduction**. Any phase-1 implementation must prevalidate all ranks and commit atomically from the caller's perspective, or surface a terminal error rather than double-apply. The meta destructor frees provider context before rank backends; reuse this lifetime.

[Upstream llama.cpp #25051](https://github.com/ggml-org/llama.cpp/pull/25051) is OPEN (2026-10-08 update, head `99becee6`) and already implements Vulkan mapped-host/timeline-FD AllReduce plus an all-COMPUTE-only F16 ring at >=2 MiB. It is a **source-level replacement candidate**, not a merged/AMD-qualified baseline. Prefer wait/adopt/verify over a parallel phase-1 transport. Its ordinary path zeroes inactive host staging; verify inactive destination initialization, import/coherency, multi-turn and post-submit failures before any port. TRVP16 owns mixed-RDNA/topology performance qualification; RRVP05 owns hardware campaigns and RRVP02's Vulkan implementation pause remains respected.

## Change Log

- 2026-10-02T12:35:07.968893+00:00 (created-by): Created by agent
- 2026-10-02T13:00:26.501279+00:00 (updated-by): Updated: section:notes
- 2026-10-08: Absorbed PRBE09/RD104 ownership and bounded phase-1 mapped-host qualification.

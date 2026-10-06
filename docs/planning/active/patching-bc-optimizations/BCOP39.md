---
id: BCOP39
order: 39
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-06T12:06:00+11:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: S
---

# Disposition R9700 PCIe-x4 expert-cache economics

## Description

Optimisation-audit disposition for host-expert caching on the current R9700 PCIe 4.0 x4 topology. MET01 remains the authoritative residency/cache policy owner; MET04 owns the EP gate; RPL01 consumes the measurements for whole-system placement scoring.

## What changed/discovered

- Local 1337 evidence supersedes the assumption that the external 4 GiB R9700 cache result should reproduce here. At ~6.7 GB/s effective H2D, 2/4/8 GiB LRU caches regress decode; 12/14 GiB cross over only after roughly 88-90% hit rate.
- Equal-VRAM resident layers beat 8 GiB LRU and preserve much higher prefill throughput near the 12 GiB comparison. With the 6900 XT MTP sidecar, equal-VRAM resident layers also match/beat 12 GiB LRU decode.
- Host-expert prefill improved 93.9 -> 420.6 t/s from ub512 -> ub4096 with decode unchanged, so transfer amortization is already the first deployment lever.
- MET04's completed hop probe rejects the proposed dense-on-XTX / experts-on-R9700 layer-split workaround for this topology.
- ap03906101/moe-hotcache provides a relevant HIP mechanism: graph-visible route counters plus persistent static/dynamic hot slots. Its own topology notes also reject a slow x4/chipset GPU as a cache tier.

## Already acted upon

- 1336 provides selective-copy telemetry/callback evidence.
- 1337 provides the current pure-LRU reference and measured cache-size curve.
- MET01 already owns static/LRU/hybrid/whole-layer comparison.
- MET04 has already rejected the R9700 layer-split hop layout.
- RPL01 already owns read-only whole-system placement scoring.

## Unresolved action

Run a hardware-free static-hot selector over existing routing traces at 8/12/14 GiB equal budgets. Promote a runtime static-hot/hybrid prototype only if held-out predicted miss bytes/token are at least 20% below pure 1337 LRU at equal VRAM without starving later layers/tensors. Reuse MET01 placement state and the existing cache storage seam.

Do not repeat 2/4 GiB pure-LRU tests, create another cache owner, or move expert-bearing layers to the R9700 via layer split.

## Terminal disposition

- **Promote to hardware:** static-hot prediction clears the >=20% miss-byte reduction gate.
- **Reject static-hot:** gate fails; retain whole-layer residency and 1337 as reference only.
- **Pure LRU:** rejected as a general policy on current x4 topology.
- **R9700 layer-split expert hop:** rejected by measured MET04 evidence.
- **Future wider-link topology:** requalify rather than carrying this rejection forward blindly.

## External evidence

- https://github.com/ap03906101/moe-hotcache
- https://github.com/ggml-org/llama.cpp/pull/29887
- https://github.com/ggml-org/llama.cpp/pull/29943
- https://github.com/ggml-org/llama.cpp/pull/29963

# 1233_rd73_stable_graph_cache_key: Replace the HIP/CUDA graph-cache key with a stable FNV-1a shape fingerprint (RD73, re-scoped from FORK-MTP-003)

**Status:** untested
**Group:** rdna-boosts
**Plan item:** RD73

## What it does

Replaces ggml_cuda_graph_get_key()'s use of the raw first-node pointer as the cuda_graphs map key with a 64-bit FNV-1a fingerprint over node count plus the first/last nodes' op/name/ne[], which is stable across allocations for a recurring shape; the existing per-node memcmp correctness check in ggml_cuda_graph_update_required() is unchanged, so a fingerprint collision only costs an extra recapture.

## Why

The raw first-node pointer is allocation-dependent, so a fresh allocation for an otherwise-identical recurring shape (e.g. repeated speculative-verify batches) caused a cold cache miss almost every time even though the shape hadn't changed; the fork measured a verify ubatch sync drop from 150ms to 57ms on a 3.8k-node graph.

## Upstream / provenance

Ported byte-for-byte from mrlordcat-rdna-lab commit 7f2e7e4a3 (https://github.com/MrLordCat/llama.cpp-rdna-lab), after an external review caught and fixed a bug in an earlier draft (hashing the whole fixed name buffer instead of its used length). Not merged into ggml-org/llama.cpp master.


## MTP validation (2026-09-04): mechanism does NOT engage on this pin

The open gate on RD73 was an MTP speculative-verify workload isolated from the
production 27B service. Done: isolated clone, isolated servers, lane
`bigcherry-native:control:linux-multi`, control vs `--experiment rd73-only`,
both under rocprofv3, identical workload (Qwen3.8-27B-Q8_0, `-sm tensor`,
`--spec-type draft-mtp`, `spec_draft_n_max=5`, same prompt, seed=42, temp=0,
n_predict=200). Patch application verified in the materialized source (FNV-1a
offset basis literal at ggml-cuda.cu:2884).

| metric | control | rd73-only |
|---|---|---|
| **graphs reused** | **65** | **65** |
| large gaps >100us (count) | 615 | 614 |
| large gaps >100us (total) | 523.9 ms | 421.7 ms |
| throughput | 51.54 tps | 52.17 tps (+1.22%) |

**Graph reuse is identical.** That is the direct test of the mechanism: an
unstable `nodes[0]` key would show as FEWER reuses in control. It does not. The
large-gap count is unchanged too; only total gap time moved, with the same
number of gaps, which is host-timing variance rather than fewer recaptures.

The +1.22% is **not** attributable to this patch -- single sample per arm,
inside documented run-to-run variance, and the causing mechanism provably did
not activate.

This confirms the earlier non-MTP finding generalises: bigcherry's pinned
llama.cpp already produces a stable `nodes[0]` per recurring shape, including
for speculative-verify shapes. The port is faithful (byte-for-byte from
`7f2e7e4a`, verified against the real diff); the difference from the fork's
150ms->57ms result is in the base tree, not the patch.

**Disposition: stays `untested`, unpromoted.** Correct and correctness-neutral,
but no measurable benefit on this hardware and pin. Cheap re-check if the pin
advances: just compare `graphs reused` between arms -- no full A/B needed.

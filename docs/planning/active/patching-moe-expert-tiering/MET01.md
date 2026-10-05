---
id: MET01
order: 1
plan: patching-moe-expert-tiering
state: pending
created-at: '2026-10-02T04:44:44.458623+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Per-layer routed-expert profile + canonical residency policy

## Description

MET01 is the single policy/accounting owner for routed-expert residency. It profiles routing, derives measured per-device expert budgets, and chooses among whole-layer residency, static-hot experts, upstream demand cache, ROCm3 auxiliary residency, and host fallback. It must not own a second scheduler, cache transport, auxiliary transport, or loader.

The immediate implementation gate is upstream llama.cpp #29943 + #29887. #29943 moves selective host-expert copying out of generic `ggml_backend_sched_compute_splits()` into a public scheduler copy callback. #29887 is explicitly intended to become user-code-only after that refactor. BigCherry should therefore qualify the callback boundary before carrying any private scheduler cache fork.

## Repository evidence

- Existing routing evidence on Flash-Next UD-IQ4_XS, ub512: mean 496/512 experts touched per layer in mixed prose+code prefill; hottest 10/25/50% cover 41.5/68.8/91.3% of picks; mean experts needed for 50/80/90/95% picks are 73/174/235/287. Static expert-count heuristics are therefore insufficient.
- MET05/patch 1328 already owns auxiliary 6900 execution and host staging. MET01 may select that tier but must not reproduce its transport.
- MET06 owns standard-GGUF materialization only if normal host-weight execution/cache mechanisms leave a measured placement gap.

## Upstream mechanism: #29943 + #29887

#29943 adds `ggml_backend_sched_copy_callback` in `ggml/include/ggml-backend.h`, stores it on `ggml_backend_sched`, and funnels split inputs through `ggml_backend_sched_copy_input()` in `ggml/src/ggml-backend.cpp`. Non-weight inputs are copied first; host-backed weight inputs are copied last so user code can inspect already-copied routing inputs. If the callback returns false, the scheduler performs the normal whole-input copy.

It also removes the scheduler's embedded `MUL_MAT_ID` expert-selection implementation and installs `llama_context::sched_copy_experts` from `src/llama-context.cpp`. This is the important consolidation boundary: selective expert-copy/cache policy belongs above generic scheduler machinery.

#29887 then supplies the policy candidate: per-layout GPU cache banks, selected-expert miss upload, remapped expert IDs and a <=32-token cache gate. Published NVIDIA Flash-Next Q4_0 results are strong for decode (4090 25.0 -> 39.4/40.7 t/s; 5090 30.8 -> 54.5/67.8 t/s) but prompt processing regresses as cache VRAM displaces whole resident layers. Treat those numbers as mechanism evidence only; AMD promotion requires gfx1100/gfx1201 measurements.

## Implementation plan

1. **Profile and budget.** Fix/retain `tools/lab/flash-next/routing-profile.sh` so `ffn_moe_topk` collection is safe under the meta callback. Capture decode and prefill separately across code, chat/prose, retrieval/long-context and MTP. Emit `hits[l][e]`, `weight_sum[l][e]`, `ubatch_present[l][e]`, assigned tokens and route mass.
2. **Adopt the callback seam, not a private cache.** On the current pin, first determine whether #29943 is present. If absent, qualify a minimal backport containing only the public copy-callback seam and moved selective-copy logic. Do not add MET-specific logic to `ggml/src/ggml-backend.cpp`.
3. **Mock callback correctness before cache policy.** Install a diagnostic callback that returns false for every host weight while counting callback invocations, backend, tensor bytes and graph first-op. Require byte/token/logit identity with callback disabled. Then enable the upstream selective-copy implementation and require identical greedy output plus nonzero selected-copy counters.
4. **Qualify #29887 at equal VRAM.** Compare whole-layer baseline, pure LRU, static-hot, and hybrid static+LRU. Preserve <=32-token gating initially. Prove the callback/cache is active under production scheduler copies; zero cache activity is a failed experiment.
5. **Canonical objective.** Rank each `(layer,expert,tier)` by avoided critical-path milliseconds per resident byte, using measured H2D, compute, sync and auxiliary staging costs. Do not optimize hit rate in isolation.
6. **Aux tier.** Feed MET05/1328 candidates from the same placement JSON. Keep 1328 whole-layer until hardware evidence proves expert-granular auxiliary placement worthwhile.
7. **Large-batch separation.** Keep pp512/2048/8192 separate from <=32-token decode/MTP. A cache configuration that improves decode by consuming VRAM but materially hurts prefill is not globally promoted; allow workload-specific feature-set policy only with explicit budgets.
8. **Only then consider MET06 slicing.** Runtime standard-GGUF slicing is blocked unless the callback/cache/static/aux matrix leaves >=5% TG/PP opportunity attributable to coarse placement or >=1 GiB avoidable resident expert memory at equal throughput.

## Code-level mock

Use the upstream seam as the test double; no cache implementation is required for the first discriminator:

```cpp
struct met_copy_probe {
    uint64_t calls = 0;
    uint64_t host_weight_bytes = 0;
};

static bool met_probe_copy(
        ggml_backend_t backend,
        const ggml_tensor * src,
        ggml_tensor * dst,
        ggml_cgraph * graph,
        void * opaque) {
    auto & p = *static_cast<met_copy_probe *>(opaque);
    ++p.calls;
    p.host_weight_bytes += ggml_nbytes(src);
    // Observation-only control: normal scheduler copy must still execute.
    return false;
}
```

Build/test this control before any cache port. Required assertions: callback fires only for host-backed weight split inputs; ordinary input tensors are already copied before callback; `return false` preserves baseline output; no callback-owned pointer survives beyond the compute call; multiple scheduler copies do not share mutable probe/cache slot state without generation/copy indexing.

For the selective path, preserve upstream MMQ safety: grouped expert copies must include required guard/padding bytes. Never infer a performance win from missing/corrupt expert work; use the BCOP/QFP28 multi-request integrity gate when qualifying Flash-Next.

## Ownership and files

- `tools/lab/flash-next/routing-profile.sh`, `tools/lab/flash-next/expert-placement/`: profiling/solver tooling.
- `ggml/include/ggml-backend.h`, `ggml/src/ggml-backend.cpp`: upstream #29943 callback seam only; no MET policy.
- `src/llama-context.cpp` / context-owned helper: selective-copy/cache user-code owner from #29943/#29887.
- MET05 / patch 1328: auxiliary ROCm3 execution and staging.
- MET06: standard-GGUF materialization fallback only after the measured gate.

No second dispatch table, residency map, cache allocator, prefetch scheduler or ROCm3 transport is permitted.

## Validation matrix

Hardware: gfx1100 XTX, gfx1201 R9700, then production 2xXTX+R9700; 6900 only through MET05 controls. No-P2P means every cache bank/upload is target-device-local.

Correctness before performance:
- callback-disabled vs observation-only callback: greedy identity and logits/KLD contract;
- selective copy/cache: multi-request same-process, cold->warm->workload-shift, MTP on/off;
- <=32-token decode plus pp512/2048/8192 bypass lanes;
- long context and scheduler-copy reuse;
- counters prove selected experts and uploaded bytes are nonzero and physically plausible.

Performance: ABBA >=5 repetitions for TG128/512 and PP512/2048/8192; report median/dispersion, static/LRU/aux GiB, route mass by tier, cache hit, H2D bytes/token, miss-upload ms/token, auxiliary service/staging, CPU fallback, peak VRAM and locked host memory.

Promotion: >=5% end-to-end TG/effective-TG or PP improvement at equal expert-VRAM budget with <=2% regression in the unaffected regime. A workload-specific decode policy may be retained if prompt regression is explicitly isolated and feature-gated. Reject any result whose implied transfer/work exceeds measured physical limits or whose correctness/integrity gate fails.

## Acceptance Criteria

- One canonical placement JSON/accounting table covers static, LRU, aux and host tiers.
- #29943 callback seam is either present upstream or qualified as a minimal temporary backport; MET logic does not live in generic scheduler code.
- Observation-only callback proves semantic transparency before selective-copy/cache testing.
- #29887-style cache proves nonzero activity on gfx1100/gfx1201 and passes correctness/integrity gates.
- Static/LRU/hybrid/aux are compared at equal expert-VRAM budget; decode and large-batch prompt effects are both reported.
- MET05 remains sole auxiliary transport owner; MET06 remains blocked until its explicit placement-gap gate passes.
- Unsupported or inactive paths fail closed rather than silently becoming baseline measurements.

## Notes

2026-10-05 audit: #29943 materially improves the integration boundary for MET. Its diff removes expert-ID parsing/copy grouping from generic scheduler compute and exposes a host-weight copy callback, while `llama_context` becomes the user-code owner. #29887 states that after #29943 it should be entirely user-code. This reduces BigCherry's reason to carry scheduler-core MoE cache patches and makes the first local mock cheap: an observation-only callback can validate ordering/lifetime on HIP before cache code is introduced.

External references: llama.cpp #29943 `ggml: refactor selective expert copying to user code`; llama.cpp #29887 `add a GPU cache for MoE experts kept in host memory`.

## Change Log

- 2026-10-02T04:44:44.458623+00:00: created.
- 2026-10-04: consolidated DwarfStar persistence and #29887 into MET01.
- 2026-10-05: consolidated MET05 auxiliary residency with canonical solver.
- 2026-10-05: made #29943 user-code copy callback the required integration seam; added observation-only mock, HIP qualification matrix and explicit no-duplicate-scheduler boundary.

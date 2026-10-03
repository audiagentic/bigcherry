---
id: PGC11
order: 0
plan: patching-gpu-collectives
state: pending
created-at: '2026-10-04T03:06:55+11:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# AllReduce config fast path: typed provider/wire IDs and one shared parser

## Description

0860 currently carries provider/wire configuration as `std::string` through backend setup and duplicates switch-byte parsing/config-application glue between `common/arg.cpp` and `llama-bench.cpp`. 0840 then performs additional provider-name string dispatch. This is cold-to-warm control-plane code, but it sits on the multi-GPU path and has grown with every provider. Reduce patch/code surface and remove repeated string comparisons/temporary strings by parsing CLI text once into compact provider/wire IDs at the backend seam, then dispatching by enum/integer IDs. Keep strings only at CLI/error/trace boundaries.

## Steps

1. Introduce private `ggml_backend_cuda_comm_provider` and `ggml_backend_cuda_comm_wire` IDs beside `ggml_backend_cuda_comm_config`; parse external strings exactly once in `ggml_backend_comm_set_config`.
2. Store typed IDs plus `switch_bytes` in the process-wide config. Replace provider string chains in 0860/0840 with `switch`/ID comparisons; expose a tiny name helper only for trace/error output.
3. Remove duplicated switch-byte parser/application machinery from `llama-bench.cpp`: route bench options through the existing common argument/config path where upstream structure permits. If llama-bench cannot consume common args cleanly, factor only the strict unsigned parser + registry apply bridge into one shared helper rather than retaining two copies.
4. Preserve ABI/proc-address input as strings so existing CLI/env and patch composition contracts do not change.
5. Measure generated diff/LOC and startup/first-collective overhead; require no steady-state regression. This item is primarily simplification, but typed dispatch must not add allocations or string compares to per-call adaptive routing.

## Detailed Solution & Technical Design

The backend config should be the normalization boundary. External text is accepted once, validated once, and converted to IDs. Adaptive dispatch reads IDs only. Do not add a public header unless reuse removes more code than it adds; prefer a private helper in the CUDA/HIP backend and make frontends call the existing proc-address setter.

The important performance invariant is that `ggml_backend_cuda_comm_try_allreduce_*` and 0840's adaptive per-call decision never construct `std::string`, call `strcmp`, or parse configuration. Provider selection should be an integer branch plus the existing size threshold.

## Code Samples & Guidance

```cpp
enum class ggml_backend_cuda_comm_provider : uint8_t {
    automatic, ccl, host, adaptive, p2p, root3, butterfly,
};

enum class ggml_backend_cuda_comm_wire : uint8_t { native, q8 };

struct ggml_backend_cuda_comm_config {
    ggml_backend_cuda_comm_provider provider = ggml_backend_cuda_comm_provider::automatic;
    ggml_backend_cuda_comm_wire wire = ggml_backend_cuda_comm_wire::native;
    size_t switch_bytes = 1 << 20;
};

static const char * provider_name(ggml_backend_cuda_comm_provider p);
```

Parse with a small exact-match table or compact if-chain in the setter; do not use maps/unordered_maps. In 0840 use `switch (config.provider)` and direct enum comparisons. Keep `ADAPTIVE_SWITCH_BYTES_DEFAULT` as the patch-authoring single source of truth until the generated code has a native constant shared by all frontends.

## Files

- `patches/0860_allreduce_provider_cli/patch.py`
- `patches/0840_hybrid_allreduce_dispatch/patch.py`
- `tools/llama-bench/llama-bench.cpp` generated target only if common parsing cannot be reused
- focused patch contract/tests for 0840/0860

## Validation

Patch compose/apply on the current pinned llama.cpp base; build HIP. CLI matrix: every provider, native/q8 wire validation, invalid provider/wire, switch-byte 0/max/overflow, env aliases, CUDA/HIP absent error. Run existing AllReduce provider tests plus dual-gfx1100 adaptive correctness. Benchmark `llama-bench` startup and first 100 small AllReduce calls with trace disabled; steady-state tg512/pp4096 must remain within noise. Report generated LOC before/after; acceptance target is net deletion in generated target code and patch authoring code combined, excluding tests/docs.

## Effort & Risk

Medium. Main risk is patch composition because 0840 extends 0860's provider set. Keep the string proc ABI unchanged and land typed normalization in 0860 first, then convert 0840 dispatch. Do not combine with threshold tuning or 3-GPU behavior changes.

## Standards

Preserve fail-closed provider validation, explicit q8/p2p compatibility checks, patch guards, deterministic trace names, and current stock fallback when BigCherry-specific providers are unavailable.

## Acceptance Criteria

- No provider/wire string operations in the per-call adaptive AllReduce path.
- One validation/normalization implementation for provider/wire configuration.
- Switch-byte parsing/application duplication reduced or documented as unavoidable with measured reason.
- Net generated+patch LOC reduction excluding tests/docs.
- Existing 0840/0860 tests pass; dual-XTX adaptive output/correctness unchanged.
- No statistically significant tg512/pp4096 regression; startup/first-collective result recorded.

## Notes

Selected from the latest active branch `agent-allreduce-final-ci` at c190b92341df2177090ea9c89409b874d7ff7ba8. The branch's latest change already centralizes the adaptive threshold constant but 0860 still emits duplicate frontend parsing/config glue and string-valued backend state. This item deliberately follows that cleanup direction without changing provider policy or PGC10's N=3 work.

## Change Log

- 2026-10-04T03:06:55+11:00 (created-by): Created from optimisation scan of latest active branch; target typed AllReduce config and parser deduplication.

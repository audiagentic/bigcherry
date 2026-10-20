---
id: BCOP99
order: 99
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-09T20:11:35+00:00'
created-by: agent
priority: P2
work: S
---

# PNRO10: hold misplaced ctx_other backend insertion pending a real separate-draft case

## Discovery / disposition

At pinned b11474, 1261's `insert_after(backends.emplace_back(backend), occurrence=0)` still lands **inside** the ordinary draft-device loop; two-device overlap can duplicate backends, and zero draft devices omit target devices. Const-correctness is already fixed; 2026-09-24 plan's "fixed both" claim was stale. `cparams.ctx_other` is gated to Gemma4Assistant and Eagle3/DFlash missing local tensors, not ordinary Qwen MTP. Upstream [llama.cpp #26636](https://github.com/ggml-org/llama.cpp/pull/26636) is open and has the correct after-loop insertion; its NVIDIA success does not qualify no-P2P RDNA.

## Ownership / 12-hour exclusion

PNRO10 owns 1261 and the conditional repair. PNRO11 owns independent DFlash lm_head replication; QFP43/1286 owns draft acceptance and must not be edited; 1252 P2P remains rejected. Latest independent PNRO10 plan work 2026-09-30, and no new independent 1261 implementation, PR, experiment or queue was found in the 12 hours before this run. Protected 1355/1356 HC/Meta promotion, 1358 split cache, QFP36 router, Radiance and Flash-Next work were not modified. Previous BCOP88-98 did not own this slice.

## Action / terminal gate

No hardware queue now. First reproduce a **supported** separate-draft target-owned shared-tensor backend failure on the current pin and record exact draft/target device sets, tensor ownership, graph abort and stock control. If absent, close PNRO10/1261. If upstream merges a correct fix, adopt it and retire the fork. Otherwise repair **only** 1261 with unique after-loop anchor, backend identity dedup, CPU-last, max-16, init failure, scheduler/allocator/graph and host-staged no-P2P correctness gates. Zero/one/two draft-device host tests, repeated requests/ubatches, full-vocab and acceptance parity must pass; no performance promotion claim without paired GPU evidence.

## References / actual validation

`PNRO10.md`, 1261 `patch.py`/`README.md`/`SUMMARY.md`; pinned `src/llama-context.cpp:146-165,335-365,402-455`, `src/models/dflash.cpp`, `ggml/src/ggml-backend.cpp:1084,1937-1946`; [llama.cpp #26636](https://github.com/ggml-org/llama.cpp/pull/26636), [vLLM draft model](https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/draft_model.py), [SGLang roadmap](https://github.com/sgl-project/sglang/issues/27462). Five pinned-source/patch-static assertions and five deterministic host-list fixtures passed. No repository pytest, C++ compile, patch composition, HIP execution or hardware benchmark.

---
id: BCOP94
order: 94
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-09T15:10:37+00:00'
created-by: agent
priority: P1
work: S
---

# PNRO09/1260: rebaseline Meta view headroom after upstream allocator change

## Discovery / disposition

Upstream llama.cpp #23671 (merged 2026-10-02) changed Meta `alloc_buffer_n` compute-pool sizing from `H * ggml_get_mem_size(ctx)` to `H * n_tensors * ggml_tensor_overhead()`. Thus b10901's measured 44/222 and 89/445 view boundaries do not qualify pinned b11474; new first-order per-container capacity is `H * n_static`, independent of the old fixture's S. Historical 4B MTP completed with H=16 and H=80, so no stock capacity failure has been established. Historical throughput deltas were confounded. No new GPU measurement.

The existing 1260 producer is explicitly non-startable: missing required `targets`, assumed source trees and nonexistent CMake target, non-capacity errors misclassified as boundaries, copied historical hardware evidence, and missing contract controls. The package stays `untested` and unpromoted.

## Ownership / 12-hour exclusion / novelty

PNRO09 is authoritative; 1260 is its sole patch. Last independent PNRO09 plan work was 2026-09-25; Oct 9 engine-directory migration preserved patch and producer blob identity and did not advance capacity logic. No 12-hour PNRO09 implementation, queued hardware lane or PR was found. Recent BCOP79-93 addressed other mechanisms. Protected independent work includes Flash-Next 1334/1347/1350 accuracy, QFP36/1357 router, QFP41/1358 Meta **split-state cache**, QFP35 fusion, Radiance gfx1100, MTP and engine benchmarking. None is modified. Upstream #30217 host-view handling (merged 2026-10-09) is a separate next-pin correctness control.

## Bounded next action / terminal

First build a real b11474 CPU+Meta host fixture through existing source/build authorities, varying `n_static` and S independently and exercising both rotating compute containers. Fail closed on missing targets, wrong error signatures, timeouts and incomplete capacity bracketing. Do not reuse historical hardware artifacts as fresh evidence. Only if a supported real recurrent+MTP graph fails at H=16 and passes H=80 with exact state/token/logit correctness, repeated reset/replay and bounded host RSS may 1260 be considered for capacity-only qualification. Otherwise close PNRO09 and retire 1260; no hardware performance campaign.

## Evidence / external references

PNRO09; 1260 patch.py, validation/producer.py, meta_boundary_test.cpp, boundary.json, hardware_d.json; `tools/bigcherry/patch/campaign/producer.py`; `tools/tests/patch/test_plan_producers.py`; pinned b11474 ggml-backend-meta.cpp and ggml.c; [upstream #23671](https://github.com/ggml-org/llama.cpp/pull/23671), [#30217](https://github.com/ggml-org/llama.cpp/pull/30217); SGLang `CudaGraphBufferRegistry` and vLLM `gpu_model_runner.py` are architecture contrasts, not transferred results. Disposable host capacity/exit-status model passed; no repo test, compilation or GPU benchmark ran.

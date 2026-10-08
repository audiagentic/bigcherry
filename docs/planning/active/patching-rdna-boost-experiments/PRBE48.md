---
id: PRBE48
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:56:49.423040+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# UP-HRX-001: AMD-native HRX backend comparative lane

## Audit disposition — 2026-10-08

**HOLD: do not create a BigCherry HRX backend, patch package, build harness, or hardware campaign yet.** This is an external backend maturity/eligibility gate, not an in-tree HIP/Vulkan optimisation. The earlier b11126 claim that no pinned HRX source was available is superseded by the exact external pins below. BigCherry's b11474 upstream pin (b9acf138a1e28ce1fc23b5a4fc4b12444b50f7ea) has no `ggml/src/ggml-hrx` directory; draft upstream PR #27218 remains open/unmerged (last updated 2026-09-03).

The October 7 AMD staging integration pins:
- `ROCm/ggml-staging-automation` commit `08264258af71` (integration staging, explicitly temporary/not a stable distribution channel).
- `AMD-Ecosystem/llama.cpp` submodule `02f2880c9f422dd1df9db11e3a0bb442bc5f11f5` (branch `hrx-graph-develop-v2`).
- `ROCm/hrx-system` submodule `c0b135a778cc1e133001ee77a5a3f74536b3f483`.
- The earlier RFC branch `users/stella/hrx-rfc-v1` / PR #27218 head `33a1f2b23975201febe33745125cd94f2c550309` is historical, **not** the latest integration baseline.

## Implementation-level finding: target dispatch is the first gate

In the pinned staging `ggml/src/ggml-hrx/dispatch_registration/dispatch-registry.cpp::find_dispatch_registry`, only **gfx1100** and **gfx1151** return registries; gfx1201 and gfx1030 return `nullptr`. `dispatch/dispatch-scheduler.cpp::schedule_graph` then fails with `no HRX dispatch registry for target`, and `DispatchScheduler::supports_node` returns false. Thus the staging build's advertised `gfx1201` *compile target* is **not proof of executable ggml HRX graph coverage** on BigCherry's R9700. gfx1030 auxiliary-device coverage is also absent. Do not allocate R9700/gfx1030 hardware lanes until a pinned revision adds and validates these dispatch paths.

On gfx1100, the implementation is a real but narrow alternative: `ggml-hrx.cpp` implements buffer/backend/device hooks and a device-local fake pointer base for GGML offset representation; `runtime/graph-executor.cpp::execute` calls `GraphProgramCache::get_or_build`, binds external values, then executes a prepared command program; `runtime/graph-program-cache.cpp` matches tensor shape/stride/type, op params, aliases and external bindings; `runtime/prepared-command-program-cache.cpp` keys prepared replay by graph UID, target, command shape and bindings hash. `dispatch_registration/dispatch-registry.cpp` composes fused Qwen attention, matmul, routed-FFN, preamble, norm and router matchers. Kernel-corpus/Loom JIT and transient-arena/buffer generations introduce different compilation, allocation, replay and host-transfer lifetimes from HIP; they are not drop-in HIP kernels.

`ggml-hrx.cpp::device_supports_op` contains generic eager capability declarations, whereas the graph scheduler additionally requires a target registry and a successful matcher. **Never equate device enumeration, successful CMake compilation or `supports_op` alone with full-model execution.** Capture the first rejected op and whether any CPU/Vulkan fallback occurred; partial offload is not an HRX speed win.

## Reuse external qualification; no duplicate BigCherry infrastructure

`ROCm/ggml-staging-automation` already owns a pinned build flow (`scripts/hrx/build/build_llama_cpp.py`), model-manifest and bounded batch runner, HRX/Vulkan PPL runner (`run_perplexity_benchmark.py`) and throughput runner (`run_lemonade_benchmark.py`). Its PPL harness distinguishes execution failure from numerical failure and tests prefill-like `-ub 512` and decode-like `-ub 1`. The October 7 model manifest marks `Qwen3.5-35B-A3B-MTP` perplexity as **expected failure** and includes `Qwen3.8-27B` only in the full tier. An XFAIL is not correctness evidence. Do not clone those runners into `tools/lab/hrx-eval/`, as the old plan proposed; use a disposable external checkout and the existing BigCherry comparison/evidence format only after Gate 0 passes.

Upstream HRX's command-buffer/graph replay plus Loom architecture JIT is a potentially transferable *mechanism* for low launch overhead, but the current runtime, dispatch registry, caches and buffer lifetimes are inseparable from HRX; **do not transplant its command scheduler, cache, allocation policy, JIT or dispatch tables into HIP/Vulkan**. The only portable hypothesis worth separately referring to an existing kernel owner is a specific measured dispatch/launch bottleneck and a minimal native mechanism, not a new framework.

External evidence only: AMD's August 17 RFC claims ~30–50% prefill and 0–15% non-MTP decode relative to the faster HIP/Vulkan backend, without absolute matched BigCherry lanes. A September 19 independent gfx1151 Qwen3-30B-A3B Q4_K_M comparison reported pp512 HRX 1724, Vulkan 1466, HIP 1650 tok/s and tg128 HRX 85.3, Vulkan 89.5, HIP 72.9; the same reporter found multi-ubatch/PPL failures, then later corrected several and identified long-context attention memory-order/dispatch costs. These are external single-device data, not gfx1100/gfx1201 evidence and not transferable to tensor-split/MTP.

## Bounded execution gate

**Gate 0 — zero hardware, repeat on a newer exact integration pin only:**
1. Record HRX llama.cpp SHA, hrx-system SHA, ROCm artifact/run ID, compiler/driver and `GGML_HRX` build flags. Confirm the pinned `find_dispatch_registry` target list and kernel-corpus coverage for the *actual* GGUF/quant; reject unsupported architectures before a build.
2. Read the staging model manifest's `expected_results`; any required model/operation marked XFAIL or skipped fails the correctness eligibility gate. Verify no in-progress BigCherry owner is already qualifying the same backend or GPU lane.
3. Do not queue a campaign until one **production-relevant single-gfx1100** model/quant has no known correctness XFAIL, a complete operation path, and an available disposable staging build. If none, close/no-campaign until upstream changes.

**Gate 1 — one disposable gfx1100 smoke only if Gate 0 passes:** use the pinned staging build/benchmark flow, same model GGUF hash, context, -b/-ub, KV type and driver for HRX vs HIP and Vulkan; force single XTX, record device list, actual offload, unsupported op/CPU fallback, PPL, greedy tokens/logits, compile/load time, peak VRAM, graph-program builds/hits, replay fallbacks and dispatch/transfer counts. Test 1- and >1-ubatch, two same-process requests, ~8K and >=32K context before throughput. Any crash, divergence beyond established matched-backend tolerance, XFAIL, partial-offload ambiguity, graph binding/lifetime failure or unsupported op => **terminal no-campaign** for this pin.

**Gate 2 — only if Gate 1 is correct:** interleaved matched pp512/2048, tg128 and production-sized server E2E with four independent sessions; compare to the **faster of HIP and Vulkan** on gfx1100. Promotion to an *optional external-backend evaluation lane* requires CI95-low >=3% E2E gain, <=1% regression in controls, no correctness/work/transfer accounting loss, and repeatable long-context stability. No production backend adoption from single-GPU evidence. The actual dual-XTX/R9700/gfx1030 no-P2P topology, MTP and Flash-Next require separate explicit multi-device coverage and correctness gates after HRX implements them; never extrapolate from W7900/Strix Halo.

**Stop states:** `no-campaign` (missing registry/ops/XFAIL), `external-smoke-rejected` (correctness/lifetime/coverage failure), `external-evaluation-only` (qualified single-gfx1100 benefit), or `upstream-wait` (no actionable supported BigCherry workload). Never introduce a third BigCherry production backend or an HRX-specific runtime selector under PRBE48.

## Ownership, dependencies and references

PRBE48 owns only this eligibility decision and evidence disposition. AMD HRX maintainers own HRX graph executor, target registry, kernel corpus and command replay. Existing BigCherry HIP/Vulkan, QFP/MET, placement, MTP and benchmark owners retain their paths. Do not modify active MTP/MoE, graph, cache, patch-system or hardware queue work. Historical RD57 successor remains PRBE48; no new technical owner is created.

- https://github.com/ggml-org/llama.cpp/pull/27218
- https://github.com/ggml-org/llama.cpp/discussions/27219
- https://github.com/ROCm/ggml-staging-automation/tree/08264258af71
- https://github.com/AMD-Ecosystem/llama.cpp/tree/02f2880c9f422dd1df9db11e3a0bb442bc5f11f5/ggml/src/ggml-hrx
- https://github.com/ROCm/hrx-system/tree/c0b135a778cc1e133001ee77a5a3f74536b3f483

## Validation recorded by this audit

Static source/manifest fixture only: 8/8 assertions passed on pinned staging target-registry coverage, fail-closed scheduler behavior and model-manifest XFAIL. No BigCherry code change, build, prototype, server run or hardware benchmark occurred. Historical b11126 plan prose was superseded; original history below is retained.

## Change Log

- 2026-09-09T10:56:49.423040+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:14:02.970413+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.345686+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.123012+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:10:25.411076+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031049_repaired-three-vulkanbackend_9010
- 2026-09-10T03:10:49.471004+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:26:04.181945+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
- 2026-09-24T02:26:30.206260+00:00 (updated-by): Updated: section:notes
- 2026-09-24T04:41:35.446106+00:00 (updated-by): Updated: section:description
- 2026-09-24T04:41:39.974354+00:00 (updated-by): Updated: section:notes

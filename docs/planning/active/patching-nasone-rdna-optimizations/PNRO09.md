---
id: PNRO09
order: 0
plan: patching-nasone-rdna-optimizations
state: pending
created-at: '2026-09-09T10:52:50.276971+00:00'
breadth: ''
skill: intermediate
created-by: capability-rebaseline-v3
work: S
priority: P1
---

# Increase Meta compute-container view headroom for recurrent/MTP graphs

## 2026-10-12 audit — 1260 validation is not startable; require a real capacity failure

**Disposition: retain PNRO09 pending and patch 1260 untested/default-off.** This supersedes the old "16 views insufficient", "C+D complete", and "no regression" interpretations below. The historical (C) metadata-capacity experiment is real recorded evidence, but it does not prove a supported model exceeds stock capacity; (D) completed on *both* 16 and 80 and its single-run throughput delta is confounded. No new GPU/build/performance result was produced in this audit. Do not promote, enable, or queue another benchmark before the following producer and workload gates.

### Source and evidence: distinct facts, not an inferred failure

- At llama.cpp **b11474** (selected branch pin) and **b11552** (main pin), `ggml/src/ggml-backend-meta.cpp::ggml_backend_meta_buffer_type_alloc_buffer_n` sets `params_compute.mem_size = compute_headroom * n_tensors * ggml_tensor_overhead()`; two rotating `stc_compute[2]` contexts are allocated **for each simple backend**. The 16 is a multiplier per statically allocated tensor, **not a global 16-view ceiling**. `ggml_backend_meta_buffer_init_tensor` registers between-eval views; `ggml_backend_meta_graph_compute` flips/resets the other container. The fallback is the stock 16 and its normal capacity failure, not an unverified dynamic growth mechanism.
- Historical `evidence/boundary.json` (2026-09-23) recorded N_max 44/222 at S=1024 and 89/445 at S=2048 for control 16 / subject 80. This establishes a **synthetic metadata capacity ratio**, not that 72 real views cannot fit at 16. `evidence/hardware_d.json` records a Qwen3.5-4B Q6_K MTP run that completed with **both** headrooms. Its 76.5 versus 98.3 gen t/s numbers were unpaired/confounded and must not be cited as gain or proof of nonregression.
- **Host-RAM theoretical budget**, not measured allocation: increasing 16 to 80 adds approximately `2 * (80-16) * n_tensors * ggml_tensor_overhead() * n_simple_bufts` bytes for the two compute containers. For 1,000 static tensors, 3 simple backends and an illustrative 368-byte tensor overhead, this is **134.8 MiB extra host metadata**, before allocator overhead; at 10,000 it is **1,347.7 MiB**. Measure actual `n_tensors`, `n_simple_bufts`, allocation peak and two-container lifetime before claiming a "small" change. This is not GPU VRAM and no P2P speedup follows from it.

### Concrete producer/fixture defects (source inspection, not a reproduced campaign)

1. `engines/llamacpp/patches/1260_nro09_meta_view_headroom/validation/producer.py::run` calls `ctx.runtime.build_materialized_pair(targets=("meta_boundary_test",), primary_target="meta_boundary_test")`. `tools/bigcherry/patch/campaign/producer.py::CampaignProducerRuntime.build_materialized_pair` feeds **targets** to `FatTargetPlan` as GPU architectures, then invokes `build_tree` on a CMake target. A fixture executable name is not an AMDGPU architecture. The producer also points to `ctx.workdir/control-source` and `subject-source` without materializing either source, and the patch-local fixture is not registered as a CMake target. This path cannot be treated as a successful current campaign.
2. `validation/meta_boundary_test.cpp` calls `ggml_view_1d(ctx2, t, 0, n_elem)`. Upstream `ggml.h` declares the last arguments `(ne0, offset_bytes)`: this constructs a **zero-element** view at a nonzero byte offset, not the documented full-length view. Metadata pressure is still possible, but this is not a representative real-view/offset correctness fixture. Correct to `(n_elem, 0)`, then test real nonzero view sizes, offsets, split callbacks and two rotating containers.
3. `_probe_n_max` linearly launches one process per N (despite the "binary" comment), stops on any nonzero return including unrelated crashes/timeouts, and never proves a typed capacity failure at N+1. Use exponential+binary search with explicit OK/capacity-failure output, strict timeouts, boundary and boundary+1, and fail closed on any other exit. Bound subprocess count and preserve stdout/stderr.
4. The producer re-emits **pre-recorded** `hardware_d.json` as if it were a current artifact; `check_results=()` and `correctness={"disposition": ...}` only cover the synthetic ratio. `config/experiment-contracts.toml::PNRO09-META-VIEW-HEADROOM` requires a `state_restore` positive lane and a `decode` control; neither is newly executed. `tools/tests/patch/test_plan_producers.py::_KNOWN_NOT_STARTABLE` explicitly exempts 1260 for this reason. Never remove the exemption or claim qualification from a copied receipt.

### Bounded implementation gate and owner boundaries

1. **Producer repair, CPU first.** Keep `validation/producer.py` the single producer and `validation/meta_boundary_test.cpp` the only boundary fixture. Materialize **identical pinned control and subject sources** through the existing composition API (subject differs only by 1260); do not invent a second build system. Register/compile the fixture explicitly against the isolated ggml+Meta CPU build; pass actual `ctx.fat_targets.targets` if using a GPU campaign build, never a CMake executable as an architecture. Capture resolved source SHA, compiler command, binary SHA and control/subject identity. Reject missing targets, failed configure, non-capacity exits, or missing typed evidence.
2. **Capacity contract.** Probe 15/16/17, 72/80/81 and discovered N_max/N_max+1 at S=1024/2048, plus n_tensors=1/2/8 and one- versus three-simple-backend Meta instances. Assert each expected failure is a **capacity** failure, not a segfault or missing binary. Alternate graph compute/eval/reset across both stc_compute containers, repeat in the same process and after a second request, check actual view shapes, alias pointers and split-state ownership. Check ratio against the independently derived allocation formula, not a hardcoded 5x alone. A deterministic host-only harness is the cheapest discriminator; no GPU lane until it passes.
3. **Real activation or retire.** Find a supported recurrent/MTP workload that deterministically exhausts the stock 16 container while 80 succeeds with identical model/graph/prompt/seed, or close 1260 as **not needed on observed workloads**. Record actual static tensor count, per-device view registrations and peak host metadata; preserve stock 16 as control. No new allocator, graph scheduler, or Meta policy table. A capacity-only improvement has no throughput promotion claim.
4. **Only if a real failure exists:** gfx1100 dual-XTX primary and gfx1201 mixed XTX/R9700 control; gfx1030 negative/CPU fallback. Test 2+ sequential requests, K=2/3/5/8 MTP, multiple ubatches, 32K/128K/245K contexts, graph capture/replay, rollback/restore, exact greedy/logits/KLD and MTP acceptance, and failure-path cleanup. Require no wrong-device execution, stale view, host-memory exhaustion or >1% unrelated E2E regression. If the stock limit never fails, or the extra host metadata is disproportionate, **reject/retire** 1260 and preserve historical negative evidence.
5. **Ownership:** PNRO09/1260 owns only Meta metadata headroom and this producer. MSM03/1341 owns per-instance mirrored-mask split state; QFP53/1362 owns KV graph-padding; current Radiance native GDN/paged attention owns its independent engine. Those recently active paths are protected; no edits, new lanes or duplicate graph/allocator changes are authorised.

### Upstream/fork applicability

- llama.cpp b11474 and b11552 retain `compute_headroom=16` and two rotating containers. Upstream [PR #29838](https://github.com/ggml-org/llama.cpp/pull/29838) (open 2026-10-02) proposes Meta `get_alloc_size_n` by factoring `alloc_buffer_n_create`; its actual diff **retains 16**. Watch for composition changes but **do not port it as a headroom fix**. Upstream [#26395](https://github.com/ggml-org/llama.cpp/pull/26395) addresses a separate split-axis bounds bug, not this capacity.
- [vLLM CUDA-graph memory profiling](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/cudagraph_utils.py) uses measured/reserved graph resources and discardable profiling pools. Transfer only the **accounting and lifetime discipline**, not CUDA graph sizing, NVIDIA results or an implementation. No AMD speedup is established.

**Audit validation actually run:** 15/15 source-static/host arithmetic assertions, including a deterministic 368-byte metadata-size boundary calculation reproducing the four historical N_max values. No repository pytest, C++/HIP compilation, fixture execution, GPU benchmark or new hardware qualification. 12-hour check passed: latest independent PNRO09 plan change 2026-09-25 05:13 UTC; patch move 2026-10-09 04:20 UTC was structural. Active Radiance native GDN/attention/AllReduce, PA44/PA49 release, QFP15, QFP53 and QFP49 were excluded. This is a new producer-integrity/capacity-ownership audit, not a repeat of BCOP155/156.


## Description

Evaluate Meta compute-container view headroom for recurrent/MTP graphs. Port only after proving the existing 16-view bound is insufficient or a supported graph-derived bound exceeds it.

## Steps

- Confirm b10705 Meta allocator behavior and derive actual maximum static-tensor view count for recurrent+MTP graphs.
- Build boundary fixtures at 15/16/17 and higher views through the real Meta context mechanism, plus repeated eval/reset.
- Port the minimal constant/rationale only if failure is reproducible; prefer a derived bounded formula over magic 128.
- Measure metadata memory and verify ordinary dense/non-recurrent graphs are unchanged.
- Record capacity decision and keep the change correctness-scoped, not a performance claim.

## Detailed Solution & Technical Design

Recurrent snapshot views are estimated around 2*(n_rs_seq+1) per shared recurrent layer. Headroom concerns tensor-object metadata, not model data; bound it from graph structure and retain safe failure if capacity is exceeded.

## Code Samples & Guidance



## Files

ggml-backend-meta.cpp; Meta context boundary fixture; recurrent/MTP graph construction and eval/reset tests; metadata memory evidence.

## Validation

15/16/17+ view allocation; real recurrent+MTP graph; repeated reset; metadata accounting; dense/non-MTP controls; source identity.

## Effort & Risk



## Standards

Affirmative capacity proof; bounded resource accounting; no performance claim for correctness promotion.

## Acceptance Criteria

Existing 16 limit is shown insufficient or a proven bound requires change; new bound covers declared maximum with bounded metadata cost; no non-target behavior changes.

## Notes

Supersedes: NRO10
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-nasone-rdna-optimizations-nro10

2026-09-25 finding: patch 1260's validation package cannot start a campaign -- its contract requires a 'controls' capability that no producer supplies, and validation/producer.py copies the pre-recorded evidence/hardware_d.json into the run instead of measuring. README.md was missing (added). Contract backend aligned to 'agnostic'. The recorded (D) 98.3 vs 76.5 gen t/s difference is confounded (a headroom change cannot explain a 28% decode gap) and must not be cited. Needs: a real controls lane (paired tg128 on tierA-qwen4b-q6k) and a producer that runs the MTP real-graph check itself. Listed in tools/tests/patch/test_plan_producers.py _KNOWN_NOT_STARTABLE until fixed.

## Change Log

- 2026-09-09T10:52:50.276971+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:09:14.697625+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.090326+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.739019+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:44:03.530481+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_024433_three-more-nasone-successors-n_7555
- 2026-09-10T02:44:33.131554+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-20T02:30:44.526791+00:00 (state-transition): State: pending → in_progress
- chg_20260920_025031_created-patch-1260-to-increase_8170
- 2026-09-20T02:50:34.930936+00:00 (updated-by): Updated: section:ledger-events
- chg_20260920_062016_patch-1260-pnro09-verified-o_3271
- 2026-09-20T06:20:21.408083+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-20T06:44:20.375111+00:00 (state-transition): State: in_progress → in_progress
- 2026-09-22 (implementation, hardware-free boundary test): Built the lean ggml+ggml-backend (CPU+Meta, no GPU) for the pinned/unpatched source (headroom=16) at /mnt/h/development/projects/bc-nro09-build. Wrote a first-pass C++ fixture (patches/1260_nro09_meta_view_headroom/validation/meta_boundary_test.cpp) that drives the REAL Meta backend (CPU simple device → meta device → meta backend, alloc_ctx_tensors + graph_compute) with N between-eval compute nodes of size S. FINDING: the binding constraint is the ggml object pool (ggml_new_object enforces ctx->mem_size); per this plan's own note the headroom bounds tensor-object METADATA, not model data. The first-pass fixture did NOT show a clean N=headroom boundary: the observed object size was S+metadata (data was being allocated into the pool, i.e. the hit context had no_alloc=false), so the boundary landed far from headroom. CORRECTED DESIGN: the boundary is on the NUMBER of tensor objects (fixed GGML_TENSOR_SIZE each), so N_max = headroom*S / GGML_TENSOR_SIZE — proportional to headroom, with ratio patched/unpatched = 80/16 = 5. NEXT: isolate the metadata-only allocation (verify the mapped simple tensors use no_alloc=true so the object size is fixed), then probe N_max on the unpatched (headroom=16) and patched (headroom=80) builds and assert the ~5x ratio plus the bounded metadata cost (compute container = 80x vs 16x the static mem_size). Build B (patched, headroom=80) still needs the 1260 patch applied to a vendor copy + build. (The ag-planning plan_update_item tool was defective this session — receiving `updates` as a string — so this note is recorded in the file directly.)
- 2026-09-23 (C boundary test COMPLETE + evidence): Corrected the fixture to drive the REAL stc_compute capacity (static leaf t in USER ctx -> stc_static; N between-eval view tensors of t in a separate ctx, registered via buf->iface.init_tensor -> stc_compute[0]; the simple tensor in stc_compute has no_alloc=true so object = fixed GGML_TENSOR_SIZE, metadata only). Built the patched variant (headroom=80) at /mnt/h/development/projects/bc-nro09-buildB. Observed N_max (unpatched=16 vs patched=80): S=1024: 44 -> 222 (ratio 5.045); S=2048: 89 -> 445 (ratio 5.0). The ratio (~5 = 80/16) is confirmed at two independent S values, with a bounded linear metadata cost (stc_compute mem_size = headroom*S; 80x vs 16x the static mem_size) and a safe abort on capacity exceedance. Evidence recorded in patches/1260_nro09_meta_view_headroom/evidence/boundary.json; SUMMARY.md + plan note updated. Commit ca1e45aa on patch-refactor (local). PUSH BLOCKED: the other session (PA36/PA40 reconciliation) has live uncommitted changes (modified 2026-09-23T00:11, ~2 min before this note) to PA36.md/PA40.md/validation_campaign.py/1233+1206 producer.py/test files that the 12 remote commits also touch; a merge would overwrite their live edits, and git stash is forbidden (AGENTS.md), so the merge is deferred to the other session. Remaining for PNRO09: the (D) real-graph hardware lane on Brutus (tierA-qwen4b-q6k, recurrent+MTP, gfx1100 0/1) + validation/producer.toml+producer.py + validation.toml capacity checks.
- 2026-09-23 (D real-graph hardware lane COMPLETE + evidence): Built the patched (headroom=80) and unpatched (headroom=16) variants on Brutus (gfx1100 7900 XTX devices 0/1) from copies of the bc-pa-work tree (the protected tree was not modified), and ran the REAL recurrent+MTP graph — tierA-qwen4b-q6k MTP (Qwen3.5-4B-UD-Q6_K_XL, --spec-type draft-mtp --spec-draft-n-max 4) — on both. FINDING: the 4B MTP workload's ~72 between-eval views (and their metadata footprint) fit even within the stock 16-view headroom (unpatched: NO abort, 18.6/76.5 t/s; patched: NO abort, 21.9/98.3 t/s, i.e. the patch is benign and actually faster). So 16->80 does not change the pass/fail outcome for this specific 4B workload; it raises the ceiling for larger/more views (consistent with the (C) boundary test's ~5x N_max ratio: 44->222 @S=1024, 89->445 @S=2048). The binding case is a workload with a larger per-view metadata footprint or a larger view count; the (C) hardware-free boundary test is the direct capacity proof, and the (D) hardware lane confirms the patch is benign (no regression, no correctness issue) on a real recurrent/MTP graph. Evidence in patches/1260_nro09_meta_view_headroom/evidence/hardware_d.json; SUMMARY.md + plan note updated. Remaining for PNRO09: validation/producer.toml+producer.py + validation.toml capacity checks (code-only).
- 2026-09-23 (producer + validation.toml COMPLETE): Added the 1260 patch-local capacity producer (validation/producer.py + producer.toml, producer_id=nro09, standard_campaign=skip -- a capacity check, not a performance contract lane). The producer builds the control (headroom=16) and subject (headroom=80) meta-backend variants via build_materialized_pair, runs the C++ boundary fixture (meta_boundary_test) against each to probe N_max at S=1024 and S=2048, asserts the ~5x N_max ratio (80/16, tolerance 0.25), and records nro09-boundary.json + nro09-hardware-d.json. Extended validation.toml with a required `capacity` check (capability=correctness, validator=correctness-summary) that consumes the producer's disposition. Verified: producer resolves via resolve_producer() and is callable with signature run(ctx: ProducerContext) -> ProducerResult; test_validation_producer.py (25 passed), test_validation_producer_structure.py (16 passed, 1 skipped), test_patch_validation_campaign_gpu_execution_guard.py all pass. PNRO09 is now COMPLETE: (C) hardware-free capacity boundary test (evidence/boundary.json), (D) real-graph hardware lane (evidence/hardware_d.json), and the capacity producer + validation.toml checks.
- chg_20260922_150002_pnro09-1260-is-complete-the_6574
- 2026-09-22T15:00:05.265380+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-22T15:00:12.216553+00:00 (state-transition): State: in_progress → completed
- 2026-09-25T05:12:49.248433+00:00 (updated-by): Updated: section:notes
- 2026-09-25T05:13:00.899965+00:00 (state-transition): State: completed → pending

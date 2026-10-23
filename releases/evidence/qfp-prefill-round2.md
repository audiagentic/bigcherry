# Qwen Flash-Next prefill round 2

Pin: llama.cpp `b11474` (`b9acf138a1e2`). Production reference: Flash-Next IQ4_XS, f16 KV, ub512, 2x gfx1100 + gfx1201 tensor split, gfx1030 draft.

Notation: **V** = verified from current BigCherry/main or pinned upstream source/evidence. **I** = inference to qualify on Brutus.

## Ranked remaining work

| Rank | Plan / mechanism | Exact target | Prefill share it can touch | Expected end-to-end gain | VRAM | Risk | Applicability / disposition |
|---:|---|---|---|---|---|---|---|
| 1 | **QFP41 per-device dispatch workers** | **V:** `ggml/src/ggml-backend-meta.cpp::ggml_backend_meta_graph_compute`, specifically the serial inner loop calling `ggml_backend_graph_compute_async(bcj.backend, bcj.cgraphs[i].cgraph_main)` for each simple backend before the subgraph AllReduce boundary. | **V:** QFP41 records ~40% split idle as the opportunity. QFP36 Gate 0 measured only ~28 s target-GPU kernel time in a 99 s profiled span (~72% non-kernel wall upper bound); that larger number is not attributable to dispatch alone. | **I:** +2-8% if host submission of the three simple-device graphs is on the prefill critical path; zero if the serial calls are already negligible relative to GPU/collective waits. | 0 GPU bytes; persistent host worker stacks/queues only. | High: backend-thread safety, graph/cache ownership, collective ordering. | **GO, first implementation.** Persistent workers only; graph rebuild, split-state cache, simple-tensor containers, arena planning/binding and collective decision remain on the caller thread. Default off during qualification. |
| 2 | **QFP35 residual HC gate -> HC_POST fusion** | **V:** `src/models/qwen4exp.cpp::build_hc_combine`: `SCALE(inject,1/hc) -> SIGMOID -> SCALE(2) -> DSV4_HC_POST`, twice/layer. 1313 already consumes the first three nodes; `ggml/src/ggml-cuda/dsv4-hc.cu::dsv4_hc_post_f32` still reads the materialized F32 `post` tensor. | **No isolated timing data.** 1313's census found ~98 SCALE launches/token/GPU, mostly HC chains; 1313 removed the scale/activation launch portion and the remaining HC_POST launch/materialization is not separately timed. Produce it with `DEPTH=24576 tools/lab/flash-next/long-ctx-profile.sh <server> <out> prefillprof` plus a graph/patch-hit census. | **I:** +0.3-1.5%; lower than 1344/1345, but exact, zero-workspace and launch/bandwidth focused. | No added VRAM; should remove one transient F32 gate materialization when fused. | Medium: graph fusion/lifetime and exact sigmoid/scale semantics. | **GO, second implementation.** New narrow consumer-fusion package requiring 1313; do not duplicate 1313's scale-act kernel. |
| 3 | **QFP36 router split-K** | **V:** `src/llama-graph.cpp::build_moe_ffn`, router `logits = build_lora_mm(gate_inp, cur)`; candidate dispatch in `ggml_cuda_mul_mat` only when that exact ordinary MUL_MAT is marked. Physical router output M remains 512 in current Meta mirrored-weight path; token N may be split. | **No router-only data.** Gate 0: float matmul family 12.4-16.0% of target-card kernel time. 1347 later identified one SGEMM family at 12.7%, 141 calls/chunk, 96 HC-inject calls; router contribution/type is still unseparated. Produce with a router-marked dispatch census + `DEPTH=24576 ... long-ctx-profile.sh ... prefillprof`. | **V plan estimate / I until census:** +0.1-0.6% if F32/F32/F32 and material; otherwise 0. | `8*M*N*sizeof(float)` split-K workspace; for M=512,N=512 ~8 MiB per active physical router call before allocator reuse. | Med/high: reduction-order changes can change top-k experts. | **DEFER pending Gate 0.** 1345 already covers QFP36's ids-helper half. |
| 4 | **QFP44 residual MMQ tile/LDS / QFP37 I=32 handoff** | **V:** `launch_mul_mat_q` / `mul_mat_q_switch_J` and RDNA config rows; validated 1237/1265 already own compact MoE block maps. 1350 in PR #22 targets ordinary few-tile Q8_0 Stream-K; QFP37 reserves 1351 for IQ4_XS expert I=32. | **V:** QFP37 Gate 0 put ordinary Q8_0 MMQ at ~15% of one XTX's kernel time; IQ4 expert MMQ ~9.4%. That is family headroom, not residual-after-compact-map opportunity. Re-census with 1237/1265/1345 and 1350 controlled. | **I:** 0-1.5% residual unless an underoccupied hot shape remains. | Normally 0 persistent VRAM; tile/LDS changes affect per-block LDS/register occupancy. | Medium, shape/arch specific. | **DEFER to QFP37/QFP44 census.** Do not duplicate compact map; close if 1350/1351 remove material under-occupancy. |
| 5 | **PRBE70 CK profiler oracle** | **V:** offline hot-signature pipeline for `ggml_cuda_mul_mat_cublas_impl` / `launch_mul_mat_q`; no runtime dependency. | Directly touches 0% at runtime. It should rank only signatures >2% kernel share. Capture with `DEPTH=24576 GGML_CUDA_OP_TIMING=1 tools/lab/flash-next/long-ctx-profile.sh <server> <out> prefillprof`, then profile those shapes offline with the installed `ckProfiler --help` syntax. | **V:** 0% direct; enables later dense-GEMM choices. | 0 runtime VRAM. | Low. | **DO as tooling before PRBE28/29/31**, but not one of the two runtime patches. |
| 6 | **PRBE28 bounded dense row padding** | **V:** weight-specific `llama_model_loader::create_tensor/load_all_data`; consumer `ggml_cuda_mul_mat_cublas_impl`. Never change generic `ggml_new_tensor_impl`. | **No current hot-signature data.** PRBE70 must identify eligible dense GEMMs and L2 alias counters first. | **I:** 0-1% E2E unless a >2% hot signature shows causal cache-set conflicts. | Hard initial cap <=256 MiB/device; production is already 92-97% full. | High layout/mmap blast radius. | **DEFER.** |
| 7 | **PRBE29 selected F16 shadow** | **V:** selected non-expert dense weights only, shadow after `load_all_data`, reuse `ggml_get_to_fp16_cuda`, dispatch before native MMQ only at measured crossover. | **No current hot-signature/crossover data.** PRBE70 first. | **I:** possible >1% only on repeatedly hot large-M dense shapes. | ~2 bytes/scalar shadow; 1B scalars ~1.86 GiB. Initial experiment <=256 MiB/device. | High due 92-97% production VRAM. | **DEFER. Whole-model shadow infeasible.** |
| 8 | **PRBE30 shadow K-padding** | **V:** only an extension of PRBE29 shadow allocation/leading dimension. | No data; blocked on PRBE29 + alias evidence. | **I:** likely sub-1% E2E. | <=64 MiB/device extra initial cap, on top of shadow. | Medium. | **BLOCKED on PRBE29.** |
| 9 | **PRBE31 hipBLASLt shadow crossover** | **V:** large-M dense shadow consumer; b11474 availability/linkage must be rechecked before coding. | No data; blocked on PRBE29 and PRBE70 signature/crossover results. | **I:** >=1% only if Lt beats native MMQ on a hot shape and workspace fits. | Shadow + Lt scratch; unsafe as broad default at current headroom. | High. | **BLOCKED on PRBE29/70.** |
| 10 | **QFP38 MTP verify launch geometry** | **V:** `mmvq.cu` already keeps current width-4/5 IQ4_XS verification on MMVQ; 1301 raw-F32 Q8 verify was measured neutral/regressive. Remaining candidates are width-specific MMVQ/MMVF/MoE launch geometry. | Primarily decode/verify, not target prefill. No current prefill materiality data. | **I:** negligible for prefill; decode-only opportunity must be census-led. | 0 persistent VRAM. | Medium. | **DEFER from prefill round.** |
| 11 | **QFP40-B residual copy audit** | **V:** validated 1308 already removes the recurrent rollback `ggml_cont` copy and owns that copy family; it accounted for ~89/108 copy launches/token and measured ~3-3.6% decode improvement. | No remaining prefill share demonstrated. | 0 until a new owner is proven. | 0. | Low audit risk. | **CLOSE AS COVERED for this prefill round.** Any residual different copy family gets a new plan; same family stays in 1308. |

## Covered mechanisms / precedents

- **QFP35 indexing:** **V covered** by validated production patch 1344. Flash-Next prefill +1.6% at 8K and +1.3% at 24K, greedy/probe identity; native flat PRE/POST kernels remain the off arm.
- **QFP36 ids helper:** **V covered** by validated production patch 1345. Gate 0 found `mm_ids_helper<10>` ~3.8% of target-card kernel time; qualified +1.6-2.1% prefill with byte-identical maps and greedy output.
- **QFP37 ordinary few-tile Q8_0:** **V measured, not yet on main** in PR #22 / 1350: +1.8/+1.2/+1.9% at 8K/24K/98K with complete separation. It does not cover MoE I geometry.
- **F32 thin HC projections:** **V covered** by 1347: the 4-row HC projections accounted for 96/141 calls in its SGEMM family; +3.1-4.6% Flash-Next prefill, with CPU-F32 fidelity no worse than production.
- **MTP prompt serialization:** **V covered in part** by 1348 at b11474: +5.4/+6.3/+9.7% prefill at 8K/24K/98K, greedy-identical. Its documented remaining blocker is the per-chunk NextN synchronization/staging path (QFP42), not QFP41.
- **QFP44 compact MoE map:** **V covered** by validated 1237 (gfx1100) and 1265 (gfx1201/gfx1030). QFP44 is residual tile/LDS evidence only.

## Round-2 implementation choice

1. **QFP41:** persistent per-simple-backend dispatch workers. Only `ggml_backend_graph_compute_async` calls for the same subgraph index move to workers. Rebuild, split-state/cache/container mutation, arena planning/binding, AllReduce selection/fallback, copies, callbacks and joins remain on the caller. This is deliberately default-off until identity/repeat/stress evidence.
2. **QFP35:** fuse the already-composed 1313 `SCALE -> SIGMOID -> SCALE` gate directly into its sole `DSV4_HC_POST` consumer. Keep the existing DSV4 arithmetic/order and native path as off arm.

Reserved patch identities were not reused: 1349 is named by QFP36 router split-K, 1350 is PR #22, 1351 is QFP37 I=32, and 1352-1354 are named by QFP38. The next unassigned identities for these slices are **1355** and **1356**.

## Qualification commands

For both runtime slices, build the production source plus only the experiment package, then use one binary and complete process separation:

```bash
# queue-env-ab.sh defines A = production environment and B = A + AB_ENV.
AB_ENV="<subject env>" \
  tools/lab/flash-next/queue-env-ab.sh <run-name> <build-run-id> 8192 24576 98304
```

Each slice README gives the exact `AB_ENV`; use `FIDELITY=1` when the mechanism changes floating-point grouping.

Required evidence for both: `BIGCHERRY_PATCH_HIT` in the subject arm, same production model/topology/f16-KV/ub512, prefill t/s at all three depths, decode control, peak per-device VRAM, greedy output identity. QFP41 additionally requires repeated-run identity/determinism and a deep-prefill + long-decode stress run with no hang/crash; host-side ThreadSanitizer where practical.

## Verification boundary

Verified source/evidence above comes from current `main`, pinned upstream `b9acf138a1e2`, and committed promotion records. Expected gains for unmeasured mechanisms are explicitly marked inference and are not promotion evidence.

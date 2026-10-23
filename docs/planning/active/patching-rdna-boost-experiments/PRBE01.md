---
id: PRBE01
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:53:31.014682+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P2
---

# Implement final native-BF16 flash-attention logical experiment

## Description

Port and qualify the folded native-BF16 flash-attention logical path from the reviewed commit series. Existing isolated evidence is directionally promising (clean decode gain and no measured prompt cost) but correctness and cross-architecture repeat remain before promotion.

## Steps

1. Port only the reviewed logical BF16 path as one precision experiment; preserve F32 accumulation and native-BF16 guards.
2. Keep unsupported hardware on the existing fallback and preserve no-F16-regression behavior.
3. Separate cleanup/operand-order evidence from the implementation arm.
4. Run the FLASH_ATTN_EXT backend correctness matrix, BF16 same-path checks, and exact output/precision checks without claiming F16 bit identity.
5. Run long/deep-context numerical-quality checks and repeat on gfx1100 and gfx1201.
6. Reproduce isolated decode/prompt characterization with balanced interleaved controls, then gate promotion on correctness, precision, quality, fallback, cross-architecture, and performance evidence.

## Detailed Solution & Technical Design

Treat this as a precision-path experiment, not six cherry-picks. Validate architecture guards, unsupported fallback, F32 accumulation, BF16 operand semantics, graph capture, and numerical stability. Existing evidence: materialized patch 1202; isolated gfx1100 decode was directionally positive and prompt deltas were noise-adjacent/indistinguishable at pp>=1024. This is not a promotion claim.

## Code Samples & Guidance



## Files

Future vendor HIP flash-attention sources; patches/1202_rd04_bf16_flash_attn_tile; correctness fixtures and backend matrix; balanced E2E evidence for gfx1100/gfx1201

## Validation

Build and apply/idempotence: `PYTHONPATH=tools python -m bigcherry patch-lint patches/1202_rd04_bf16_flash_attn_tile`; `PYTHONPATH=tools python -m bigcherry patch-rebase-check --focal-overlay 1202_rd04_bf16_flash_attn_tile --source bigcherry-tuning`; package pytest offline. CORRECTED (source-verified): 1202's 35 source anchors match uniquely at b11126, but the patch has NO BIGCHERRY_PATCH_HIT/BIGCHERRY_PATCH_TRACE marker while validation.toml's activation-probe check requires one -- add a BIGCHERRY_PATCH_TRACE-gated GGML_LOG_WARN marker at the actual native-BF16 branch inside `ggml_cuda_flash_attn_ext_tile(...)` (ggml/src/ggml-cuda/fattn-tile.cu, the real dispatch switch on K->ne[0] -- NOT a `..._tile_case_type` function, which does not exist in the source) when the BF16 K/V case dispatches; require subject-hit/control-miss evidence per validation.toml's existing (currently BLOCKED) marker-probe check. The correctness producer is NOT test-backend-ops -o FLASH_ATTN_EXT and NOT the old `--overlay --arch` campaign invocation -- per PA35/PA36 and this patch's own validation.toml, correctness is a real whole-model PPL comparison (backend_reference + ppl_equality) driven by the package-local producer, invoked as `--validation-producer 1202_rd04_bf16_flash_attn_tile/rd04`. Hardware (dev, gfx1100 AND gfx1201 both required): run the package-local producer per validation.toml (apply/build/performance/controls/correctness checks), reproducing isolated decode/prompt characterization with balanced interleaved controls; promotion requires correctness+precision+quality+fallback+cross-architecture+performance evidence together, not isolated throughput alone.

## Effort & Risk



## Standards

Correctness before performance; explicit precision contract; fallback preservation; no claim of F16 bit identity.

## Acceptance Criteria

All RD04 requirements are either passed with evidence or explicitly dispositioned: F32 accumulation/native-BF16 guards, unsupported fallback, FLASH_ATTN_EXT correctness matrix, BF16 same-path checks, long/deep-context quality, gfx1100/gfx1201 coverage, unsupported fallback, no F16 regression, and balanced performance. No promotion on isolated throughput alone.

## Notes

Supersedes: RD04
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd04

Supersedes: RD04
Inherited semantic scope: preserve every actionable RD04 gate; historical evidence remains on completed predecessor.
Migration: capability-rebaseline-v3-2026-09

2026-09-24 relevance at b11126: IMPLEMENTED-AS-PATCH. patches/1202_rd04_bf16_flash_attn_tile exists, state=untested. Item's own detailed_solution already records isolated gfx1100 evidence as directionally positive (decode gain, prompt deltas noise-adjacent at pp>=1024) but explicitly not a promotion claim -- cross-architecture (gfx1201) repeat and the full correctness/quality gate matrix are still outstanding. No upstream b11126 native-BF16 flash-attention path found matching this scope (b11126 ggml-cuda flash-attn is FP16/F32-path generic, not this project's folded BF16 logical-path experiment). Disposition: validate/qualify existing patch; no GPT design needed (patch already implements the described precision-path experiment).

2026-09-24 GPT review req_215c89d0b13a4bb7 applied: verified 1202's 35 anchors are unique at b11126 but confirmed (grep) it has no activation marker although validation.toml requires one for its marker-probe check -- added a required BIGCHERRY_PATCH_TRACE-gated WARN marker at the real dispatch function `ggml_cuda_flash_attn_ext_tile` in fattn-tile.cu (not the nonexistent `..._tile_case_type`). Removed the stale test-backend-ops/old campaign validation language in favor of the actual package-local producer workflow (`--validation-producer 1202_rd04_bf16_flash_attn_tile/rd04`) already declared in validation.toml.

## 2026-10-08 b11474 experiment scope

**Rank 6, BF16 attention qualification.** `1202_rd04_bf16_flash_attn_tile` is **untested** (patch.toml), not production; dispatch seam `ggml/src/ggml-cuda/fattn-tile.cu::ggml_cuda_flash_attn_ext_tile` BF16 K/V tiles with F32 accumulation. The b11474 upstream #29901 tiled lightning **indexer** optimizes indexing, not the native BF16 precision path, so it is not evidence of equivalent BF16 arithmetic. Scope `K/V=BF16, head 64/128/256, GQA 1/4/8, q_rows=1..16 and 512-token prefill, ctx=8K/24K/98K`, gfx1100/gfx1201 with gfx1030 fallback. Gate with existing patch activation / `BIGCHERRY_PATCH_TRACE=1` and a new test-only `BIGCHERRY_BF16_FA_TILE=0|1` only if 1202 lacks a production-independent A/B switch (do not assume such a switch exists). Expected gain: uncertain, target >=1% BF16 decode with no prefill regression; FP16 KV production is a non-activating control, no benefit claim on it. Run FLASH_ATTN_EXT/backend reference, long-context PPL, graph capture, native-vs-1202 `tools/lab/flash-next/queue-env-ab.sh`; require marker subject only and matched BF16 numerics before adoption.

Recheck every historical b11126 anchor on the composed b11474 source before coding; no GPU run or patch implementation is claimed by this plans-only triage.

## Change Log

- 2026-10-08 (triage): Kept experiment pending (P2); source/shape, coverage against #29901/1202/1253, env gate, expected effect and separated hardware test defined above.

- 2026-09-09T10:53:31.014682+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:10:07.152011+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.129465+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.797954+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:10:06.643778+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes
- chg_20260910_021313_the-semantic-audit-is-now-trac_4827
- 2026-09-10T02:13:13.308366+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:26:45.540247+00:00 (updated-by): Updated: section:validation, section:notes
- 2026-09-24T04:40:44.032183+00:00 (updated-by): Updated: section:validation, section:notes


## 2026-10-11 b11474 audit: qualify actual native-BF16 TILE routing before performance

**Disposition: pending / default-off; no new implementation or GPU run authorised by this audit.** Patch `1202_rd04_bf16_flash_attn_tile` already implements the fork's native BF16 TILE arithmetic; upstream b11474 has native BF16 **VEC** support independently (merged llama.cpp #20525). The optimisation is specifically the TILE BF16-K/V branch, not every `-fa on -ctk bf16 -ctv bf16` invocation. The b11474 native selector, rather than the requested KV dtype, determines whether it can activate.

### Implementation and lifetime boundary (pinned b11474)

- `ggml/src/ggml-cuda/fattn.cu::ggml_cuda_get_best_fattn_kernel` chooses VEC/TILE/MMA_F16 using head dimensions, query rows, GQA, mask, KV padding and GPU capability. `ggml_cuda_flash_attn_ext_get_alloc_size` stages K/V for the selected kernel; 1202 changes only the TILE case to avoid F16 staging for **both** BF16 K and BF16 V on BF16-capable devices. MMA_F16 still stages F16; VEC already supports BF16 natively.
- `ggml/src/ggml-cuda/common.cuh::amd_mfma_available` is CDNA-only; `amd_wmma_available` is true for RDNA3/RDNA4. On masked, aligned GQA=4/8, head 64/128/256, query rows=512, the RDNA WMMA gate selects **MMA_F16**, so the planned pp512 GQA-positive lane does **not** exercise 1202. With GQA=1 the same pp512 cases select TILE; with query rows=1, GQA=1 selects VEC and GQA=4/8 selects TILE. Mask absence, ALiBi, padding and different head sizes change this; never infer route from token count alone.
- `1202/patch.py` creates `ggml_cuda_flash_attn_ext_tile_case_type` in `fattn-tile.cu`, dispatching BF16 only when K and V are both BF16 and `bf16_mma_hardware_available` passes. `fattn-tile.cuh` implements BF16 dot2 and packed PV with F32 accumulation; `common.cuh` gates the RDNA3/4 `v_dot2_f32_bf16`/permutation helpers. The branch must preserve F16 fallback, softmax/recurrent output ownership, scratch bounds and graph replay. gfx1030 is a nonactivating fallback, not a BF16 TILE performance target.
- The patch is a generated 35-edit net fork diff. Upstream #26289 (merged 2026-09-27) changed F16 TILE configuration code in `fattn-tile.cuh`; it is **not** evidence of an AMD BF16 gain. Run actual sequential patch composition, anchor/cardinality/idempotence and build on the b11474 pin before trusting old b11126 anchor checks. Do not claim that an unrun patch composition passed.

### Evidence contract correction (source and tracked records)

- `1202/validation/producer.py::run` forces BF16 KV flags but **does not prove TILE activation**. It derives both `backend_reference` and `ppl_equality` from one `PerplexityComparison.ok` boolean; those are not independent checks and `backend_reference` does **not** currently compare token logits. Keep PPL as the PPL signal; require a separately evidenced per-token/full-vocabulary logits/KLD reference before using the backend-reference name as numerical proof.
- `1202/validation/producer.toml` explicitly has `trace_probe = "skip"`, and the patch has no `BIGCHERRY_PATCH_HIT` marker despite `validation.toml` requiring one. Existing activation remains **BLOCKED**. The historical plan's statement that the marker was added is incorrect; its implementation and actual subject-hit/control-miss evidence are outstanding.
- Nine tracked `1202/evidence/validation.json` records all have `eligible_for_validated_state=false`. The three b11126 architecture records mark correctness **FAIL** because `llama-perplexity` returned no parseable final PPL line; that is a **measurement/producer failure**, not demonstrated logits or PPL numerical divergence. Earlier b10901 gfx1100 PPL passes are historical, not b11474 qualification.
- External 2026-08-20 gfx1100 7900 GRE BF16-KV decode measured **+3.39% tg128** (Ministral-3-14B Q4_K_M); pp>=1024 showed no measured cost. This is an earlier pin/hardware/model, not a current BigCherry promotion. No new GPU performance was measured here.

### Cheapest discriminator and exact implementation gate

1. **Host selector fixture before hardware:** use the pinned `ggml_cuda_get_best_fattn_kernel` predicates, with explicit head, GQA, q_rows, mask, ALiBi, K padding and CC. This audit's 108-case illustrative RDNA3/RDNA4 host grid (head 64/128/256, GQA 1/4/8, q_rows 1/2/4/8/16/512, mask present, K padded) produced 52 TILE, 6 VEC and 50 MMA_F16 classifications; five route assertions passed. This is a source-derived fixture, **not a compiled backend selector test**. Add the smallest actual selector regression to existing 1202 mechanics tests, not a new routing subsystem.
2. **Patch activation:** add one trace-gated marker at the **executed BF16 TILE dispatch** in `fattn-tile.cu::ggml_cuda_flash_attn_ext_tile_case_type` (introduced by 1202), not merely at KV flag parsing. Require subject hit/control miss, actual kernel family and staging-byte counts. Use existing trace/rocprof evidence and per-run identity; a once-per-process hit cannot prove every request/shape. Check K/V aliases, staging scratch lifetime, graph capture/replay, repeated requests, multi-ubatch and no-P2P multi-GPU controls.
3. **Numerics:** separate true full-vocabulary/token logits (defined tolerance, KLD and greedy/MTP acceptance) from corpus PPL; compare native b11474 BF16-VEC and F16-staged TILE controls where appropriate, and F32 as a quality reference. Do not demand F16 bit identity from a BF16 precision experiment. A missing PPL output is an invalid run, not an accuracy failure.
4. **Cost and promotion:** first attribute executed BF16 TILE time and removable F16-staging bytes for real production shapes. Reject if no TILE hits or exposed removable E2E time <2.9126% (theoretical time reduction needed for +3% throughput even with a free replacement). Only then qualify isolated gfx1100/gfx1201 BF16-KV decode and a **verified TILE** prefill shape, plus F16-KV, MMA_F16, VEC and gfx1030 fallback controls. Minimum four independent sessions/architecture and ten paired ABBA rounds; CI95-low >=3% E2E improvement, <=1% control regression, no missing work or numerical/graph regressions. Keep 1202 untested/default-off until all gates pass; otherwise close or retain blocked with the exact missing receipt.

**Ownership/consolidation:** PRBE01/1202 owns native BF16 TILE only; upstream #20525 owns native BF16 VEC, native b11474 owns MMA_F16/route selection, PRBE54/1271 owns Q5 staging, and active QFP49/QFP41/QFP48 own their own prefill/Meta/kernel profiling lanes. No new dispatcher, allocator, telemetry system or competing hardware queue. External reference: [vLLM ROCm attention backend](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/rocm_attn.py) gates head/layout support rather than treating requested dtype as proof of a kernel path; this is a routing-contract reference, not transferable AMD speedup. Upstream sources: [#20525](https://github.com/ggml-org/llama.cpp/pull/20525), [#26289](https://github.com/ggml-org/llama.cpp/pull/26289), [b11474 selector](https://github.com/ggml-org/llama.cpp/blob/b11474/ggml/src/ggml-cuda/fattn.cu).


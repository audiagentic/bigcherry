# BigCherry Experimental-Intent & External Optimization Guide

Agent handoff for `tuning-code-rebase` — 20 August 2026

Reformatted from the original pandoc export (`BigCherry_Experiment_Contract_Implementation_Guide.md`, downloaded 2026-08-20) for readability; no content changed. Tracked as plan `experiment-contracts` (items EC01-EC12) in `docs/planning/`.

## 1. Mission

Extend BigCherry in-place so external or experimental optimizations can be expressed as small, durable experiment contracts and evaluated by the existing autotune/campaign/evidence machinery. Do not build a second benchmark framework, a second candidate schema, or a model-name-based runtime dispatcher. The missing layer is experimental intent: what an optimization claims to improve, the signatures/workloads that should trigger it, controls that must not regress, boundary cases that define its safe envelope, correctness requirements, and promotion thresholds.

## 2. Non-negotiable architecture

- Keep runtime candidate identity separate from source provenance and from experiment identity.
- Keep BigCherry kernel families unchanged: mmvq, mmq, mmvf, mmf, blas.
- Workload labels (decode, prefill, mtp_verify, moe_prefill, moe_decode, long_context, gdn_prefill, multi_gpu_copy) are experiment metadata, not kernel-family identity.
- Prefer captured canonical signatures over hand-authored M/N/K shapes.
- Do not add model architecture/name to runtime dispatch identity unless a real semantic requirement proves it is necessary.
- External PRs/branches are locators only; immutable commit SHAs are source identity.
- Every imported optimization must be atomic enough to benchmark and reject independently. Split mixed upstream PRs into logical BigCherry transforms.
- Promotion requires correctness, causal performance evidence, controls, boundary evidence, provenance, and existing BigCherry promotion gates.

## 3. Existing machinery to reuse

| Capability | Current implementation | Action |
| --- | --- | --- |
| Canonical operation identity | `hip-autotune-signature.*`; `0820_measurement_signature_shapes.py` | Reuse |
| Candidate manifest/catalog | `autotune_schema.py`; `autotune_catalog.py` | Reuse; do not duplicate |
| Campaign planning/execution | `campaign_*.py` | Extend with contract expansion |
| A/B benchmarking | `ab_benchmark.py` | Reuse |
| Replay | `hip-autotune-replay.*`; `replay_cache.py` | Reuse |
| Correctness/parity | `parity.py`; `parity_loaders.py`; HI correctness tests | Reuse |
| Generalisation/holdout | `generalise.py` | Reuse |
| Promotion | `promotion.py`; `tune_promotion.py` | Add contract gate |
| Provenance/evidence | `experiment_bundle.py`; `provenance.py`; `source_identity.py` | Reuse |
| Multi-GPU | `multi_gpu_validate.py`; split/reduce telemetry | Reuse |
| Resource evidence | `resource_report.py`; SMI; pool/workspace metrics | Reuse |
| External source registry | `external-sources.toml`; `source_audit.py` | Extend |
| Patch transforms | `patcher.py`; `patchset.py`; `transform_loader.py` | Reuse |

## 4. New component: Experiment Contract

Create a sibling module (preferred: `tools/bigcherry/experiment_contract.py`) rather than bloating `autotune_schema.py`. The latter remains the runtime candidate-manifest contract.

### Minimum contract fields

```
id: RDNA-EXT-001
title: tiny-M Q8 MMQ specialization
source:
  source_id: <external-sources.toml id>
  commits: [<immutable SHA>]
  atomic_part: tiny-m-q8-gate
hypothesis:
  family: mmq
  expected_effect: performance
  rationale: <why this should win>
prerequisites: []
scope:
  backend: hip
  architectures: [gfx1100, gfx1201]
  weight_types: [q8_0]
positive:
  models: [<recipe/model refs>]
  workloads: [small_m, mtp_verify]
controls:
  models: [<control refs>]
  workloads: [decode, prefill]
boundary:
  dimensions:
    physical_m: [1, 2, 3, 4, 8, 16, 32, 64, 128]
correctness:
  backend_reference: required
  greedy_parity: required
acceptance:
  target_kernel_gain_pct: 5
  end_to_end_gain_pct: 1
  max_control_regression_pct: 1
```

## 5. Identity model

Preserve three distinct identities:

1. **Source idea** — e.g. an AMD/stew675 PR or commit.
2. **Atomic optimization** — BigCherry's independently testable port/transform.
3. **Runtime candidate(s)** — executable MMQ/MMVQ/MMF/MMVF/BLAS alternatives exposed to the tuner.

Never put a GitHub PR number into a runtime candidate name. A source optimization can expose multiple runtime candidates, and source history can rebase while the candidate identity remains stable.

## 6. Signature and metadata policy

`0820` already serializes the canonical signature into tuning JSONL. Contracts should select/capture real signatures from model workloads rather than duplicate their shapes. Extend the signature only for values that genuinely determine dispatch. Keep contextual facts as measurement metadata when possible.

| Class | Examples |
| --- | --- |
| Likely measurement metadata | model/architecture, layer role, workload tag, context depth, MTP verify width, MoE token/expert distribution, split mode, device count, output owner, peer capability |
| Potential dispatch refinements only if proven | physical-M semantics, divisibility class, bounded width, layout/alignment refinements |
| Do not blindly key on | model name, benchmark recipe name, upstream PR/commit, campaign ID |

## 7. Required implementation work

| ID | Work | Definition of done |
| --- | --- | --- |
| EC01 | Experiment contract schema/validator | Add closed schema, immutable contract identity/hash, source linkage, prerequisites, positive/control/boundary/correctness/acceptance sections. |
| EC02 | Contract registry/loading | Load contracts deterministically; reject duplicates, unknown source IDs, unknown family/workload tags, invalid thresholds and cycles. |
| EC03 | Campaign expansion | `campaign_planner.py` expands a contract into positive, control and boundary lanes while preserving build/campaign identity. |
| EC04 | Lane metadata | `campaign_lane.py` carries contract ID, optimization ID, role (positive/control/boundary/holdout), workload tag and model/recipe reference. |
| EC05 | Evidence binding | Bind contract identity/hash into experiment bundles and reports without contaminating runtime signature identity. |
| EC06 | A/B aggregation | `ab_benchmark.py`/comparisons aggregate target gains and control regressions per contract. |
| EC07 | Correctness gates | Wire parity/reference/bit-identity requirements declared by the contract. |
| EC08 | Generalisation handoff | Feed exact winning signatures to existing `generalise.py`; do not invent hand-coded dispatch envelopes first. |
| EC09 | Promotion gate | `promotion.py`/`tune_promotion.py` must require all declared contract evidence and holdout/generalisation proof where applicable. |
| EC10 | Reporting | `report.py` shows hypothesis, source, exact winners, losing/non-trigger envelope, controls, boundaries, correctness, generalised rule, and promotion decision. |
| EC11 | CLI | Add validate/list/plan/run/report entry points under existing bigcherry CLI conventions. |
| EC12 | Tests | Schema, hash stability, dependency cycles, campaign expansion, identity separation, negative controls, boundary expansion, evidence completeness, promotion rejection, report output. |

## 8. Current patch inventory on tuning-code-rebase

The entries below are patch IDs. Each implementation is package-owned at
`patches/<patch-id>/patch.py`; package validation, evidence, and fixtures live
alongside it when present.

- 0100_cmake_options.py
- 0200_dispatch_hook.py
- 0300_mmq_forced_j.py
- 0400_mmvf_forced_block.py
- 0500_mmf_forced_nwarps.py
- 0600_mmvq_geometry.py
- 0650_mmvq_native_variant.py
- 0700_coverage_counters.py
- 0800_server_shutdown_endpoint.py
- 0810_replay_hit_diagnostics.py
- 0820_measurement_signature_shapes.py
- 0830_split_reduce_telemetry.py
- 0900_pool_workspace_metrics.py
- 1000_rdna4_mmq_q2k_q6k_fix.py
- 1002_hip_unsafe_math_opt_in.py
- 1003_quantized_cpy_thread_block_fix.py
- 1004_rms_norm_mul_rope_fusion.py
- 1005_prompt_cache_checkpoint_selection.py
- 1100_hi70_direct_op_evidence.py
- 1200_rd19_single_gpu_meta_bypass.py
- 1201_rd20_attn_gate_tp_split.py
- 1202_rd04_bf16_flash_attn_tile.py
- 1203_rd050607_rdna4_wmma_fa_q6k_mmq.py
- 1204_rd08_q6k_mmvq_vdr2.py
- 1205_rd12_paired_mmvq_dual_output.py
- 1206_rd13_mul_mat_add_view_fusion.py
- 1207_rd17_moe_topk_down_fold.py
- 1208_rd21_gfx1151_mmvq_nwarps_table.py
- 1209_rd22_integrated_gpu_host_buffer_backout.py
- 1210_rd26_bitidentical_decode_verify_standalone.py

## 9. Historical/legacy patch work that must not be lost

| Patch | Purpose |
| --- | --- |
| 0001 | recurrent state shrink/expand |
| 0002 | recurrent CMake gates |
| 0003 | RDNA4/GFX1201 MMVQ warp experiment |
| 0004 | AMD checkpoint override |
| 0005 | recurrent cache dirty |
| 0006 | cache generations |
| 0007 | rollback/release tracking |
| 0008 | packed Q8_1 HIP MMVQ |
| 0009 | Vulkan K-quant transpose |
| 0010 | Vulkan MTP device embeddings |
| 0011 | AMD Vulkan MTP device detection |
| 0012 | pending hidden-host-write avoidance |
| 0013 | unsafe Vulkan backend-copy disable |
| 0014 | Vulkan MTP graph-ring shadow path |

Historical validation note: application was previously validated cleanly through 0014. The Flash-Attention mask fix `a4837577a` / PR #12853 was already ancestral to the older b10173 base and must not be duplicated. The Vulkan K-quant transpose port had conflicts in four core Vulkan files and was treated as a manual-port blocker.

## 10. RDNA boost plan catalogue and sequencing

The repository contains RD01-RD27 planning items. Completed: RD01, RD02, RD03, RD14, RD16, RD23 (RD14/RD16 are specifically `superseded`, a closed-but-distinct state from `completed`). Active/pending or in progress: RD04-RD13, RD15, RD17-RD22, RD24-RD27. Agents must read the individual RD plan before changing a related transform; `external-sources.toml` is the source-of-truth for immutable source snapshots and status.

| Plan | Agent guidance |
| --- | --- |
| RD04 | BF16 flash-attn tile path; 1202 exists and has local isolated evidence. Cross-arch/correctness work remains before promotion. |
| RD05/RD06/RD07 | RDNA4 WMMA flash-attn + Q6_K MMQ prefill work; combined materialization 1203, but experiment contracts should preserve logical hypotheses even if one source transform is combined. |
| RD08 | Q6_K MMVQ VDR2; 1204. |
| RD09 | Q8_1 graph cache foundation; prerequisite for RD10/RD11/RD15 and RD24. |
| RD10 | Fold norm-input Q8_1 quantize into RMS norm; conflicts with patch 1004 at RMS_NORM; explicit node-ownership decision required. |
| RD11 | Write Q8_1 in gating-MUL; depends on RD09. |
| RD12 | Paired MMVQ dual-output; 1205. |
| RD13 | mul_mat + add through view fusion; 1206. |
| RD14 | Superseded by RD24. |
| RD15 | Shared-expert output-chain fusion; planned; use branch-tip fixed kernel state per RD25. |
| RD16 | Superseded by RD24. |
| RD17 | MoE top-k weight MUL folded into down projection; 1207. |
| RD18 | IMRoPE + set_rows for BF16 KV cache; extends patch 1004 and must be ported on top of it. |
| RD19 | Single-GPU metadata bypass; 1200. |
| RD20 | Attention gate TP split; 1201. |
| RD21 | gfx1151 MMVQ nwarps=2 Q8_0 decode; 1208; hardware-gated; take branch-tip fixed table per RD25. |
| RD22 | Integrated-GPU HIP host-buffer backout; 1209; hardware-gated. |
| RD23 | Completed plan item; retain its evidence/decision in history. |
| RD24 | SSM pre-scan 16-node fusion; supersedes RD14/RD16; depends on RD09; port branch-tip post-RD25 state. |
| RD25 | Not a standalone patch. Bake-in rule: RD21/RD24/RD15 (and dependent RD26 regions) must be taken from branch-tip post-fix state. |
| RD26 | Five-commit decode/speculative-verify determinism cluster. 1210 contains standalone base-anchoring pieces; FA pieces are composition-gated after 1202/1203; full cluster must be validated together. |
| RD27 | Fold SSM conv_input concat into qkv MMVQ + rpb=2 for small-K MoE; dependency on RD09/RD24 must be proven, not assumed. |

## 11. External-source rules

- Use `external-sources.toml` as the authoritative registry.
- Branch/PR names are locators only. Active reviewed snapshot SHAs are identity.
- A branch rebase creates a new reviewed snapshot; never silently mutate the prior snapshot.
- Every logical imported change must link source commit(s), RD/EX plan item, status, and patch module when materialized.
- Before pin bumps, check whether a source change landed upstream and retire redundant transforms rather than stacking duplicates.
- For stew675 rdna-boosts, snapshot v2 is the active reviewed snapshot; its v1 logical changes were patch-id checked as content-identical after rebase, and new v2 changes are tracked separately.

## 12. Test workflow for every optimization

1. Apply only the atomic optimization and required prerequisites to a clean generated tree.
2. Build with the correct BigCherry variant; record source revision, manifest hash, build descriptor hash and model hash.
3. Run positive model/workload lanes and capture canonical signatures.
4. Filter/identify the exact signatures the hypothesis intends to improve.
5. Benchmark native plus eligible candidate alternatives using existing tuner/A-B machinery.
6. Probe boundary values around the claimed trigger, not only the expected winning point.
7. Run negative/control models and workloads to prove non-trigger or non-regression behavior.
8. Run correctness/parity/bit-identity checks appropriate to the optimization.
9. Run end-to-end PP/TG/MTP/MoE/multi-GPU measures when the optimization can affect them.
10. Pass exact winners into `generalise.py`; let holdout proof determine whether a broader rule is safe.
11. Promote only if contract thresholds, provenance, evidence completeness and existing promotion gates all pass.
12. Report failures and losing envelopes as first-class results; rejected optimizations are useful evidence.

## 13. Generalisation policy

Reuse the existing generalized-key proof. Current policy defaults require at least 3 exact signatures, 100 holdout calls, 5% added coverage, ≤0.5% median regret, ≤1.0% upper regret, and ≤0.5% exact-signature regression. Do not weaken these globally to make a patch pass; a contract may be stricter.

(Verified 2026-08-20 against `tools/bigcherry/generalise.py`'s actual constants: `min_holdout_calls=100`, `min_added_coverage_pct=5.0`, `max_median_regret_pct=0.5`, `max_upper_regret_pct=1.0`, `max_exact_regression_pct=0.5` — this section is accurate to the exact decimal.)

## 14. Known design findings that remain relevant

- Recording is a separate capability from tuning; an inventory build must be able to record without enabling the tuner.
- MMQ J values must be derived from sparse upstream config tables, not the apparent switch range.
- MMVQ small_k is a real geometry dimension; fusion is operation/signature state, not a duplicate compiled-candidate dimension.
- Screening must retain native, top 3, and every candidate within 10% of best median.
- GPU MAD and host median are important; tiny kernels can win GPU time and lose wall-clock time.
- Capture all required collection points, including lower-level public MMQ/MMF family entry points.
- Use bounded-width array caching where width is tiny and dynamic (not a hash lookup on the hottest path).
- Resource/spill blacklist must happen before tuning.
- Replay cache reload remains restart-only until graph/binding invalidation is proven.
- Manual replay should remain proven before tuner complexity is relied upon.
- Production hot-path overhead, timing/concurrency, replay provenance, statistics, MMVF identity, build/hardware provenance and release validation remain areas where audits have historically found P0/P1 risk.

## 15. Agent execution order

1. Read `AGENTS.md` plus `docs/standards/HIP_AUTOTUNE_STANDARDS.md`.
2. Read `docs/reference/OVERVIEW.md`, `DESIGN_DECISIONS.md`, `FAMILY_MODEL.md`, `BUILD.md`, `TEST.md` and `PACK_REVIEW.md`.
3. Read `external-sources.toml` and the RD/EX plan(s) for the optimization being touched.
4. Implement EC01-EC02 first (contract model/registry), then EC03-EC05 (campaign/evidence wiring), then EC06-EC10 (evaluation/promotion/reporting), then CLI/tests.
5. Do not port additional RD patches merely to exercise the framework. First make existing 120x transforms expressible as contracts.
6. Backfill contracts for already materialized 1200-1210 transforms before adding new external optimizations.
7. Then materialize remaining RD items in dependency-safe waves, with RD25 branch-tip bake-in rules enforced.
8. Run host tests on every change; run GPU/correctness/regression gates before changing a plan item to validated/promoted.
9. Commit only after APPLY -> REVIEW invariants -> TEST host/GPU/regression gates are complete.

## 16. Definition of done for this initiative

- Every materialized external optimization has an experiment contract.
- Every contract has immutable source provenance, explicit prerequisites, positive/control/boundary cases and acceptance gates.
- Campaign planning can expand contracts without hand-authoring cross-product tests.
- Canonical signatures remain the basis of runtime winner identity.
- Generalisation and holdout prove dispatch envelopes.
- Reports show both where an optimization wins and where it must not be selected.
- Promotion cannot bypass contract correctness/control/boundary evidence.
- Existing BigCherry build/replay/campaign/release workflows continue to pass.
- No duplicate framework or model-specific runtime dispatch layer is introduced.

## 17. Primary repository references

- docs/standards/HIP_AUTOTUNE_STANDARDS.md
- docs/reference/OVERVIEW.md
- docs/reference/DESIGN_DECISIONS.md
- docs/reference/FAMILY_MODEL.md
- docs/reference/BUILD.md
- docs/reference/TEST.md
- docs/reference/PACK_REVIEW.md
- docs/reference/PIN_REBASE_REVIEW_B10502.md
- external-sources.toml
- tools/bigcherry/autotune_schema.py
- tools/bigcherry/autotune_catalog.py
- tools/bigcherry/campaign_planner.py
- tools/bigcherry/campaign_lane.py
- tools/bigcherry/ab_benchmark.py
- tools/bigcherry/experiment_bundle.py
- tools/bigcherry/generalise.py
- tools/bigcherry/promotion.py
- tools/bigcherry/tune_promotion.py
- tools/bigcherry/report.py
- tools/bigcherry/source_identity.py
- tools/bigcherry/identity_separation.py
- patches/0820_measurement_signature_shapes/patch.py
- docs/planning/active/rdna-boost-experiments/RD24.md
- docs/planning/active/rdna-boost-experiments/RD25.md
- docs/planning/active/rdna-boost-experiments/RD26.md
- docs/planning/active/rdna-boost-experiments/RD27.md

## Appendix A. Patch-aware implementation map

The patch modules are not merely mechanics. Their module docstrings already carry experimental and maintenance intent that should be harvested into Experiment Contracts wherever applicable. Agents should read the patch module before the corresponding RD/HI plan and preserve its provenance, isolation, correctness, composition and promotion notes. The table below classifies every patch currently present on `tuning-code-rebase`.

| Patch | Module | Class | State | What it establishes | Contract / agent consequence |
| --- | --- | --- | --- | --- | --- |
| 0100 | cmake_options | Core / build contract | Validated | Defines measured-dispatch, replay, record, workspace and routing-transform build surfaces; enforces invalid build combinations; accepts documented variant-set aliases. | Contract framework prerequisite. Do not model as an optimization. |
| 0200 | dispatch_hook | Core / dispatch seam | Validated | Adds guarded measured-dispatch hooks while retaining untouched upstream selectors as the fallback; exposes BLAS forwarder and effective-call metadata. | Key invariant: decline must execute real upstream path. Contract tests should never bypass this fallback. |
| 0300 | mmq_forced_j | Core / candidate surface | Validated | Separates native MMQ J scan from launch and allows explicit J; sparse config-table eligibility prevents invalid J/type combinations. Forced J==native J is a built-in noise canary. | Use native-equivalent canary in experiment quality checks. |
| 0400 | mmvf_forced_block | Core / candidate surface | Validated | Makes MMVF block size and accumulator mode explicit, appended/defaulted parameters; native path has no override read. Hard eligibility mirrors warp/shared-memory/fusion constraints. | Contracts may select MMVF alternatives; parity against native-selected equivalent is mandatory. |
| 0500 | mmf_forced_nwarps | Core / candidate surface | Validated | Makes MMF nwarps explicit without touching native path; forced value is applied before shared-memory calculation so allocation matches geometry. | Contracts may select MMF nwarps; reject geometries exceeding architecture/shared-memory constraints. |
| 0600 | mmvq_geometry | Core / compiled candidate surface | Validated | Adds explicit MMVQ nwarps/rows-per-block/small_k template geometry while default zero preserves the exact upstream instantiation. | Foundation for MMVQ experiments. Invalid generated geometry must fail at compile/catalog time. |
| 0650 | mmvq_native_variant | Core / candidate routing | Validated | Routes forced MMVQ geometry through upstream marshalling and diverges only at final launch. Missing compiled instance is fatal; fusion remains signature state, not candidate identity. | Experiment contracts must never interpret silent native fallback as a candidate result. |
| 0700 | coverage_counters | Core / evidence | Validated | Counts real family-entry traffic and catches fused graph paths that bypass dense matmul selection; quantifies what fraction of actual work is visible to measured dispatch. | Coverage evidence is required before claiming a model/workload is 'tuned'. |
| 0800 | server_shutdown_endpoint | Core / harness | Validated | Opt-in graceful /shutdown endpoint for benchmark automation so backend destruction flushes buffered HIP tuning state. | Harness prerequisite on platforms where process termination loses evidence. |
| 0810 | replay_hit_diagnostics | Core / evidence | Validated | Compile-time optional replay-hit JSONL diagnostics; production replay has no diagnostics branch/synchronization cost. | Use to prove winner-cache coverage without imposing production hot-path overhead. |
| 0820 | measurement_signature_shapes | Core / evidence identity | Validated | Persists canonical signature JSON beside tuning results. | Experiment contracts should bind observed canonical signatures rather than inventing manual M/N/K identities. |
| 0830 | split_reduce_telemetry | Core / multi-GPU evidence | Validated | Observes requested/actual SPLIT_REDUCE provider and meta handoff; can explicitly exercise auto/rccl/meta plans without mutating shared provider selection. | Required evidence for multi-GPU contracts; keep provider selection separate from observation. |
| 0900 | pool_workspace_metrics | Core / resource evidence | Validated | Measures real per-candidate scratch high-water via the CUDA/HIP pool rather than declared workspace or global free-memory deltas; includes cache-isolation protocol. | Use measured workspace in Pareto/promotion evidence; do not rank on declared constant bounds. |
| 1000 | rdna4_mmq_q2k_q6k_fix | Upstream fix / correctness+performance | Validated | Backports ROCm codegen fixes for RDNA4 Q2_K/Q6_K MMQ; deliberately excludes upstream heuristic thresholds because BigCherry measures family competition directly. | Treat as an independently selectable upstream-fix baseline, not an autotuned candidate. |
| 1002 | hip_unsafe_math_opt_in | Upstream fix / determinism | Active opt-in | Makes HIP unsafe-math explicitly opt-in after hardware evidence showed speculative-vs-nonspeculative temp-0 divergence on gfx1100; keeps an escape hatch for non-MTP workloads. | Correctness policy input: MTP contracts must record unsafe-math state and require bit-identical acceptance. |
| 1003 | quantized_cpy_thread_block_fix | Upstream provenance / retired | Rejected | Historical backport found already ancestral to the pinned base; kept only for provenance and rejected from selection. | Do not create experiments around it. Rebase tooling should treat it as retired evidence. |
| 1004 | rms_norm_mul_rope_fusion | Upstream provenance / retired | Rejected | Historical fusion backport found already present in the pinned base; retained for provenance only. | Important conflict note for RD10/RD18 planning, but not a selectable enhancement. |
| 1005 | prompt_cache_checkpoint_selection | Upstream fix / server correctness | Active | Fixes hybrid/recurrent exact-position checkpoint validity and prompt-cache entry selection; server-side and backend-independent. | Use as baseline correctness requirement for hybrid/recurrent model experiments; not a kernel candidate. |
| 1100 | hi70_direct_op_evidence | Core / deterministic correctness corpus | Validated | Adds direct MUL_MAT shapes for candidates production models may structurally never reach (MMQ fallback and MMF widths 1..16). | Acceptance runs should regenerate correctness evidence against the exact build instead of depending on lucky local models. |
| 1200 | RD19 single_gpu_meta_bypass | RD experiment / orchestration perf | Untested | Skips Meta wrapper for single-GPU tensor split; fork claims ~1-2% decode gain. No multi-GPU behavior change. | Positive: single-GPU -s tensor. Controls: normal single-GPU and 2+ GPU meta path. Reproduce fork claim before promotion. |
| 1201 | RD20 attn_gate_tp_split | RD experiment / multi-GPU correctness | Untested | Aligns attn_gate TP split granularity with attn_q to prevent incompatible split states, especially 3+ GPU zero-share layouts. | Correctness-first contract. Reproduce pre-fix failure where possible; prove 1/2/3+ GPU equality. |
| 1202 | RD04 BF16 flash-attn tile | RD experiment / FA precision+performance | Untested | Seven-commit net port: native BF16 K/V tile path with FP32 accumulation, BF16 KQ/PV handling, SRAM changes and broader head-size coverage. | Positive: BF16 KV, long-context decode/prefill. Correctness: FA backend ops + PPL/quality gate. Preserve cross-arch fallback. |
| 1203 | RD05/06/07 RDNA4 WMMA FA + Q6_K MMQ | RD experiment / mixed kernel perf | Untested | One external commit containing logically separable WMMA FA race/config/head-size work, Q6_K sub-scale fold, and diagnostics. | Backfill separate Experiment Contracts for RD05, RD06 and RD07 even though materialized in one patch; avoid treating the patch file as one hypothesis. |
| 1204 | RD08 Q6_K MMVQ VDR2 | RD experiment / decode perf | Untested | Processes two Q6_K chunks per vec-dot to amortize loads; fork claims bit-identical output and modest DRAM-bound decode gain. | Positive: Q6_K decode. Gate on bit-identity vs VDR1; test op-timing graph-capture interaction with 1203. |
| 1205 | RD12 paired MMVQ dual output | RD experiment / graph+kernel fusion | Untested | Fuses adjacent K/V-style MMVQ matmuls sharing an activation into one dual-output launch. | Positive: attention K/V projection patterns. Gate on bit-identity. Composition with RD17 currently intentionally fails loudly. |
| 1206 | RD13 mul_mat+add through view | RD experiment / graph fusion | Untested | Extends matmul+bias fusion across one RESHAPE/view, targeting SSM model graph shape. | Positive: SSM/Mamba-family pattern. Controls: non-SSM and no-view path. Validate fused-node bookkeeping. |
| 1207 | RD17 MoE top-k down fold | RD experiment / MoE fusion | Untested | Folds per-token top-k weight scaling into MMVQ down-projection epilogue; expected to remove many per-token MUL kernels. | Positive: MoE decode. Gate on PPL/output equality and kernel-count/timing. RD12 composition needs explicit future compatible anchors. |
| 1208 | RD21 gfx1151 MMVQ nwarps table | RD experiment / hardware-scoped perf | Untested / hardware deferred | Adds dedicated gfx1151 table with nwarps=2 Q8_0 decode; deliberately bakes RD25 branch-tip consistency fix into table semantics. | No performance promotion without gfx1151 evidence. On other architectures require explicit non-selection/no-regression proof. |
| 1209 | RD22 iGPU host-buffer backout | RD experiment / hardware correctness | Untested / hardware scoped | Disables HIP integrated host-buffer path after fork evidence of async corruption on Strix Halo; discrete GPUs should be behavior-neutral. | Promote only with integrated-GPU output/PPL evidence on our hardware; never generalize globally from discrete tests. |
| 1210 | RD26a decode/verify bit identity | RD experiment / determinism partial | Untested / partial cluster | Ports standalone non-FA/CPU pieces of five-commit decode-vs-speculative-verify determinism cluster; FA and MMVQ/fused pieces remain composition-gated. | Do not claim RD26 complete from 1210 alone. Final contract must validate all five logical commits together after required 1202/1203/RD25-tip composition. |

## Appendix B. Rules for deriving Experiment Contracts from patch modules

- **Harvest, do not duplicate**: where a 12xx module already states source-id, plan-item, fork commit(s), snapshot head/base, adaptations, isolation recipe and maintenance rules, the contract loader/backfill tool should import or cross-check those fields rather than asking an agent to retype them.
- **One patch file is not necessarily one hypothesis.** 1203 contains RD05, RD06 and RD07 and therefore needs at least three logical contracts/evidence tracks even though the code is materialized atomically.
- **One logical optimization is not necessarily one patch file.** RD26 is a five-commit determinism cluster; 1210 is only the base-standalone subset. The contract must remain incomplete until the composition-gated pieces are present and validated together.
- **States matter.** Core validated patches establish the laboratory. Untested rdna-boosts patches are specimens. Rejected upstream-fix patches are provenance only and must never silently re-enter a recipe.
- **Composition failures are evidence, not patcher defects.** RD12/RD17 intentionally collide today because first-sweep isolation forbids unproven experimental composition. Add compatible anchors only when a combined experiment is explicitly planned.
- **Branch-tip bake-in rules are part of correctness.** RD21 and future RD24/RD15/RD26b ports must take the fixed branch-tip region defined by RD25; never recreate a known-broken intermediate commit state merely to preserve historical patch order.
- **Hardware-scoped findings stay hardware-scoped.** RD21 cannot be promoted without gfx1151; RD22 requires an integrated-GPU reproduction. A no-regression result on another architecture is not positive evidence.
- **A correctness patch may have no expected throughput gain.** RD20, RD22 and RD26 should use correctness acceptance as the primary gate and performance only as a regression budget.
- **Evidence-enabling patches** (0700, 0810, 0820, 0830, 0900, 1100) should be referenced by contract requirements so an agent knows exactly which instrumentation proves each claim.
- **External fork performance numbers are hypotheses only.** Contract reports must clearly distinguish source-reported numbers from BigCherry-reproduced results.

## Appendix C. Recommended automated backfill

Add a small patch-metadata reader to the Experiment Contract tooling. It should load patch modules without executing edits, extract GROUP/STATE/PROVENANCE plus structured header metadata where available, and cross-check `external-sources.toml`. For 12xx patches, generate a draft contract skeleton with source identity and plan linkage pre-filled. The human/agent then supplies only hypothesis-specific workload, control, boundary and acceptance fields. Do not parse arbitrary prose as authoritative identity. Machine-readable PROVENANCE and `external-sources.toml` win; docstrings provide rationale, testing guidance and maintenance notes.

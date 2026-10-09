---
id: PRBE12
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:54:19.359011+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: P2
---

# Evaluate MUL_MAT plus RESHAPE plus ADD fusion

## Description

Current 2026-10-09 disposition: patch 1206 extends native HIP `ggml_cuda_try_fuse` from direct MUL_MAT->ADD to MUL_MAT->RESHAPE/qualified VIEW->ADD. The September 2025-style TODO text below was superseded by implementation in September 2026: WARN-level activation markers, a bound `validation.toml`, positive/negative probes, full-vocabulary backend-reference producer, and paired positive/control benchmark lanes already exist. The patch remains `untested`, not promotable by documentation alone. The unresolved question is clean, current-pin, architecture-specific end-to-end benefit with safe view aliasing, not building another producer.

Measured history: gfx1100 b11126 four independent tg128 sessions +0.66/+0.53/+0.47/+0.65% with flat controls and exact 64x248,320 full-vocabulary comparison; gfx1201 four noisy sessions +0.2/-0.2/+0.3/+0.4% are inconclusive; gfx1030 has activation and historical correctness but no qualifying performance series. Those results are not a b11474 promotion verdict.

## Steps

1. Reuse `tools/tests/patch/test_patch_1206_rd13_mul_mat_add_view_fusion.py` for the existing exact-RESHAPE, reversed ADD, null addend, extra consumer and wrong intermediate near misses. Add bounded host/real-backend cases for PRBE39's qualified `GGML_OP_VIEW`: offset 0 vs nonzero; contiguous vs strided; equal vs unequal byte span; source identity; and disjoint vs overlapping output. Do not duplicate the existing matcher or build a new dispatch table.
2. Rebaseline against pinned b11474 with the existing patch-local producer `validation/producer.py::run` and `validation.toml` (`--validation-producer 1206_rd13_mul_mat_add_view_fusion/rd13`). One physical architecture per invocation; registered positive `tierA-qwen4b-q6k`, registered negative `tierM-gptoss20b-q6k`, and required `control_model` producer input. Before timed work require subject-only WARN marker, then run the existing paired tg128 positive and negative control, followed by the producer's 64-step, 248,320-vocabulary deterministic server comparison. Validate exact source/build/PCI locator identity and retained artifacts.
3. Exercise repeated same-process capture/replay and multi-request reset with active fusion; compare greedy tokens and pre-sampling full-vocabulary logprobs against unfused, and assert no changed MTP acceptance/work if an optional MTP control is used. Do not let one-shot `std::atomic_flag` activation logs masquerade as a per-layer invocation count.
4. For promotion use the **existing frozen** `RD13-MUL-MAT-ADD-VIEW-FUSION` contract: `improvement_no_regression_v1`, >=4 independent sessions per architecture, >=10 paired rounds/session, session-bootstrap CI95 lower bound >0%, and <=1% control regression. Do not substitute a generic 3% materiality rule or stop after a favourable session. Run gfx1100/gfx1201/gfx1030 sequentially on isolated hardware; gfx1201's noisy historical lane is not evidence of benefit.
5. If an architecture has no reproducible improvement, retain patch 1206 unpromoted there; an architecture-specific gate requires an explicit owner/contract decision, not an implicit change to `validation-architectures`. Do not enqueue duplicate QFP/MMQ, Meta, MTP or Radiance work.

## Detailed Solution & Technical Design

Authoritative source: `patches/1206_rd13_mul_mat_add_view_fusion/patch.py::_NEW` replaces the direct-fusion block inside pinned `ggml/src/ggml-cuda/ggml-cuda.cu::ggml_cuda_try_fuse`. It checks the next graph node for RESHAPE or a VIEW whose `view_src` is the matmul, offset is zero, both tensors are contiguous and `ggml_nbytes` agrees. `ggml_can_fuse_subgraph` enforces the three-node span/consumer rule; ADD must consume the intermediate at `src[0]`, have a non-null addend and equal operand shapes; `ggml_cuda_check_fusion_memory_ranges` rejects overlapping output. Existing `ggml_cuda_should_fuse_mul_mat_vec_f/q` gates choose F32/F16/BF16 or quantized MMV paths. The fused kernel writes the ADD output and elides two intermediate graph nodes; no new persistent buffer or allocator is needed. Direct MUL_MAT->ADD remains the legacy two-node path and must never emit RD13's activation marker.

The producer's `run` first executes subject/control trace probes, then `run_paired_llama_benchmark` on positive and negative models, then builds one attested server pair for full-vocabulary comparison. It returns `promotion_lane_effects`, `promotion_trigger_evidence`, `contract_correctness_results` and `rd13-performance.json`; `performance_benchmark_cli = "forbid"` prevents a **second** generic benchmark authority, not the producer's own performance measurement. `validation.toml` already requires apply/build/activation/correctness/performance/controls.

Theoretical opportunity per fused ADD is at most one avoided separate ADD launch and approximately two intermediate-output byte transfers (matmul output write/read), with addend read and final output write still required. This is a ceiling, not a measured traffic saving. Derive `output_bytes`, actual fusion invocation count, launch time and E2E share from an isolated profiling lane before interpreting sub-percent gains. Existing once-per-process WARN marker proves activation, not frequency.

Upstream: ggml-org/llama.cpp PR #29633 changed `ggml_cuda_should_fuse_mul_mat_vec_f`'s MMVF eligibility signature to include `warp_size` (merged 2026-10-05), so re-evaluate actual F/Q branch selection at b11474. The current upstream CUDA direct MUL_MAT+ADD matcher remains a direct-only baseline. Vulkan PR #27220 demonstrates reusable `ggml_can_fuse_subgraph`/shape/consumer gating but is a different UNARY+MUL operator, not a transplant. AMD's TLX fused GEMM epilogue analysis motivates measuring avoided intermediate traffic, but its hardware results do not transfer to RDNA3/4.

Ownership: PRBE12/1206 owns this fusion and performance contract; PRBE39's VIEW/overlap hardening is already folded into 1206 (no second patch); PVPS15 owns producer activation-before-timing policy. Active QFP, MTP and engine-registry development must not be edited or benchmarked by this slice.

## Code Samples & Guidance



## Files

Existing owner: `PRBE12.md`, `patches/1206_rd13_mul_mat_add_view_fusion/{patch.py,patch.toml,validation.toml,validation/producer.py,validation/producer.toml,README.md,SUMMARY.md,TESTING.md}`; tests `tools/tests/patch/test_patch_1206_rd13_mul_mat_add_view_fusion.py`, `test_patch_validation_campaign_rd13_backend_reference.py`, `test_patch_validation_campaign_rd13_contract_cli.py`; `config/experiment-contracts.toml`. No new plan, producer, scheduler, allocator or configuration flag.

## Validation

Host: current patch dry-run against b11474 vendor source, near-miss/VIEW/overlap fixtures, producer mock for subject-only activation, typed positive/control lane effects and fail-closed model/attestation checks; existing tests are present but were NOT executed in this documentation audit. Hardware: 1x gfx1100, 1x gfx1201, 1x gfx1030 serial independent lanes, paired ABBA tg128, full-vocab reference, graph replay and MTP-control parity where relevant. Capture `rd13-backend-reference.json`, `rd13-performance.json`, trace logs, build/source hashes and physical locator; do not claim a new b11474 result from older b11126 evidence.

## Effort & Risk

Small remaining documentation/fixture work; hardware cost only after source+host gates. Primary risks are false activation attribution, reshape/view aliasing, MMVF dispatch changes and concurrent hardware noise. Retain `untested` and default-off fallback until contract gates pass.

## Standards

Exact pattern; no false positives; causal isolation; preserve legacy direct-ADD semantics.

## Acceptance Criteria

Exact and negative selector fixtures, safe view/output lifetimes, deterministic full-vocab correctness, same-process replay and valid producer evidence must pass. Frozen contract requires four independent sessions with ten paired rounds each, CI95-low >0% positive, <=1% negative-control regression and no work disappearance. Failed/noisy lanes terminate the current decision without optional stopping; no architecture is promoted by extrapolation.

## Notes

**Current authoritative note (2026-10-09):** The older notes below preserve historical chronology. Their statements that WARN markers, activation, contract binding, `validation.toml`, performance and control producers are missing are superseded by the September 2026 code and this audit. Historical b11126 results do not validate b11474. PRBE39 is folded into 1206 and has no separate active plan.


Supersedes: RD13
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd13



REAL HARDWARE CORRECTNESS CHECK RUN (2026-09-11/12), first-ever real execution of this patch's correctness proof: authored a real ppl_equality correctness producer (patches/1206_rd13_mul_mat_add_view_fusion/validation/rd13_correctness.py, reusing the shared tools/bigcherry/experiment/perplexity.py primitive extracted from RD17's own producer -- RD13 needs no bespoke control-variant worktree, since its patch.py is one self-contained block replacement with no separate struct/kernel-plumbing edits that would stay compiled-but-inert under a partial revert; the project's normal baseline composition with RD13 excluded IS the correct control). Not yet CLI-wired for the same reason as RD17 (require_execution_package()'s unconditional gate); callable directly via run_rd13_ppl_check().

Ran for real on Brutus: gfx1201, tierM-gptoss20b-q6k (gpt-oss-20B, the contract's own declared model), real wikitext2 corpus. Result: PASS -- subject PPL=561.6933+/-1.67373, control PPL=561.6933+/-1.67373, delta=0.0, sigma=0.0 (exact match, same pattern as RD17's own fix-verification run).

HONEST CAVEAT (same class as RD17's, recorded rather than overclaimed): delta=0.0 exactly proves NO REGRESSION from RD13's patch -- it does NOT by itself prove the RESHAPE-mediated fusion actually activated during this run. RD13's own patch.py docstring says the view pattern needs "an SSM/Mamba-family model" to fire; gpt-oss-20b is MoE (the contract's own model choice) but whether it actually contains a RESHAPE-mediated residual add in its real graph is unverified here -- RD13 has real BIGCHERRY_PATCH_HIT activation trace markers (unlike RD17) that were NOT checked in this run (this producer only proves PPL equality, not activation). A real activation check (trace-marker probe under BIGCHERRY_PATCH_TRACE=1) is separate, not-yet-done follow-up work needed to know whether this PASS reflects "fusion fired and is safe" or "fusion never fired, so trivially safe."

Also worth flagging honestly, not blocking: the absolute PPL magnitude (561.69) is unusually high for a 20B model on wikitext2 (typically low double digits) -- possibly a real quirk of this specific quantization/corpus/context-length combination, possibly an unrelated tokenization/chat-template mismatch in how llama-perplexity was invoked here. Since it's IDENTICAL between subject and control, it does not indicate an RD13-specific defect, but the absolute number itself was not independently sanity-checked against a known-good baseline for this exact model/corpus/args combination -- worth a separate look before treating 561.69 as a meaningful baseline for anything else.

Remaining real work: real activation-trace verification (has real markers already, just not exercised by this specific run), the real performance claim (this contract has none -- CORRECTNESS/DETERMINISM... actually check: RD13's own contract IS performance-bearing, moe_decode workload declared), full validation.toml authoring + contract binding (blocked the same way RD17's is). Patch state remains "untested" -- this is real evidence, not a promotion.

2026-09-24 relevance at b11126: TODO, narrowed -- do not repeat the already-real PPL-equality/no-regression evidence; focus on activation proof, negative fixtures, performance, and contract binding. GPT design request submitted (req_a8361cdd54af4bd5, batched with PRBE11); gateway congested at submission -- authored directly against this item's own existing real-hardware notes and patches/1206.../patch.py as a fallback.

2026-09-24 GPT review req_7f4dea253b7247f0 applied: verified via grep that patch 1206's two activation markers use GGML_LOG_INFO (patch.py lines ~222, ~235) -- changed step 1 to require both be changed to GGML_LOG_WARN. Corrected the mediating-node terminology from generic "VIEW" to the patch's actual match target GGML_OP_RESHAPE, and clarified that direct MUL_MAT->ADD (no RESHAPE) should retain legacy fusion behavior without emitting the 1206_rd13 marker rather than being treated as a rejected pattern.

2026-09-25 implementation: 1206 markers moved to GGML_LOG_WARN (commit after cd35b01f); PRBE39 extension (VIEW + memory-range check) landed in the same package. b11126 gfx1100 campaign (work/runs/prbe12-1206-gfx1100 on Brutus): ELIGIBLE, 0 blocking reasons. tg128 positive (tierA-qwen4b-q6k) +0.552% CI95 [0.152, 1.065] n=10; control (tierM-gptoss20b-q6k) -0.001% [-0.105, 0.106]; backend_reference 64 steps x 248320 full-vocab logprobs max diff 0; activation marker subject-only. Caveat: ran while another campaign was building on the host (paired interleaving mitigates); quiet rerun required before promotion. gfx1201 run launched.

2026-09-25 RESULT: RD13 (1206) gfx1100 PASS (4 sessions: +0.66/+0.53/+0.47/+0.65% tg128, every CI above 0, controls flat, full-vocab logprobs identical, activation proven). gfx1201 FAIL/not established (point estimates +0.2/-0.2/+0.3/+0.4%, control CIs up to +-4% -- sessions ran concurrently with the gfx1100 lane; later sessions run serially). Not re-running gfx1201 to chase the verdict (no optional stopping); a serial-lane re-measurement needs an explicit decision.

## Change Log

- 2026-10-08 (triage): pending; 1206_rd13_mul_mat_add_view_fusion state=untested (patch.toml); gfx1100 recorded positive, gfx1201 qualification incomplete (item notes). No in-flight branch on main; remaining VIEW-negative fixture and controlled repeat. Reset stale in_progress.

- 2026-09-09T10:54:19.359011+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:11:26.671901+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.183883+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.881480+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:47:41.484974+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_024759_three-rdna-fusion-successors-n_3469
- 2026-09-10T02:47:59.468336+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-11T14:55:48.900915+00:00 (updated-by): Updated: section:notes
- chg_20260911_145554_ran-the-first-ever-real-correc_4120
- 2026-09-11T14:55:54.334136+00:00 (updated-by): Updated: section:ledger-events
- chg_20260911_212428_finished-documenting-one-more_3069
- 2026-09-11T21:24:28.100057+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:34:46.365428+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:effort_risk, section:notes
- 2026-09-24T04:39:02.337298+00:00 (updated-by): Updated: section:steps, section:notes
- 2026-09-24T14:09:37.091460+00:00 (state-transition): State: pending → in_progress
- 2026-09-24T14:09:40.423483+00:00 (updated-by): Updated: section:notes
- chg_20260924_141016_five-experimental-rdna-patches_5706
- 2026-09-24T14:10:19.108890+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-25T04:23:57.043827+00:00 (updated-by): Updated: section:notes

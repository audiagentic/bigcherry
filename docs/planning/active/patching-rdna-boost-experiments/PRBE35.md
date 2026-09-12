---
id: PRBE35
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:55:53.286868+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# AMD-STREAM-005: Protect concurrent-region join node from fusion

## Description

Protect the concurrent-region join node from graph fusion after PRBE34 shared-expert overlap; this is a mandatory prerequisite for graph-opt defaulting.

## Steps

- Require PRBE34 exact shared-expert concurrency identity.
- Guard graph fusion so the concurrent-region join remains present and aux stream rejoin semantics are preserved.
- Run repeated graph capture/replay on MoE decode with graph-opt on, plus graph-opt-off and dense controls.
- Verify no capture abort, output parity and no material ordinary regression.
- Do not enable PRBE36 default-on until this protection passes.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

Graph fusion eligibility around concurrent-region join; PRBE34 scheduler; capture/replay fixtures; MoE/dense controls; trace and output evidence.

## Validation

Repeated capture/replay; no abort; output parity; graph-opt off/dense controls; no material regression; join/rejoin trace.

## Effort & Risk



## Standards

Concurrency join correctness; dependency-aware defaulting; no fusion false positives.

## Acceptance Criteria

Join protection prevents capture failure and preserves output under eligible concurrency; PRBE36 remains blocked until this gate passes.

## Notes

Supersedes: RD43
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd43



CORRECTION (2026-09-11/12): an earlier session's survey of this item incorrectly reported "no patch materialized yet -- pre-authoring backlog work." That was wrong -- patch 1216_rd43_concurrent_join_fusion_guard is a real, materialized patch package (patch.py/patch.toml/README.md all exist), with patch.toml's REQUIRES correctly declaring its real hard prerequisite (1215_rd394041_amd_stream_moe_overlap, itself a real materialized patch). Corrected here rather than left standing.

REAL HARDWARE CORRECTNESS CHECK RUN (2026-09-11/12), first-ever real execution of this patch: authored patches/1216_rd43_concurrent_join_fusion_guard/validation/rd43_correctness.py (reusing the shared tools/bigcherry/experiment/perplexity.py primitive, its third real caller) plus run_rd43_ppl_check() in validation_campaign.py. Subject = 1215+1216 applied; control = 1215 alone (same "no bespoke control-variant worktree needed" reasoning as RD13 -- one self-contained ggml-cuda.cu edit, no other-file plumbing left compiled-but-inert under a partial revert). Crucially, BOTH runs set GGML_CUDA_GRAPH_OPT=1 -- the exact env var this patch's own docstring names as the real-hardware reproduction condition for the crash it exists to fix ("capturing stream has unjoined work" at cudaStreamEndCapture).

Ran for real on Brutus: gfx1201, tierM-gptoss20b-q6k (gpt-oss-20B, the contract's own declared model), real wikitext2 corpus. Result: PASS. Both subject and control builds completed AND RAN TO COMPLETION under GGML_CUDA_GRAPH_OPT=1 -- no capture abort, the primary proof this patch's own acceptance criteria call for. Secondary proof: exact PPL match (561.6933 both arms, delta=0.0) -- no output regression.

Honest scope note: this proves RD43 does not crash and does not regress output for this one real MoE-with-shared-expert workload/model/architecture. PRBE35's full acceptance criteria (repeated capture/replay cycles, graph-opt-off and dense-model controls, output parity specifically attributable to the fusion-cap fix rather than absence of the failure mode in this exact run) are broader than this one check -- this closes the first real data point, not the full item. Patch state remains "untested" -- this is real evidence, not a promotion. PRBE36 (default-on) stays blocked pending the item's full remaining scope.

1215's README.md authored (2026-09-11), documenting real indirect evidence it is exercised (builds, doesn't corrupt PPL, doesn't abort under GGML_CUDA_GRAPH_OPT=1) via RD43/1216's and RD44/1217's own already-run campaigns, since both require it. Core +7.4% tg128 performance claim still not independently reproduced. Ledger event chg_20260911_221000_found-and-documented-real-evid_3917 (mis-linked to PRBE13 initially, corrected here).

### 2026-09-12: real performance + activation + cross-architecture evidence (GPT-reviewed, req_893357b8c0bd4a4e / req_58a8c6465fd249f1 / req_40c25c3069ab4852)

First-ever dedicated performance campaign for the combined 1215+1216 unit (required together per this item's own PRBE34 dependency). Corrected methodology after GPT review: RD42/1215's shared-expert overlap is intra-token, single-GPU, batch-1-decode, opt-in behind GGML_CUDA_GRAPH_OPT=1 -- an initial dual-GPU/-sm-tensor/graph-opt-unset test structurally could not exercise it.

**gfx1100 single-GPU, corrected condition** (Qwen3.6-35B-A3B-UD-Q4_K_M): 6-round interleaved 2x2 (OFF=GGML_CUDA_GRAPH_OPT=0 control, ON==1 subject). OFF mean -0.23% (flat, as expected). ON mean +2.47%, all 6 rounds positive -- real, reproducible, non-regressing gain (smaller than the fork's own +7.41%, measured on a different architecture gfx1151/RDNA3.5).

**Direct profiler proof**: rocprofv3 kernel-trace, 103,768 real dispatches across 4 HSA queues. Computed real wall-clock union-intersection between the two populous queues (82,750 vs 20,170 dispatches) -- 82.28% of the auxiliary (shared-expert) stream's busy time genuinely overlaps in real time with the main (routed-expert) stream. Not just the 'Adding shared-expert stream at node...' marker firing (also independently confirmed via debug-verbosity log) -- actual concurrent GPU kernel execution.

**Cross-architecture**: gfx1201 (single-GPU, same model) shows a directionally similar but noisier/weaker signal (ON +2.59% mean, but OFF control itself non-flat at +1.38%, higher baseline noise on this device). gfx1030 (single-GPU, smaller IQ3_S quant since Q4_K_M doesn't fit in 16GB) shows a noisy NET NEGATIVE (ON -1.19% mean, one severe -6.25% outlier) -- no clear benefit, possibly harmful, possibly just measurement noise from less VRAM headroom on this older/smaller GPU. Not conclusive either way.

**Split-mode impact** (gfx1100 dual-GPU, GRAPH_OPT=1): `-sm layer` shows a real, consistent REGRESSION (~-4.4% mean, all 3 rounds negative). `-sm tensor` is flat/neutral (no benefit, no harm).

**Production guidance**: real benefit is conditional on single-GPU deployment with GGML_CUDA_GRAPH_OPT=1 explicitly set; must NOT be enabled for multi-GPU -sm layer (real regression); safe-but-inert under -sm tensor. Generalization beyond gfx1100 is unproven/mixed.

**GPT-reviewed disposition**: gfx1100 single-GPU performance/mechanism qualification is substantively complete (real E2E gain + activation marker + direct profiler overlap proof satisfy RD42's own claimed gate) -- recorded as a real positive finding on its own merits, independent of any specific percentage threshold (per explicit user correction this session: do not gate promotion on a fixed percentage bar). Do NOT promote to validated yet: this project's policy requires a complete validation package and patch-verify-evidence-passing persisted evidence; 1215's own contract requires a bit_identical correctness check and 1216's requires backend_reference (existing PPL-equality evidence does not formally substitute for that named check). GPT: 'stop hardware profiling now -- remaining gaps are package/contract work, not more performance evidence.' The cross-architecture/split-mode sweep was completed as a direct answer to a generalization question, not further promotion-path profiling.

Patches 1215 and 1216 stay state=untested. Real, substantial evidence now exists; the formal validated-state package (validation.toml, bound Experiment Contract with its two named correctness checks) remains the concrete next step if pursued.

### 2026-09-12: concrete next-step scoping for the formal validated-state package

Checked what the pre-authored contracts (`config/experiment-contracts.toml`) actually require, to make the remaining gap concrete rather than abstract:

- `[contract.RD39-42-STREAM-MOE-OVERLAP]` (1215): model `tierM-gptoss20b-q6k`, workloads `moe_decode`/`decode`, correctness=`bit_identical` (required), architectures `[gfx1100, gfx1201, gfx1030]`, acceptance `target_kernel_gain_pct=1, max_control_regression_pct=1` -- easily cleared by the already-measured +2.47% gfx1100 result, so this is not the blocker.
- `[contract.RD43-CONCURRENT-JOIN-FUSION-GUARD]` (1216): same model/workloads, correctness=`backend_reference` (required), `prerequisites = ["RD39-42-STREAM-MOE-OVERLAP"]`.

Both contracts already exist pre-authored -- the real remaining gap is that no qualification RUNNER exists to bind and execute them. `validation_campaign.py` has exactly two patch-specific qualification functions today (`run_rd08_contract_qualification`, `run_rd73_contract_qualification`), no generic one. Writing `run_rd39_42_contract_qualification()` (and wiring 1216's `backend_reference` check, which per its prerequisite likely composes with 1215's rather than needing a fully separate runner) following that same real, careful pattern -- anchor-tested, GPT-reviewed, offline-tested, then real-hardware-run on `tierM-gptoss20b-q6k` -- is genuine new authoring work, not a quick script. Deliberately NOT rushed into this already-long session; scoped here so the next session picking this up has the exact target (function name, contract IDs, model, correctness-check kind) rather than needing to re-derive it.

Note: PA33 (patch-validate orchestrator) would eventually make this kind of per-patch qualification-function authoring unnecessary by consuming the contract generically -- but PA33 itself doesn't exist yet either, so for 1215/1216 specifically the nearer-term path is still a dedicated qualification function matching the existing RD08/RD73 pattern.

### 2026-09-12: full GPT design for the formal qualification runners (req_3e42043eb71a4a92)

Got a design review before writing any code, given RD08's single-op test-backend-ops pattern is a category error for a graph-scheduling patch like 1215/1216. Key correction: completion text/token equality (what I used for 1207) is TOO WEAK for a bit_identical claim -- different logits can produce identical greedy tokens. Full per-step raw logits comparison is the right bar, not text diffing.

**Design (GPT-approved, concrete, ready to implement)**:

1. One shared low-level primitive, `_run_decode_logit_reference_pair(...)`: materializes/builds two isolated compositions, fixed model/prompt/context/seed, 64 deterministic decode steps (sampling disabled), captures the FULL raw logits buffer after every decode step, computes a per-step SHA256 digest over a documented dtype/byte representation, records generated token IDs as diagnostics, returns exact equality + first mismatching step + per-step digests + aggregate digest + full command/env/build identities. **Caveat**: if `/completion`'s HTTP API cannot expose raw logits (it likely cannot beyond n_probs top-k, not the full vocab), a small bespoke llama_decode/logits-capture helper is needed -- NOT a test-backend-ops-style tool, something closer to `examples/eval-callback`.

2. `run_rd39_42_contract_qualification()` (1215): control=baseline (no 1215/1216), subject=baseline+1215+1216 (1216 included because it's operationally inseparable from RD42's execution path -- document this explicitly in evidence), single gfx1100, GGML_CUDA_GRAPH_OPT=1 both arms, require the subject's shared-expert activation marker to fire, `CorrectnessResult(check_id="bit_identical", passed=<all 64 per-step digests match exactly>, method="full-model-decode-logit-digest", details={...})`.

3. `run_rd43_contract_qualification()` (1216): causal control=baseline+1215 (NOT plain baseline -- isolates 1216's own marginal effect), subject=baseline+1215+1216, GGML_CUDA_GRAPH_OPT=1 both arms, reuse the existing PPL-equality primitive (RD19 already establishes PPL subject/reference comparison as legitimate backend_reference methodology) but emit `check_id="backend_reference"` explicitly -- do NOT silently relabel the existing 561.6933==561.6933 historical PPL result as satisfying this; that result stands as corroboration only, a fresh formal run under this exact check_id is required. A stronger version could reuse the logit-digest primitive for an exact 1215-only-vs-1215+1216 match, but GPT says this isn't required -- the contract asks backend_reference, not bit_identical, for 1216.

4. 1215 and 1216 must each independently obtain their own contract verdict and `eligible_for_validated_state` -- a convenience `run_amd_stream_contract_qualification()` may call both runners, but their evidence records stay separate.

**Not implemented this session** -- this is genuine new engineering (a real logits-capture mechanism plus two carefully-designed contract-specific runner functions, each needing GPT-reviewed implementation, offline tests, and real hardware execution) deliberately left for a dedicated future pass rather than rushed at the end of an already very long session. This note carries the complete, ready-to-implement design so the next session/agent can go straight to authoring.

## Change Log

- 2026-09-09T10:55:53.286868+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:13:05.382498+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.285865+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.037669+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:59:39.595087+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_030005_amd-streamfus-successors-prbe_8761
- 2026-09-10T03:00:05.752127+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-11T15:32:01.953695+00:00 (updated-by): Updated: section:notes
- chg_20260911_153207_real-hardware-test-confirms-a_1251
- 2026-09-11T15:32:07.442514+00:00 (updated-by): Updated: section:ledger-events
- chg_20260911_212645_documented-two-more-optimizati_8645
- 2026-09-11T21:26:45.045884+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-11T22:10:30.357360+00:00 (updated-by): Updated: section:notes
- chg_20260911_230135_found-and-documented-a-real-sa_9076
- 2026-09-11T23:01:35.306535+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T09:17:39.246458+00:00 (updated-by): Updated: section:notes
- chg_20260912_091821_validated-a-gpu-concurrency-op_8669
- 2026-09-12T09:18:21.067473+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T09:49:37.378827+00:00 (updated-by): Updated: section:notes
- 2026-09-12T09:58:46.097569+00:00 (updated-by): Updated: section:notes

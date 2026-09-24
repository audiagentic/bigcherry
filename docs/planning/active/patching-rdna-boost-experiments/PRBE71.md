---
id: PRBE71
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:58:31.188986+00:00'
breadth: ''
skill: ''
created-by: capability-rebaseline-v3
priority: null
---

# Qualification-only ablation knobs for RD39-44 (stream/concurrency) and RD50-53 (GDN)

## Description

TODO. RD39-44 live in patches/1215_rd394041_amd_stream_moe_overlap (plan-item "RD39/RD40/RD41/RD42", state=untested) and RD50-53 live in patches/1221_rd50_gdn_chunked_recurrence (plan-item "RD50/RD51/RD52/RD53", state=untested). Both are coupled/production-shaped bundles; this item asks for temporary, qualification-only ablation knobs layered on top of each without splitting the production patches themselves.

## Steps

1. Read patches/1215_rd394041_amd_stream_moe_overlap/patch.py and patches/1221_rd50_gdn_chunked_recurrence/patch.py in full to enumerate their real Edit() ids before adding any ablation knob -- do not assume which edits correspond to 'safe plumbing' vs 'overlap' (RD39-41) or 'generic shuffle+expf+default launch' vs 'DPP/exp2/launch-bounds' (RD50-53) without re-reading the actual current code (PRBE101's own investigation of 1215 already found the RD42 activation gate at ggml-cuda.cu's `ggml_cuda_graph_optimize`, static GGML_CUDA_GRAPH_OPT lambda -- reuse that as the RD39-41-vs-RD42 boundary reference point).
2. Add a SEPARATE qualification-only env var per cluster, distinct from each patch's own production env var(s) (GGML_CUDA_GRAPH_OPT for 1215; whatever 1221 uses for GDN -- confirm by grep), e.g. BIGCHERRY_QUAL_RD3944_VARIANT=native|plumbing_only|plumbing_plus_overlap and BIGCHERRY_QUAL_RD5053_VARIANT=generic|dpp|exp2|launch_bounds, read once via the same static-lambda getenv pattern already used for GGML_CUDA_GRAPH_OPT, defaulting to the patch's real production behavior when unset (never silently changes default behavior).
3. Gate each ablation branch with compile-time #ifdef/constexpr where possible (cheaper, easier to verify no runtime cost when unset) and runtime env checks only where compile-time separation isn't feasible (e.g. the DPP/exp2 GDN micro-decisions inside a single kernel body).
4. For RD39-44: three variants (native = neither RD39-41 nor RD42 active; safe-plumbing-only = RD39-41 without RD42's stream overlap; safe-plumbing+overlap = full 1215 as shipped). For RD50-53: four variants (generic shuffle+expf+default launch = RD50/51 baseline without RD52/53's compiler-specific tricks; +DPP; +exp2; +launch-bounds), each isolated so kernel time deltas attribute to exactly one component.
5. Run each variant through backend-reference correctness (test-backend-ops) before any benchmark; record PP/TG, kernel time, and component deltas vs native in tools/lab/rdna-boost-ablation/results/ as machine-readable JSON plus a short markdown summary.
6. After qualification, either remove the ablation knobs entirely (preferred -- they are explicitly temporary) or, if kept, quarantine them behind a clearly-named BIGCHERRY_QUAL_* prefix that is never read outside this qualification path, and never expose them as a documented production API.

## Detailed Solution & Technical Design

This does not split or replace either production patch -- it adds parallel, removable instrumentation so causal attribution of each already-coupled cluster's components becomes measurable without disturbing the coupled production contract PRBE101 and others depend on. Ablation knobs must fail closed to production behavior when unset.

## Code Samples & Guidance

Env-gated knob pattern (extending the existing GGML_CUDA_GRAPH_OPT precedent in patches/1215's patch.py, ggml-cuda.cu):
```cpp
    static bool enable_graph_optimization = [] {
        const char * env = getenv("GGML_CUDA_GRAPH_OPT");
        return env != nullptr && atoi(env) == 1;
    }();
    static int qual_variant = [] {
        const char * env = getenv("BIGCHERRY_QUAL_RD3944_VARIANT");
        if (env == nullptr) return 2; // default: full production behavior (plumbing+overlap)
        if (!strcmp(env, "native")) return 0;
        if (!strcmp(env, "plumbing_only")) return 1;
        return 2; // plumbing_plus_overlap
    }();
    if (qual_variant < 2 && !enable_graph_optimization) { /* native / plumbing-only: skip overlap-specific work below */ }
```
(exact insertion point requires reading 1215's real patch.py Edit bodies first, per step 1 -- not fabricated here.)

## Files

patches/1215_rd394041_amd_stream_moe_overlap/patch.py (temporary qualification Edit additions, or a sibling qualification-only overlay); patches/1221_rd50_gdn_chunked_recurrence/patch.py (same); tools/lab/rdna-boost-ablation/results/*.json,*.md.

## Validation

Each variant passes existing backend-reference correctness (test-backend-ops) before benchmarking; PP/TG, kernel-time, and component-delta report per variant with native controls in tools/lab/rdna-boost-ablation/. Hardware runs (not performed here) via python -m bigcherry.patch.validation_campaign on gfx1100 (RD39-44, where PRBE101 found the real overlap win) and gfx1201/gfx1030 (RD50-53 GDN).

## Effort & Risk

M; touches two already-untested production-shaped patches, so must be added additively (new Edit ids, no modification of existing ones) to avoid perturbing their own pending qualification.

## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Obtain causal attribution for both clusters while preserving the coupled production contract; no variant may bypass safety prerequisites or be promoted without correctness.

## Notes

Supersedes: RD91
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd91

2026-09-24 relevance at b11126: TODO, both target patch bundles (1215, 1221) confirmed present and untested; no existing ablation knobs found in either patch.py (grep for BIGCHERRY_QUAL_ finds nothing). GPT design request: gateway rejected all submissions this session (VAL-AGW-025 / EXT-GPTAUTO-003); plan authored directly, cross-referencing PRBE101's own findings about 1215's GGML_CUDA_GRAPH_OPT gate -- no GPT request id.

## Change Log

- 2026-09-09T10:58:31.188986+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:15:42.529821+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.448926+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.286324+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:20:14.039303+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_032047_repaired-five-more-active-succ_6361
- 2026-09-10T03:20:47.933544+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:36:46.875374+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes

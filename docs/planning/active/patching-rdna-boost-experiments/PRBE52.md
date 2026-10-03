---
id: PRBE52
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:57:05.575180+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# UP-MTP-001: Adaptive MTP draft depth

## Description

Materialize runtime wiring for the pure `bigcherry_nro06_adaptive_mtp` controller already supplied by `1255_nro06_adaptive_mtp_depth`. Add a separate disabled-by-default adaptive floor; preserve fixed-depth MTP exactly when disabled. The optimization only changes speculative work, never target-token semantics.

## Steps

1. Require 1255 and add `n_min_adaptive=0` to `common_params_speculative_draft` plus `--spec-draft-n-min-adaptive` / `LLAMA_ARG_SPEC_DRAFT_N_MIN_ADAPTIVE`.
2. Store controller state and last drafted count per sequence beside `pending_h`.
3. Reset state in `common_speculative_impl_draft_mtp::begin`; validate the adaptive floor after any chain-head cap of `n_max`.
4. Replace only the fixed `params.n_max <= result.size()` stop test with the controller's current depth when enabled.
5. Record only drafts surviving the existing `n_min` filter; call `update(last_n_draft, n_accepted, ...)` from `accept()` using the real accepted count.
6. Emit one activation marker only on the enabled adaptive path.
7. Validate deterministic greedy token identity and server throughput against fixed-depth control.

## Detailed Solution & Technical Design

Patch `1268_prbe52_adaptive_mtp_wiring`, order 1268, `requires=["1255_nro06_adaptive_mtp_depth"]`. The new floor uses 0 as the sole disabled value; enabled values must be <= the effective post-constructor `n_max`. Each sequence owns independent controller/counter state. `begin()` is the request reset boundary. Drafting clears stale accounting before each attempt, applies the adaptive cap only after a token is actually drafted, and stores the final submitted draft count. `accept()` updates the controller after pending hidden-state selection. No global/static adaptive state is allowed. Activation logging uses only a once-per-process atomic flag and does not affect state.

Full-vocabulary comparison is intentionally not the MTP correctness oracle: accepted draft tokens do not expose complete vocabulary rows. Correctness is identical greedy token IDs for equal prompt/model/seed; backend/kernel correctness remains covered by the prerequisite path.

## Code Samples & Guidance

```cpp
int32_t n_min_adaptive = 0; // 0 disables
const int effective_n_max = params.n_min_adaptive > 0
    ? adaptive_state[seq_id].n_cur : params.n_max;
...
adaptive_state[seq_id].update(last_n_draft[seq_id], n_accepted,
                              params.n_max, params.n_min_adaptive);
```

Use `LLAMA_ARG_SPEC_DRAFT_N_MIN_ADAPTIVE=1` for qualification so the unchanged control binary ignores the unknown environment variable while the subject opts in; do not pass a subject-only CLI flag to both arms.

## Files

- `common/common.h`
- `common/arg.cpp`
- `common/speculative.cpp`
- `patches/1268_prbe52_adaptive_mtp_wiring/*`
- `tools/tests/patch/test_1268_prbe52_adaptive_mtp_wiring.py`
- `config/experiment-contracts.toml`

## Validation

Offline: patch lint/rebase-check and mechanics tests for apply, idempotence, missing anchors, explicit opt-in, per-sequence state, and fixed-depth default. Hardware: gfx1100/gfx1201, `tierM-qwen35b-a3b-moe-mtp`, four independent sessions, ten paired rounds/session. Positive lane is batched MTP server throughput (`server_requests_per_start=5`) with adaptive floor 1 and fixed max 4; correctness is 64 greedy tokens identical between arms. Control lane is ordinary `llama-bench` tg128. Promotion policy: `improvement_no_regression_v1`, target CI95 low > 0, control CI95 high <= 1%.

## Effort & Risk

L / medium-high. Main risks are sequence-boundary state leakage, stale draft-count feedback, and accidentally changing fixed-depth behavior. All are contained by per-sequence vectors, reset-before-use, zero-disabled gating, and deterministic token identity.

## Standards



## Acceptance Criteria



## Notes

Source audit against b11126 confirmed the concrete hooks: `pending_h`, `begin()`, `result.push_back(id)` followed by `params.n_max <= result.size()`, the final `params.n_min` filter, and `accept(seq_id, n_accepted, ...)`. The previous placeholder package id 1256 is obsolete because that order is already occupied; materialized id is 1268. Keep plan state `pending` until hardware evidence promotes/rejects the patch.

Source audit against b11126 confirmed the concrete hooks: `pending_h`, `begin()`, `result.push_back(id)` followed by `params.n_max <= result.size()`, the final `params.n_min` filter, and `accept(seq_id, n_accepted, ...)`. The previous placeholder package id 1256 is obsolete because that order is already occupied; materialized id is 1268. Keep plan state `pending` until hardware evidence promotes/rejects the patch.

FINDING 2026-09-27 (hardware, gfx1100, 3/3 real sessions): backend_reference correctness FAILS -- greedy 64-token MTP output diverges between control and subject at token index 36, identically across all 3 sessions run so far (not noise; deterministic). This directly violates this item's own stated invariant ('the optimization only changes speculative work, never target-token semantics' / 'deterministic greedy token identity'). Speculative decoding's acceptance test is supposed to guarantee the accepted stream always matches the target model's own greedy output regardless of draft depth/dynamics -- a divergence here means the adaptive-depth controller (n_cur tracking, or its interaction with accept()'s last_n_draft update) is corrupting generation, not just its speed. The GPT code-review-confirmed floor invariant (n_min_adaptive >= n_min, fixed 2026-09-27) did not fix this -- it prevents a stuck-drafting state, not this semantic divergence. Remaining 6 hardware sessions (gfx1100 s3/s4, all 4 gfx1201) paused in the queue (perf-only.txt) rather than burning further GPU time reproducing the same deterministic failure. Needs an author-level re-look at the adaptive controller's interaction with the accept-loop hidden-state selection (pending_h/verify_h indexing) before more hardware time is spent -- correctness gate already blocks promotion so nothing unsafe reached production, this is purely about not implementing/testing efficiency.

2026-09-27 (later): GPT diagnosis (req_9d9f95bc2be242f6) landed as commit d5ce7866, adding requires=["1210_rd26_bitidentical_decode_verify_standalone"] to 1268's patch.toml -- the correctness divergence above was caused by decode-batch vs speculative-verify-batch logits not being bit-identical, which 1210 fixes. Deployed and confirmed: t-1265b-gfx1201-s3 (an unrelated patch, but same scaffold path) passed cleanly, proving the framework wiring works.

However the first real 1268 requeue (t-1268b-*, all 8 sessions, gfx1100+gfx1201) all failed in ~13s each with 'ValueError: 1268_prbe52_adaptive_mtp_wiring requires explicitly selected module(s): 1210_rd26_bitidentical_decode_verify_standalone' -- adding `requires` to patch.toml only enforces that 1210 must be present in the resolved exact set IF 1268 is selected; it does NOT auto-compose 1210 into the baseline. The queue lines still only had `--common-patches 1255_nro06_adaptive_mtp_depth` (unchanged from before the requires fix), so resolve_exact failed closed correctly. Fixed by adding 1210 to --common-patches explicitly (`--common-patches 1255_nro06_adaptive_mtp_depth,1210_rd26_bitidentical_decode_verify_standalone`) and requeuing as t-1268c-* (8 sessions, gfx1100 s1-4 + gfx1201 s1-4). Verified locally via resolve_source_composition() before requeuing. GPU time wasted: ~13s x 8 = ~1.7min, negligible.

GPT review (req_28ab0790a31443df) separately reviewed the 1265 demotion (d2c2365d) and PVPS13 locator-fix centralization (364157ff): APPROVE both, no corrective changes required.

2026-09-27 GPT adversarial review verdict (req_28ab0790a31443df, incorporated): APPROVE both commit d2c2365d (1265 demotion) and commit 364157ff (PVPS13 locator-fix centralization); no corrective changes required. Specific findings: (1) 1265 diagnosis AGREE -- resolve_source_composition()'s rejection of an already-present focal patch is intentional isolation behavior; demotion, not special-casing composition resolution, is the correct lifecycle repair. (2) 1241/1252 correctly left on expected= AGREE -- their attested identity is a composite two-device/tensor-split identity that a single ProducerDeviceContext cannot faithfully derive; device= is only appropriate for the 5 single-physical-device producers actually migrated. (3) mtp_server_lane() exactly-one-of device/expected API AGREE, sound design. One MINOR non-blocking note: the demotion comment in recipes.toml would benefit from recording the superseded implementation digest/evidence ID for unambiguous later archaeology, though current text is operationally adequate -- not required, no action taken.

## Change Log

- 2026-09-27T11:18:43.972819+00:00 (updated-by): Updated: section:notes
- 2026-09-27T13:25:45.046397+00:00 (updated-by): Updated: section:notes
- 2026-09-27T13:28:13.703319+00:00 (updated-by): Updated: section:notes

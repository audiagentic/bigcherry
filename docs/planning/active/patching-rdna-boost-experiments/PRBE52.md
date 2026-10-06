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

The speculative baseline must also contain upstream llama.cpp PR #29924 (or an equivalent merged fix) before adaptive-depth performance is interpreted for temperature > 0 / mixed drafter lanes. #29924 fixes truncated n-gram drafts being converted into non-empty lists of empty candidate sets, which incorrectly routes verification through rejection sampling with q=0. `draft-mtp` alone is not affected, but `ngram-mod,draft-mtp` combinations are, so the bug can masquerade as an adaptive/MTP throughput or acceptance problem.

## Steps

1. Require 1255 and add `n_min_adaptive=0` to `common_params_speculative_draft` plus `--spec-draft-n-min-adaptive` / `LLAMA_ARG_SPEC_DRAFT_N_MIN_ADAPTIVE`.
2. Store controller state and last drafted count per sequence beside `pending_h`.
3. Reset state in `common_speculative_impl_draft_mtp::begin`; validate the adaptive floor after any chain-head cap of `n_max`.
4. Replace only the fixed `params.n_max <= result.size()` stop test with the controller's current depth when enabled.
5. Record only drafts surviving the existing `n_min` filter; call `update(last_n_draft, n_accepted, ...)` from `accept()` using the real accepted count.
6. Emit one activation marker only on the enabled adaptive path.
7. Before any temperature>0 or hybrid-drafter performance run, resolve #29924 at the current pin. Preserve the exact upstream semantic rule: truncation may resize `result_q` only when the drafter actually populated candidate lists; an empty candidate vector must stay empty.
8. Validate deterministic greedy token identity and server throughput against fixed-depth control. Add a separate temperature 0.7 regression lane near max_tokens/context-end for `ngram-mod`, `draft-mtp`, and `ngram-mod,draft-mtp` at np=1/2.

## Detailed Solution & Technical Design

Patch `1268_prbe52_adaptive_mtp_wiring`, order 1268, `requires=["1255_nro06_adaptive_mtp_depth"]`. The new floor uses 0 as the sole disabled value; enabled values must be <= the effective post-constructor `n_max`. Each sequence owns independent controller/counter state. `begin()` is the request reset boundary. Drafting clears stale accounting before each attempt, applies the adaptive cap only after a token is actually drafted, and stores the final submitted draft count. `accept()` updates the controller after pending hidden-state selection. No global/static adaptive state is allowed. Activation logging uses only a once-per-process atomic flag and does not affect state.

Full-vocabulary comparison is intentionally not the MTP correctness oracle: accepted draft tokens do not expose complete vocabulary rows. Correctness is identical greedy token IDs for equal prompt/model/seed; backend/kernel correctness remains covered by the prerequisite path.

For #29924, keep drafter candidate ownership explicit. Pseudocode shape:

```cpp
if (draft.size() > n_max) {
    draft.resize(n_max);
    if (!result_q.empty()) {
        result_q.resize(n_max);
    }
}
```

Do not "simplify" this back to unconditional `result_q.resize(n_max)`: empty means the drafter supplied no rejection-sampling distribution and must remain a semantic state, not just a capacity/detail.

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

#29924 regression lane: temperature 0.7, fixed seeds, output cap that forces at least one draft truncation near max_tokens/context-end. Report drafted count, accepted count, acceptance ratio, effective t/s, and whether `result_q` was empty/non-empty at the truncate boundary. Upstream reported Qwen3.6-35B-A3B `ngram-mod,draft-mtp` np=1 301.7 -> 333.9 t/s and np=2 187.0 -> 329.4 t/s, with np=2 acceptance 0.538 -> 0.670; BigCherry must reproduce the semantic fix before using the magnitude as an expectation on AMD.

## Effort & Risk

L / medium-high. Main risks are sequence-boundary state leakage, stale draft-count feedback, accidentally changing fixed-depth behavior, and confounding an upstream n-gram truncation bug with MTP/adaptive policy. All are contained by per-sequence vectors, reset-before-use, zero-disabled gating, deterministic token identity, and the explicit #29924 baseline gate.

## Standards

Upstream semantic fixes before local policy tuning; keep drafter candidate-state semantics explicit; one speculative verification owner; no local fork of a two-line upstream fix once merged.

## Acceptance Criteria

- Fixed-depth disabled path remains unchanged and greedy-correct.
- Adaptive lane passes the existing 1210 bit-identical prerequisite and deterministic token gate.
- #29924/equivalent is present before temperature>0 mixed-drafter measurements.
- Truncated n-gram drafts with no candidate distributions remain on sample-and-match verification rather than rejection sampling.
- np=1/2 mixed `ngram-mod,draft-mtp` lanes show no end-of-generation acceptance collapse.

## Notes

Source audit against b11126 confirmed the concrete hooks: `pending_h`, `begin()`, `result.push_back(id)` followed by `params.n_max <= result.size()`, the final `params.n_min` filter, and `accept(seq_id, n_accepted, ...)`. The previous placeholder package id 1256 is obsolete because that order is already occupied; materialized id is 1268. Keep plan state `pending` until hardware evidence promotes/rejects the patch.

FINDING 2026-09-27 (hardware, gfx1100, 3/3 real sessions): backend_reference correctness FAILS -- greedy 64-token MTP output diverges between control and subject at token index 36, identically across all 3 sessions run so far (not noise; deterministic). This directly violates this item's own stated invariant ('the optimization only changes speculative work, never target-token semantics' / 'deterministic greedy token identity'). Speculative decoding's acceptance test is supposed to guarantee the accepted stream always matches the target model's own greedy output regardless of draft depth/dynamics -- a divergence here means the adaptive-depth controller (n_cur tracking, or its interaction with accept()'s last_n_draft update) is corrupting generation, not just its speed. The GPT code-review-confirmed floor invariant (n_min_adaptive >= n_min, fixed 2026-09-27) did not fix this -- it prevents a stuck-drafting state, not this semantic divergence. Remaining 6 hardware sessions (gfx1100 s3/s4, all 4 gfx1201) paused in the queue (perf-only.txt) rather than burning further GPU time reproducing the same deterministic failure. Needs an author-level re-look at the adaptive controller's interaction with the accept-loop hidden-state selection (pending_h/verify_h indexing) before more hardware time is spent -- correctness gate already blocks promotion so nothing unsafe reached production, this is purely about not implementing/testing efficiency.

2026-09-27 (later): GPT diagnosis (req_9d9f95bc2be242f6) landed as commit d5ce7866, adding requires=["1210_rd26_bitidentical_decode_verify_standalone"] to 1268's patch.toml -- the correctness divergence above was caused by decode-batch vs speculative-verify-batch logits not being bit-identical, which 1210 fixes. Deployed and confirmed: t-1265b-gfx1201-s3 (an unrelated patch, but same scaffold path) passed cleanly, proving the framework wiring works.

However the first real 1268 requeue (t-1268b-*, all 8 sessions, gfx1100+gfx1201) all failed in ~13s each with 'ValueError: 1268_prbe52_adaptive_mtp_wiring requires explicitly selected module(s): 1210_rd26_bitidentical_decode_verify_standalone' -- adding `requires` to patch.toml only enforces that 1210 must be present in the resolved exact set IF 1268 is selected; it does NOT auto-compose 1210 into the baseline. The queue lines still only had `--common-patches 1255_nro06_adaptive_mtp_depth` (unchanged from before the requires fix), so resolve_exact failed closed correctly. Fixed by adding 1210 to --common-patches explicitly (`--common-patches 1255_nro06_adaptive_mtp_depth,1210_rd26_bitidentical_decode_verify_standalone`) and requeuing as t-1268c-* (8 sessions, gfx1100 s1-4 + gfx1201 s1-4). Verified locally via resolve_source_composition() before requeuing. GPU time wasted: ~13s x 8 = ~1.7min, negligible.

GPT review (req_28ab0790a31443df) separately reviewed the 1265 demotion (d2c2365d) and PVPS13 locator-fix centralization (364157ff): APPROVE both, no corrective changes required.

2026-09-27 GPT adversarial review verdict (req_28ab0790a31443df, incorporated): APPROVE both commit d2c2365d (1265 demotion) and commit 364157ff (PVPS13 locator-fix centralization); no corrective changes required. Specific findings: (1) 1265 diagnosis AGREE -- resolve_source_composition()'s rejection of an already-present focal patch is intentional isolation behavior; demotion, not special-casing composition resolution, is the correct lifecycle repair. (2) 1241/1252 correctly left on expected= AGREE -- their attested identity is a composite two-device/tensor-split identity that a single ProducerDeviceContext cannot faithfully derive; device= is only appropriate for the 5 single-physical-device producers actually migrated. (3) mtp_server_lane() exactly-one-of device/expected API AGREE, sound design. One MINOR non-blocking note: the demotion comment in recipes.toml would benefit from recording the superseded implementation digest/evidence ID for unambiguous later archaeology, though current text is operationally adequate -- not required, no action taken.

Upstream reference verified 2026-10-04: https://github.com/ggml-org/llama.cpp/pull/29924 . Do not implement it as a BigCherry-specific speculative algorithm; rebase or carry the minimal semantic fix until upstream merge reaches the pin.

2026-10-06 RECONCILED TO b11402 AND QUEUED (steps 1-2 of the corrected execution gate done, 3-4 queued, 5 not started). (1) Reconciliation: against pin d89651a7b205, 1210 and 1255 apply cleanly; 1268 failed on exactly the three anchors the disposition named (prbe52-begin-reset, prbe52-draft-reset, prbe52-depth-limit) because upstream #27694 made begin() reset the per-sequence sampler first, moved the greedy/probabilistic decision ahead of any sampler reset at the draft start, and put the candidate capture between result.push_back and the depth cap. The three edits are re-anchored on that shape and changed from whole-block replacement to inserting only their own lines (begin: after the sampler reset, identified by the !is_mem_shared position check that only the MTP class has; draft start: after 'drafting[seq_id] = true', identified by the !params.probabilistic block and the pending_h embedding; depth limit: the cap block followed by the chain_heads branch). No behavioural change: floor 0 keeps fixed-depth MTP. (2) Tests: the patch's offline fixture updated to the pin's shape, 3/3 pass (apply + idempotence, noise-stripped anchors, missing depth anchor fails closed); 1210 + 1255 + 1268 apply in order on the pristine pin AND on the production-patched tree; patch-lint clean; the three inserted sites inspected in the output. The known_broken disposition (bound to pin 0504396 and the old digest) is cleared. (3) Queued on Brutus behind the MET01 lanes: tools/lab/flash-next/queue-adaptive-mtp.sh b11402i b-fadef-b11402e 8192 32768 - build b-adaptmtp-b11402i (experiment adaptive-mtp: production + 1210 + 1255 + 1268), then per depth on Flash-Next production (2x XTX + R9700 tensor split, MTP draft on the 6900 XT, --spec-draft-n-max 4): (a) adaptive OFF vs the production build b-fadef-b11402e, ABBA - greedy identity plus the cost of carrying 1210 + 1255; (b) adaptive floor 1 vs fixed depth 4 on the one binary, ABBA - greedy identity is the correctness gate, effective decode t/s and acceptance are the result, activation by the BIGCHERRY_PATCH_HIT patch=1268 marker and depth_change events. Deviation from the item's validation text, stated: the first lane is the Flash-Next deployment, not tierM-qwen35b-a3b-moe-mtp / four sessions; that is the lightweight tier - the full contract lanes and the WHIRL calibration (step 5) are not started and nothing here is promotion evidence for them. #29924 is not needed for these lanes (draft-mtp only, greedy).

2026-10-06 FIRST HARDWARE RESULT ON b11402 - NOT PROMOTABLE AS IS (chain-b11402h, sequential; build b-adaptmtp-b11402i = production + 1210 + 1255 + 1268; Flash-Next production, 2x XTX + R9700 tensor split, MTP draft on the 6900 XT, --spec-draft-n-max 4, 512 decode tokens). Lane 'on' = same binary, A fixed depth 4, B LLAMA_ARG_SPEC_DRAFT_N_MIN_ADAPTIVE=1, ABBA. 8K depth (10,102 prompt tokens): A 75.6 / 77.3 t/s (accepted 367/573, 369/565); B 75.6 t/s (340/480) in the run that completed; the other B run CRASHED - 'ROCm error: an illegal memory access was encountered ... in function ggml_cuda_graph_update_executable ... hipGraphExecDestroy(graph->instance)' (log adaptmtp-b11402i-on-d8192/d8192-3-B/timing.server.log). 32K depth (47,090 prompt tokens): A 58.7 / 57.6 t/s (349/644, 346/656); B 61.4 / 62.6 t/s (345/547, 336/509) - +6.7%, complete separation in this one ABBA, no crash. Activation proven: BIGCHERRY_PATCH_HIT patch=1268 in every B server log and none in A; depth_change events 1->2 (x4), 2->3 (x2), 3->4 (x2), 4->3 (x2) at 32K. CORRECTNESS GATE FAILS: greedy text differs between A and B at both depths, and at 32K the two B runs differ from EACH OTHER (md5 11bf7b16... and 1c0a412f...), while the two A runs agree. So with adaptive depth the output is not reproducible run to run. Not yet attributed: the stack now selects kernels by batch size (1334 sparse attention above 4 queries, MMVQ/MMQ thresholds), so a different verify-batch size can change logits by rounding and flip a near-tie; a fixed-depth-3 vs fixed-depth-4 control on the same binary is needed to separate 'depth changes text on this stack' from a 1268 defect. Lane 'off' (production build b-fadef-b11402e vs this binary with the floor unset): 32K A 61.9 / 70.5 t/s (accepted 347/654, 367/576 - the production arm itself varies run to run), B 57.9 / 58.8 t/s (348/648, 349/644); the 8K off lane on disk is from the earlier overlapped run and is WITHDRAWN (the queue skipped it as finished). Reading: carrying 1210 + 1255 with adaptive off costs decode against production (about -5% or more at 32K), adaptive on earns about that much back at 32K and nothing at 8K. Net against production is not positive, plus one crash and non-reproducible text. Next, in order: (1) root-cause the hipGraphExecDestroy crash (graph cache vs changing verify-batch shape); (2) the fixed-3 vs fixed-4 text control; (3) a clean 8K off lane; (4) measure 1210 alone against production. No WHIRL calibration until 1-2 are resolved.

2026-10-06 CONTROLS DONE - REJECTED FOR THE FLASH-NEXT DEPLOYMENT ON b11402 (run chain2, queue-adaptive-controls.sh b11402i; build b-adaptmtp-b11402i; production = b-fadef-b11402e; ABBA; 512 decode tokens). (1) Fixed depth 4 (A) vs fixed depth 3 (B), same binary, adaptive off: 8K 76.3 / 77.6 vs 80.6 / 80.7 t/s (accepted 368/569, 369/565 vs 349/485, 350/482); 32K 58.6 / 58.9 vs 65.3 / 65.5 t/s (349/644 x2 vs 339/515 x2). Greedy text DIFFERS between fixed 3 and fixed 4 at both depths (md5 A != B), and each arm is reproducible. So on this stack the greedy text depends on the draft depth with no adaptive code involved (kernel selection by verify-batch size, e.g. 1334's sparse attention above 4 queries); the adaptive arm's text difference is therefore NOT a 1268 defect, and PRBE52's 'exact greedy IDs' gate cannot be met by any depth-changing policy on Flash-Next. The adaptive arm's run-to-run text variation (two texts in two runs at 32K) follows from the same cause plus acceptance-dependent depth history. (2) Best fixed depth beats adaptive: fixed 3 gives 80.6 t/s at 8K and 65.4 t/s at 32K; adaptive (floor 1, max 4) gave 75.6 and 61.4 / 62.6. The production deployment already runs depth 3. (3) Cost of the prerequisites, adaptive off, depth 4: 8K production 76.9 / 82.1 vs this build 77.3 / 77.0 t/s; 32K production 65.2 / 67.5 vs 58.4 / 58.7 t/s (-11%) - carrying 1210 + 1255 costs decode at depth. (4) One adaptive run crashed at 8K (hipGraphExecDestroy illegal memory access); not root-caused. VERDICT: do not adopt 1268 for Flash-Next - it loses to the best fixed depth, its prerequisites cost up to 11% at 32K, and it crashed once. Patch state stays 'evaluated' (not promoted); the reconciliation to b11402 is kept so the WHIRL cost-model comparison can still be run on tierM-qwen35b-a3b-moe-mtp per the item's own gate, where the result may differ. If that is pursued: root-cause the crash first, and replace the identity gate by a fidelity gate for stacks whose kernels depend on batch size.

2026-10-06 VERDICT NARROWED (owner challenge; GPT req_dd331459b65244d1 and survey req_a66df6ec7eed4403). The earlier 'rejected' applies ONLY to the configuration tested (floor 1, max 4, 512-token requests, with 1210), not to adaptive depth as such. GPT's findings: (1) the test was cold-start heavy - the controller resets to the floor in begin() every request (nasone32's own default floor is 3; WHIRL cold-starts at min(max, 3)); adaptive vs fixed-3 cost ~0.42 s per 512-token request at both depths, i.e. a start-up cost that amortises over long generations; upstream PR #27210 states max < 7 is outside the controller's preferred regime. (2) Reported gains elsewhere are workload dependent and mostly against one fixed depth, not the best fixed depth: PR #27210 (Qwen3.8-27B Q8_0, 2x R9700, >= 4K-token generations, adaptive 3..12 vs fixed 3): reasoning -0.5%, prose +1.3%, code +12.6%, recall +80%; WHIRL auto 31.36 vs best-fixed 31.28 t/s (+0.3%). (3) 1255's rule: climb after 2 / 4 / 10 / 6 / 3 / 2 consecutive full accepts at depth 1 / 2 / 3 / 4 / 5 / >=6, any miss resets the climb, drop when accumulated misses reach max(5*depth, 20); on our trace it settles around 3 with excursions to 4 - and since fixed 3 beats fixed 4 here, an acceptance-only controller cannot know 3 is throughput-optimal. (4) 1210 is required by 1268's manifest, not by the controller; it normalises the verify path toward decode numerics (MMVF / fusion normalisation, small SGEMM and AMD WMMA attention disabled, decode-style tiles) which explains the ~11% cost; where depth-dependent greedy text is already accepted it can be tested without. (5) Other methods: upstream draft-mtp ALREADY has a per-token confidence gate, --spec-draft-p-min (default 0 = off, never set here) - GPT's pick for this setup because it avoids paying the next slow-GPU draft pass inside the round, whereas acceptance-history controllers only learn after paying it; also proposed upstream: an entropy gate (#29875, MTP gain unverified); WHIRL's E/T cost model; suffix / n-gram co-drafting as a complementary mechanism (stew +13.6% over adaptive-MTP alone, mostly recall). QUEUED (chain4 / chain6 on Brutus, production setup with MTP on the 6900 XT, A = fixed depth 3): confidence gate p_min 0.6 / 0.75 / 0.9 with caps 5 / 5 / 6 and 0.75 with cap 3 on the production build; then on the 1268 build with 2,048-token generations: adaptive floor 3 max 4, floor 3 max 8, and floor 3 max 8 with p_min 0.7. The crash (hipGraphExecDestroy) is still not root-caused.

2026-10-06 CONFIDENCE-GATE SCREEN - LARGE LOSS AS CONFIGURED, CAUSE NOT YET ATTRIBUTED (run chain9, queue-pmin-screen.sh b11402 on b-ixtile-b11402f = production set; Flash-Next production, MTP draft on the 6900 XT, 512 decode tokens, ABBA, A = fixed depth 3, B = --spec-draft-p-min with a cap). 8K depth, A 84-86 t/s (accepted ~349/485, 12.4 ms per step): p_min 0.6 cap 5 -> 49.6 / 49.6 t/s (360/470, 361/465; ~22 ms per step); 0.75 cap 5 -> 43.1 / 44.7 (347/410, 351/416; ~28 ms); 0.9 cap 6 -> 41.3 / 41.3 (336/357, 340/361; ~35 ms); 0.75 cap 3 -> 45.6 / 50.1 (323/361, 330/357; ~30 ms). 32K depth, A 65.7-66.6 t/s (327/547): 0.6 cap 5 -> 40.2 / 40.4 (335/437, 338/465); 0.75 cap 5 -> 36.1 / 37.8 (307/384, 322/382); 0.9 cap 6 -> 32.8 / 32.2 (274/298, 271/289); 0.75 cap 3 -> 36.4 / 35.7 (286/329, 286/332). The gate works as a policy - accepted tokens per step rise from 0.60-0.71 to 0.75-0.93 - but time per step is 2-3x higher, so decode is 40-50% slower in every setting, including cap 3 (the production cap) where no extra draft pass can occur. Greedy text also varies between the two B runs of a setting. The draft sampling code path is the same with and without a threshold (common_sampler_sample + get_candidates in both), so the step cost is not the confidence computation. HYPOTHESIS (not confirmed): with a threshold the number of drafted tokens changes from round to round, so the target's verify batch changes shape every round and the HIP execution graph is re-captured instead of reused; fixed depth always produces the same shape. The same mechanism would explain the hipGraphExecDestroy crash seen with adaptive depth and why adaptive depth showed no gain at 8K. If confirmed, variable-length drafting of any kind (confidence gate, adaptive depth, WHIRL) is penalised on this stack until variable verify shapes are made cheap (pad the verify batch to a fixed size, or keep one captured graph per shape), and PRBE52's verdict has to be re-taken after that fix. DIAGNOSTIC QUEUED (chain11): fixed depth 3 and p_min 0.75 cap 5, both with GGML_CUDA_DISABLE_GRAPHS=1, at 8K - if the gated arm is no longer slower than fixed depth without graphs, re-capture is the cause. Adaptive floor-3 lanes (2,048-token generations) are running now.

## Change Log

- 2026-09-27T11:18:43.972819+00:00 (updated-by): Updated: section:notes
- 2026-09-27T13:25:45.046397+00:00 (updated-by): Updated: section:notes
- 2026-09-27T13:28:13.703319+00:00 (updated-by): Updated: section:notes
- 2026-10-04 (agent): Added upstream #29924 as a prerequisite correctness/performance baseline for temperature>0 n-gram + MTP lanes; preserved draft-mtp-only semantics.

## 2026-10-06 WHIRL cost-model audit - corrected execution gate

External mechanism provenance: `tsaipifong/whirl-llm` v0.1.3. Historical BigCherry provenance must also be preserved: 1255 is a local staged adaptation of `nasone32/llama.cpp-RDNA3-7900xtx-opt@10579a7365a3bc86c4f8e41aaab20e73e1571e5e`. WHIRL is a candidate refinement of the cost model, not the origin of adaptive MTP.

### Blocking prerequisite

Do not start WHIRL-policy implementation or hardware comparison from the current 1268 package. `dispositions/1268_prbe52_adaptive_mtp_wiring.json` currently marks it `known_broken / FAILED_NEEDS_RECONCILIATION`: upstream probabilistic-MTP changes broke its begin-reset, draft-reset and depth-limit anchors, and it is not in the current build recipe.

Execution order is mandatory:
1. reconcile 1268 against the current source pin;
2. restore apply/idempotence/composition tests with 1255 + explicit 1210;
3. prove adaptive-off equals fixed-depth behavior;
4. pass 1210 greedy identity and repeated same-process correctness;
5. only then collect adaptive-policy calibration/performance data.

Historical 1268 throughput records whose validation contract failed greedy correctness are motivation only and must not be used as promotion evidence.

### Correct discriminator: sampled depth calibration, not synthetic trace replay

1317/1318 provide useful observed-round timing, but they do not record the counterfactual acceptance/time of draft depths that were not executed. Therefore an arbitrary trace cannot be replayed faithfully through fixed depth, current 1255 and WHIRL E/T policies.

Initial calibration must use `parallel=1` because 1317's file-static accumulator is documented for single-slot interpretation. Deliberately exercise every candidate depth, initially `k={1,2,3,4}` or the supported range, and persist same-round tuples including:

`{session,workload,context_bucket,k,n_drafted,n_accepted,cycle_us,draft_us,target_submit_us,target_sync_us,process_us,sample_us}`.

Do not synthesize unobserved `T(k)`. Require adequate samples for every candidate action in each workload/context bucket used for policy fitting.

Use calibration/train rounds to estimate conditional acceptance and `T(k)`; score held-out rounds. Compare:
- best fixed depth;
- current 1255 climb/drop;
- WHIRL-style `argmax E(tokens|k)/T(cycle|k)` with 3% hysteresis and bounded neighbour probes.

Score accepted target tokens per measured cycle time and regret versus best fixed. Report sample counts/uncertainty. This is an offline discriminator, not counterfactual proof of hardware throughput.

If held-out predicted advantage over 1255 is <5%, stop and retain 1255. If >=5%, replace **only 1255 internals**; retain 1268's existing opt-in/configuration surface and do not add a second controller/mode.

### Internal evidence supporting the objective

FMTP03 already measured the relevant failure mode: representative ahead lanes reduced step time roughly 5-8% while effective throughput remained approximately neutral because promoted-front acceptance fell. That result strengthens the use of accepted target tokens per wall-clock cycle over acceptance-only or latency-only control.

It does not authorize restarting FMTP02-FMTP07; that pipeline remains paused by owner decision and is independent of this front-depth experiment.

### Hardware gate after offline promotion

gfx1201 then gfx1100; fixed-best vs current adaptive vs E/T adaptive; ABBA >=5 repetitions across coding, prose, repetitive/tool-call and representative 8K/64K/128K contexts. Require exact greedy IDs, repeated same-process correctness, >=5% median effective-TG improvement over current adaptive, and <=2% regression in every held-out representative lane.

WHIRL n-gram co-drafting remains out of scope. After #29924/equivalent, use upstream mixed proposer support as the control. No local proposer arbitration without a separate measured residual >=5%.

Traceability:
- https://github.com/nasone32/llama.cpp-RDNA3-7900xtx-opt commit 10579a7365a3bc86c4f8e41aaab20e73e1571e5e
- https://github.com/tsaipifong/whirl-llm
- https://github.com/tsaipifong/whirl-llm/blob/main/src/model/spec.cpp
- https://github.com/tsaipifong/whirl-llm/blob/main/docs/guide/en/speculative-decoding.md
- https://github.com/ggml-org/llama.cpp/issues/24507
- https://github.com/ggml-org/llama.cpp/issues/23184
- 2026-10-06T00:23:02.188814+00:00 (updated-by): Updated: section:notes
- 2026-10-06T01:25:44.356993+00:00 (updated-by): Updated: section:notes
- 2026-10-06T02:50:02.731751+00:00 (updated-by): Updated: section:notes

## Ledger-events

- chg_20261006_025048_adaptive-mtp-depth-wiring-appl_5444
- 2026-10-06T02:50:52.507468+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-06T03:31:46.193215+00:00 (updated-by): Updated: section:notes
- 2026-10-06T05:25:37.876948+00:00 (updated-by): Updated: section:notes

---
id: RCS12
order: 0
plan: build-radiance-components
state: pending
created-at: '2026-11-04T08:38:41.932631+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: L
---

# Model-constant execution guard test and the audit's execution-path fixes (staging planes, GEMV K bound, top-k, histogram, layout resolver)

## Description

Owner requirement: BigCherry must not hardcode model-based implementation into the core; Qwen3.8 Flash-Next (MoE) and Qwen3.8-27B (dense) are both live targets, and a new model must be addable by data (container, manifest, layout) not by editing core code. A read-only audit on 2026-11-04 (55 classified findings, each with file:line and a quoted source line) found that the requirement is not yet met. The helper derives shape from the container (Container::open, segment_shape, layout_for), but several live consumers still use Flash-Next constants for byte offsets, route capacity, heat accounting and matrix math, and the bc-serve layout resolver only knows the single-card Flash-Next recipes. This item owns the fixes ranked P0/P1 by the audit plus the guard test that stops new model literals entering execution paths. It sits beside RCS09 (shape negotiation, per-kernel limits) and RCS10 (generic-model coverage) and does not replace them: each fix lands in the slice that already owns it, and this item is the umbrella plus the guard.

## Steps

1. Guard test first (card-free, CPU CI): model_constant_execution_guard with two checks. STATIC: scan engines/radiance/components/**, patch payloads, tools/lab/radiance launch scripts, engine_adapters and the layout CLI for model-shaped literals (2560, 640, 384, 512, 10, 49, 160 and the n-gram sizes), byte-offset multiplications, kernel dispatch predicates on fixed shapes, std::getenv("R3_*") reads and hard-coded .rad or heat paths; an allowlist entry per accepted use as path:anchor:category:reason:owner:expiry; only bounded capacities that fail closed (kMaxLayers, kMaxTopK, hardware tile sizes, checked K bounds), isolated test/replay files and comments may stay long-term; an unexpired execution-path or configuration-debt entry may not outlive its owning change. DYNAMIC: instantiate Container/fake geometry and layout_for for Flash-Next {hidden 2560, top_k 10, experts 512, intermediate 640, layers 49} plus two synthetic shapes ({2048, 4, 256, 768, 32} and {2560, 16, 1024, 3072, 64}) and assert non-overlapping in-bounds activation planes, max_rows*top_k capacity, histogram coverage, consistent encoder/decoder fields and fail-closed refusal of unsupported kernel dispatch.
2. Fix P0 MC-001..003: remove fixed kActivationBf16Bytes/kActivationQBytes plane-address math from the engine mailbox ops and the helper (moe_mailbox_ops.hip, moe_remote_gpu.h); derive offsets from the segment layout; test with different hidden widths on both sides, mutation-checked.
3. Fix P0 MC-004..005: stop the GEMV silently returning early for K > 2560 (moe_lane_gemm.h, moe_remote_gpu.h): choose the kernel from the true K or fall back to the verified GEMM; startup must also check the full-width intermediate K.
4. Fix P0 MC-016..019 (with RSC05/RSC08): replace the Flash-Next-only recipe_for, fixed GPU selection, TP=1 and plugin list in serving.cpp with validated layout and arch data; cover dense qwen27b-2xtx and an invented third layout in card-free tests.
5. Fix P1: MC-010..011 (no top_k=10 default when metadata is absent: resolve and reject ambiguity before allocating the segment); MC-008..009 (the histogram counts all runtime expert IDs, bounded/chunked scratch, shape-aware heat-file provenance, with RHP02); MC-006..007 (replace kMaxRoutes = kMaxRows*10 with rows*actual_top_k plus slot capacity, including CPU and replay variants); MC-012..014 (one activation-quantisation group and one CPU/GPU TP slice policy from LayoutShape, guarding unsupported formats and splits).
6. P1/P2 follow-ups through their owning items: typed-manifest startup end to end with no R3_* fallback (RSC02/RSC06/RSC08, MC-022..039), placement cost from actual top-k and calibrated policy (RSC03, MC-015/024/025), n-gram width/vocab registry constraints and the TP all-reduce cutover as calibrated arch/layout policy (RCS10/RAD19, MC-020/021/040).
7. Each fix removes or expires its allowlist entry in the same change; no compatibility path stays.

## Detailed Solution & Technical Design

Classification used by the audit: A bounded capacity that validates the real shape and refuses; B diagnostic or test-only; C a model-specific value that affects an execution, selection or planning branch (a defect for any other shape, not necessarily for today's two models); D configuration that belongs in the manifest/layout/arch data; E naming or comment only. Numbers 512, 128 and 256 are often MXFP4 tile, FP8 group or attention warp sizes, not model dimensions: do not replace them mechanically; examine the dataflow and the checked dispatch. Where shape comes from today: Container::open reads layers/experts (max directory IDs), hidden and intermediate (down-projection shape) and top_k from metadata n_expert_used (0 means absent, not 10); segment_shape builds LayoutShape and layout_for derives route, activation, result, map and bitmap bytes with checked arithmetic; Runner::set_shape stores the runtime shape; the engine patch rad0002 declares mailbox ops with the engine's n_embd and top_k. What the audit found outside that path: staging Q and scale planes still offset by 2560-wide constants in producer and helper (MC-001..003); GEMV kernels return early for K > 2560 after the dispatcher chose them by row count only (MC-004..005); top_k fallback 10 and kMaxRoutes derived from 10 (MC-006..011); a fixed 512-expert engine histogram and 128-group activation assumption (MC-008..009, 012); CPU helper reference split 384+256 for width 640 without the refusal the GPU path has (MC-013..014); serving.cpp layout names, device types, rank count and plugin list fixed for Flash-Next (MC-016..019); n-gram kernel registration requires an exact embedding width and vocabulary (MC-020..021). Qwen3.8-27B (dense, no remote experts) is not exposed to the MoE/helper defects; it is exposed to the bc-serve layout resolver blocker and to the tuned TP all-reduce cutover (a policy value, MC-040), and the dense qwen27b-2xtx layout is the first non-Flash-Next layout the guard must pass.

## Code Samples & Guidance



## Files

engines/radiance/components/moe-remote/{moe_mailbox_ops.hip,moe_remote_gpu.h,moe_lane_gemm.h,worker_shape.h,moe_remote_shm.h,worker_hostref.h,moe_remote_hook.cpp}, engines/radiance/components/serving/{serving.cpp,experts.h}, engines/radiance/components/ngram-helper/r3_ngram_host.cpp, engines/radiance/patches/rad0002_moe_remote_handoff/, tools/lab/radiance/launch-flashnext-best.sh, tools/tests/ (the new guard test and its allowlist), component.toml test declarations and tools/tests/release/test_bump_gates.py.

## Validation

The guard test runs in CPU-only CI and is required by patch promotion; it fails when a model-shaped literal is added to an execution path without an allowlist entry, and when an allowlist entry no longer matches the source. The dynamic check passes for all three geometries after the fixes. Mutation-check each fix (restore the constant and show the test fails). Hardware: Flash-Next production layout unchanged in decode ms/step and KLD (the fixes are behaviour-neutral for 2560/640/512/10), checked with the standard A/B; qwen27b-2xtx unchanged.

## Effort & Risk

Large. Risk: the P0 fixes touch the engine/helper shared-memory contract, where an offset mistake corrupts silently, so each change needs both-sides layout tests and a hardware KLD and speed check; changing kernel dispatch by true K must not slow the Flash-Next decode path (gate on the same A/B as every RHP slice). The guard test needs a careful allowlist or it becomes noise: allowlist by anchor and reason, not by line number.

## Standards

No hand-picked constants (derive from the container, layout and runtime); no compatibility shims or default-preserving parameters; fail closed; engine changes only as managed patches with thin calls; GEMM and kernel rewrites only as managed patches or bigcherry components; each fix migrates every caller in the same change.

## Acceptance Criteria

1. model_constant_execution_guard exists, is required in CI, and has an allowlist with owner and expiry on every entry. 2. No unexpired C-class entry remains for MC-001..014 and MC-016..019; their fixes are merged with mutation-checked tests. 3. bc-serve resolves qwen27b-2xtx and an invented third layout from data. 4. A synthetic model geometry different from Flash-Next passes the dynamic layout and capacity checks. 5. Flash-Next and 27B production numbers unchanged.

## Notes

Source: model-constants audit 2026-11-04 (55 findings; report and findings.json kept in the work root, not in git). Plan ownership per the audit: RCS09 shape negotiation and per-kernel limits, RCS10 generic-model coverage, RSC05 layout and device resolution, RSC06 typed helper loader, RSC07 engine loader, RSC08 unified manifest and launch cutover, RSC03 calibrated resource plan, RAD19 performance experiments. These are assignments, not completed items.

2026-11-04: step 1 STATIC half done. tools/bigcherry/model_constants.py scans engines/radiance/components, patch payloads and tools/lab/radiance for model-shaped literals (2560, 640, 384, 160, 49, .rad paths) and the symbols the audit found them behind (kActivationBf16Bytes, kActivationQBytes, kLegacyIntermediate, kExpertWidth, kRankWidth, kHidden, kMaxRoutes); engines/radiance/model_constant_guard.toml classifies the 99 current hits (94 entries, A/B/C/D/E). C and D entries name an OPEN plan item as owner and become invalid when it completes, so fixing a finding means deleting its entry in the same change. tools/tests/core/test_model_constants.py runs it on the real tree and mutation-checks the checker on a synthetic tree. NOT done: the DYNAMIC half (Container/fake geometry and layout_for at Flash-Next plus two synthetic shapes, a C++ card-free test), the getenv(R3_*) and numeric 512/10 scans (declared settings are audited by the component gate; the bare numbers are too ambiguous to scan without dataflow), and steps 2-7 (the fixes).

## Execution staging (2026-11-13)

Split out as separate items: GEMV K bound (step 3) -> [RCS13](RCS13.md); remote-MoE topology bounds (audit C05) -> [RCS14](RCS14.md). RHP01/RHP02 carry the heat-feed work; the shm protocol is shared, so order matters:

1. Stage 1, parallel and ABI-free: RCS13 (kernel/dispatch, card-free then a short hardware identity) and RHP01 stage 1-2 (rad0010 current-pin tests, missing hardware cells); RHP02 slice A (inactive observer ABI and engine seam).
2. Stage 2: RCS14 gates A-E card-free in order, then gate F hardware parity on Flash-Next helper layouts, then TP3 remote helpers.
3. Stage 3: RHP02 slice B (transport on protocol v26), then slice C (shadow parity, needs RHP01 accepted; covers RHP08), then slice D (atomic cutover, deletes the duplicate collector).
Each stage ends with the standard decode and prefill A/B and KLD; nothing in a later stage starts on hardware before the earlier stage's identity result. RCS13 and RCS14 both touch `moe_remote_gpu.h` and `moe_mailbox_ops.hip`: RCS13 lands first.

## Change Log

- 2026-11-04T08:38:41.932631+00:00 (created-by): Created by agent
- 2026-11-04T08:39:20.421299+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:effort_risk, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20261104_083930_a-plan-item-now-tracks-making_2415
- 2026-11-04T08:39:34.906520+00:00 (updated-by): Updated: section:ledger-events
- 2026-11-04T22:37:02.085119+00:00 (updated-by): Updated: section:notes
- chg_20261104_223702_a-new-test-blocks-model-specif_4673
- 2026-11-04T22:37:06.168768+00:00 (updated-by): Updated: section:ledger-events
- 2026-11-13T03:54:41.818055+00:00 (updated-by): Updated: section:notes

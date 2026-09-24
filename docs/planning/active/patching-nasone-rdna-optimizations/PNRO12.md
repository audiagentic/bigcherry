---
id: PNRO12
order: 0
plan: patching-nasone-rdna-optimizations
state: pending
created-at: '2026-09-09T10:53:04.447530+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P1
---

# Qwen4exp gather-based sparse QSA decode

## Description

Resolve upstream ancestry/backport and qualify Qwen4exp gather-based sparse QSA decode; do not duplicate an equivalent upstream implementation.

TODO, but the 'resolve PR #28213 ancestry' step is superseded by a stronger finding: upstream already ships a gather-based sparse QSA decode fusion, but ONLY for the Vulkan backend, not CUDA/HIP. Verified in b11126: `conversion/qwen4exp.py:63` already writes `indexer_top_k` GGUF metadata (`add_indexer_top_k(hp['indexer_budget'])`). `ggml/src/ggml-vulkan/ggml-vulkan.cpp` fully implements the op-fusion: `topk_qsa_pattern` (a `GGML_OP_GET_ROWS, GGML_OP_PERMUTE, ...` op-sequence constant, ggml-vulkan-types.h:530), `topk_qsa_edges` (edge constraints, :535), `ggml_vk_can_fuse_topk_qsa()` (matcher, ggml-vulkan.cpp:13589) and `ggml_vk_topk_qsa()` (the fused gather kernel dispatch, :11173), wired into the graph-fusion pass around :14178. Grepped `ggml/src/ggml-cuda` for 'qsa': zero matches -- no CUDA/HIP equivalent. The correct scope for this item is therefore: PORT the Vulkan fusion's op-pattern design into ggml-cuda's own fusion mechanism (verified real API: `ggml_cuda_can_fuse()`, ggml-cuda.cu:3180, already used for other fusions e.g. `ggml_cuda_can_fuse(cgraph, i, { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})` at :3544) -- not scan for a separate CUDA-side upstream PR. Masked full-cache attention remains the correct nonqualifying-context fallback either way.

## Steps

- Resolve PR #28213/equivalent head and ancestry against current/candidate pin before authoring a patch.
- If absent, port with QSA/decode predicate, n_kv>=4*width threshold, runtime escape hatch, and selected K/V/bias indices preserving rotation/cache semantics.
- Validate padded top-k width, -inf masking and single-token-per-stream restriction; nonqualifying contexts retain masked full-cache path.
- Sweep below/at/above threshold and compression ratios; compare selected-index and attention outputs to masked reference.
- Measure KV bytes, gather/dequant/cast cost, FA cost and total decode; retire local patch when equivalent becomes baseline.

## Detailed Solution & Technical Design

Gathering bounds attention work near indexer_top_k+ratio but adds GET_ROWS/dequant overhead, so a crossover selector is required. Bias and position/stream semantics are load-bearing; model-specific Qwen4exp scope must not generalize to ordinary attention.

Read `ggml_vk_can_fuse_topk_qsa()` (ggml-vulkan.cpp:13589) and `ggml_vk_topk_qsa()` (:11173) in full first -- their exact op-pattern/edge-check logic and gather-kernel structure is the reference design to port, not to reinvent. In ggml-cuda.cu, add a `ggml_cuda_can_fuse_topk_qsa(cgraph, i)` predicate mirroring the Vulkan matcher's op-sequence check but using `ggml_cuda_can_fuse()`'s existing op-list-matching primitive (same style as the ROPE/VIEW/SET_ROWS example at :3544), gated additionally on this item's own required conditions: QSA/decode predicate (single-token-per-stream, i.e. decode not prefill), `n_kv >= 4*width` threshold (only worth gathering above this ratio -- crossover selector), and a runtime escape hatch (env var, mirroring other ggml-cuda opt-outs) to force the ungathered masked path. On match, dispatch a new gather kernel (new .cu function, e.g. `ggml_cuda_topk_qsa_gather` in a new file `ggml/src/ggml-cuda/topk-qsa.cu`) that: computes the padded top-k selected K/V/bias indices from the indexer scores (respecting `indexer_top_k` GGUF metadata read the same way Vulkan's path does), applies -inf masking for padded/invalid slots, and preserves RoPE/cache rotation and position/stream semantics exactly as the existing masked full-cache FlashAttention path does (do not reimplement rotation math -- gather indices then feed the existing attention kernel path, mirroring how Vulkan's fusion consumes GET_ROWS+PERMUTE as pre-stages rather than a monolithic new attention variant). Below the n_kv threshold or outside decode, or with the escape hatch set, or for any architecture/model outside Qwen4exp's indexer-based QSA, fall through unchanged to the existing masked full-cache attention path -- this must remain the default and the only path for ordinary (non-QSA) attention models.

## Code Samples & Guidance

Target files (b11126): `ggml/src/ggml-cuda/ggml-cuda.cu` -- verified anchor: `static bool ggml_cuda_can_fuse(const struct ggml_cgraph *                cgraph,` at line 3180 (existing general fusion predicate to build the new QSA-specific matcher alongside, following the precedent call `ggml_cuda_can_fuse(cgraph, i, { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})` at line 3544 -- mode="insert_after" a new `if (ggml_cuda_can_fuse_topk_qsa(cgraph, i)) { ... }` branch modeled on that call site). New file `ggml/src/ggml-cuda/topk-qsa.cu` for the gather kernel (no existing anchor -- new file, added via patch package file addition, not an Edit). NEEDS-VERIFICATION before authoring patch.py: (1) full body of `ggml_vk_can_fuse_topk_qsa`/`ggml_vk_topk_qsa` (ggml-vulkan.cpp:13589,11173) to get the exact op-pattern/edge list to mirror; (2) `ggml_cuda_can_fuse`'s full signature/semantics (read lines 3180+ in full, not just the call-site precedent) before writing the new predicate; (3) how `indexer_top_k` metadata is read/threaded into the graph at inference time (grep llama.cpp's model-build code for `indexer_top_k` consumption, likely near Qwen4exp's attention graph construction).\nPatch package sketch: `patches/1265_pnro12_qsa_gather_cuda/{patch.toml,patch.py,SUMMARY.md,README.md,TESTING.md}` (next free id -- adjust for actual availability at authoring time). patch.toml: kind="enhancement", state="untested", requires=[].

## Files

Qwen4exp graph/model declarations; graph-input mask setup; upstream ancestry record; gather implementation/patch if absent; exact attention/reference and long-context campaign evidence.

ggml/src/ggml-cuda/ggml-cuda.cu (fusion dispatch); ggml/src/ggml-cuda/topk-qsa.cu (new gather kernel file); ggml-vulkan.cpp (reference only, not modified); patches/1265_pnro12_qsa_gather_cuda/*; selected-index/masked-reference correctness fixtures; KV-bytes/gather-cost/FA-cost decode campaign evidence.

## Validation

Ancestry; selected-index equality; exact/near-exact attention; threshold boundary; multi-stream; quantized KV; long-context memory/latency; nonqualifying original path.

Offline: `PYTHONPATH=tools python -m bigcherry patch-lint patches/1265_pnro12_qsa_gather_cuda`; `PYTHONPATH=tools python -m bigcherry patch-rebase-check --focal-overlay 1265_pnro12_qsa_gather_cuda --source bigcherry-tuning`; test-backend-ops-style fixtures: selected-index equality vs masked full-cache reference at padded top-k widths, -inf masking correctness, single-token-per-stream restriction enforced (multi-token-per-stream must fall back), threshold boundary (just below/at/above n_kv>=4*width), nonqualifying contexts (prefill, non-Qwen4exp, escape hatch set) retain the unmodified masked path exactly. Hardware (Brutus, gfx1100/gfx1201, not run here): `python -m bigcherry.patch.validation_campaign --overlay 1265_pnro12_qsa_gather_cuda --arch gfx1100` sweeping below/at/above-threshold compression ratios, measuring KV bytes, gather/dequant/cast cost, FA cost, total decode vs masked baseline; retire this local patch if/when a real upstream CUDA/HIP QSA fusion equivalent lands and becomes baseline.

## Effort & Risk

Work=M (unchanged). Correctness risk centers on exactly matching the Vulkan reference's masking/index semantics and not disturbing RoPE/cache rotation -- gather indices then reuse the existing attention kernel path rather than reimplementing rotation math.

## Standards

Upstream-before-backport; attention correctness first; explicit crossover selector; no ordinary-attention generalization.

## Acceptance Criteria

Correctness matches masked QSA over threshold matrix; nonqualifying contexts retain original path; long-context decode shows positive effect after gather overhead; local patch is retired once upstream equivalent is baseline.

## Notes

Supersedes: NRO13
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-nasone-rdna-optimizations-nro13

2026-09-24 relevance at b11126: TODO, redirected from 'resolve PR #28213 ancestry' to 'port the existing Vulkan topk_qsa fusion (ggml-vulkan.cpp:11173/13589) into ggml-cuda using its own ggml_cuda_can_fuse() mechanism (ggml-cuda.cu:3180, precedent call at :3544)' -- this is a materially better-grounded design than the item's original premise since it found a real reference implementation to mirror instead of an assumed-absent upstream PR. GPT design request req_fd0a2a33c0804146 (batched PNRO11+PNRO12) was still running/had not returned a terminal response by the time this item needed to be finalized in this session; design completed directly from source instead. If/when that GPT response lands later, a follow-up session should read it and reconcile.

## Change Log

- 2026-09-09T10:53:04.447530+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:09:32.138424+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.103800+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.758828+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:45:42.383626+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_024630_the-remaining-nasone-successor_5195
- 2026-09-10T02:46:30.038448+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:35:47.115828+00:00 (updated-by): Updated: section:description, section:detailed_solution, section:code_samples, section:files, section:validation
- 2026-09-24T02:36:56.930450+00:00 (updated-by): Updated: section:effort_risk, section:notes

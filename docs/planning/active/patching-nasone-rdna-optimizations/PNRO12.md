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

TODO, NOT-READY (Vulkan reference misinterpreted, per GPT review). CORRECTED: `ggml_vk_can_fuse_topk_qsa()`/`ggml_vk_topk_qsa()` fuse the INDEXER's score-expansion->TOP_K chain (selecting which K/V positions to attend to) -- they do NOT themselves gather K/V or reduce FlashAttention's actual KV length. Verified at b11126: `src/models/qwen4exp.cpp::graph::build_attn_qsa()` (line 695, called at line 841) still feeds the FULL cached K/V plus a sparse mask into attention -- the compaction this item wants (actually shrinking the K/V tensors FlashAttention operates on) is not what the Vulkan fusion does. The proposed `n_kv >= 4*width` crossover criterion is not Vulkan's real gate (that gate was invented, not sourced). The proposed new `topk-qsa.cu` file is also incompatible with BigCherry's anchor-only patcher (Edit() operates on anchored text in existing files, not whole-new-file additions without a corresponding edit-based integration).

## Steps

1. Implement the actual compact-K/V graph change at `src/models/qwen4exp.cpp::graph::build_attn_qsa()` (verified real function, line 695, called at line 841), AFTER the `mctx_cur->get_k(...)`/`get_v(...)` calls (verify exact call sites at implementation time) -- this is where full K/V currently gets fed into attention; the real gather/compaction must happen here, not by fusing the indexer's own TOP_K op.
2. Preserve the selected-index set, masks, stream, and RoPE/cache rotation semantics exactly as the existing masked full-cache path does when building the compacted K/V tensors.
3. IF a separate port of Vulkan's indexer-TOP_K fusion is also wanted (a distinct, smaller optimization from the K/V compaction in step 1): dispatch it from `ggml_cuda_try_fuse()` (verified real function, ggml-cuda.cu:3432) rather than a new bespoke predicate, and put the implementation/declaration into the EXISTING `top-k.cu`/`top-k.cuh` files (extending them via anchored Edit()s), not a new `topk-qsa.cu` file, since BigCherry's patcher is anchor-only against existing files.
4. Use the REAL crossover predicate for when compaction is worthwhile -- derive it from measurement (KV-bytes/gather cost vs FA cost trade-off), not the invented `n_kv >= 4*width` rule, which has no basis in the Vulkan source.
5. Validate padded top-k width, -inf masking and single-token-per-stream restriction; nonqualifying contexts retain the masked full-cache path.
6. Sweep below/at/above threshold and compression ratios; compare selected-index and attention outputs to masked reference.
7. Measure KV bytes, gather/dequant/cast cost, FA cost and total decode; retire local patch when an equivalent becomes upstream baseline.

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

2026-09-24 GPT review req_215c89d0b13a4bb7 applied: corrected the misinterpretation of the Vulkan reference -- ggml_vk_can_fuse_topk_qsa/ggml_vk_topk_qsa fuse indexer score-expansion->TOP_K only, they do not gather K/V or shrink FlashAttention's KV length (verified build_attn_qsa at qwen4exp.cpp:695/841 still feeds full K/V+mask). Moved the real compaction work to build_attn_qsa() itself, after get_k/get_v; rescoped any indexer-TOP_K-fusion port to dispatch via the verified real ggml_cuda_try_fuse (ggml-cuda.cu:3432) into the existing top-k.cu/.cuh files rather than a new file the patcher cannot integrate; removed the unsourced n_kv>=4*width rule in favor of a measured crossover.

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
- 2026-09-24T04:50:47.897561+00:00 (updated-by): Updated: section:description, section:steps, section:notes

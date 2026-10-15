---
id: QFP05
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-03T15:20:39.453089+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# 1297 MTP draft vocabulary trim and context-adaptive draft LM-head follow-ups

## 2026-10-14 implementation audit: prefer an offline GGUF d2t control before adaptive vocabulary

**Disposition:** QFP05 stays pending, but the first unresolved implementation is a **single static frequency-ranked, offline-trimmed MTP sidecar**. The existing validated 1297 runtime first-N trim stays the production control; do not replace it on external performance claims. Dynamic per-step vocabulary selection, a second runtime cache, and new TOP_K/scheduler APIs are **not admitted** without the measured gates below. No Radiance helper/mailbox or actively developed engine work is in this scope.

### Source-confirmed execution and memory ownership

- BigCherry `engines/llamacpp/patches/1297_draft_vocab_trim/patch.py` builds `bc_trim_ctx/bc_trim_buf` at `llama_context` construction, **after** `model.output` has been loaded. It reads the entire original output tensor with `ggml_backend_tensor_get` into a host `full` vector, copies selected quantized rows into `rows`, then uploads `bc_out_trim` and `bc_out_trim_ids`. It does **not** release the canonical `model.output`. Thus this is a draft-GEMV compute trim, **not** a model/VRAM trim.
- For 248,320 × 2,560 Q8_0 weights (34 bytes per 32), full output payload = **675,430,400 bytes (644.14 MiB)**; a nominal 65,536-row subset = **178,257,920 bytes (170 MiB)**. Runtime 1297 keeps both, at least **814.14 MiB** of head payload on the owning backend (plus buffer padding and special-token rows), and temporarily allocates roughly that much host payload in `full + rows`. These are **static byte calculations, not measured VRAM or transfer timings**.
- The 1297 graph in `src/models/qwen4exp.cpp::graph_mtp` computes compact logits, constructs full-vocabulary `-inf` logits and scatters with `ggml_set_rows`; the sampler/target see canonical token IDs. Its empty-output-row fallback keeps the ordinary head. Preserve this verified graph contract and the default-unset path.
- Pinned upstream b11552 `src/models/qwen4exp.cpp::load_arch_tensors` still loads `output.weight` as `{n_embd,n_vocab}`; it has no Qwen4Exp `d2t` consumer. `src/models/eagle3.cpp` already loads a `d2t` tensor and scatters compressed logits. Upstream issue [#25187](https://github.com/ggml-org/llama.cpp/issues/25187) has a Qwen35 MTP-only GGUF `d2t` prototype at `avifenesh/llama.cpp@047bfa508`. The **Qwen4Exp-specific** external port `drluoto/llama.cpp@fb367b8cf2cf439326c8f77477ff65d01ba4816f` changes `load_arch_tensors` and the **MTP graph head**, not the trunk head. This is a testable implementation reference, **not upstream-merged code**.
- The external `scripts/frspec/trimma-sidovagn.py` emits an offline reduced Q8_0 `output.weight` and `d2t` I64 map. As inspected, its frequency-map loop checks `i < n_vocab` but **not `i >= 0`**, and does not guarantee EOG/control/user-defined token inclusion. A negative token ID can index the wrong Python/NumPy row. Do **not** copy this producer without strict range, uniqueness, required-token and exact-row-order checks. The fork also supports an inverse `t2d` variant selected by map length; BigCherry should qualify only the unambiguous forward `d2t[compact_row] = canonical_id` contract first.

### One bounded static experiment; exact owner and implementation

**Owner:** QFP05 owns vocabulary membership, offline producer, map and performance disposition. QFN01/1297 owns the already-validated runtime deployment control. FMTP02/03 and QFP08 own draft scheduling/prefill; QFP13/QFP17 own generic TOP_K. No new sampler, scheduler, allocator or registry.

1. **Pre-hardware fixture:** add a disposable GGUF producer/validator using existing `gguf-py`. Input: immutable original sidecar GGUF SHA256, frozen tokenizer/corpus/hash, selected K=65,536 (optional 48K control), and a deterministic ranked-ID list. Form `required = control ∪ user_defined ∪ eog` from the exact tokenizer; reject missing/out-of-range/duplicate IDs, negative IDs, mismatched tokenizer and head dimensions. Union required tokens **before** applying the K cap; if required exceeds K, reject. Sort selected canonical IDs once and gather quantized rows in precisely that order; write `d2t` I64 of the same length. Verify GGUF readback of every selected row (byte-equal quantized block), mapping, type, shape, metadata, output hash and token coverage. Do not modify the trunk GGUF.
2. **Consumer change only if fixture passes:** extend `src/models/qwen4exp.cpp::load_arch_tensors` with an explicit MTP-only predicate based on `nextn_flags`/existing sidecar identity. If and only if a valid `d2t` tensor is present, size `output.weight` to `n_draft_vocab`, load the map, and fail **before graph execution** on invalid map/type/shape or missing independent output head. In `graph_mtp` after the MTP `build_lora_mm`, reuse EAGLE3's `ggml_fill(-inf) + ggml_set_rows` to return `[n_vocab,n_outputs]`; do not edit the trunk graph. Keep full-head/no-map behavior byte-equivalent.
3. **Mutual exclusion:** a GGUF `d2t` and `BIGCHERRY_DRAFT_VOCAB_N>0` must be rejected together, not double-trimmed or silently reinterpreted. The existing 1297 runtime trim stays available for legacy full-head sidecars. No runtime row gather, dynamic bucket cache or additional device allocator in this static experiment.
4. **Shape/lifetime:** the GGUF owns reduced weights and `d2t` for the model lifetime; the draft graph borrows immutable tensors. Reuse existing GGML buffer/graph lifetimes; no per-request mutation or host sync. Full-vocab scatter remains `-inf` for excluded IDs and must handle `n_outputs=1`, multi-row MTP, no-output prompt replay, graph rebuild and repeated same-process requests.
5. **Controls:** on gfx1030 (6900 XT draft), matched b11552, same target and exact seed/prompt, compare (A) full head, (B) 1297 first-N 65,536 + specials, (C) offline GGUF with the **same** row IDs as B, (D) offline frequency-ranked 65,536 + required IDs. Only C vs B isolates load-time residency; D vs C tests policy/acceptance. Include 10K/80K/160K, code/prose/multilingual and an EOS-heavy control; log actual draft-head GPU time, total draft service, accepted/drafted by position, effective target tok/s, model-load wall time, host peak, device allocations, `d2t` map/hash and executed graph route.
6. **Promotion:** first require exact canonical logits/scatter, greedy target-text identity, target-logit/KLD controls, deterministic seeds, multi-request and MTP rollback, and no graph/copy lifetime violations. Require >=4 independent sessions, >=10 paired ABBA rounds per depth, >=3% effective decode gain over **best working 1297 static control** at 10K and 80K (CI95 lower bound >0), <=2% long-context regression, and no acceptance loss that erases throughput. The offline route can separately qualify as a **VRAM enabler** if measured resident savings permit a previously failing workload without regression; do not call that a decode speedup.
7. **Stop gate for adaptive policy:** only reopen per-step NanoSpec/SpecVocab-style buckets after an exact-pin trace of the best offline static route proves >=3% *exposed* end-to-end perfect-removal head opportunity and demonstrates material acceptance misses recoverable by context. Measure gather+selection+remap and graph-shape reserve before designing any dynamic implementation. A simple 170/644 × 4.05-ms head-time extrapolation suggests ~1.07 ms/step at 64K (~2% of a 54-ms 80K step), **hypothesis only**, not a measured ceiling; reject only using observed timing.

**External performance is not a BigCherry result:** upstream #25187's 5090-laptop Qwen35 map improved mean 83.9→85.1 tok/s on its public corpus; an independent 70K mixed-workload test in the issue found trimmed 32K slower than a full-head control; the Qwen4Exp Strix Halo fork reported 27.4→29.9 warm tok/s but 29.3→28.9 cold tok/s with a 65K sidecar. These establish sensitivity to workload/architecture, not expected gains on gfx1030.

**Cheapest discriminator performed in this audit:** 13/13 actual-source string checks (1297, pinned b11552, external Qwen4Exp port); 12/12 host-only map/scatter/admission/byte-accounting tests. No GGUF conversion, patch composition, HIP compile or GPU benchmark was run.

## Description

Patch 1297_draft_vocab_trim (evaluated, production profile, opt-in `BIGCHERRY_DRAFT_VOCAB_N=65536`) trims the MTP draft `output.weight` (Q8_0, 675 MB, 248K vocab), which was ~31% of 6900-XT draft time at 80K. The current implementation keeps the first N rows and scatters `-inf` for the rest. Evidence is already strong enough to keep 1297 as the baseline: ~8% lower ms/step at 10K and ~7% at 80K.

The next optimisation is **not** another arbitrary smaller prefix. Replace prefix membership with a measured vocabulary policy and separate three mechanisms:

1. **static frequency-ranked subset** (FR-Spec/VocabTrim control);
2. **context-adaptive active vocabulary** (NanoSpec/MicroSpec-style temporal locality control);
3. **smaller/requantized LM-head rows** (storage/bandwidth control).

QFP05 owns draft-vocabulary selection, row remapping, draft-head service time and acceptance trade-offs. It does not own generic `TOP_K` kernels, MTP scheduling/overlap (FMTP), or target verification. Any generic multi-row TOP_K optimisation belongs upstream/QFP17/QFP13 as appropriate.

## Steps

1. Instrument real production-like traffic and record, per drafted position: target token id, draft top-k ids/probabilities, whether the accepted target token is inside candidate subsets, draft LM-head kernel time, gather/remap time and total draft service time.
2. First qualify the offline GGUF d2t control above at equal 65,536-row membership against runtime 1297, then compare frozen frequency-ranked membership against the same-size prefix control. Require canonical row/ID mapping and EOG/control coverage.
3. Only if the offline static route passes its correctness gate **and** measured exposed head/acceptance opportunity clears the >=3% E2E admission gate, prototype a context-adaptive **union vocabulary**: `V = V_static_hot ∪ V_recent ∪ V_prompt ∪ V_draft_top/history`, capped to a fixed bucket size. Start with 8K/16K/32K buckets, not per-step arbitrary tensor shapes.
4. Keep selection state GPU-resident where practical. Build/remap indices asynchronously and double-buffer them so selection/gather for step n+1 can overlap draft transform work for step n. Do not add a host synchronization to save a GEMV.
5. Compare two implementation forms: (A) gather selected rows into a contiguous scratch head then GEMV; (B) indexed/sparse row GEMV directly from the canonical head. Select by **total head service time**, including gather and remap, not GEMV microbenchmark alone.
6. Separately test output-head requantization. Do not combine quantization and vocabulary-policy changes until each has an independent ABBA result.
7. Re-measure 1297/48K/64K/static-frequency/adaptive at 10K and 80K, then one long-context lane. Promote only a Pareto improvement in end-to-end effective TG.

## Detailed Solution & Technical Design

### Fixed-shape adaptive vocabulary

Dynamic vocabulary must not recreate QFP22's runtime-shape/scheduler-reserve problem. Candidate counts use a small finite bucket set (initially 8K/16K/32K/48K/64K). Graph topology and LM-head output shape remain constant within a bucket; invalid/padded slots receive `-inf`. Bucket transitions are explicit graph variants reserved before execution, or remain outside graph capture until proven safe.

Maintain `selected_token_ids[bucket]` and an inverse token-to-slot map. The draft sampler operates on compact logits. Before target verification, proposed slot ids are mapped back to canonical vocabulary ids; the target continues to verify against the full vocabulary. No target distribution is modified.

The adaptive set should be a union rather than a replacement for the static hot set. Measure marginal coverage from each source (`static`, prompt tokens, recent generated tokens, recent draft top-k/history). Only keep a source if its incremental accepted-token coverage justifies its row/gather cost.

### Decision model

For policy P and bucket B measure:

`benefit = target_verify_steps_saved_by_acceptance - draft_head_ms - selection_ms - gather_ms - remap_ms`

The primary metric is effective accepted target tokens / wall-second. Acceptance alone is insufficient; LM-head microseconds alone are insufficient.

Early-stop adaptive work if frequency-ranked 48K/64K already reduces LM-head share below 10% of total speculative step time or adaptive selection+gather consumes >50% of the LM-head time it removes.

## Code Samples & Guidance

Prefer one vocabulary-policy interface feeding 1297 rather than separate trim implementations. Conceptually:

```cpp
struct draft_vocab_view {
    ggml_tensor * compact_head;
    ggml_tensor * canonical_ids;
    uint32_t valid_rows;
    uint32_t bucket_rows;
};
```

The sampler sees compact logits plus `canonical_ids`; target verification always sees canonical ids. Keep the existing plain/full-head fallback for prompt replay and any invalid policy state.

Do not fork or duplicate generic TOP_K. Upstream llama.cpp PR #29883 demonstrates that multi-row TOP_K can be dominated by repeated per-row launches and host synchronization, but QFP05's decode path should first prove TOP_K is material after LM-head trimming before importing that mechanism.

## Files

Exact surfaces: `engines/llamacpp/patches/1297_draft_vocab_trim/patch.py` is the runtime control; pinned `src/models/qwen4exp.cpp::load_arch_tensors` and `graph_mtp` are the proposed GGUF d2t consumer; `src/models/eagle3.cpp` is the canonical scatter reference. Use existing `gguf-py` for an offline sidecar producer and lab tooling for coverage/head timing. Do not alter target model logits or generic CUDA/HIP TOP_K.

## Validation

ABBA at ~10K/~80K plus one long-context lane; >=3 independent requests per arm. Record draft head ms, total draft ms, target ms, accepted/drafted tokens, effective TG, candidate-set coverage, selection/gather/remap overhead and VRAM. Greedy target output must remain identical for deterministic lanes. Validate bucket tails, prompt replay, cache reuse, request reset, and fallback to full vocabulary.

## Effort & Risk

Medium. Static frequency ranking is low risk. Dynamic vocabulary is higher risk because sparse row access/gather can erase GEMV savings and runtime-dependent graph shapes can invalidate scheduler reservation. Keep it behind the same draft-vocab capability boundary and fixed bucket shapes.

## Standards

One canonical vocabulary-policy interface; no duplicated sampler or TOP_K implementation; fail closed to full draft head; benchmark artifacts identify corpus/hash, policy, bucket and exact row ordering.

## Acceptance Criteria

Keep 1297 as baseline. Promote a follow-up only if it improves effective TG by >=3% versus the best static 64K control at both 10K and 80K, with no >2% regression on the long-context lane and no target-output change. For adaptive vocabulary, require >=95% of the static-64K accepted-token coverage or enough lower draft latency to produce the same-or-better accepted-tokens/s. Reject any path that adds an unconditional host synchronization or unreserved runtime graph topology.

## Notes

Existing evidence: 1297 gives ~-8% ms/step at 10K and ~-7% at 80K (`flashnext-trim-ab-2`). Draft is now ~6.5 ms per 3-token draft at 10K and ~9.2 ms at 80K (15-17% of step).

External mechanism evidence: FR-Spec reports 75% LM-head computation reduction and ~1.12x average speedup using a frequency-ranked compressed draft vocabulary. VocabTrim independently uses frequently sampled target tokens for a static reduced head. 2026 work goes further: NanoSpec/MicroSpec reports context-sensitive active vocabularies below ~3K average and 51.6% average draft-latency reduction / 1.12-1.32x end-to-end versus EAGLE-2; SpecVocab reports up to 8.1% average throughput improvement over EAGLE-3 by selecting a vocabulary subset per decoding step. These are mechanism references, not expected AMD gains; BigCherry must qualify gather locality and GPU-resident selection on gfx1030.

Upstream llama.cpp research issue #25187 already tracks FR-Spec-style native-MTP vocabulary trimming. Reuse/contribute to that seam where possible rather than creating a second generic llama.cpp vocabulary-pruning API.

Upstream PR #29883 is also relevant as a warning: its multi-row TOP_K change batches segmented sorting and removes repeated host synchronization, reporting very large prefill gains on H20 but unchanged decode. Therefore QFP05 must profile the trimmed draft decode before spending effort on TOP_K; the LM head remains the proven bottleneck here.

Related: FMTP01/FMTP03 (draft/verify scheduling), QFP08 (draft-prefill overlap only), QFP13 (generic launch reduction), QFP15 (acceptance nondeterminism/benchmark hygiene), upstream llama.cpp #25187 and #29883.

## Change Log

- 2026-10-03T15:20:39.453089+00:00 (created-by): Created by agent
- 2026-10-05: Deep optimisation review: replaced prefix-only follow-ups with staged static-frequency -> fixed-bucket context-adaptive vocabulary -> requantization plan; consolidated TOP_K and MTP scheduling ownership; added promotion/early-stop gates and external 2025-2026 mechanism evidence.

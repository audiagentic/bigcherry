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

## Description

Patch 1297_draft_vocab_trim (evaluated, production profile, opt-in `BIGCHERRY_DRAFT_VOCAB_N=65536`) trims the MTP draft `output.weight` (Q8_0, 675 MB, 248K vocab), which was ~31% of 6900-XT draft time at 80K. The current implementation keeps the first N rows and scatters `-inf` for the rest. Evidence is already strong enough to keep 1297 as the baseline: ~8% lower ms/step at 10K and ~7% at 80K.

The next optimisation is **not** another arbitrary smaller prefix. Replace prefix membership with a measured vocabulary policy and separate three mechanisms:

1. **static frequency-ranked subset** (FR-Spec/VocabTrim control);
2. **context-adaptive active vocabulary** (NanoSpec/MicroSpec-style temporal locality control);
3. **smaller/requantized LM-head rows** (storage/bandwidth control).

QFP05 owns draft-vocabulary selection, row remapping, draft-head service time and acceptance trade-offs. It does not own generic `TOP_K` kernels, MTP scheduling/overlap (FMTP), or target verification. Any generic multi-row TOP_K optimisation belongs upstream/QFP17/QFP13 as appropriate.

## Steps

1. Instrument real production-like traffic and record, per drafted position: target token id, draft top-k ids/probabilities, whether the accepted target token is inside candidate subsets, draft LM-head kernel time, gather/remap time and total draft service time.
2. Establish equal-size controls at 48K/64K: first-N versus frequency-ranked. Frequency order must be generated from a frozen corpus/traffic artifact, versioned by hash; never silently learn from the benchmark requests.
3. If static 48K/64K frequency ranking still spends >=20% of draft service time in the LM head, prototype a context-adaptive **union vocabulary**: `V = V_static_hot ∪ V_recent ∪ V_prompt ∪ V_draft_top/history`, capped to a fixed bucket size. Start with 8K/16K/32K buckets, not per-step arbitrary tensor shapes.
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

Expected owner surfaces: patch 1297 draft-vocab code; speculative sampler/common MTP glue needed for canonical-id remap; lab tooling for vocabulary coverage/head timing. Avoid changes to target model logits or generic CUDA/HIP TOP_K unless separately owned.

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

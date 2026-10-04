---
id: FMTP06
order: 0
plan: patching-flash-next-mtp-pipeline
state: pending
created-at: '2026-10-04T00:48:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: L
---

# Extend ahead reuse to probabilistic rejection and multi-slot service

## Description

After greedy single-slot qualification, preserve exact speculative proposal semantics for temperature > 0 and isolate ahead state across multiple server sequences. This is not required for the first Flash-Next performance proof.

## Steps

1. For probabilistic MTP, return one proposal distribution `q` for every ahead token, including bridge.
2. Preserve token/q alignment through promotion: consume bridge token and bridge q together; attach only tail q rows to promoted `spec_draft_q`.
3. Preserve draft sampler/RNG state exactly. Ahead execution must not perturb the sampler state required by a later fresh authoritative draft after a flush.
4. Prefer cloning/copying sampler state for ahead execution when supported; otherwise add an explicit snapshot/restore primitive. Do not reconstruct RNG state from token history.
5. Keep bridge equality as the prefix-validity gate even in probabilistic mode. Proposal q rows are additionally required for correct rejection sampling of the tail.
6. Multi-slot: make epoch, frontier, seed, result and controller state per sequence.
7. A single worker may batch continuation jobs only if current `common_speculative_draft` batching semantics can be preserved. Otherwise serialize bounded jobs rather than introduce a second draft context prematurely.
8. Bound storage to at most one pending and one completed result per active sequence; no unbounded queue.
9. Sequence A reset/rejection must not invalidate sequence B.
10. Do not copy speculative ahead state into shared-prompt child slots or prompt-cache state unless a separate proof shows the frontier and proposal state are identical.

## Detailed Solution & Technical Design

Probabilistic result shape:

```cpp
struct common_mtp_ahead_result {
    uint64_t epoch = 0;
    llama_seq_id seq_id = -1;
    llama_pos base_pos = -1;
    llama_tokens tokens; // bridge + tail
    std::vector<std::vector<llama_token_data>> q; // same length
    bool ok = false;
};
```

Invariant:

```cpp
GGML_ASSERT(result.q.empty() || result.q.size() == result.tokens.size());
```

Promotion:

```cpp
if (bridge_valid) {
    promoted.tokens.assign(result.tokens.begin() + 1, result.tokens.end());
    promoted.q.assign(result.q.begin() + 1, result.q.end());
}
```

Every `promoted.tokens[i]` must retain the proposal distribution generated for that exact token position. Rejection sampling must never pair a token with a shifted q row.

### Sampler isolation

The current MTP implementation uses per-sequence draft samplers. The ahead chain is speculative relative to the speculative decoder itself: if it is flushed, its sampler advancement must disappear too.

Preferred shape:

```cpp
common_sampler_ptr ahead_sampler(common_sampler_clone(smpls[seq_id].get()));
// resume uses ahead_sampler only
```

If clone does not preserve the necessary RNG/candidate state, introduce/test a sampler copy-state operation before enabling probabilistic ahead.

### Multi-slot epochs

```cpp
struct seq_ahead_state {
    uint64_t epoch = 0;
    std::optional<job> pending;
    std::optional<result> complete;
    controller ctrl;
};
std::vector<seq_ahead_state> ahead;
```

Never use one global generation counter for all slots.

## Code Samples & Guidance

Randomized multi-slot state-machine tests should intentionally complete jobs out of order:

```text
seq0 job e7 starts
seq1 job e2 starts
seq0 resets -> e8
seq1 completes e2 -> may publish
seq0 old e7 completes -> must be rejected
```

## Files

- `common/speculative.cpp/.h`
- sampling implementation only if exact clone/copy support is missing
- `tools/server/server-context.cpp`
- multi-sequence concurrency/state tests
- experiment contracts for `--parallel > 1`

## Validation

Probabilistic:
- token/q length and position assertions;
- deterministic seeded replay where baseline is deterministic;
- statistical distribution comparison against ahead-disabled speculative rejection;
- flush restores sampler state exactly.

Multi-slot:
- randomized 2..N sequence interleavings;
- reset/reject/stop one sequence while another completes;
- stale completions cannot cross slot reuse;
- TSAN fake-worker stress.

Hardware multi-slot work starts only after FMTP07 greedy single-slot proof.

## Effort & Risk

L / high. Sampler-state correctness and cross-sequence lifecycle are significantly harder than greedy single-slot overlap.

## Acceptance Criteria

- Proposal distributions stay aligned with their tokens after bridge removal.
- Flushed ahead work leaves no sampler/RNG side effect.
- Sequence-local invalidation cannot affect another sequence.
- Storage/worker queues remain bounded.
- Distribution correctness matches ahead-disabled control within the existing speculative-rejection test methodology.

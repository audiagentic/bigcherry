---
id: FMTP02
order: 0
plan: patching-flash-next-mtp-pipeline
state: pending
created-at: '2026-10-04T00:48:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: L
---

# Add live-continuation and forced-front MTP primitives

## Description

Refactor c061 single-head non-shared MTP into one per-step primitive that supports both:

1. continuing immediately beyond a freshly drafted front; and
2. replaying a known promoted front from the authoritative target seed, then continuing beyond it.

No scheduling change in this item.

## Key review finding

A copied `(id, h_row, pos)` is not a self-contained durable seed. Continuation also depends on the live `ctx_dft` KV frontier and draft sampler state. Treat a fresh-front continuation record as an **ephemeral lease**: any rollback, process, reset, state restore, other draft mutation or sampler reset invalidates it.

Promoted tails cannot use that old lease on the next round because authoritative rollback/reseed intentionally destroys the speculative branch. They need forced-front replay.

## Steps

1. Extract the single-head MTP transition into one helper that decodes an input pair, obtains output hidden state/candidates and optionally samples/accepts a token.
2. Fresh-front path: at the normal effective depth stop, capture only enough metadata to resume immediately from the live frontier: seq, epoch, next pos, last front token, copied `h_row`, frontier generation.
3. Add `continue_live_front(...)` returning `bridge + tail`; it is callable only before any `ctx_dft`/sampler mutation.
4. Add `replay_known_front_and_continue(...)` for promoted fronts:
   - start from authoritative `(id_last, pending_h)`;
   - run MTP sequentially through the known promoted tokens, forcing those tokens rather than using sampled proposals;
   - update sampler state consistently with forced tokens where relevant;
   - after the last known token, sample bridge and future tail normally.
5. Keep all ahead tokens disjoint from the current `dp.result`/`dp.result_q`.
6. p-min applies only to newly sampled bridge/tail, not to forced promoted-front tokens that are already being target-verified.
7. Decode failure invalidates the live lease/result but does not corrupt the ordinary target path.
8. Clear/increment a draft-frontier generation on begin/reset/process/rollback/state restore and any operation that makes a live lease invalid.

## Detailed Solution & Technical Design

At c061 the depth stop occurs after the sampled `id` and its `h_row` exist, but before `(id, h_row)` is added as the next MTP input. That is the fresh-front continuation seam.

For a promoted front, do not pretend the prior speculative KV still exists. Reconstruct it:

```text
authoritative seed: (Z, target_h_prev)
force B1 -> obtain draft h(Z)
force B2 -> obtain draft h(B1)
...
force BM -> obtain draft h(BM-1)
decode BM -> sample bridge, then continue normally
```

Exact indexing should follow the existing MTP loop rather than this shorthand; the invariant is that every forced token is fed with the hidden row produced by the preceding MTP input.

Suggested internal API shape:

```cpp
struct mtp_live_lease {
    uint64_t epoch = 0;
    uint64_t frontier_gen = 0;
    llama_seq_id seq_id = -1;
    llama_pos next_pos = -1;
    llama_token input_id = LLAMA_TOKEN_NULL;
    std::vector<float> input_h;
    bool valid = false;
};

bool continue_live_front(const mtp_live_lease &, int32_t n_tail, mtp_ahead_result &);
bool replay_known_front_and_continue(
    llama_seq_id seq_id,
    llama_token id_last,
    llama_pos pos0,
    const std::vector<float> & pending_h,
    const llama_tokens & known_front,
    int32_t n_tail,
    mtp_ahead_result & out);
```

The implementation should share one per-step decode helper; do not fork a second MTP loop.

### Semantics of hidden-state mismatch

A continued/promoted tail may differ from what a fresh next-round target-reseeded MTP would have proposed. That is not a greedy correctness failure: proposals remain proposals and the target verifies them. The strengthened mock explicitly exercises this case.

## Files

- `common/speculative.h/.cpp`
- patch package + focused mechanics tests

## Validation

1. Fresh live continuation: uninterrupted `N+1+M` chain equals `front(N) + continue_live_front(1+M)` from identical state.
2. Lease invalidation: any rollback/process/reset/sampler reset makes the old lease unusable.
3. Forced replay: starting from authoritative seed, replay known front tokens and verify the resulting live position/hidden progression is deterministic and can generate a bridge+tail.
4. Force tokens that differ from what MTP would greedily sample; replay must still advance state without committing them to target output.
5. N=1..max, promoted front lengths 1..max, p-min early stop, decode failure.

## Acceptance Criteria

- Ahead disabled leaves front draft unchanged.
- One shared per-step implementation drives fresh draft, continuation and forced replay.
- No context-owned hidden pointer survives a decode.
- Live lease cannot outlive its exact draft KV/sampler frontier.
- Promoted-front replay works after authoritative rollback/reseed; no second draft-context snapshot is required.

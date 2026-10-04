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

# Refactor native MTP into a resumable continuation primitive

## Description

Prove, synchronously, that c061's single-head non-shared MTP loop can stop at the normal verification depth and later resume the same speculative branch to produce `bridge + tail`. No threads in this item.

The normal front draft returned to the server must remain byte-for-byte/ID-for-ID equivalent to control.

## Steps

1. Extract the single-head per-step body from `common_speculative_impl_draft_mtp::draft()` into an internal helper that can execute a bounded number of additional steps without changing the public draft result.
2. At the normal depth stop, capture a continuation seed only for eligible mode. Copy, never alias:
   - `seq_id` and epoch;
   - next input token (the last front token);
   - next position;
   - the `h_row` produced for that token;
   - draft limit/p-min status.
3. Do not capture a seed if drafting stopped because of p-min, decode failure, `n_min` filtering, unsupported mode, or empty front.
4. Add synchronous `resume_ahead(seed, n_tail)` returning at most `1 + n_tail` tokens. Element 0 is the bridge; elements 1.. are future tail.
5. Keep bridge/tail storage disjoint from `dp.result` and `dp.result_q`; current target verification must see exactly the original front.
6. `resume_ahead()` must use the same decode path, p-min check and greedy sampler semantics as the front loop.
7. Clear/invalidate continuation on `begin()`, context/state restore, decode failure and destructor.
8. Add a mechanics test comparing a single uninterrupted MTP chain with `front(N) + resume(1+M)` from identical initial state.

## Detailed Solution & Technical Design

At c061, the relevant sequence is:

```cpp
int ret = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
...
common_sampler_sample(smpl, ctx_dft, i_last[seq_id], true);
const float * h_row = llama_get_embeddings_nextn_ith(ctx_dft, i_last[seq_id]);
...
const llama_token id = cur_p->data[0].id;
common_sampler_accept(smpl, id, true);
result.push_back(id);

if (params.n_max <= (int) result.size()) {
    // current code stops here
}
...
const int32_t idx = batch.add(id, dp.pos0 + i + 1, seq_id, true);
batch.set_embd(idx, { h_row, 1, (size_t) n_embd });
```

The depth stop happens after `id` and `h_row` exist but before they are enqueued as the next input. That is the exact continuation seam. Save a copy of that omitted next-input pair.

Recommended records:

```cpp
struct common_mtp_ahead_seed {
    uint64_t epoch = 0;
    llama_seq_id seq_id = -1;
    llama_pos next_pos = -1;
    llama_token input_id = LLAMA_TOKEN_NULL;
    std::vector<float> input_h;
    bool valid = false;
};

struct common_mtp_ahead_result {
    uint64_t epoch = 0;
    llama_seq_id seq_id = -1;
    llama_pos base_pos = -1;
    llama_tokens tokens; // bridge, then future tail
    bool complete = false;
};
```

Do not store `h_row` as a pointer: it refers to context-owned output storage that can be overwritten by the next `llama_process()`.

Refactor shape:

```cpp
bool draft_mtp_step(
        llama_seq_id seq,
        llama_token input_id,
        llama_pos pos,
        const float * input_h,
        mtp_step_result & out);
```

The helper should own exactly one decode/sample transition. The existing loop then becomes repeated calls, minimizing divergence between front and ahead paths.

### State/KV rule

The synchronous resume deliberately mutates the same speculative branch in `ctx_dft`; after the result is copied out, the existing server rollback/replay path still restores/truncates the draft context before authoritative verification processing. FMTP03 will move the timing of that rollback, not remove it.

### Interaction with 1268

Use the already-computed effective front depth from 1268. Do not have the continuation helper reinterpret adaptive depth. `n_tail` is independent future work and bounded by FMTP ahead policy.

### Interaction with 1308

1308 changes Qwen4Exp target recurrent rollback snapshot graph construction, not this MTP decoder loop. Mechanics tests must cover both env arms later, but FMTP02 should not depend on 1308 being enabled.

## Code Samples & Guidance

Capture only on a depth stop, not p-min:

```cpp
const bool hit_front_limit = effective_n_max <= (int) result.size();
if (hit_front_limit && mtp_ahead_eligible) {
    auto & s = ahead_seed[seq_id];
    s.epoch = epoch[seq_id];
    s.seq_id = seq_id;
    s.next_pos = dp.pos0 + i + 1;
    s.input_id = id;
    s.input_h.assign(h_row, h_row + n_embd);
    s.valid = true;
}
```

## Files

- `common/speculative.h`
- `common/speculative.cpp`
- new patch package + focused mechanics tests

## Validation

CPU/pure bookkeeping tests:
- seed copied, not pointer-retained;
- seed only captured on eligible depth stop;
- p-min/failure/filter/reset invalidate it;
- result remains separate from front.

Real MTP mechanics test:
- start from identical draft context/sampler state;
- arm A drafts `N+1+M` uninterrupted;
- arm B drafts front `N`, captures seed, resumes `1+M`;
- compare token IDs for continuation and, where practical, resulting draft memory/position frontier;
- test N=1..configured max and M=1..configured max.

## Effort & Risk

L / medium. The core arithmetic is simple; risk is subtle draft KV/sampler state drift from refactoring the loop.

## Acceptance Criteria

- Ahead disabled: front draft is unchanged.
- Split front+resume token sequence equals uninterrupted native MTP for the same starting state.
- No hidden-state pointer survives a decode.
- Unsupported modes cannot create a valid seed.
- Existing rollback/replay restores authoritative draft state after the synchronous experiment.

---
id: FMTP04
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

# Promote prefetched tails only across a verified bridge

## Description

Connect the asynchronously computed ahead result to the next speculative round without allowing any ahead token to bypass target verification. A future tail is reusable only when target verification fully accepts the current front and the MTP bridge equals the target's authoritative extra token.

## Steps

1. Add per-slot ahead metadata separate from existing `spec_draft`, `spec_draft_q`, `spec_i_batch`, replay state and speculative checkpoint.
2. Store at minimum: epoch, base position, front length, bridge token, tail tokens, completion/failure state and timing/accounting.
3. In target acceptance handling compute `front_full = accepted_front == front_len` using the existing verification result, taking replay semantics into account.
4. Identify the authoritative target-extra token `Z` from the accepted result on the ordinary full-accept path.
5. Promote only when all are true:
   - front fully accepted;
   - ahead result succeeded;
   - ahead epoch equals slot epoch;
   - ahead base/frontier equals the current slot frontier;
   - bridge exists and `bridge == Z`;
   - tail is non-empty and meets existing minimum-draft policy.
6. Consume bridge. It must never be inserted into the next `spec_draft`; the target already authored/emitted `Z`.
7. Store only the remaining tail as a **future proposal**. At the next `pre_decode()`, install it into the ordinary verification draft and skip fresh MTP front generation for that slot.
8. Create/update the normal speculative checkpoint at the authoritative frontier before verifying a promoted tail, preserving partial-accept replay semantics.
9. Flush and increment epoch on partial acceptance/rejection, bridge mismatch, replay restore, prompt/state restore, context shift, stop/EOS, decode error, request reset/release, sequence reuse, base-position mismatch or worker failure.
10. A late result carrying an old epoch must be ignored without touching slot or context state.

## Detailed Solution & Technical Design

The bridge is mandatory because target verification supplies one authoritative token beyond the draft:

```text
current MTP front:      A1 A2 ... AN
MTP continuation:       B0 B1 B2 ... BM
target full verify:     A1 A2 ... AN Z
```

`B1..BM` were generated conditioned on `B0`; they are reusable only if `B0 == Z`.

Suggested slot state:

```cpp
struct server_mtp_ahead {
    uint64_t epoch = 0;
    llama_pos base_pos = -1;
    int32_t front_len = 0;
    llama_token bridge = LLAMA_TOKEN_NULL;
    llama_tokens tail;
    bool ready = false;

    void clear() {
        base_pos = -1;
        front_len = 0;
        bridge = LLAMA_TOKEN_NULL;
        tail.clear();
        ready = false;
    }
};

uint64_t spec_epoch = 0;
server_mtp_ahead spec_ahead;
```

Promotion predicate:

```cpp
const bool promote =
    !slot.spec_is_replay &&
    n_accepted == n_draft &&
    slot.spec_ahead.ready &&
    slot.spec_ahead.epoch == slot.spec_epoch &&
    slot.spec_ahead.base_pos == expected_base_pos &&
    slot.spec_ahead.front_len == (int32_t) n_draft &&
    slot.spec_ahead.bridge == ids.back() &&
    slot.spec_ahead.tail.size() >= (size_t) effective_n_min;
```

Use a named helper for invalidation:

```cpp
void invalidate_mtp_ahead(server_slot & slot) {
    ++slot.spec_epoch;
    slot.spec_ahead.clear();
}
```

Do not increment epoch for an ordinary successful promotion until the promoted tail has been transferred to its new round metadata; otherwise the new proposal becomes stale immediately. Make the epoch transition explicit in one helper rather than open-coding increments.

### Existing replay interaction

Current server speculative verification can restore a checkpoint on partial acceptance where sequence removal cannot represent the rollback cheaply. That restore is authoritative. Any ahead branch based on the pre-restore speculative frontier is invalid and must be flushed before returning through replay.

### Qwen4Exp recurrent state

Target recurrent rollback snapshots (including 1308's direct-copy variant) are target-side verification/replay state. The ahead tail never replaces these snapshots. Both 1308 off/on must therefore yield the same promotion decisions and target token IDs.

## Code Samples & Guidance

Install promoted tail only at the next normal draft-building boundary:

```cpp
if (slot.has_valid_promoted_tail()) {
    slot.spec_draft = std::move(slot.spec_ahead.tail);
    slot.spec_ahead.clear();
    // normal target verification/checkpoint logic continues
} else {
    common_speculative_get_draft_params(spec.get(), slot.id).drafting = true;
    drafting.push_back(&slot);
}
```

Never append a promoted tail to the batch that just verified its parent front.

## Files

- `tools/server/server-context.cpp`
- `common/speculative.h/.cpp` result collection API
- patch package + server-state mechanics tests
- `mock_pipeline.py` as high-level oracle

## Validation

Exhaustive deterministic tests for front depth 1..8:
- reject at every position;
- full accept + bridge match;
- full accept + bridge mismatch;
- empty/short tail;
- stale epoch completion;
- base-position mismatch;
- checkpoint replay/restore;
- stop/EOS in front or as target-extra token;
- context shift/reset/slot reuse.

For all cases, emitted greedy target IDs must equal ahead-disabled control.

## Effort & Risk

L / high. A single incorrect frontier or bridge condition can cause semantic divergence; fail closed everywhere.

## Acceptance Criteria

- No ahead token is emitted without ordinary target verification.
- Reuse requires full-front acceptance plus bridge equality.
- Bridge is consumed, never duplicated as next draft token.
- All invalidation boundaries fence stale work.
- Existing checkpoint/replay remains authoritative.
- 1308 off/on produces identical token semantics and promotion predicates.

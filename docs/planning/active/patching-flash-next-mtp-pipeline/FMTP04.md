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

# Promote tails only across a verified bridge and replay them under overlap

## Description

Reuse ahead tokens without allowing any token to bypass target verification. Promotion requires full-front acceptance plus bridge equality. On the next round, a promoted front skips **serial fresh drafting**, but it must be replayed through MTP during target verification to reconstruct a live draft frontier for the next ahead continuation.

## Steps

1. Keep per-slot ahead metadata separate from `spec_draft`, `spec_draft_q`, replay state and checkpoints.
2. Store parent epoch, absolute base position, front length, bridge, tail, completion/failure and timing.
3. In target acceptance handling determine `front_full` from the existing speculative result, excluding replay paths.
4. Identify target-authoritative extra token `Z` on ordinary full acceptance.
5. Promote only when: full front, result success, parent epoch match, exact base/frontier match, bridge exists and equals `Z`, and tail length meets existing `params.n_min` policy.
6. Consume bridge; only tail tokens become the next verification draft.
7. Advance to a new child epoch when promotion is committed; retag the promoted front to that child round. Every round transition gets a new epoch so an old result cannot match a later round accidentally.
8. At next `pre_decode()`, install promoted tail into ordinary target verification batch and skip serial `common_speculative_draft()` for that slot.
9. After target verification is asynchronously submitted, FMTP03 must call FMTP02 forced-front replay on the promoted tokens, then continue to a new bridge+tail before target sync.
10. After target sync, run existing rollback/truncate, `common_speculative_process()` and target accept/reject unchanged.
11. Flush on partial acceptance/rejection, bridge mismatch, replay restore, prompt/state restore, context shift, stop/EOS, decode error, request reset/release, sequence reuse, base mismatch or ahead failure.

## Detailed Solution & Technical Design

Promotion rule:

```text
current front:        A1 ... AN
MTP ahead:            B0 B1 ... BM
target full verify:   A1 ... AN Z
```

Only if `B0 == Z` may `B1..BM` become the next target-verified front.

Suggested metadata:

```cpp
struct server_mtp_ahead {
    uint64_t parent_epoch = 0;
    llama_pos base_pos = -1;
    int32_t front_len = 0;
    llama_token bridge = LLAMA_TOKEN_NULL;
    llama_tokens tail;
    bool ready = false;
};

uint64_t spec_epoch = 0;
```

Fail-closed predicate:

```cpp
const bool promote =
    !slot.spec_is_replay &&
    n_accepted == n_draft &&
    slot.spec_ahead.ready &&
    slot.spec_ahead.parent_epoch == slot.spec_epoch &&
    slot.spec_ahead.base_pos == expected_base_pos &&
    slot.spec_ahead.front_len == (int32_t) n_draft &&
    slot.spec_ahead.bridge == ids.back() &&
    slot.spec_ahead.tail.size() >= (size_t) params.speculative.draft.n_min;
```

On success, increment epoch once and move the tail into child-round state. On any failure/invalidation, increment epoch and clear it. Centralize this transition; do not open-code epoch changes across reset paths.

### Critical review correction: tokens are not enough for steady-state continuation

The previous ahead branch's `ctx_dft` state is discarded by authoritative rollback/reseed after each target verification. Therefore a promoted tail can be used immediately as a **target proposal**, but there is no valid live MTP continuation seed for that child round.

Do not preserve the speculative KV by skipping rollback: authoritative replay/reseed is a correctness invariant. Instead, FMTP03 replays the promoted tokens from authoritative MTP seed while the target verifies them. That reconstructs the live MTP frontier and keeps the pipeline continuous without a second context snapshot.

### Hidden trajectory need not match fresh MTP

A promoted tail may have been generated from recursively propagated draft hidden state and differ from what fresh target-seeded MTP would have sampled next round. Greedy correctness is still preserved because the target verifies every proposal. Forced replay uses the authoritative seed plus the promoted token values to reconstruct a valid live state for further proposals.

## Code Samples & Guidance



## Files

- `tools/server/server-context.cpp`
- `common/speculative.h/.cpp`
- patch package + state tests
- `mock_pipeline.py`

## Validation

Exhaustive/state tests:
- reject at every front position;
- full accept + bridge match/mismatch;
- stale parent epoch;
- base/front-length mismatch;
- empty/short tail;
- replay restore, stop/EOS, context shift, reset/reuse;
- successful promotion receives a new child epoch;
- promoted child invokes forced-front replay rather than stale live continuation.

Greedy target IDs must match ahead-disabled control in every case.

## Effort & Risk



## Standards



## Acceptance Criteria

- No ahead token is emitted without target verification.
- Full-front + bridge equality + exact parent metadata are mandatory.
- Bridge is consumed, never duplicated.
- Every child round is epoch-fenced.
- Promoted fronts skip serial fresh drafting but are replayed through MTP under target verification before further ahead generation.
- Existing checkpoint/replay remains authoritative.
- 1308 off/on does not change token semantics or promotion predicates.

## Notes

Prep 2026-10-04 - code anchors (vendor/llama.cpp/tools/server/server-context.cpp): server_slot fields spec_draft (~L213) / spec_i_batch (~L215), cleared ~L342; verify batch built from spec_draft ~L477-511; pre_decode() ~L2907, drafting set + common_speculative_draft ~L3044 (skip for slots with a promoted front); accept path ~L3910-3966: common_sampler_sample_and_accept_n -> n_rollback -> common_speculative_accept(spec, slot.id, accepted.size()-1) -> slot.spec_draft = accepted - promotion predicate (full front + bridge == ids.back()) goes right after the accept. Patch number: 1323_mtp_ahead_promotion. FMTP05 controller: 1324_mtp_ahead_controller.

2026-10-05 reconciliation with what landed: promotion is implemented inside patch 1322 (not a separate 1323). 1322 promotes only when the whole front was accepted AND the target sampled the tail's first token (full front + bridge equality), consumes the bridge, promotes only a full-length tail, skips the serial fresh draft for the promoted round and replays the promoted front as a forced front (1321 dp.forced) during the next target verify. Every promoted token is still target-verified (greedy identical in the 24K / 80K screen, 39% / 25% of rounds promoted). Not yet checked against this item's acceptance: explicit parent metadata / epoch fencing of child rounds (single-slot greedy only so far), and the offline mechanics tests listed here. Open performance problem is owned by FMTP03 (promoted fronts are accepted less than fresh ones, so t/s is neutral despite -7% ms/step). The planned patch number 1323 is free; 1324 remains reserved for the FMTP05 controller, which depends on 1268 - and 1268 currently fails to apply over the build selection (common/speculative.cpp anchors, found by the 2026-10-05 pin-bump), so it must be repaired before FMTP05 starts.

## Change Log

- 2026-10-04T06:32:31.159768+00:00 (updated-by): Updated: section:notes
- 2026-10-05T10:04:13.100614+00:00 (updated-by): Updated: section:notes

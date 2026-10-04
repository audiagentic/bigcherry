# 1321_mtp_ahead_primitives

**Status:** untested
**Plan item:** FMTP02

## What it does

Adds the two MTP drafting primitives the FMTP ahead pipeline needs, inside the existing single-head MTP draft loop
(no second loop, no scheduling change):

- **Forced promoted front** (`dp.forced`): known tokens are fed through the MTP head from the authoritative seed
  `(id_last, pending_h @ pos0)` in order, each paired with the hidden row of the previous step, before sampling
  resumes. p_min does not apply to them. Greedy drafting only; at most `n_max` tokens.
- **Live tail** (`dp.n_tail`, `dp.tail`): drafting continues past the front (`n_max`) for up to `n_tail` tokens into
  `dp.tail`, disjoint from `dp.result` (the verify set is unchanged). The tail is produced in the same `draft()` call
  from the live draft KV/sampler frontier, so no continuation lease can outlive a rollback/reseed. If the front is
  dropped by `n_min`, the tail is dropped too.

Defaults (`forced` null/empty, `n_tail` 0) keep today's draft exactly. Chained-head and shared-KV MTP assert the new
fields are unused. The server does not set them yet; FMTP03 schedules the ahead draft.

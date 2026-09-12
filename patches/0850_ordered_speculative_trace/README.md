# 0850: ordered per-verify-step speculative-decode acceptance trace

## Scope

Records one `(draft_n, accepted_n)` pair per speculative-decode verify step
(not just the aggregate totals `llama-server` already exposed) and
serializes it as `draft_trace` in the `/completion` response's `timings`
object. Full design rationale and implementation notes are in
`SUMMARY.md` -- do not duplicate here.

## Why

BigCherry's pre-promotion behavioral gate compared native vs. candidate
runs using only aggregate `(draft_n, draft_n_accepted)` scalars, which
two runs with genuinely different per-step work schedules can sum to
identically -- already produced one real false signal (HI141): a promoted
cache's acceptance silently shifted 0.94734 -> 0.86391, looking like a
throughput regression but actually a work-equivalence violation the old
gate couldn't see.

## Real hardware evidence

Per HI166's own plan item (state `completed`): real Brutus MTP trace
verified ordered `draft_trace` emission and reconciliation; the HI141
hard-fail witness remains intact under the new gate; focused
behavioral/patch offline suites pass. Design reviewed by `dev-gpt-agent`
(`req_afbaf0f27c6d4511`) against the actual pinned source before
implementation -- caught a real defect in the first proposed recording
point (would have used a rollback-truncated draft width on a
checkpoint-replay's resumed iteration) before any code was written.

## Lifecycle: deliberately still `untested`

This is not an oversight. HI166's own notes are explicit: "patch 0850
remains experiment-only/untested because normal composition evidence
binding and dual-XTX native/control/replay qualification are not
complete. HI166's own ordered-trace implementation, gate semantics, and
hardware producer evidence are complete; no further work remains under
HI166." A separate plan item (PHA07) recorded this as the deliberate
lifecycle decision required before any promotion: "Future promotion
belongs to a dedicated qualification composition and must not weaken
provenance gates."

In other words: the mechanism works and is proven on real hardware, but
this project's own promotion process (composition evidence binding,
dual-XTX qualification) is a separate, not-yet-run gate -- correctly kept
separate from "does the instrumentation work."

## Known limitations

- No `validation.toml` adapter; production consumption is via
  `experiment.hi166-ordered-trace-only` in `config/recipes.toml`, not a
  normal recipe.
- Do not promote without running the dedicated qualification composition
  HI166/PHA07's notes describe -- this README existing is not itself that
  qualification.

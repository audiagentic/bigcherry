# 0850_ordered_speculative_trace: Ordered per-verify-step speculative-decode acceptance trace

**Status:** untested
**Group:** core
**Plan item:** HI166

## What it does

Records one (draft_n, accepted_n) pair per speculative-decode verify step (not just the aggregate totals llama-server already exposed) and serializes it as `draft_trace` in the `/completion` response's `timings` object, alongside the existing `draft_n`/`draft_n_accepted` scalars.

## Why

BigCherry's pre-promotion behavioral gate (`tools/bigcherry/tuning/behavioral_gate.py`) compared native vs. candidate runs using only aggregate `(draft_n, draft_n_accepted)` scalars. Two runs with different per-step work schedules can sum to the same totals -- e.g. accepting `[4,4]` then `[4,0]` across two verify steps vs. `[4,2]` then `[4,2]` both aggregate to `draft_n=8/accepted=4` and were indistinguishable to the gate, despite doing different work. This already produced one real false signal (HI141): a promoted cache's acceptance silently shifted 0.94734 -> 0.86391, which looked like a throughput regression but was a work-equivalence violation the gate couldn't see.

HI166 requires ordered-trace equality (not just aggregate-scalar equality) before any further MMQ/candidate-set expansion may proceed, precisely to prevent a wider search from manufacturing apparent winners by doing less/different speculative work.

## Design notes

- Trace lives on `server_slot_stats` (the per-task-result struct), the same place `n_draft_tokens`/`n_draft_accepted`/`n_draft_verif_steps` already live -- deliberately NOT on `server_slot` alongside `n_accepted_per_pos`, which that struct's own comment says is excluded from task results.
- Acceptance in this codebase's accept-and-verify design is always a prefix accept (both `common_sampler_sample_and_accept_n` and `server_sample_and_accept_synth` stop at the first rejected position), so a bare `(draft_n, accepted_n)` pair per step is sufficient -- no bitmap needed.
- Recorded immediately after `accepted` is computed, before the checkpoint-rollback early return, guarded by `!slot.spec_is_replay` -- the one point that captures each true logical verification decision exactly once, using the original draft width rather than a rollback-truncated replay width.
- The digest that makes gate equality cheap (`ordered_trace_digest`) is computed client-side (Python), not here -- the server cannot know in advance whether a comparison will match.

## Upstream / provenance

Local design (dev-gpt-agent review req_afbaf0f27c6d4511, verified against the actual current-pin source before implementation). Not based on an upstream PR/commit.

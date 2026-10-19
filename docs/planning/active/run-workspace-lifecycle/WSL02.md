---
id: WSL02
order: 2
plan: run-workspace-lifecycle
state: pending
created-at: '2026-10-19T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: M
---

# Harvest: extract what we keep, then mark the raw output expendable

## Description

A run is finished when what we need from it has been extracted to a durable, small record. Each run kind declares a HARVEST RECIPE (a template, in code and documented): which files form the evidence
(run.json, summaries, window reports, kernel-fraction summaries, stream-bench records, acceptance.json), which are copied into docs/evidence/<plan-item>/ (with the README and the claims they support),
and which are raw bulk (server and helper logs, rocprof traces, KL log-prob caches, dumps, captures). `bigcherry run harvest <run>` validates the evidence against its schema, copies it (or writes the
bundle to evidence-hold/), records content hashes and the destination in the manifest, and sets the run to harvested.

Rules: nothing is deleted by harvest; the raw bulk becomes eligible for cleanup only after harvest plus its retention period (WSL03); a run that failed, or is flagged suspect (for example measured while
the host was loaded), is harvested as a record of the failure, not as evidence. Summaries that are derived from bulk (kernel fraction from a rocprof trace, trace windows from helper logs) are produced by
harvest before the bulk can go, so the stats are never lost.

## Steps

1. Recipe schema, and the recipes for ab, accept, stream, replay, capture, kl, profile.
2. Move the summarising steps that are ad hoc today (kernel-fraction text, trace_report windows, KL result extraction) into the recipes.
3. `bigcherry run harvest` and `run status` (what is unharvested, what is waiting for retention).
4. Evidence README generation from the manifest (what, why, how to read), so the human-written part is only the interpretation.

## Validation

Harvest of the 1.8.0 acceptance, stream, small-lane and traced runs reproduces the evidence already in docs/evidence/RAD10 and RAD06; a run missing a required evidence file fails harvest with a clear message.

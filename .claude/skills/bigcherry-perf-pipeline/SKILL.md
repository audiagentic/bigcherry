---
name: bigcherry-perf-pipeline
description: Drive workload-targeted BigCherry performance work end to end (triage -> author -> mechanics -> queued build -> firing preflight -> balanced measurement -> lifecycle), including keeping reviewer-agent (GPT gateway) sessions continuously busy. Use when asked to find, develop, test, or ship patches that speed up a specific model/hardware configuration, or to run/maintain the GPT review loop.
---

# BigCherry Performance Pipeline

Purpose

Run the repeatable loop in docs/reference/patches/PATCH_DEVELOPMENT_PIPELINE.md for a named
workload (model + quant + architectures + split mode + decode/prefill + MTP settings), and
keep it moving. This skill orders the work and enforces the gates; the substantive work
belongs to the owning skills:

- authoring/repair -> bigcherry-patch-author
- hardware-free mechanics -> bigcherry-patch-verify
- comparative measurement -> bigcherry-benchmark
- contract evidence -> bigcherry-patch-qualification
- promotion/rejection -> bigcherry-patch-lifecycle

Triggers

- "find/develop/test the patches that help <model> on <cards>"
- "queue the remaining tests", "what's left for <workload>"
- "keep GPT busy", "work with gpt on these patches", reviewer-loop maintenance
- adding a new card/topology or quant sweep to an existing workload

Non-triggers

- a single-domain task already owned by one of the skills above
- pin bumps (bump-llamacpp)

Source of truth

- docs/reference/patches/PATCH_DEVELOPMENT_PIPELINE.md (stages, rules, reviewer loop)
- docs/reference/build/BUILD.md "Queued qualification campaigns and the firing pre-flight"
  (queue.sh row syntax: campaign, PROFILE, PREFLIGHT, BUILD, AB, VIS=, REQUIRES=)
- tools/lab/plan-qualification/queue.sh, locked-run.sh
- tools/lab/plan-qualification/ queue/preflight helpers; topic-specific active job scripts live under their owning `tools/lab/<topic>/` directory
- config/recipes.toml ([experiment.*] for single-patch builds, [patch-set.validated-enhancements]
  for production)

## Workflow

1. **Pin the workload.** Write down model file, quant, archs and device set, split mode,
   server args, and which phase matters (decode/prefill). Every later step uses exactly this.
2. **Triage.** For each candidate patch: does its gate admit this workload (arch, quant type,
   ncols/width, split mode)? What marker proves it fired? Rate TEST-NOW / TEST-LATER / SKIP.
   Ask the reviewer for breadth; verify gate claims against patch.py yourself.
3. **Author/repair** missing pieces via bigcherry-patch-author (marker, validation.toml,
   experiment entry). Mechanics via bigcherry-patch-verify. Commit and push.
4. **Queue, never ad hoc.** Write/extend a job script under tools/lab/<topic>/:
   `BUILD` each variant (arch list for multi-arch), `PREFLIGHT` on the pinned model/config,
   then `REQUIRES=<preflight> AB ...` or contract campaign rows. Pull on the bench host and
   start it detached; it waits on locks behind running work.
5. **Read results only when they finish.** Check activation per arm, work equivalence (MTP
   acceptance), per-position means, CI. Hand interpretation rules to bigcherry-benchmark.
6. **Decide via bigcherry-patch-lifecycle.** Winners: fresh contract sessions at the current
   pin, then promote into validated-enhancements with a comment stating what the evidence
   covers. Neutral: leave untested, record it. Regression: reject with evidence.
7. **Record.** Update the plan item and record an ag-ledger change event.

## Reviewer-agent loop (when asked to keep GPT working)

- Maintain two sessions, one per workstream; schedule a recurring check (~12 min, CronCreate)
  whose prompt names both session ids and each session's backlog.
- Each check: running -> leave; finished -> apply, test, commit+push, queue rows, submit the
  next backlog item; failed/interrupted -> resubmit.
- Before every request: commit and push (the reviewer only sees pushed files). For vendor
  anchors, materialize the configured pinned `vendor/llama.cpp` checkout and name the exact revision/path/function in the request; do not commit copied upstream source snapshots.
- Request complete applyable files with anchor match counts plus queue rows; never apply
  reviewer output without local lint/tests.

## Stop conditions

Stop and report instead of proceeding when:

- the workload is not pinned (step 1);
- a timed row would run without a passed PREFLIGHT for that patch on that config;
- arms differ in more than one variable, or activation/work-equivalence evidence is missing;
- a promotion would rely on evidence from an older pin or a changed patch.py;
- a job script would be edited or pulled on the bench host while it executes in place;
- a patch's gate cannot admit the workload (retarget or drop; do not "measure harder").

## Safety rules

- Never run hardware jobs outside queue.sh locks; never overlap a timed run.
- Scripts in tools/lab/<topic>/ only; models under /mnt/data/llm-models/<model>/gguf/<vendor>/.
- No git stash; stage explicit paths; commit trailer per repo instructions.
- Campaigns never mutate lifecycle; decisions are explicit (bigcherry-patch-lifecycle).

## Self-validation

Before reporting: workload pinned? every timed result has firing proof and equal work?
everything ran through the queue? reviewer sessions both busy (if the loop is active)?
plan item and ledger updated?

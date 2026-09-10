---
id: RHA01
order: 10
plan: run-hip-autotune
state: pending
created-at: '2026-09-09T10:48:34.167227+00:00'
breadth: ''
skill: intermediate
created-by: capability-rebaseline-v3
work: M
priority: P2
---

# profile-campaign CPU call-graph capability via perf (enhancement, deferred -- perf currently non-functional on Brutus)

## Description

Optional, activation-gated CPU call-graph profiling capability for profile-campaign; implementation remains deferred until a usable perf path and a concrete unresolved CPU-attribution question exist.

## Steps

1. Freeze the configured direct perf binary path and explicit sudo policy; use a capability preflight rather than inferring from perf_event_paranoid. 2. Implement perf.py with perf record -F 199 -e cpu-clock:u --call-graph dwarf,16384, perf script/report normalization, tool-version and exact-command capture, and two profile passes for call-stack reproducibility only. 3. Wire the CPU stage into workflow.py's existing interleaved-control sequence and keep perf mutually exclusive with rocprofv3. 4. Require a concrete unresolved CPU-attribution question selected at implementation time; do not default to retracted THA16 without re-justification and do not use PHC03's GPU crash localization as a synthetic target. 5. Treat failed sampling as failed diagnostic evidence, never as a silent control replacement; keep profiles diagnostic-only and separate from throughput comparisons. 6. Implement only after both perf capability and target question are confirmed on the real host.

## Detailed Solution & Technical Design

The direct Brutus binary /usr/lib/linux-tools-6.8.0-139/perf is a configurable path, not a committed server detail. The normalized artifact schema is symbol,dso,self_samples,inclusive_samples,self_pct,inclusive_pct,estimated_on_cpu_ms plus tool version and exact command; values are sampling estimates, not exact call counts. `--profile-passes 2` tests call-stack distribution reproducibility, not performance power. Perf and rocprofv3 never run simultaneously; default thread inheritance covers the server process. If prerequisites are absent, report unavailable honestly and leave this item pending.

## Code Samples & Guidance



## Files

tools/bigcherry/profiling/perf.py; tools/bigcherry/profiling/workflow.py integration; profiling configuration/schema; normalized artifact and preflight tests; real-target diagnostic evidence.

## Validation

Preflight the configured direct binary and a real cpu-clock sampling pass under the explicit sudo policy. Validate normalized output with tool/version/command provenance, two-pass reproducibility, and failure behavior. Run against a currently unresolved CPU-attribution question on a real target; ensure failed sampling is marked failed and never replaced by control. Verify perf/rocprofv3 mutual exclusion and diagnostic-only status.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Do not close while either perf capability or a legitimate unresolved CPU-attribution target is absent. When activated, perf.py is wired into the existing workflow, emits reproducible normalized sampling artifacts, preserves failure truth, and remains mutually exclusive with rocprofv3 and non-promotable for throughput. THA16/PHC03 are not assumed targets without fresh justification.

## Notes

Supersedes: HI133
Migration: capability-rebaseline-v3-2026-09
Successor key: run-hip-autotune-hi133

Evaluated against the completed RHA04/RHA10/RHA11 production admission path. This is an optional profiling enhancement, not a prerequisite for the parity/admission mission. Brutus currently exposes no usable perf events for the requested CPU call graph; implementing perf.py integration without a functioning target would produce no decision-grade evidence. Leave pending/deferred until a perf-capable host or kernel configuration is available; do not reopen the completed GPU admission items.

External dev-gpt holistic review (2026-09-10, req_9f60aaa2ae5f4b88): GO after a tiny contract freeze — design is settled enough to implement now. Freeze before coding: (1) configured direct perf binary path + explicit sudo policy (per this item's own prior verified findings: /usr/lib/linux-tools-6.8.0-139/perf with sudo); (2) normalized perf artifact schema including tool-version/command capture; (3) a failed sampling pass is failed diagnostic evidence, never silently replaced by the control run. Keep perf and rocprofv3 mutually exclusive per the existing design. THA16 (dispatch-resolver-overhead question) confirmed as the right first real validation target. Execution order: ranked #3.

CORRECTION from deeper repo-validated dev-gpt review (2026-09-10, checked against planning-refactor HEAD 429134745b04b7b96c2e86ad1c18625cbcbb4ff3): the prior GO verdict understated two real problems. (1) This item's own Description says perf is unblocked while its Validation/older Notes still say Brutus has no usable perf -- self-contradictory as currently written, fix before treating as implementation-ready. (2) The prior review's "THA16 is the right first validation target" claim is NOT well-supported by the repo: THA16 is primarily the already-existing GPU trace/tuning-divergence investigation built on HI132 primitives, not a CPU-attribution question this item's perf capability was designed to answer. RETRACT that specific target; instead validate perf.py against a concrete, currently-unresolved CPU-attribution question (not yet identified -- pick one when implementing, do not default back to THA16 without re-justifying it). Execution order DROPS to #10 (after RRBC02), not #3 -- the original priority reflected an unsupported blocker claim.

RE-ASSESSED 2026-09-10 against the WHOLE project run history including tonight's new RU01/PHC03 finding (per user directive). Verdict: CONFIRMED KEEP DEFERRED, still no legitimate CPU-attribution target. Specifically checked whether PHC03 (the real META D=3 segfault found by RU01 tonight) could be RHA01's real target -- it cannot: RU01 already localized the deterministic crash into ggml_backend_cuda_cpy_tensor_async during META's fold/copy-back path via a symbol-resolved gdb backtrace, and PHC03's own scope (find the exact copy/fallback branch responsible) is crash/control-flow localization work, not CPU on-CPU-time attribution -- sampling CPU call graphs would add little over gdb/core-dump/source instrumentation and would not answer PHC03's actual question. Every previously-considered candidate (THA16/THA13/THA09/THA10-class questions) remains either a non-CPU-attribution question or already resolved by direct evidence, confirmed again with the fuller history in view. No other currently open item supplies a genuine unresolved host-CPU attribution question. Repo evidence is sufficient to justify continued deferral -- but completing RHA01 cannot be justified from the repo alone regardless: Brutus still lacks a system-default usable perf event source (the workaround binary found earlier this project is a manual path, not a default-available one) AND the project currently lacks a real validation target; both conditions would need to hold before reopening. No action taken; correctly remains an optional, activation-gated capability, not scheduled work. Do NOT use PHC03 as a synthetic justification to force this open.

Supersedes HI133. Preserve the corrected self-consistency: direct perf was verified but integration and target evidence remain undone; this remains optional and does not block GPU admission items.

## Change Log

- 2026-09-09T10:48:34.167227+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:04:37.889270+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.818915+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T21:02:58.731913+00:00 (updated-by): Updated: section:validation, section:notes
- chg_20260909_210309_the-only-remaining-active-plan_2565
- 2026-09-09T21:03:09.280906+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:07:32.121390+00:00 (updated-by): Updated: section:notes
- 2026-09-10T00:08:04.125826+00:00 (updated-by): Updated: order=3, priority='P2'
- chg_20260910_000828_reviewed-and-re-planned-all-pe_3612
- 2026-09-10T00:08:28.930653+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.306685+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:18:48.452755+00:00 (updated-by): Updated: order=10, section:notes
- 2026-09-10T00:27:44.126726+00:00 (updated-by): Updated: section:files, section:validation
- 2026-09-10T02:55:58.252588+00:00 (updated-by): Updated: section:notes
- chg_20260910_025604_confirmed-via-a-fresh-whole-pr_7836
- 2026-09-10T02:56:04.082942+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:48:20.752589+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria, section:notes
- chg_20260910_034835_repaired-the-final-six-live-su_3009
- 2026-09-10T03:48:35.225417+00:00 (updated-by): Updated: section:ledger-events

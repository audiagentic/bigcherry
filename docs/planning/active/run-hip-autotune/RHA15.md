---
id: RHA15
order: 0
plan: run-hip-autotune
state: pending
created-at: '2026-09-15T02:19:11.114837+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
priority: P1
---

# Correctness-evidence FAIL for every candidate in real tune-campaign runs (not reproducible in isolation)

## Description

A real tune campaign (tierA-qwen4b-q6k, production-safe-single, gfx1100, current pin b10901) runs record+tune successfully (67 dispatch rows, 31 real candidates found) but EVERY candidate fails correctness-evidence generation with seed=1/2/3 native=ok candidate=failed. Confirmed non-transient across two independent full campaign runs (byte-identical failure). Manually reproducing the exact same binary+signature+candidate+environment in isolation PASSES cleanly, both times tested.

Discovered while trying to produce a fixed promoted-winners corpus for PA26 (docs/planning/active/patching-patch-system/PA26.md) -- see that item's notes history for the full multi-pass investigation chain and exact commands/evidence. This blocks winners-corpus production project-wide, not just PA26.

Investigation so far (3 forked passes, see PA26.md notes for full detail):
1. First pass: deep-dived one specific failing row/signature, manually reproduced the exact binary+signature+candidate+environment in isolation -- passed cleanly both times. Ruled out: registry mismatch, environment stripping, wrong binary.
2. Second pass: traced hip-autotune-dispatch.cu, found a real previously-undetected second GGML_HIP_FORCE_CANDIDATE_STRICT abort path ("is not eligible for this signature", distinct from "not found in registry") that was silently folding into a generic undiagnostic "failed" status. Landed DIAGNOSTIC HARDENING (new abort-path detection + extended preflight matching the fused-GLU path's existing signature preflight) as commits 4a1939d0/8ce29dfc -- deliberately not claimed as a fix, since the original failure's temp artifacts had been auto-cleaned so the new diagnostics couldn't be checked against the real failure.
3. Third pass: re-tested with the new diagnostics in place. Directly tested and RULED OUT cwd as a variable (byte-identical SIGABRT in ggml_hip_dispatch_resolve from both the binary's own bin/ directory and the real campaign's bc-pa-work working directory, using the -p ".*" fixed synthetic corpus). This conclusively eliminates cwd but did not independently reproduce the original bug (used a synthetic corpus, not the original failing signature's exact --test-file line).

Remaining live hypothesis: an order/state-dependent effect across the ~31 sequential correctness-evidence subprocess calls the real campaign makes back-to-back, that a single isolated manual invocation cannot reproduce. Untested. Needs either a full campaign re-run under the now-landed diagnostic hardening (which should surface the actual abort reason/preflight mismatch this time instead of a generic FAIL), or a dedicated multi-invocation stress harness that calls the correctness-evidence path N times in the same process/sequence a real campaign does.

## Steps

1. Run a full real tune campaign (or a scoped repro harness making the same ~31 sequential correctness-evidence calls in order) with the diagnostic hardening from 4a1939d0/8ce29dfc in place, and capture the actual abort-path/preflight diagnostic this time (don't let temp artifacts get auto-cleaned before inspection).
2. If the diagnostics reveal a genuine state-dependent bug (e.g. a stale/shared GPU context, cached device state, or a preflight that only breaks after N prior invocations), design and land a real fix -- consult GPT (session ses_c2892cdae7f14feb or a fresh one) on the fix design given this touches the general dispatch signature-matching path used far beyond PA26.
3. Verify the fix with a clean full campaign re-run: correctness-evidence should genuinely PASS (not just avoid crashing) for real candidates.
4. Once fixed, PA26's corpus-generation and two-arm hardware comparison can proceed (currently blocked on this).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Standing user authorization (2026-09-15): "start it - always start hardware test when needed" -- no need to ask before running real hardware repro attempts on Brutus for this investigation, only verify it's actually idle first.


## 2026-09-15: real orchestration bug found and fixed (not root-caused by more hardware repro)

Re-read the actual call path (tools/bigcherry/hi80_generate_correctness_evidence.py and tools/bigcherry/tuning/workflow.py) instead of running another hardware repro pass, and found the real bug via code inspection, confirmed offline (a standalone script replicating the exact loop shape showed an EvidenceError on row1 aborted before row2/row3 were reached under the pre-fix code) and via dev-gpt-agent design review (session ses_c2892cdae7f14feb, req_7e5ae686e055404e):

`generate_for_row()`'s own docstring: "raises CliError, scm.SignatureMappingError or ce.EvidenceError on failure -- the caller decides whether that is fatal to the whole run." Two callers exist -- the standalone CLI's `main()` already catches SignatureMappingError (skip) and (CorrectnessGateError, EvidenceError, CliError) (count as failed, continue), so every row gets an independent attempt. `workflow.py::_stage_correctness_evidence()` -- the function the REAL `bigcherry tune-campaign` actually calls -- only caught SignatureMappingError. An EvidenceError/CliError on the FIRST row processed propagated straight out and aborted the ENTIRE stage before any subsequent row was even attempted.

This matches the original PA26 failure shape exactly: the campaign's first promotion pass correctly classified 31 rows as needing correctness evidence, the stage was invoked, and it most likely aborted on the first problematic row it reached -- reported as "FAILED at the correctness-evidence stage", not 31 independent failures. The manual isolated repro from an earlier pass ("the top provisional winner" from promoted.jsonl) may simply never have tested the row that actually raised.

**Fixed** (commit `3af8ad2f`, pushed both remotes): centralized `hi80.ROW_FAILURE_EXCEPTIONS`, `_stage_correctness_evidence` now attempts every row and raises the existing `TuneCampaignError` only after a full pass if any failed (with per-row dispatch+candidate identity in the message), matching the CLI's own already-established per-row-tolerant design. An unclassified exception still propagates immediately (never widened to generic Exception, per GPT's explicit instruction). 7 new tests; full tools/tests/tuning suite 1230 tests, only the same 1 pre-existing unrelated failure (HI104) noted throughout this session.

**This fixes the confirmed orchestration bug. It does NOT explain WHY the original first-row EvidenceError/CliError fired** -- per GPT: "This fixes the confirmed orchestration bug without claiming it fixes the underlying candidate/signature failure." Next step (per GPT and this item's own step 1): a real tune-campaign rerun, now with BOTH this fix and the earlier diagnostic hardening (4a1939d0/8ce29dfc) in place -- the campaign should now report the REAL per-row reason for any genuine failures instead of a whole-stage abort, finally letting root-cause analysis proceed with real per-row evidence. Not yet attempted in this pass (a fresh hardware pass is the natural next step, not bundled into this fix-landing pass).

## Change Log

- 2026-09-15T02:19:11.114837+00:00 (created-by): Created by agent

## Ledger-events

- chg_20260915_023104_fixed-a-real-bug-where-a-singl_7530
- 2026-09-15T02:31:07.679293+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-15T02:31:25.750063+00:00 (updated-by): Updated: section:notes

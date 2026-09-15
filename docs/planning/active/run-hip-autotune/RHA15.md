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

## Change Log

- 2026-09-15T02:19:11.114837+00:00 (created-by): Created by agent

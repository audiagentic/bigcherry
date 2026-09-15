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

ROOT CAUSE FOUND AND CONFIRMED (2026-09-15, cross-validated by two independent investigations): `tools/bigcherry/tuning/workflow.py::_stage_correctness_evidence()` (line ~488) builds its own `test-backend-ops` lane with `build_name="tune"` and passes that binary into `hi80.generate_for_row()`. That function internally calls `signature_digest_verification.observed_test_backend_ops_signature_hex()` with `GGML_HIP_DISPATCH_MODE=record` to independently verify the signature hex before trusting any correctness comparison (HI121/HI125 gate). But `config/recipes.toml`'s `[build.tune]` only sets `GGML_HIP_AUTOTUNE=ON` / `GGML_HIP_WORKSPACE_METRICS=ON` -- never `GGML_HIP_AUTOTUNE_RECORD=ON`. The tune-lane binary is therefore NEVER compiled with record capability, so `dispatch_mode="record"` silently falls back to native mode (`ggml_hip_parse_mode: this build cannot record (configure with GGML_HIP_AUTOTUNE_RECORD=ON); using native`) and never writes a dispatch_db -- guaranteed `EvidenceError: signature-verification record-mode run failed ... cannot independently observe the real signature hex` for every row, unconditionally. This explains the 31/31 (now 31/36) consistent failure and why isolated manual repros using `dispatch_mode=replay` (a different code path) never caught it.

The codebase ALREADY has the correctly-built binary for this: `_stage_signature_verifier()` (workflow.py ~345) builds a dedicated `build_name="record"` lane specifically because record-mode capability is required for signature verification -- but that lane is only wired into the separate ingest-time `signature_digest_verifier` hook, never into `_stage_correctness_evidence`'s own `generate_for_row()` calls, which still (wrongly) use the tune-lane binary for the same kind of record-mode preflight probe.

Discovered while trying to produce a fixed promoted-winners corpus for PA26 (docs/planning/active/patching-patch-system/PA26.md). Blocks winners-corpus production project-wide, not just PA26.

## Steps

1. DONE: land the real orchestration fix (commit 3af8ad2f) so the stage attempts every row instead of aborting on the first failure -- this was necessary to even SEE the real per-row diagnostic.
2. DONE: real hardware re-run with both the diagnostic hardening (4a1939d0) and the orchestration fix (3af8ad2f) confirmed the exact root cause above via a live Python repro on Brutus reproducing the identical EvidenceError plus the binary's own plaintext stdout confirming no-record-capability.
3. NEXT: design and land the actual fix -- route `hi80.generate_for_row()`'s internal `_observed_signature_hex` preflight through a record-capable binary (reuse `_stage_signature_verifier()`'s already-built `build_name="record"` lane, e.g. by threading its `binary_ref.path`/`source_root` into `_stage_correctness_evidence` as a separate `verifier_binary`/`verifier_vendor_root` parameter distinct from the main candidate-replay binary, which still needs the tune-lane build to have the tuned candidate registered/resolvable). Consult GPT (session ses_c2892cdae7f14feb) on the exact fix design before implementing, since this touches the shared dispatch signature-matching path used well beyond PA26/RHA15.
4. Verify the fix with a clean full campaign re-run: correctness-evidence should genuinely PASS for real candidates (not just avoid the EvidenceError).
5. Once fixed, PA26's corpus-generation and two-arm hardware comparison can proceed.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Standing user authorization (2026-09-15): "start it - always start hardware test when needed" -- no need to ask before running real hardware repro attempts on Brutus for this investigation, only verify it's actually idle first.

Standing user authorization (2026-09-15): "start it - always start hardware test when needed" -- no need to ask before running real hardware repro attempts on Brutus for this investigation, only verify it's actually idle first.

Investigation chain (chronological): diagnostic hardening (4a1939d0/8ce29dfc) -> cwd ruled out -> orchestration bug found+fixed (3af8ad2f/1ab802be, the whole-stage-abort-on-row-1 bug) -> fresh campaign re-run with both fixes launched on Brutus (pa26-rha15-verify1, artifacts preserved at /home/audumla/bc-pa-artifacts/pa26-rha15-verify1/, 13M) -> real per-row diagnostics obtained (31/36 rows failed with the IDENTICAL EvidenceError) -> root cause confirmed via code inspection + live repro (this update). Full narrative detail from the investigation preserved in this item's change history (the ad-hoc field names from an earlier malformed update are superseded by this consolidated description/steps).

## 2026-09-15 (continued): REAL ROOT CAUSE FOUND AND VERIFIED (not HIP_VISIBLE_DEVICES)

The campaign (pa26-rha15-verify1) ran to completion with both fixes in place: build succeeded (506/506), record+tune succeeded (67 rows, 31 candidates), and correctness-evidence now correctly attempted all 36 rows and reported a real aggregate: `correctness-evidence generation failed for 31/36 row(s)` with full per-row detail (the orchestration fix from 3af8ad2f working exactly as designed -- no more single-row abort). Artifacts preserved at `/home/audumla/bc-pa-artifacts/pa26-rha15-verify1/` (workdir + full campaign log, 13M) before any cleanup.

Every one of the 31 failing rows hit the IDENTICAL new diagnostic (from the 4a1939d0/8ce29dfc hardening): `EvidenceError: signature-verification record-mode run failed (exit 0) or produced no dispatch_db -- cannot independently observe the real signature hex`.

**Investigated and found the real root cause via code inspection + a live repro on Brutus** (not another blind hardware pass): `tools/bigcherry/tuning/workflow.py::_stage_correctness_evidence()` (line ~488-529) calls `hi80.generate_for_row(conn, row, binary=lane_result.binary_ref.path, ...)` using the **`build_name="tune"`** lane's `test-backend-ops` binary, with NO `runner=` override (defaults to bare `subprocess.run`). That binary internally calls `signature_digest_verification.observed_test_backend_ops_signature_hex()` with `GGML_HIP_DISPATCH_MODE=record` to independently verify the signature hex BEFORE trusting any correctness comparison (HI121/HI125 gate). Ran this exact call directly against the real tune-lane binary and a real row's canonical signature from the campaign's own `promoted.jsonl` (op=29/MUL_MAT, m=2560/n=512/k=4096) via a live Python repro on Brutus -- reproduced the EXACT same EvidenceError, and the binary's own stdout revealed the real cause in plain text:
```
ggml_hip_parse_mode: this build cannot record (configure with GGML_HIP_AUTOTUNE_RECORD=ON); using native
```
**The tune-lane binary was never compiled with `GGML_HIP_AUTOTUNE_RECORD=ON`, so `dispatch_mode="record"` silently falls back to native mode and never writes a `dispatch_db` -- guaranteed EvidenceError for every single row, unconditionally.** This is a build-capability mismatch, not a state/order/env effect -- explains why it's 31/31 identical and why a prior isolated manual repro (which used `dispatch_mode=replay` + `FORCE_CANDIDATE_STRICT`, a different code path that doesn't need record capability) never caught it.

Also separately confirmed (via a second repro leg) that `HIP_VISIBLE_DEVICES` scoping is NOT the cause -- the error reproduces identically with or without it set to match `_stage_signature_verifier`'s existing device-scoped runner. That comment/reasoning in workflow.py (about env= replacing ambient env) is real and correctly motivated `_gpu_scoped_test_backend_ops_runner`, but is not what's breaking this call site.

**The codebase already has the correctly-built binary for this need**: `_stage_signature_verifier()` (workflow.py ~345-392) builds a dedicated **`build_name="record"`** lane specifically because record-mode capability is required for signature verification -- but that lane's binary/verifier is only wired into the SEPARATE ingest-time `signature_digest_verifier` hook (`inventory.load_measurements`), never into `_stage_correctness_evidence`'s own `generate_for_row()` calls, which still use the tune-lane binary for the exact same kind of record-mode preflight probe.

**Fix direction (not yet implemented, pending GPT design review)**: `_stage_correctness_evidence` (or `hi80.generate_for_row`/`generate_for_candidate`) needs to route the `_observed_signature_hex` preflight specifically through a record-capable binary (the same one `_stage_signature_verifier` already builds), while the actual correctness candidate replay (`dispatch_mode=replay` + `FORCE_CANDIDATE_STRICT`) still needs the tune-lane binary (to have the tuned candidate registered/resolvable). This likely means adding a separate `verifier_binary`/`signature_verifier_runner`-shaped parameter through `generate_for_candidate` -> `_observed_signature_hex`, distinct from the main `binary` used for the candidate run itself -- consulting GPT (session ses_c2892cdae7f14feb) before implementing, since this touches the shared dispatch signature-matching path.

## 2026-09-15 (continued): fresh hardware campaign launched with both fixes in place

Started a fresh session to re-run RHA15 step 1 (real tune-campaign with both the diagnostic-hardening commits 4a1939d0/8ce29dfc AND the orchestration fix 3af8ad2f/1ab802be in place). Verified Brutus idle (0% GPU all 4 devices, no live processes) before touching anything.

Found `/home/audumla/bc-pa-work` had leftover untracked debris (`pa26-repro3/`, 528K) from a prior pass, which made the tree dirty and caused `tune-campaign`'s materialize stage to fail closed ("BigCherry repository is dirty; use explicit development override") -- correct fail-closed behavior, not a bug. Preserved it (moved to `/home/audumla/bc-pa-artifacts/pa26-repro3-prior-pass/`, not deleted) rather than removing it, per artifact-preservation discipline. Tree now clean.

Also found the real invocation needs the venv python (`/home/audumla/bc-pytest-venv/bin/python`, not bare `python`/`python3`) with `PYTHONPATH=tools` from within `bc-pa-work`.

Launched (PID 2089964, nohup, log at `/home/audumla/bc-pa-work/pa26-rha15-verify1.log`):
```
cd /home/audumla/bc-pa-work && PYTHONPATH=tools /home/audumla/bc-pytest-venv/bin/python -m bigcherry tune-campaign --platform linux-multi --model /mnt/vault/llm-models/qwen3.5-4B/gguf/mtp/Qwen3.5-4B-UD-Q6_K_XL.gguf --devices 0 --runtime-profile production-safe-single --source bigcherry-native --run-id pa26-rha15-verify1 --json
```
Build in progress as of this note (HIP object compile, ~125/506). Will let it run to completion (build + record + tune + correctness stages), preserve artifacts before any cleanup, and report real per-row diagnostics once correctness-evidence runs.

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
- 2026-09-15T03:31:00.025020+00:00 (updated-by): Updated: section:title, work=None, skill=None, priority=None, section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:standards, section:acceptance_criteria, section:notes
- 2026-09-15T03:44:03.613854+00:00 (updated-by): Updated: section:title, section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:standards, section:acceptance_criteria, section:notes
- chg_20260915_034425_found-and-verified-the-real-ca_3031
- 2026-09-15T03:44:28.747732+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-15T03:44:53.844246+00:00 (updated-by): Updated: section:title, work='M', skill='advanced', priority='P1', section:description, section:steps, section:notes

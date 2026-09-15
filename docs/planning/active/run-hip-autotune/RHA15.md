---
id: RHA15
order: 0
plan: run-hip-autotune
state: pending
created-at: '2026-09-15T02:19:11.114837+00:00'
breadth: ''
skill: null
created-by: agent
work: null
priority: null
---

# 

## Description



## Steps



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

## 2026-09-15 (continued): fix implemented per GPT design review, landed, real re-verification campaign launched

GPT design review (req_f7f5a793c0ce4244) confirmed the root-cause analysis and gave a concrete threading design: keep the tune-lane binary (build.tune, variant-set="workload-max") for the actual native-vs-forced-candidate correctness execution (it must resolve the tuned candidate; build.record's variant-set="inventory" is native-only and cannot), but route the mandatory HI121/HI125 signature-verification preflight through the already-built record-capable `signature_digest_verifier` callable from `_stage_signature_verifier()` instead of letting `_observed_signature_hex()` re-derive its own probe against the tune binary.

**Implemented** (commit `c2ee3a99`, pushed to both remotes):
- `hi80_generate_correctness_evidence.py`: `generate_for_candidate()`/`generate_for_row()` gain an optional `signature_digest_verifier` parameter; when supplied, it replaces the `_observed_signature_hex()` fallback for both the GLU and ordinary MUL_MAT/MUL_MAT_ID branches. Standalone/direct-call compatibility preserved (falls back to the original probe when omitted).
- `workflow.py::_stage_correctness_evidence()`: now takes `signature_digest_verifier`, `signature_verifier_result`, and `devices`; passes the verifier into every `generate_for_row()` call, uses a GPU-scoped candidate runner (RHA15 secondary gap: was implicitly depending on physical GPU 0) via the campaign's real device selection, and added a provenance guard -- raises `TuneCampaignError` closed if the tune lane's and verifier lane's `source_slice_id` ever differ, before combining their evidence.
- `recovery.py::AssignmentExecutor`: gained the same optional `signature_digest_verifier` field, threaded into its own `generate_for_candidate()` call -- lazy recovery-alternative qualification had inherited the identical bug once the mandatory preflight was generalized.
- `workflow.py::_stage_replay_validate()`/`run_tune_campaign()`: thread the same memoized verifier instance through to recovery.

7 new offline tests added (verifier receives canonical signature, `_observed_signature_hex` fallback skipped when verifier supplied, forced-candidate execution still uses tune binary, source-mismatch fails closed, recovery receives verifier). Full `tools/tests/tuning` suite: 1238 passed, only the same 1 pre-existing unrelated HI104 failure noted throughout this session.

**Re-verification launched**: fast-forwarded `/home/audumla/bc-pa-work` to `c2ee3a99` and launched a fresh real campaign (`pa26-rha15-verify2`, log at `/home/audumla/bc-pa-work/pa26-rha15-verify2.log`) with the fix in place. Result pending -- will report the real per-row outcome once it completes (expected per GPT: the 31 no-dispatch_db failures should disappear; any remaining per-row failures would be genuine RHA15 correctness/signature/candidate issues, not build-capability artifacts).

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
- 2026-09-15T04:02:55.575012+00:00 (updated-by): Updated: section:title, work=None, skill=None, priority=None, section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:standards, section:acceptance_criteria, section:notes

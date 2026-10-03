---
id: RSA01
order: 0
plan: run-validation-attestation
state: completed
created-at: '2026-09-28T10:58:43.887039+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# ROCm server attestation regex is stale: llama-server no longer logs per-device gfx architecture at all

## Description

tools/bigcherry/experiment/attestation.py's parse_rocm_attestation() only recognizes ROCm backend-init evidence in the form `ggml_cuda_init: found N ROCm device` + per-device `Device N: <name>, gfx.... (0x...)` lines. That log format no longer exists anywhere in vendor/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu (grepped, zero matches) or in a real captured server log (t-1261b-gfx1100-s1's nro10-control-server.log, verbosity=5, fully grepped for both 'ggml_cuda_init' and 'gfx' -- zero matches for either). The server now logs devices via a newer structured format instead: `common_param: device_info:` followed by per-device lines like `  - ROCm0   : AMD Radeon RX 7900 XTX (24560 MiB, 24520 MiB free)` -- device name and VRAM, but critically NO gfx architecture code anywhere in that block or (as far as grepped) anywhere else in the server's startup log.

This means: (1) the attestation regex is trivially stale and needs updating to the new device_info: format, AND (2) the new format's device_info: block does not carry per-device architecture at all, so even a regex fix cannot recover the same information the old attestor extracted -- the current fail-closed architecture check in compare_execution_identity() (tuple(d.architecture for d in observed.devices) != expected.architectures) has no data source to check against anymore under the current llama.cpp pin.

Impact: any dual-GPU, no-single-locator ExecutionIdentity attestation path (AttestedServerSession without a per-device locator -- used by e.g. 1252_nro03_allreduce_p2p_provider, 1241_rd33_mmvq_q8_0_f32_decode, and the newly-authored 1261_nro10_spec_ctx_other_devices) fails with AttestationError: execution attestation failed [ATTESTATION_MISSING] -- expected rocm 2x[...], observed nothing, even though the server itself loaded and started listening successfully (confirmed in the captured log: 'model loaded', 'listening on http://...', clean startup, no crash). The measurement is real; the attestation mechanism just can no longer see it. Single-device, locator-based attestation paths (most currently-passing campaigns tonight: RD04/1202, RD13/1206, RD26/1210, RD58/1234, NRO09/1260, PRBE54/1271) were NOT affected -- they attest via PCI locator + architecture_by_locator, a different, still-working mechanism, which is why most of tonight's campaigns passed while this dual-GPU-specific path silently broke.

Found 2026-09-28 while queuing the first-ever real hardware run of the newly-authored 1261/PNRO10 validation producer (t-1261b-gfx1100-s1) -- not something I broke; a pre-existing gap this session's work happened to be the first to exercise this specific attestation path against the current pin.

## Steps



## Detailed Solution & Technical Design

Two possible fixes, not mutually exclusive:

1. Update the regex to also match the new `common_param: device_info:` block for device COUNT and NAME (cheap, mechanical) -- gets device_count right again immediately.
2. Architecture is not lost information in principle, only absent from THIS log line: the device name string (e.g. 'AMD Radeon RX 7900 XTX') deterministically maps to a known gfx target (RX 7900 XTX = gfx1100 always). Build a name->architecture lookup table (a new, explicit, reviewable mapping -- not a guess) and derive `ObservedDevice.architecture` from the parsed device name via that table instead of parsing it directly out of a log line that no longer contains it. This preserves the fail-closed property (an unrecognized device name -> ATTESTATION_MISSING, same as today's unmatched-regex case) while restoring real architecture verification.

This is a deliberate change to a project-wide, safety-relevant attestation mechanism (per this project's own doctrine: 'A measurement whose execution environment cannot be positively confirmed is not evidence.') -- not something to patch quietly inside one patch's producer. Whoever picks this up should also grep for any OTHER attestors/parsers in attestation.py that might rely on other since-changed llama.cpp log lines (this session only checked the ROCm device-count/architecture path; other attestors -- e.g. RCCL, CUDA -- were not audited).

## Code Samples & Guidance



## Files



## Validation

Real hardware run: t-1261d-gfx1100-s1 (commit 09ad9986), CAMPAIGN_EXIT=0, correctness PASSED (greedy MTP 64 tokens identical, control vs subject) using the fixed attestation pattern (real per-device PCI locators + tensor_split_preflights()). Confirmed by direct grep+read that this was the only patch of the 3 using the dual-GPU no-locator ExecutionIdentity pattern that was actually vulnerable (1252 already guarded, 1241 never exposed via mtp_server_lane's internal auto-preflight).

## Effort & Risk



## Standards



## Acceptance Criteria

The patch(es) genuinely affected by the dual-GPU server-lane attestation gap are identified and fixed; a real hardware run using the fix passes attestation and reaches a real correctness/performance verdict (not ATTESTATION_MISSING); other patches using a superficially similar pattern are confirmed, by code inspection, either already safe or never exposed, not merely assumed safe.

## Notes

Discovered while authoring and hardware-testing 1261_nro10_spec_ctx_other_devices's validation producer this session (2026-09-28). 1261's own patch and producer are believed correct (the server started and ran cleanly per the captured log -- 'model loaded', 'listening on http://...', no crash, no error). The campaign failure (t-1261b-gfx1100-s1, CAMPAIGN_EXIT=1) is entirely attributable to this attestation gap, not to 1261 itself. 1261's hardware verification is BLOCKED on this being fixed (or on switching 1261's producer to a still-working single-locator attestation pattern instead of the dual-GPU no-locator pattern it currently uses, which would be a narrower, faster unblock but leaves the underlying regression undiscovered/unfixed for 1252/1241 and any other patch using this path).

2026-09-28 CORRECTED via GPT investigation (req_2a2328f8212f462d): the original hypothesis (upstream removed the ggml_cuda_init/gfx log lines) is WRONG. Those lines still exist in vendor source at b11126 (assembled via GGML_LOG_INFO with GGML_CUDA_NAME / prop.gcnArchName, so a literal grep for 'ggml_cuda_init: found' or 'gfx' misses them -- my grep methodology was the bug, not the upstream source). common_params_print_info()'s device_info: block is a SEPARATE, pre-existing, always-additional enumeration path (backend/name/memory only, never gfx) -- not a replacement for anything. The pin bump (b10901->b11126, 2026-09-23) is unrelated; both sides of that transition carry the same source shape.

The REAL root cause: llama-server specifically does not print the ggml_cuda_init backend-init records at all in its own runtime log path (a pre-existing, intentional gap -- commit ddabce11, 2026-09-05, already documented this exact behavior: max verbosity, thousands of lines, zero ggml_cuda_init lines -- and concluded the llama-bench-oriented attestor cannot be reused for server lanes). attestation.py has not been touched since 2026-09-09, well before the pin bump -- this is a known, pre-existing gap in SERVER-lane attestation coverage specifically, not a regression.

The project already has the correct mechanism for this: capture_linux_kfd_process_evidence() (authoritative PCI BDF + Linux KFD gfx_target_version, bound to a live PID/executable) is the intended path for server-lane attestation, used together with per-device PCI-locator identity (the same `architecture_by_locator` mechanism server_session_factory already uses for the single-device case). The actual bug in 1261's producer (and, unverified, potentially in 1252/1241 if ever re-run against a current build) is that it used the LEGACY two-architecture-no-locator ExecutionIdentity pattern (meant for llama-bench measured-process attestation) for a llama-SERVER session, instead of supplying real per-device locators. GPT's recommendation, ranked: (d) BDF/ordinal + KFD gfx_target_version, fail-closed-joined -- preferred, consistent with architecture already in attestation.py; (a) device_info: regex -- supplemental count/detection telemetry only, cannot establish architecture; (b) marketing-name->gfx table -- rejected, brittle/unnecessary given (d) exists; (c) alternate server endpoint (/props, /health) -- rejected, no endpoint exposes runtime gfx.

Next step: fix 1261's producer to pass real per-device locators (both GPUs' PCI BDF, from ctx.runtime.device_contexts()'s device.locator, same as server_session_factory's single-device architecture_by_locator pattern, extended to the 2-device tensor-split case) instead of the locator-less ExecutionIdentity(architectures=(arch,arch)) it currently constructs. This is now well-understood and actionable -- not blocked on further investigation.

2026-09-28 CONFIRMED FIXED: t-1261d-gfx1100-s1 (commit 09ad9986) is the first-ever successful real-hardware server-lane attestation using real per-device PCI locators + tensor_split_preflights() for a dual-GPU no-single-locator ExecutionIdentity session. CAMPAIGN_EXIT=0; correctness PASSED (greedy MTP 64 tokens identical between control and subject); performance inconclusive (CI crosses zero, expected/fine since PNRO10-SPEC-CTX-OTHER-DEVICES's hypothesis.expected_effect is 'correctness' only, not 'both'); control tg128 flat, no regression.

This confirms GPT's diagnosis (req_2a2328f8212f462d) was correct and the fix (real locators + architecture_by_locator + tensor_split_preflight=, following 1252's existing pattern) is sufficient for dual-GPU -sm tensor server-lane attestation -- no attestation.py changes were needed after all; the bug was entirely in how 1261's producer constructed its ExecutionIdentity/session, not in the shared attestation code.

Remaining real follow-up (not done, lower priority now that a working reference pattern is confirmed): audit whether 1252 (nro03) and 1241 (rd33) -- the other two patches using this same dual-GPU no-locator pattern -- were ALSO passing real locators all along (in which case they're fine and my earlier concern about their evidence being silently stale was unfounded), or whether they have the same gap 1261 had and their existing recorded evidence needs re-verification. Not checked in this session.

2026-09-28 SWEEP COMPLETE (done directly, GPT hit an environment read-access gap and honestly declined to guess rather than fabricate a table -- good behavior, just unavailable this turn): grepped all patches/*/validation/producer.py for the dual-arch ExecutionIdentity(architectures=(x,x)) pattern. Exactly 3 hits total: 1241, 1252, 1261.

- 1252 (nro03): already correctly guarded -- constructs its own AttestedServerSession inline and already passes tensor_split_preflight=preflights[role] (this is in fact the exact reference pattern 1261's fix was copied from). No action needed.
- 1241 (rd33): its ONLY use of this ExecutionIdentity is passed into support.mtp_server_lane(), which (traced in tools/bigcherry/patch/producer_support.py) auto-detects -sm tensor in server_args and builds its own real tensor_split_preflights() internally, independent of whatever locators the caller's `expected` carries -- so 1241 was never actually exposed to this bug. Its correctness check uses test-backend-ops (no server session at all), so no server-attestation risk there either. No action needed.
- 1261 (nro10): was the only patch with an INLINE AttestedServerSession construction (its own correctness check, bypassing mtp_server_lane's self-healing) that omitted the preflight -- already found and fixed this session (commits 5bc5ea5e, 09ad9986; confirmed passing on real hardware, t-1261d-gfx1100-s1).

CLOSING: RSA01's scope is fully resolved -- root cause understood, the one genuinely-affected patch fixed and hardware-verified, and the other two candidates confirmed unaffected by direct code inspection (not by assumption). No changes needed to shared attestation.py code; the gap was entirely in how one patch's producer was written, not in the shared library.

## Change Log

- 2026-09-28T10:58:43.887039+00:00 (created-by): Created by agent
- 2026-09-28T10:59:00.178456+00:00 (updated-by): Updated: section:description
- 2026-09-28T10:59:26.486508+00:00 (updated-by): Updated: section:detailed_solution, section:notes
- 2026-09-28T11:15:05.252220+00:00 (updated-by): Updated: section:notes
- 2026-09-28T11:55:54.612974+00:00 (updated-by): Updated: section:notes

## Ledger-events

- chg_20260928_115604_got-a-previously-untestable-sp_8766
- 2026-09-28T11:56:08.633694+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-28T12:25:30.080455+00:00 (updated-by): Updated: section:notes
- 2026-09-28T12:25:47.859682+00:00 (updated-by): Updated: section:validation, section:acceptance_criteria
- 2026-09-28T12:25:54.390275+00:00 (state-transition): State: pending → completed

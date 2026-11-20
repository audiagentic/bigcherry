---
id: PVPS01
order: 0
plan: patching-validation-package-standard
state: pending
created-at: '2026-09-09T11:01:25.029179+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P1
---

# PQM-v1: standard Patch Qualification Matrix (4 arms x N architectures, isolated + in-situ)

## Description

Complete PQM-v1 as the standard patch qualification matrix: four contrast arms across gain/safety architectures, isolated and in-situ, with fail-closed planning, execution, evidence, and admission gates.

## Steps

1. Generate a versioned matrix from existing contract and architecture declarations: llama-native, BigCherry-native, BigCherry-native+P, and full release R/R+P; derive gain_arches from contract scope and safety_arches from validation coverage. 2. Resolve release-delta subject as exact current release plus P's dependency closure; fail closed on unknown/unavailable architecture, contract mismatch, or non-composable shipped patch. 3. Run gain architectures with 3v2 superiority and 4v4-P non-inferiority; safety-only architectures get correctness, activation, attestation, one smoke each, and a preregistered escalation trigger. 4. Cache binaries/build artifacts but never reuse historical measurements as paired evidence; rerun controls inside every paired session and rerun 2v1/4v1 only at release/pin/framework changes. 5. Execute all cells even after a failure for complete diagnostics; merge correctness so any failing result wins and any cell error yields invalid fail-closed verdict. 6. Require VA25 execution attestation and a complete generated matrix before release admission; defer multi-GPU/topology until all single-GPU gates and identity/visibility/topology prerequisites pass.

### Inherited qualification gates from PVPS03, PVPS05, PVPS06, PVPS07, PVPS09 and PVPS11

- PVPS03: preserve a non-gating `ladder.json` and per-architecture session history after each producer, with paired/interleaved and order-rotated stock, base BigCherry, validated BigCherry and validated BigCherry + focal-patch `llama-bench` arms on the producer model (tg128, pp512). Build the base arm only when its composition digest differs; keep the verdict unchanged. Test validated-set inclusion, focal exclusion and deduplication.
- PVPS05/PVPS07: retain the production dual-XTX `-sm tensor` MTP no-regression lane, draft-acceptance observation and paired/interleaved `mtp_wall_tps` comparison; include the pp4096 prefill sentinel. Promotion gate only applies to declared TP/all-reduce/device-placement/speculative-decode subsystems. Extend through a predeclared `max_rounds` only when CI straddles -1%; include the explicit draft-acceptance equivalence bound (e.g. absolute delta <= 0.02). Existing production-lane integration does not close its outstanding environment, queue, noise-floor and hardware gates.
- PVPS06: preserve exact all-vocabulary comparison for bit-identity contracts; for non-bit-identical contracts require finite values, margin-aware top-1 identity, KL(control||subject), JS divergence, top-p 0.95 mass agreement and max log-probability difference only where either arm has p > 1e-4. Thresholds are declared in the contract; synthetic tests must distinguish tail noise from top-1/KL failures and replay 1207's recorded divergence as a failure.
- PVPS09: paired-session statistics must be robust to a single transient catastrophic round; use synthetic series to show a -30% transient does not dominate the estimated session effect. Preserve the specified 1263 series-2 validation as future evidence, not a cleanup pass.
- PVPS11: extend the same qualification owner to an explicit backend-typed Vulkan campaign, retaining four arms, GGML_VULKAN=ON/GGML_HIP=OFF builds keyed by backend+driver, Vulkan device/driver/UUID-or-PCI attestation, GGML_VK_VISIBLE_DEVICES mapping to configured physical locators, backend-selected scaffold/producers and queue `--backend vulkan`. Validate patch 1269 on RADV gfx1100 and gfx1201. This remains an open implementation/validation requirement; current campaign construction observed in `campaign/build.py` and `validation_campaign.py` is HIP-only.

### Inherited timed-process identity and withdrawal requirements from PVPS08/PVPS12

- PVPS08: every timed HIP process must attest the physical PCI bus ID resolved by the HIP backend at CUDA/GGML init (via `hipDeviceGetPCIBusId`), so validation records bind the process's actual device rather than relying only on a prior membership preflight. Remove the preflight only after exact process-level attestation is present.
- PVPS12: each patch-validation record must persist an immutable `run_id` (workdir identity + start timestamp + host) and `session_uuid` at write time. Migrate the schema and readers; `withdraw_discarded.py` must select by explicit `run_id`, retaining pair-ratio equality as a consistency check only. Preserve historical records and reject mismatched IDs.

## Detailed Solution & Technical Design

PQM-v1 is a global policy identity bound by version/hash. The planner emits patch_id, contract_id/hash, gain/safety architectures, isolated and release compositions, release_composition_hash, and policy. The generic composition resolver uses exact overlays and dependency closure, never patch-specific campaign engines. The execution orchestrator runs every cell serially, separates inferential effects from safety smoke cells, merges correctness across cells with failures taking precedence, and produces invalid on infrastructure errors. The gate distinguishes isolated superiority from in-situ non-inferiority. Multi-GPU phases are explicitly later and require device/topology/P2P/collective identity.

## Code Samples & Guidance



## Files

campaign/resolution.py; campaign/qualification_matrix.py; campaign/qualification_execution.py; campaign/qualification_rd08.py; patch validation/evidence schema and admission gates; tools/tests/campaign/test_qualification_matrix.py, test_qualification_execution.py, test_qualification_rd08.py; VA25 attestation integration.

## Validation

Run offline matrix/execution/adapter tests and a real RD08 gfx1100 parity adapter run using an isolated clone, with control/subject compositions and attestation. Verify the known bit-identical correctness failure and near-zero combined effect reproduce the existing hand-rolled verdict, proving orchestration parity rather than falsely passing. Verify fail-closed planner paths, stale release composition detection, non-composable counterfactual handling, smoke exclusion from promotion, complete diagnostics after a cell failure, and no historical measurement reuse. Do not claim full PQM or multi-GPU completion until live adapters and all declared architectures are run.

### Inherited qualification validation

Retain existing PQM parity/fail-closed checks. Add PVPS03 composition and ladder-artifact tests plus a non-gating four-arm hardware ladder. Preserve PVPS05/07 activation, lane-noise, acceptance-bound, adaptive-round and pp4096 tests and the 1216 lane as a qualification fixture; run the dual-XTX production lane only with recorded settings and authorized hardware. Add PVPS06 synthetic probability distributions and offline re-evaluation of 1207. Add PVPS09 transient-outlier simulations. For PVPS11, add Vulkan attestation parser/build-plan/device-map tests and an end-to-end 1269 campaign with four sessions per architecture; no Vulkan support or hardware acceptance is claimed from the current HIP-only code.

Add a synthetic HIP-device log test and a dual-XTX validation session showing each timed process attests its PCI bus ID without preflight. Add withdrawal-by-run-id tests, mismatched-ID rejection and a producer-written record carrying both immutable identifiers. Existing recorded withdrawals remain historical and are not re-linked by guesswork. These validations are future acceptance gates; none were run in this planning cleanup.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

PQM-v1 matrix generation is deterministic, version-bound, fail-closed, and derived from existing declarations. Four arms and isolated/in-situ contrasts are represented with correct superiority/non-inferiority semantics. Execution is attested, runs all cells, merges failures safely, and rejects infra errors; evidence verifier requires complete matrix. Historical measurements never substitute for paired observations. Multi-GPU remains gated behind single-GPU correctness/attestation and topology identity.

Every inherited source gate is either already verified by current tests/CLI or remains explicit here as pending work. The matrix preserves PVPS03's four-arm reference ladder without changing verdicts; PVPS05/07's production MTP no-regression threshold, acceptance bound, adaptive rounds, pp4096 sentinel and subsystem-scoped promotion gate; PVPS06's bit-identity versus distribution-level correctness distinction; PVPS09's resistance to single-round transients; and PVPS11's backend-typed Vulkan build, device attestation, queue, correctness and per-architecture campaign requirements. No hardware proof is inferred from plan transfer. PVPS14's greedy divergence remains independently open.

Every timed validation process records the actual HIP-resolved physical PCI bus ID and its immutable run_id/session_uuid. The preflight is removable only when timed-process identity is proven. Withdrawals resolve uniquely by run_id; pair-ratio vectors serve only as a consistency check, and mismatched identity is refused. Historical records remain readable. No hardware or record migration is claimed by the transfer.

## Notes

Supersedes: VA26
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-validation-package-standard-va26

Supersedes VA26. Preserve VA25 dependency, RD08 parity evidence, release composition hashes, admission ordering, and explicit deferred work (multi-GPU, calibration scheduling, cache layer, automatic escalation, reverse-dependency index, and schema enforcement) until separately implemented.

Consolidation update 2026-11-20: PVPS03, PVPS05, PVPS06, PVPS07, PVPS09 and PVPS11 are incoming source gates. Their exact requirements and unresolved hardware/implementation work are preserved above; none is represented as completed. PVPS04 remains a separate active owner because current code uses the tune binary for activation probes and the campaign has tune/replay execution paths.

Consolidation update 2026-11-20: PVPS08's timed-process PCI identity and PVPS12's explicit withdrawal record identity are incoming gates because proposed MRC02 is now completed and does not retain these patch-validation acceptance details. PVPS01 is the active patch qualification/evidence owner; both gates remain pending there.

## Change Log

- 2026-09-09T11:01:25.029179+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:18:22.560586+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.611228+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:47.484054+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:48:13.282693+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria, section:notes
- chg_20260910_034835_repaired-the-final-six-live-su_3009
- 2026-09-10T03:48:35.219417+00:00 (updated-by): Updated: section:ledger-events
- 2026-11-20T04:08:05.221841+00:00 (updated-by): Updated: section:steps, section:validation, section:acceptance_criteria, section:notes
- 2026-11-20T04:23:00.757219+00:00 (updated-by): Updated: section:steps, section:validation, section:acceptance_criteria, section:notes

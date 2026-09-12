---
id: RHA13
order: 1
plan: run-hip-autotune
state: pending
created-at: '2026-09-11T22:57:02.601017+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: M
---

# Windows mixed gfx1100+gfx1201 HIP multi-GPU correctness gate (#28676)

## Description

Validate and gate the reported Windows mixed gfx1100+gfx1201 HIP silent-corruption topology. This is environment-blocked qualification, not a harness to build speculatively.

## Steps

1. Freeze the platform/device contract: Windows version, ROCm/runtime, exact device order, layer/tensor split modes, model/build/pin, fixture hashes and environment selectors.
2. Resolve issue #28676 and capture exact reproduction details.
3. Run single-device controls and mixed HIP layer/tensor split cases; Vulkan remains a separate comparative control.
4. Compare deterministic token/output hashes and correctness, not throughput alone; persist complete provenance.
5. Keep mixed HIP performance non-admissible until the gate passes; preserve a failure as regression evidence.

## Detailed Solution & Technical Design

Reuse campaign identity, correctness evidence, receipts and validation-package machinery; do not add a separate benchmark oracle. Required evidence must distinguish a true pass from fluent but corrupt output and must retain safe fallback/quarantine behavior.

## Code Samples & Guidance



## Files

tools/bigcherry/campaign/**; tools/bigcherry/tuning/correctness_evidence.py; tools/bigcherry/core/provenance.py; tools/tests/campaign/**; tools/tests/tuning/**; docs/evidence/<run-id>/; docs/reference/FINDINGS.md

## Validation

Issue state/reproducer; five-cell control/subject matrix; deterministic hashes; zero corruption; complete runtime/build/device provenance; layer and tensor split; no llama-bench-only admission.

## Effort & Risk



## Standards

Environment contract first; correctness before throughput; no unsupported extrapolation; Vulkan boundary independent.

## Acceptance Criteria

A committed reproducible verdict exists for the mixed topology; no performance result is trusted without output validation; reproduced failures remain gated/quarantined with fallback and regression evidence.

## Notes

Provenance: shared ChatGPT conversation 'Daily AMD Updates Review', 12 Sep 2026, section '#28676 — mixed XTX + RDNA4 Windows HIP can silently generate garbage'; source link https://github.com/ggml-org/llama.cpp/issues/28676. External report says Windows HIP dual-GPU corrupts while Linux HIP and Windows Vulkan controls pass. Current BigCherry status: no matching PR/issue-specific plan or committed validation found by scan; must revalidate against current main/pin before implementation.

Provenance: upstream issue #28676 and shared review. GPT says implementation is blocked until actual host/device identity and fixture contract are available.

## Change Log

- 2026-09-11T22:57:02.601017+00:00 (created-by): Created by agent

## Ledger-events


- chg_20260911_225756_added-six-provenance-rich-buil_2622
- 2026-09-11T22:57:56.698658+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T10:31:48.117034+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:validation, section:standards, section:acceptance_criteria, section:notes
- chg_20260912_103200_updated-the-active-buildrunp_8222
- 2026-09-12T10:32:01.038027+00:00 (updated-by): Updated: section:ledger-events

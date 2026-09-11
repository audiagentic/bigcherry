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

Validate the reported Windows HIP silent-corruption case on mixed RX 7900 XTX gfx1100 plus RX 9070 XT gfx1201. Throughput alone is insufficient because the reported failure produces fluent but incorrect output with normal PP/TG.

## Steps

1. Resolve the exact upstream issue and capture its reproduction details. 2. Run single-device XTX and R9700 controls, mixed HIP layer-split and tensor-split cases, and the same mixed pair under Vulkan as a control. 3. Use a deterministic prompt and compare token/output hashes plus correctness, not only throughput. 4. Record OS, ROCm/runtime, device order, split mode, model, build/pin, and all environment selectors. 5. Keep Windows mixed HIP multi-GPU evidence non-admissible until the gate passes; preserve the failure as a regression guard if reproduced.

## Detailed Solution & Technical Design

This is a run/qualification gate, not a kernel optimization. The shared chat reports llama.cpp issue #28676 as open on 12 Sep 2026, with corruption on Windows 11 and official ROCm 10.0 binaries for RX 7900 XTX + RX 9070 XT, while single-XTX, Linux HIP, and Windows Vulkan controls were correct. Treat those facts as external provenance only until reproduced against the current BigCherry pin. Reuse existing campaign identity, correctness evidence, receipts, and validation-package machinery; do not add a separate benchmark oracle.

## Code Samples & Guidance



## Files

tools/bigcherry/campaign/**; tools/bigcherry/tuning/correctness_evidence.py; tools/bigcherry/core/provenance.py; tools/tests/campaign/**; tools/tests/tuning/**; docs/evidence/<run-id>/; docs/reference/FINDINGS.md

## Validation

Resolve issue #28676 state and exact reproducer; run five-cell control/subject matrix; require deterministic output/token hashes, zero corruption, and complete runtime/build/device provenance. Confirm llama-bench-only success cannot admit the result. Test both layer and tensor split and preserve failed evidence if reproduced.

## Effort & Risk



## Standards



## Acceptance Criteria

A committed, reproducible correctness verdict exists for the mixed Windows HIP topology. No performance result from that topology is treated as trustworthy without output validation. If reproduced, the topology remains gated/quarantined with a clear safe fallback and regression evidence; if not reproduced, the negative evidence and exact environment are retained.

## Notes

Provenance: shared ChatGPT conversation 'Daily AMD Updates Review', 12 Sep 2026, section '#28676 — mixed XTX + RDNA4 Windows HIP can silently generate garbage'; source link https://github.com/ggml-org/llama.cpp/issues/28676. External report says Windows HIP dual-GPU corrupts while Linux HIP and Windows Vulkan controls pass. Current BigCherry status: no matching PR/issue-specific plan or committed validation found by scan; must revalidate against current main/pin before implementation.

## Change Log

- 2026-09-11T22:57:02.601017+00:00 (created-by): Created by agent

## Ledger-events

- chg_20260911_225756_added-six-provenance-rich-buil_2622
- 2026-09-11T22:57:56.698658+00:00 (updated-by): Updated: section:ledger-events

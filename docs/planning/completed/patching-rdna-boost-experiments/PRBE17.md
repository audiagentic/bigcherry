---
id: PRBE17
order: 0
plan: patching-rdna-boost-experiments
state: superseded
created-at: '2026-09-09T10:54:37.108784+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# Defer HIP integrated-GPU host-buffer backout

## Description

UPSTREAM-ABSORBED. RD22/patch 1209 forced `integrated=false` on HIP to back out upstream PR #24233's UMA host-buffer path. Verified at b11126: ggml/src/ggml-cuda/ggml-cuda.cu:307 still unconditionally sets `info.devices[id].integrated = false; // Temporarily disabled due to issues with corrupted output (e.g. #15034)` -- i.e. upstream's own PR #28604 revert (already noted superseded at b10900 in patch 1209's SUMMARY.md) remains in effect at the current pin. No code path exists to restore. Disposition: superseded, no implementation plan needed.

## Steps

N/A -- superseded, no implementation.

## Detailed Solution & Technical Design

N/A

## Code Samples & Guidance

N/A

## Files

HIP host-buffer policy source; integrated-GPU fixture; async/blocking/load and PPL/output tests; ownership/lifetime evidence; discrete-GPU non-selection controls.

## Validation

N/A -- confirm via `git -C work/upstream/llama.cpp.git grep -n integrated b11126 -- ggml/src/ggml-cuda/ggml-cuda.cu` (line 307 unconditional false) if re-checking at a future pin bump.

## Effort & Risk

None -- no work remains.

## Standards

Hardware-scoped change; no global workaround without evidence.

## Acceptance Criteria

Promotion requires affirmative integrated-GPU evidence with no discrete regression; until then this remains explicitly deferred and no global workaround is introduced.

## Notes

Supersedes: RD22
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd22

append

2026-09-24 relevance at b11126: UPSTREAM-ABSORBED. ggml-cuda.cu:307 unconditionally sets integrated=false at b11126 (same as patch 1209's own effect); patch 1209 SUMMARY.md already records superseded at b10900 via upstream PR #28604. No GPT design request needed (no code change required).

## Change Log

- 2026-09-09T10:54:37.108784+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:11:50.083646+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.205711+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.915767+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:50:24.214993+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_025049_rdna-successors-prbe1719-now_5726
- 2026-09-10T02:50:49.204728+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:23:13.807490+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:validation, section:effort_risk, section:notes
- 2026-09-24T02:23:29.330802+00:00 (updated-by): Updated: section:notes
- 2026-09-24T02:23:34.930490+00:00 (state-transition): State: pending → superseded

---
id: PRBE48
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:56:49.423040+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# UP-HRX-001: AMD-native HRX backend comparative lane

## Description

TODO (scoping/research, not yet code). Scope an AMD-native ggml-hrx backend as a separate experimental lane vs upstream PR #27218, kept fully isolated from HIP patches. Relevance at b11126: confirmed ABSENT -- `git -C work/upstream/llama.cpp.git grep -ln hrx b11126` returns no backend source hits (only an unrelated tools/ui/package-lock.json string match). No ggml-hrx tree exists in-repo at this pin, so there is nothing to grep for absorption and no existing code to design edits against; GPT design step was skipped for this reason (no real anchors exist yet) per brief step 4's judgment allowance.

NON-PATCH-IMPLEMENTATION ITEM (research/tracking only), confirmed by GPT review. No pinned HRX source revision, no in-tree files, no anchors, and no patches/ package can be implemented at b11126 -- ggml-hrx is absent from tree (confirmed via `git -C work/upstream/llama.cpp.git grep -ln hrx b11126`, only an unrelated package-lock.json hit). Concrete next action: pin the exact upstream PR #27218 source commit/snapshot first (network step, outside this offline mirror), THEN specify concrete CMake/backend files and anchored integration edits -- do not treat this as implementation-ready until that pin exists. Kept pending as a tracked research item, not reclassified to a different state.

## Steps

1. Pull upstream PR #27218 (ggml-hrx) description/diff from GitHub (network-enabled step, not the offline mirror) and record: which ops/kernels it implements, which gfx targets it claims support for, and its CMake/build integration surface.
2. Confirm absence at pin: re-run `git -C work/upstream/llama.cpp.git grep -rn hrx b11126 -- ggml/` before starting (must stay empty until/unless upstream merges it).
3. Define the backend maturity matrix: Qwen3.6 dense Q8/Q4 and MoE Q8/Q4, on each of gfx1100/gfx1201/gfx1030, columns = {compiles, op-coverage %, PPL/temp-0 match, server-protocol pass, PP, TG, MTP TPS, VRAM, load-time}.
4. Stand up ggml-hrx as an OUT-OF-TREE optional backend build (separate CMake option, e.g. -DGGML_HRX=ON) so it never touches the HIP patch surface; do not cherry-pick any HRX kernel into ggml-cuda/HIP.
5. Run the maturity matrix against matched HIP and Vulkan controls on the same hardware/model/quant combos.
6. Decide: keep experimental (most likely outcome given HRX is early-stage) unless correctness parity + repeatable PP/TG/MTP advantage over both HIP and Vulkan is shown.
7. If HRX proves immature or abandoned upstream, deprecate this item with that evidence.

## Detailed Solution & Technical Design

HRX is scoped as a fully separate third backend lane, not a HIP enhancement -- this is a build/eval harness item, not a kernel patch. No BigCherry patch package is proposed here because there is no in-tree HRX source to anchor edits against; the deliverable is a build+bench harness plus a go/no-go writeup. Revisit this plan once ggml-hrx lands in a future llama.cpp pin (re-run the grep above after each pin bump per the standing 'freeze source identity, re-audit ancestry after each pin bump' doctrine already in PRBE52's sibling item).

## Code Samples & Guidance

None -- no HRX source exists in this repo to anchor against. Do not write speculative Edit()/FilePatch anchors for code that isn't present.

## Files

New: tools/lab/hrx-eval/ (bench harness, maturity-matrix runner, comparison report template). No patches/ package until HRX is in-tree and a concrete kernel-level change is identified.

## Validation

Backend-op correctness tests (once ggml-hrx builds), PPL/temp-0 identity, server protocol compatibility smoke test where applicable. Performance: PP/TG/MTP TPS, VRAM, load-time, op-coverage % vs matched HIP/Vulkan on Brutus (not run by this agent).

## Effort & Risk

L / medium risk -- depends entirely on external, possibly-unstable upstream backend; scope creep risk if kernels get cherry-picked into HIP against the item's own explicit prohibition.

## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Keep HRX experimental unless matched backend correctness/feature parity and repeatable PP/TG/MTP advantage are demonstrated; do not merge backend-specific kernels into HIP without a separate decision.

## Notes

Supersedes: RD57
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd57

append

2026-09-24 relevance at b11126: confirmed ggml-hrx backend absent from tree (`git -C work/upstream/llama.cpp.git grep -ln hrx b11126` only hits tools/ui/package-lock.json, no ggml source). GPT design request: skipped, no in-tree anchors exist for HRX yet.

2026-09-24 GPT review req_2b65d50ebe9547fd applied: NOT-READY -- confirmed non-patch-implementation item, description made explicit, concrete next action (pin PR #27218 source first) stated, kept pending.

## Change Log

- 2026-09-09T10:56:49.423040+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:14:02.970413+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.345686+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.123012+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:10:25.411076+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031049_repaired-three-vulkanbackend_9010
- 2026-09-10T03:10:49.471004+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:26:04.181945+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
- 2026-09-24T02:26:30.206260+00:00 (updated-by): Updated: section:notes
- 2026-09-24T04:41:35.446106+00:00 (updated-by): Updated: section:description
- 2026-09-24T04:41:39.974354+00:00 (updated-by): Updated: section:notes

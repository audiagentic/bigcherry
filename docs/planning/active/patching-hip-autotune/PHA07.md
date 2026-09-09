---
id: PHA07
order: 7
plan: patching-hip-autotune
state: pending
created-at: '2026-09-09T12:34:07.591900+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
priority: P1
---

# 0850 ordered speculative trace patch lifecycle and production integration

## Description

Own the patch-lifecycle boundary left open by HI166. Patch 0850 implements and has real hardware evidence, but remains state=untested and is only exposed through experiment.hi166-ordered-trace-only. Establish whether the ordered trace is safe and appropriate for the normal framework composition, or retain it explicitly as experimental with a documented integration contract.

## Steps

1. Inspect patch 0850's current package, source anchors, behavioral_gate consumer, recipe membership, and dependency closure.\n2. Define the production composition required for normal tune-campaign/replay runs: build selection, server response schema, missing-trace fail-closed behavior, and compatibility with non-MTP lanes.\n3. Run hardware-free patch-lint, patch-verify, focused patch tests, and a clean normal-composition build/test path.\n4. Run the required real MTP qualification on the supported dual-XTX topology, including native/control and replay/candidate lanes, ordered-trace reconciliation, and the HI141 regression witness.\n5. Make an explicit lifecycle decision: promote only with current-pin evidence and required architecture/activation gates, or retain untested/experimental with the reason and a safe plan for consumers.\n6. Update HI166 only after this lifecycle boundary is resolved; preserve all historical evidence.

## Detailed Solution & Technical Design

The patch lifecycle and behavioral-gate semantics are separate axes. PHA07 must not treat isolated implementation tests or one successful hardware trace as promotion evidence. It must prove the exact patch composition used by normal production tuning, validate current-pin applicability and recipe membership, and ensure any lane that needs ordered traces fails closed when the producer is absent. If the patch remains experiment-only, behavioral_gate must not be silently required for unrelated production lanes; the experiment path must carry an explicit contract and receipt identity.

## Code Samples & Guidance



## Files

patches/0850_ordered_speculative_trace/patch.toml; patches/0850_ordered_speculative_trace/patch.py; config/recipes.toml; tools/bigcherry/tuning/behavioral_gate.py; tools/tests/patch/test_hi166_ordered_speculative_trace.py; tools/tests/tuning/test_behavioral_gate.py; docs/planning/active/hip-autotune/HI166.md

## Validation

Hardware-free: patch-lint, bigcherry check, patch-verify-evidence/patch-rebase-check for the current pin, focused HI166 patch and behavioral-gate tests, and a clean normal-composition build. Hardware: real supported dual-XTX MTP native/control/replay comparison with complete ordered traces and HI141 witness hard-fail preservation. No promotion or completion claim without all required evidence.

## Effort & Risk

M: lifecycle decision spans patch mechanics, recipe composition, behavioral-gate contracts, and real MTP hardware. Main risks are accidentally promoting an experimental server-schema dependency or changing non-MTP lanes; preserve fail-closed and explicit experiment boundaries.

## Standards

PATCH_VALIDATION.md; PATCH_SYSTEM.md; config/external-sources.toml; docs/reference/tooling/TOOLING.md; bigcherry-patch-lifecycle skill

## Acceptance Criteria

A current-pin lifecycle decision is recorded for patch 0850. If promoted, the patch has current validation architecture/evidence, correct normal recipe membership, passing mechanics and qualification gates, and the resulting normal composition is tested. If not promoted, it remains explicitly experiment-only with a documented consumer contract, fail-closed behavior, and no claim that HI166 is fully retired.

## Notes

Created from dev-GPT audit req_ca99651c6e134bc9. HI166's ordered-trace implementation is functionally validated, but patch 0850 remains untested and experiment-only. PHA07 owns the residual lifecycle/integration boundary; do not close HI166 until PHA07 records a deliberate result.

## Change Log

- 2026-09-09T12:34:07.591900+00:00 (created-by): Created by agent

## Ledger-events

- chg_20260909_123508_added-pha07-so-hi166s-remaini_9277
- 2026-09-09T12:35:08.318139+00:00 (updated-by): Updated: section:ledger-events

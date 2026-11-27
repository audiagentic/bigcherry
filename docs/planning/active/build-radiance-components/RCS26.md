---
id: RCS26
order: 0
plan: build-radiance-components
state: pending
created-at: '2026-11-26T01:36:23.064898+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Resolve kernel family, ownership and naming from existing metadata

## Description

Reuse r3_groups.cmake/tsv, patch registry, component.toml targets and origins.toml. Distinguish build group, qualification family and runtime row. Resolve complete canonical inventory including exclusions, affected units via compose_check and dependency closure. Unknown scope widens or blocks. Expand naming suffixes literally from validated owner/target metadata; parse lexical context, not isolated edit braces. Audit exceptions.

## Steps

1. Validate full build inventory, exclusions and origin row mapping. 2. Resolve affected units and transitive header dependencies. 3. Replace naming suffix regex with metadata-resolved literals; scope-aware declarations and exception audit. 4. Integrate existing compose and pin-triage gates.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

See docs/design/radiance-kernel-composition.md; extend existing owners.

## Validation

Actual configure inventory, synthetic new group/target, excluded and overlay units, shared headers, conditional branches, raw strings, nested namespaces, macros, attributes and ordinary helpers. Real lint and compose gates.

## Effort & Risk



## Standards



## Acceptance Criteria

One source per dimension; no duplicate taxonomy; no unsupported scope silently passes; existing build and release gates unchanged.

## Notes

2026-11-27: [Enhancement parity guide](../../../design/radiance-enhancement-parity.md) defines source counterexample, old/new actual recipe and per-block rewrite inventory and owner/row/closure proof before adoption. Use existing pin-triage, compose_check and origins, no second catalogue. Unknown scope widens affected compile closure; unresolved capability blocks retirement, not automatic full-engine assessment. Keep archived/shelved ownership evidence and advance last-integrated only after verified integration.

Detailed design and boundaries: docs/design/radiance-kernel-group-qualification.md. No active related item edited.

2026-11-26: PR1644 merged as f29a806e8. Follow-up uses compose_check local include closure, widens unresolved/header/removed paths to full inventory, imports existing pin-triage JSON changed_files. Real 115 canonical/109 configured units; stale manifest refused (six excluded units). Private composed build scratch prevents source-cache edits. Remaining: compiler dependency/complete lexical exception audit and automatic pin gate integration.

RV4754: use existing affected owner/row/family metadata for necessary baseline selection and coverage; broad catalogue redesign after baseline. No new composition framework.

## Change Log

- 2026-11-26T01:36:23.064898+00:00 (created-by): Created by agent
- 2026-11-26T01:37:48.639355+00:00 (updated-by): Updated: section:steps, section:notes
- 2026-11-26T05:06:55.737981+00:00 (updated-by): Updated: section:notes
- 2026-11-27T15:14:17.841665+00:00 (updated-by): Updated: section:notes

---
id: BHK03
order: 3
plan: build-host-kernel
state: pending
created-at: '2026-11-18T01:29:42.197170+00:00'
breadth: ''
skill: advanced
created-by: codex
priority: P2
work: M
---

# Establish a verified Linux runtime timing-write path before any tuning

## Description

Depends on BHK02. BIOS realtime timing is enabled, but no supported Alder Lake DDR5 write implementation has been established. Document writable fields, controller/DRAM synchronization, live versus retraining settings, exact driver interface and recovery. Keep prototype writes disabled until code review and a maintenance window authorize a named change.

## Steps

- Use BHK02's active readout and exact controller revision as the prerequisite; identify a supported writer implementation, not just a list of offsets.
- For each proposed setting, document register/bitfield source, permissions/lock state, controller-versus-DRAM update sequence, required quiescence, readback and whether retraining is necessary.
- Limit the first investigation to one justified field; separate mode-register-dependent timings and frequency/gear changes from controller-only updates.
- Keep the candidate tool read-only/dry-run by default during investigation. A future write requires a named setting, maintenance window, recovery access and explicit authorization.
- Conclude with a supported smallest experiment or a no-verified-writer disposition; prefer BIOS/reboot changes when the update sequence cannot be established.

## Detailed Solution & Technical Design

Realtime Memory Timing is already enabled in BIOS, but that only removes a hardware/firmware gate. Root, /dev/mem or a newer kernel does not supply the write sequence. Read-only telemetry code is a register-discovery source, not a safe writer. Do not expose writable controller BARs to the agent container or write SPD to change live timings.

If implementation is justified, keep it as a narrow host helper with validated CPU identity and a whitelist of supported operations. No universal register poke endpoint, automatic tuning loop or fuzzing. Preserve unrelated register bits, handle lock/refusal states, and prove readback on the exact operation. A reversible bitfield write may still corrupt executing memory, so restoring old bytes is not a guaranteed recovery plan.

## Code Samples & Guidance

Design record for one candidate operation:
```text
CPU/IMC revision -> documented field -> required quiescence
-> controller/DRAM update order -> allowed value domain
-> readback check -> failure/recovery decision
```
If any arrow is unsupported, stop at feasibility documentation instead of guessing a write.

## Files

- docs/planning/active/build-host-kernel/BHK02.md
- docs/evidence/2026-11-18-memory-baseline/README.md

## Validation

Static/source review of one bounded operation and its dry-run/readback path. No live mutation during planning. Only add an actual hardware test once a verified implementation and authorized window exist.

## Effort & Risk

M investigation, high consequence live mutation: unknown update ordering can corrupt running memory. Stop at feasibility unless an exact operation and authorized recovery window are established.

## Standards



## Acceptance Criteria

The plan either establishes a defensible single-operation writer experiment with explicit limits or records that Linux live writes remain unsupported for this host. No kernel update or telemetry install is represented as enabling edits.

## Notes

Plan-only follow-up to the earlier RAM baseline and timing investigation. BHK01 already owns kernel recovery; do not duplicate it or modify live host settings as part of planning.

Prerequisite BHK02. Feasibility only until a named write is authorized; do not install a driver, reboot or change RAM timings in the planning/detail pass.

## Change Log

- 2026-11-18T01:29:42.197170+00:00 (created-by): Created by codex
- 2026-11-18T01:35:34.385264+00:00 (updated-by): Updated: section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:acceptance_criteria, section:notes
- 2026-11-18T01:44:07.424265+00:00 (updated-by): Updated: section:effort_risk

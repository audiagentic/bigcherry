---
id: BHK02
order: 2
plan: build-host-kernel
state: pending
created-at: '2026-11-18T01:29:35.902016+00:00'
breadth: ''
skill: advanced
created-by: codex
priority: P2
work: M
---

# Read active memory timings through an ABI-matched host diagnostic

## Description

Depends on BHK01's verified matching headers and Module.symvers. Audit a pinned read-only Alder Lake DDR5 reader (RamSleuth/CoreFreq candidate), compile it separately, and export a host snapshot for the container. Preserve BIOS-reported speed versus SPD capabilities versus live register data. No force-load, kernel replacement, container restart or timing writes during planning.

## Steps

- Read BHK01 and verify what was actually recovered: patched source reference/config is not proof that installed module CRCs and generated headers are reproduced.
- Locate the exact installed-kernel build identity. Build a pinned read-only reader only after matching generated headers and Module.symvers exist; do not symlink generic headers into the dangling custom build link.
- Audit the reader: MCHBAR mapping read-only, supported Alder Lake DDR5 register offsets, bounded reads, no iowrite/MSR write/module auto-install path.
- Prepare a one-shot host export distinguishing SMBIOS configured speed/voltage, SPD advertised profiles and active register timings. Keep module-load execution in a separately authorized maintenance step.
- After readout, store compact per-field provenance/confidence with the comparison baseline; expose only the snapshot to the container, not writable raw controller access.

## Detailed Solution & Technical Design

A recent kernel is already present; updating Linux is not the feature needed. The dependency is ABI-matched host access to memory-controller registers plus supported decode logic. Candidate RamSleuth ramsleuth_intel exports read-only sysfs registers; CoreFreq also requires a host kernel module. Select one only after checking its pinned source and actual DDR5 field decoding. Do not add both stacks merely for redundancy.

BHK01 forbids kernel installation/loading/reboot as part of recovery. This item may prepare source and a build receipt without changing that restriction. Live readout requires a named authorized load step after ABI verification. A failed build remains a documented prerequisite, never a reason to force module loading.

Use small records with field, value, units, source, tool version and confidence. BIOS/SMBIOS can report 6000 MT/s and 1.3 V without exposing tCL/tRCD/tRP; SPD is capability data. Unknown fields remain unknown. Existing read-only SMBIOS snapshot binding is adequate container integration once refreshed; no new daemon is needed for a baseline.

## Code Samples & Guidance

Proposed snapshot shape:
```json
{"field":"tCL","value":null,"unit":"cycles","source":"active_imc","reason":"reader_not_yet_loaded"}
```
Capture the reader commit, kernel release/vermagic and generated-header/Module.symvers hashes before the load step. A version-string match alone is insufficient.

## Files

- experiments/host-kernel-gfxoff-7.0.13/README.md
- docs/evidence/2026-11-18-memory-baseline/README.md
- docs/planning/active/build-host-kernel/BHK01.md

## Validation

Source audit and one build-identity check first. After authorized load, one snapshot with plausible supported fields; compare configured speed with SMBIOS, not SPD maximum. No soak, frequency changes, device writes or container recreation for this readout.

## Effort & Risk

M: correct kernel ABI and trustworthy active-register decoding are the blockers; latest Linux or SPD maximum is not a substitute. Preserve protected build artifacts and do not force-load.

## Standards



## Acceptance Criteria

Either active timing fields have a supported decoder and traceable live source, or the concrete ABI/register blocker is recorded. Snapshot distinguishes advertised and actual values. Installed kernel and other actors' services remain protected.

## Notes

Plan-only follow-up to the earlier RAM baseline and timing investigation. BHK01 already owns kernel recovery; do not duplicate it or modify live host settings as part of planning.

Prerequisite BHK01 exact ABI artifacts or an explicitly approved alternate boot. No authorization for deep testing or module loading is implied by plan creation.

## Change Log

- 2026-11-18T01:29:35.902016+00:00 (created-by): Created by codex
- 2026-11-18T01:35:27.906711+00:00 (updated-by): Updated: section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:acceptance_criteria, section:notes
- 2026-11-18T01:43:58.354221+00:00 (updated-by): Updated: section:effort_risk

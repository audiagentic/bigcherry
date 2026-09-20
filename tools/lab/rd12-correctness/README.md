# rd12-correctness: ad-hoc real-hardware driver for the 1205 RD12 producer

Plan item: PA36 (1205 RD12 sub-slice 2)
Status: active
Owner: PA36 (patching-patch-system)
Question state: open

## Question

On real hardware, does the 1205 RD12 patch-local producer
(`--validation-producer 1205_rd12_paired_mmvq_dual_output/rd12`) run and
record correctly through the GENERIC dispatcher, one contract architecture
per invocation? This replaces the deleted `run_rd12_correctness_check()`
driver (which called the deleted CLI path directly). The generic dispatcher
owns the five-build standard-campaign scaffold, the producer's own build
pair, evidence binding, and the tracked record.

## Inputs

- Per-architecture invocation (the historical RD12 rule); shared
  worktree/build roots so the fat multi-arch binaries are built once
  (cmake-cache reuse) and reused while each `run_dir` stays per-architecture.

## Outputs

Per-architecture tracked RD12 records in the campaign workdir.

## Runtime

GPU required: yes (per RD12 contract architecture)
Real compilation required: yes (5-build scaffold + producer pair)
Mutates canonical BigCherry state: no (tracked records only)

## Safety

- Drives the generic `validation_campaign` dispatcher; not production
  tooling and not the evidence authority.
- Check for live build/lease state before running on a shared tree.

## Disposition

Retained (TRANSITIONAL) as the RD12 real-hardware driver; diagnostic hardware
evidence, not a maintained tool.

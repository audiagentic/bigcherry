# VA26 RD08 orchestrator parity check

Plan item: VA26
Status: answered
Owner: Claude Sonnet 5 (session)
Question state: answered

## Question

Does the new campaign.qualification_execution + campaign.qualification_rd08
execution pipeline reproduce the SAME substantive verdict as RD08's
existing, hand-rolled `--run-rd08-contract` qualification path, when run
against real hardware?

## Inputs

- Patch `1204_rd08_q6k_mmvq_vdr2`, contract `RD08-Q6K-MMVQ-VDR2`.
- gfx1100 (2x RX 7900 XTX), model `tierM-gptoss20b-q6k`.
- An isolated clone of the repo on the build server (never the shared
  checkout), base_repo resolved via
  `context.ProjectContext.resolve().upstream_repo`.

## Outputs

Console output only (printed plan/promotion result); no generated
artifacts retained outside the isolated clone's own throwaway work root.

## Runtime

GPU required: yes
Real compilation required: yes (4 real llama-bench builds + 2 real
test-backend-ops correctness builds per run)
Mutates canonical BigCherry state: no

## Safety

- Canonical-state mutation: none. Reads the real patch catalog/config;
  writes only to its own isolated clone's work directories.
- Do not import this script from `bigcherry` production, tests, or
  maintained analysis -- it is a diagnostic driver, not a library.
- Requires `HIP_VISIBLE_DEVICES`/`ROCR_VISIBLE_DEVICES` set explicitly
  (refuses to run otherwise, matching this project's documented
  multi-GPU-visibility safety rule).

## Disposition

**Answered, 2026-09-08**: live run confirmed the new orchestrator
reproduces RD08's real, independently-confirmed correctness failure
(byte-identical output digests to the existing evidence.json) plus a new
finding (near-zero combined-release gain). See
`docs/planning/completed/validation-package-standard/VA26.md` for the full
result.

Graduating: no durable capability graduation decision made yet -- this
script stays as a lab driver for now. Revisit if/when VA26's execution
phase gets a maintained CLI entry point (per its own design review,
deliberately not added yet).

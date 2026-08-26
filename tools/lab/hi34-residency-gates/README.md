# HI34 residency gates

Plan item: RA12
Status: completed; compatibility implementation retained
Owner: rationalisation / HI34
Question state: answered

## Question

Can the HI34 residency-gate checks classify the recorded residency evidence
without making lab code importable as production tooling?

## Inputs

Residency-gate command arguments and the evidence files named by the caller.

## Outputs

A deterministic gate report and exit status; generated evidence remains under
the caller's artifact directory.

## Runtime

GPU required: no
Real compilation required: no
Mutates canonical BigCherry state: no

## Safety

- Canonical-state mutation: none.
- The root `tools/residency_gates.py` wrapper remains the supported legacy CLI.
- This lab directory is not evidence authority outside the gate contract.

## Disposition

Delete this plan-owned implementation after the compatibility CLI contract is
retired; any durable gate capability must be graduated separately.

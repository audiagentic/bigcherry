# rd30-correctness: ad-hoc real-hardware driver for RD30 correctness

Plan item: RD30 / PA36
Status: active
Owner: PA36 (patching-patch-system)
Question state: open

## Question

On real hardware (Brutus, single `gfx1100` XTX — the contract's own
`scope.architectures` is `gfx1100`-only), does
`run_rd30_correctness_check()` produce the RD30 correctness evidence? No
`--run-rd30-correctness` CLI flag exists yet (RD58-style wiring into
`run()`'s mutually-exclusive execution modes is separate follow-up scope).
This driver mirrors the RD13 real-hardware driver pattern — a direct call to
the real producer function.

## Inputs

- `/home/audumla/rocm-shim`, `AMDGPU_TARGETS=gfx1100`.

## Outputs

RD30 correctness evidence in the campaign workdir.

## Runtime

GPU required: yes (gfx1100)
Real compilation required: yes (control + subject builds)
Mutates canonical BigCherry state: no

## Safety

- Direct call to the real producer function; not production tooling and not
  the evidence authority.
- Never run on a shared/ambiguous tree without checking for live leases first.

## Disposition

Retained (TRANSITIONAL) as the RD30 real-hardware driver; diagnostic hardware
evidence, not a maintained tool.

# rd13-backend-reference: ad-hoc real-hardware driver for RD13 backend reference

Plan item: RD13 / PA36 (1206 migration)
Status: active
Owner: PA36 (patching-patch-system)
Question state: open

## Question

On real hardware (Brutus, 2× `gfx1100` XTX or single `gfx1100`), does
`run_rd13_backend_reference_check()` produce the RD13 backend-reference
evidence? No `--run-rd13-backend-reference` CLI flag exists yet in
`validation_campaign.py` (RD58-style wiring is separate follow-up scope), so
this script invokes the real producer function directly — exactly the way
RD58's real evidence was first gathered before its own CLI flag existed.

## Inputs

- `/home/audumla/rocm-shim`, `AMDGPU_TARGETS=gfx1100`, `tierA-qwen4b-q6k`
  model, base revision resolved from the active pin.

## Outputs

RD13 backend-reference evidence in the campaign workdir.

## Runtime

GPU required: yes (gfx1100)
Real compilation required: yes (control + subject builds)
Mutates canonical BigCherry state: no

## Safety

- Direct call to the real producer function; not production tooling and not
  the evidence authority.
- Never run on a shared/ambiguous tree without checking for live leases first.

## Disposition

Retained (TRANSITIONAL) as the RD13 real-hardware driver; diagnostic hardware
evidence, not a maintained tool.

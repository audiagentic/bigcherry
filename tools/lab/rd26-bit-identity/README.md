# rd26-bit-identity: ad-hoc real-hardware driver for RD26 decode/verify bit-identity

Plan item: RD26 / PA36 (1210 migration)
Status: active
Owner: PA36 (patching-patch-system)
Question state: open

## Question

On real hardware, does `run_rd26_decode_verify_bit_identity_check()` pass
for `RD26-DECODE-VERIFY-BIT-IDENTITY`? No `--run-rd26-bit-identity` CLI
flag exists yet (separate follow-up scope). The contract scopes
`gfx1100`/`gfx1201`/`gfx1030`; this driver runs `gfx1100` first (simplest
single-GPU topology) — `gfx1201`/`gfx1030` are real follow-up runs.

**A real run against the CURRENT 1210 patch may legitimately FAIL**: patch
1210 contains only 2 of a 5-commit determinism cluster (the "base-standalone"
hunks), and the complete decode/verify determinism property belongs to the
full cluster. A FAIL here is real, useful evidence of that documented
remaining gap, not a bug.

## Inputs

- `/home/audumla/rocm-shim`, `AMDGPU_TARGETS=gfx1100`;
  `HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0`.

## Outputs

RD26 decode/verify raw-F32-logit byte-identity result in the campaign workdir.

## Runtime

GPU required: yes (gfx1100)
Real compilation required: yes (control + subject builds)
Mutates canonical BigCherry state: no

## Safety

- Direct call to the real producer function; not production tooling and not
  the evidence authority.
- Check for live build/lease state before running on a shared tree.

## Disposition

Retained (TRANSITIONAL) as the RD26 real-hardware driver; diagnostic hardware
evidence, not a maintained tool.

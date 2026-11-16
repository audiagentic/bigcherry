# VTP44 baseline status

## Valid now (no device execution)
- `baseline-libr4d-gfx1201-catalogue.txt`, `baseline-libr3-gfx1100-catalogue.txt`: the row catalogue each plugin prints before any kernel runs (595 rows each).
- `baseline-libr4d-host-symbols.txt`: exported `r4d_*` host symbols (1,065).
Taken from the rad0038 build (`radiance_home/kernels/libr4d.so`, `libr3.so`).

## Pending: device smoke-test baseline
The first attempt ran `r4d_selftest` with no device pinned, so HIP used device 0 (gfx1100) against gfx1201-only
device code and the launch segfaulted inside `libamdhip64.so.7` (ROCm 7.2.4 does not return an error for an
incompatible code object). 201 rows reported SIGSEGV, 40 passed. Pinned to the gfx1201 card
(`HIP_VISIBLE_DEVICES=2`) the rows checked (`relu_bf16`, `clamp_bf16`) pass. The other rows were NOT re-run, so the
201 failures are not yet all attributed to device selection. That run is discarded, not kept as a baseline.
Independent confirmation and the vendored-runtime comparison: `feat/iq6r-v4-qualification`,
`docs/evidence/2026-11-16-iq6r-v4-kickoff/CRASH-DIAGNOSIS.md` and `ROCM-VENDOR-ABI.md`.

The corrected baseline (run by the selftest agent, replacing the discarded one) must record:
- the physical GPU and its detected GFX arch, the ROCm version, and the exact libr4d / libr3 build identity;
- executed / passed / failed / skipped counts, with numeric mismatches kept separate from unsupported-device
  failures and real crashes;
- each remaining failure with a reproducing command (`r4d_selftest <plugin.so> --case <row>`).
Run libr4d on the gfx1201 card and libr3 on a gfx1100 card.

## Refactor acceptance (patch A)
Catalogues and host symbols identical before/after; the corrected smoke baseline identical before/after.

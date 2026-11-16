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

## Corrected baseline (2026-11-16, run by the selftest agent)
Pinned `HIP_VISIBLE_DEVICES=2` (physical gfx1201 R9700; the selftest reports `device 0: gfx1201`); the loaded `libr4d.so` holds
gfx1201 offload images; the running catalogue matches the 595 row names above. Build `vtp43/hb-i8` (rad0038, no rad0040/41), source
`vtp43/src-i8`. Result: **0 failures, 246 skipped** (`r4d-selftest-gfx1201-raw.log`, `receipt.json` under
the work root's `r4d-gfx1201-baseline/rad0038-i8/` folder). The skipped rows are the moe_gemm_w4a8, attn_qrow, attn_window and
attn_anchor_window families and the like; none is a bf16_fp8, bf16_i8 or gqa12 row.
Patch A and B runs use the same invocation on the same card (`vtp44/run_selftest_b.sh`, results in
the work root's `r4d-gfx1201-baseline/vtp44-ab/` folder), queued behind other jobs on the `gpu` lease.

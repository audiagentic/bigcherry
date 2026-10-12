# 0003_mtp_full_head_switch

**Status:** untested
**Plan item:** RAD01

## What it does

`RAD_MTP_FULL_HEAD=1` bypasses the U2 draft-head copy and uses the trunk head; unset/0 preserves the default. Correct condition is `std::atoi(e) != 0`, not character comparison. Original diff: `feat/rad11-gfx1100-bootstrap:engines/radiance/engine-patches/0003-mtp-full-head-switch.patch`.

## Why and evidence

Diagnostic to isolate draft-head quantization from MTP acceptance. Feature-branch `engines/radiance/kernels/README.md`, section **MTP against llama.cpp on the same workload**: runs `fnmh3-1`/`fnmh3-0` found 0.63-0.65 acceptance using trunk int8 vs 0.62-0.64 with U2. No accepted improvement or new package validation; keep untested.

## Upstream

Radiance 1.3.0, `89cee7ce414486c717aa2085c99e1f522e4c0ea6`; BigCherry-local diagnostic switch.

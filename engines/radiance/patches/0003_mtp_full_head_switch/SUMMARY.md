# 0003_mtp_full_head_switch

**Status:** untested
**Plan item:** RAD01

## What it does

Adds `RAD_MTP_FULL_HEAD=1` to make MTP draft rounds use the trunk language-model head instead of the container's two-bit draft-head copy. The switch is off by default.

## Why

This isolates draft-head precision as a possible cause of Radiance's lower MTP acceptance on the measured Flash-Next workload.

## Evidence

On `feat/rad11-gfx1100-bootstrap`, `engines/radiance/kernels/README.md`, subsection “MTP against llama.cpp on the same workload”, records `RAD_MTP_FULL_HEAD=1` acceptance at 0.63-0.65 versus 0.62-0.64 with the u2 head; it did not explain the acceptance gap. The switch is not qualified for adoption, so the package remains `untested`.

## Upstream

Local BigCherry change against Radiance 1.3.0, commit `89cee7ce414486c717aa2085c99e1f522e4c0ea6`; source diff: `engines/radiance/engine-patches/0003-mtp-full-head-switch.patch` on `feat/rad11-gfx1100-bootstrap`.

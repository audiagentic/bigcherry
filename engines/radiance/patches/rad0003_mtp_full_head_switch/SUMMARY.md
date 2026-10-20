# rad0003_mtp_full_head_switch

**Status:** untested
**Plan item:** RAD01

## Summary

RAD_MTP_FULL_HEAD=1 drafts MTP rounds through the trunk LM head instead of the container's 2-bit draft head.

## Outcome

untested: off by default, not yet measured.

## Detail

See below.

## What it does

Adds `RAD_MTP_FULL_HEAD=1` to make MTP draft rounds use the trunk language-model head instead of the container's two-bit draft-head copy. The switch is off by default.

## Why

This isolates draft-head precision as a possible cause of Radiance's lower MTP acceptance on the measured Flash-Next workload.

## Evidence

On `feat/rad11-gfx1100-bootstrap`, `engines/radiance/kernels/README.md`, subsection “MTP against llama.cpp on the same workload”, records `RAD_MTP_FULL_HEAD=1` acceptance at 0.63-0.65 versus 0.62-0.64 with the u2 head; it did not explain the acceptance gap. The switch is not qualified for adoption, so the package remains `untested`.

## Upstream

Local BigCherry change, authored against Radiance 1.3.0 (`89cee7ce414486c717aa2085c99e1f522e4c0ea6`) and re-anchored on Radiance 1.7.0 (pin `48f7b8eb6d703cd5318346b5a351a292afaf4bf9`, MEN12). The anchor (`if (!head || !lm) return false;` in `mtp_draft_head_q2`) is unchanged and unique on 1.7.0; the patch now also adds the `#include <cstdlib>` that 1.4.0+ removed from the header. 1.7.0 has no equivalent switch. Original source diff: `engines/radiance/engine-patches/0003-mtp-full-head-switch.patch` on `feat/rad11-gfx1100-bootstrap`.

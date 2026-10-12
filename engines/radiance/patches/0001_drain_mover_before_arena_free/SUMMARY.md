# 0001_drain_mover_before_arena_free

**Status:** evaluated
**Plan item:** RAD01

## What it does

Drains each rank's mover stream before `Ctx::shutdown()` frees the activation arena. No inference-step arithmetic changes. Original source diff: `feat/rad11-gfx1100-bootstrap:engines/radiance/engine-patches/0001-drain-mover-before-arena-free.patch`.

## Why and evidence

The mover can still copy into arena-backed slabs or prefill-stager buffers during teardown. In two-XTX Flash-Next runs without the fix, 2/3 `fnh1`-`fnh3` executions hung; with the patched engine, `fnp1`-`fnp3` exited cleanly (3/3). See feature-branch `engines/radiance/kernels/README.md`, section **Flash-Next: radiance's own prefill stager, switched on**. This is historical hardware evidence, not a current-package validation or production qualification.

## Upstream

Radiance 1.3.0 at `89cee7ce414486c717aa2085c99e1f522e4c0ea6`; BigCherry-local change. Package state `evaluated` is supported by the live registry for hardware-tested but not fully qualified candidates. No G0-G4 promotion claim.

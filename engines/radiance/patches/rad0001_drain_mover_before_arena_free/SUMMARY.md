# rad0001_drain_mover_before_arena_free

**Status:** superseded
**Plan item:** RAD01

## Summary

Drains each rank's mover stream before the activation arena is freed at shutdown.

## Outcome

superseded: fixed a real two-XTX teardown hang (3/3 clean exits vs 2/3 hangs unpatched); superseded by later teardown work.

## Detail

See below.

## What it does

Drains each rank's mover stream before `Ctx::shutdown()` frees the activation arena. No inference-step arithmetic changes. Original source diff: `feat/rad11-gfx1100-bootstrap:engines/radiance/engine-patches/0001-drain-mover-before-arena-free.patch`.

## Why and evidence

The mover can still copy into arena-backed slabs or prefill-stager buffers during teardown. In two-XTX Flash-Next runs without the fix, 2/3 `fnh1`-`fnh3` executions hung; with the patched engine, `fnp1`-`fnp3` exited cleanly (3/3). See feature-branch `engines/radiance/kernels/README.md`, section **Flash-Next: radiance's own prefill stager, switched on**. This is historical hardware evidence, not a current-package validation or production qualification.

## Superseded

Upstream radiance 1.7.0 (release commit `1e968f6`, pin `48f7b8e`) fixes the same teardown race: `Engine::~Engine` now syncs every rank's `mover.transfer_stream()` in a loop before the `ctx.shutdown()` loop, with the same diagnosis (the prefill stager ring and lent slab slots copy into the arena). It was never wrong; the engine implemented the fix. The package is kept as the record of the original fix and is retired from the radiance recipes' patch sets. Its anchor still matches once on 1.7.0, so applying it would double the drain: do not re-add it. Differences from upstream's form: upstream does not call `rad_dev_set(r->device)` first. The Flash-Next two-XTX 3/3-clean-exit check should be repeated once on the 1.7.0 pin to confirm the upstream form.

## Upstream (original pin)

Radiance 1.3.0 at `89cee7ce414486c717aa2085c99e1f522e4c0ea6`; BigCherry-local change. It was `evaluated` with the historical hardware evidence above. No G0-G4 promotion claim.

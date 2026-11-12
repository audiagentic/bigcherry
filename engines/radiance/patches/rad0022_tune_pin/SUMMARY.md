# rad0022_tune_pin

**Status:** untested
**Plan item:** VTP31

Scope: radiance's kernel tune pin (`RADIANCE_TUNE`) as an engine option, `--tune-pin SPEC`, so a layout or manifest
can carry it.

## Summary

The same text and checks as `RADIANCE_TUNE` (an axis or value a kernel does not offer is an error), set before the
plugins load. A different `RADIANCE_TUNE` already in the environment is refused. Without the option nothing changes.

## Outcome

Untested (not promoted). The value it carries first, `all_reduce:nblocks=4,nthreads=256` on MXFP4 TP3 (XTX, XTX,
R9700), measured through `RADIANCE_TUNE`: +1.7% / +6.7% / +3.5% / +5.7% at 1/4/6/8 streams against the default grid,
and above dual-XTX TP2 at every stream count; greedy output identical (VTP31.md).

## Detail

See README.md.

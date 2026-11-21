# rad0052_b16_kv_geometries

**Status:** validated
**Plan item:** RUM07

Scope: Add the remaining tested B16 geometries and isolate Qwen35 DFlash2.

## Summary

The patch adds Qwen4exp GQA6/page4 and Qwen35 GQA6/page16 B16 rows while keeping H128/GQA4/page16 DFlash2 on native E4M3.

## Outcome

The Radiance 2.2 production attachment passes exact replay. The [carry-forward record](README.md) links the historical behavior to the focused RUM07 port checks. RUM09 owns integrated deployment quality and parity.

## Detail

Geometry-specific rows keep the DFlash2 route on its existing E4M3 cache while enabling B16 on the retained trunk layouts.

[Patch implementation](patch.py).

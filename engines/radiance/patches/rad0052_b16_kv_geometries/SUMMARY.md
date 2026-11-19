# rad0052_b16_kv_geometries

**Status:** validated
**Plan item:** RUM07

Scope: Add the remaining tested B16 geometries and isolate Qwen35 DFlash2.

## Summary

The patch adds Qwen4exp GQA6/page4 and Qwen35 GQA6/page16 B16 rows while keeping H128/GQA4/page16 DFlash2 on native E4M3.

## Outcome

The retained behavior was tested on the old Radiance base and its historical device, codec, KLD, serving, and performance evidence remains authoritative for those semantics. On Radiance 2.1.0, this port passed ordered exact apply/reapply, focused host and architecture fixtures, selected object builds, the complete component build, ABI44 plugin checks, and the serving/libr3 component suites. This validated migration is included in the production recipe. The final paired deployment quality/performance gate remains RUM09.

# rad0052_b16_kv_geometries

**Status:** untested
**Plan item:** RUM07

Scope: Add the remaining tested B16 geometries and isolate Qwen35 DFlash2.

## Summary

The patch adds Qwen4exp GQA6/page4 and Qwen35 GQA6/page16 B16 rows while keeping H128/GQA4/page16 DFlash2 on native E4M3.

## Outcome

The old-base behavior is covered by the retained BigCherry evidence. ABI44 attachment checks and the final production validation remain part of RUM07/RUM09 before this package enters the bumped production recipe.

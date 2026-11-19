# rad0054_attn_prefill_fp8

**Status:** untested
**Plan item:** RUM07

Scope: Restore independent E4M3 prefill precision modes for historical rows.

## Summary

The patch restores off, QK, PV and combined prefill modes on the supported Qwen35 H256/GQA6/page16 E4M3, V4 and K6V4 rows; decode and unsupported geometries do not accept the prefill selector.

## Outcome

The old-base behavior is covered by the retained BigCherry evidence. ABI44 attachment checks and the final production validation remain part of RUM07/RUM09 before this package enters the bumped production recipe.

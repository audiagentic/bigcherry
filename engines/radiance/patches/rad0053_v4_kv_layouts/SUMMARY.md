# rad0053_v4_kv_layouts

**Status:** untested
**Plan item:** RUM07

Scope: Restore fp8_v4 and k6_v4 on the shared ABI44 packed-KV descriptor.

## Summary

The patch ports the historical V4/K6 stores, readers and geometry rows using descriptor-owned offsets and preserves the tested standalone and fused scale arithmetic.

## Outcome

The old-base behavior is covered by the retained BigCherry evidence. ABI44 attachment checks and the final production validation remain part of RUM07/RUM09 before this package enters the bumped production recipe.

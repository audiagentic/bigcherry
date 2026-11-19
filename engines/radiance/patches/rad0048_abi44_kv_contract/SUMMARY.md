# rad0048_abi44_kv_contract

**Status:** untested
**Plan item:** RUM03/RUM07

Scope: Define the local ABI44 KV layout contract and canonical packed-format descriptors.

## Summary

The patch appends and fingerprints the local KV layout/capability fields and defines one descriptor table for V4, K6V4, B16F8 and B16I8.

## Outcome

The old-base behavior is covered by the retained BigCherry evidence. ABI44 attachment checks and the final production validation remain part of RUM07/RUM09 before this package enters the bumped production recipe.

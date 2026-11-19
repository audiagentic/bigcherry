# rad0049_qwen4exp_e4m3_kv

**Status:** untested
**Plan item:** RUM07

Scope: Restore native E4M3 Flash-Next cache rows on the ABI43 base.

## Summary

The patch restores the deployed H256/GQA12 and GQA6/page4 E4M3 rows, store and attention wiring without aliasing the format to I8.

## Outcome

The old-base behavior is covered by the retained BigCherry evidence. ABI44 attachment checks and the final production validation remain part of RUM07/RUM09 before this package enters the bumped production recipe.

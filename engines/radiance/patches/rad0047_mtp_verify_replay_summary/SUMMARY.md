# rad0047_mtp_verify_replay_summary

**Status:** untested
**Plan item:** RUM08

Scope: Add safe replay handling for wide speculative verification with missing summaries.

## Summary

The patch disables recorded-pass reuse for an unproven nonpositive full-KV summary, preserving live execution rather than forming a potentially unsafe replay key.

## Outcome

The old-base behavior is covered by the retained BigCherry evidence. ABI44 attachment checks and the final production validation remain part of RUM07/RUM09 before this package enters the bumped production recipe.

## Detail

The escape applies only to wide full-KV verification without an authoritative positive summary. It does not replace the upstream replay scheduler.

[Patch implementation](patch.py).

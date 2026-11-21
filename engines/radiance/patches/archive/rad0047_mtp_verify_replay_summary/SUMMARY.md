# rad0047_mtp_verify_replay_summary

**Status:** rejected
**Plan item:** RUM08

Scope: Proposed live-issue escape for a wide speculative verify without a proven full-KV host summary.

## Summary

The uncomposed patch would refuse wide verify recording if the full-KV summary was absent or nonpositive. It is rejected on the 2.2 production path because the scheduler cannot generate that case for a valid Qwen4exp wide verify.

## Outcome

For pinned 2.2.0 source b6b8751, `BatchBuilder::build` refuses empty plans. A real wide verify contains positive query tokens; its full-attention group is non-windowed, so `stage_kv` computes `used = ctx_len + n_tokens`, with `ctx_len >= 0`, and `kv_used_max_h` is positive. `RadBatch::kv_used_max_h` is always bound to this vector. A direct test callback can be passed a zero/null summary, but this is not a scheduler-produced verify. No additional guard is required for the fixed-depth production mode.

The original patch remains archived with its source intact. This is a host-code reachability decision, **not** a GPU replay performance or full-mode qualification. See `docs/evidence/2026-11-21-rum08-replay/README.md`; RUM09 owns the hardware canary.

## Detail

Its guard would have made the artificial missing-summary case issue live rather than sharing the upstream above-bound selector key. No standalone engine policy or legacy fallback is introduced.

[Patch implementation](patch.py).

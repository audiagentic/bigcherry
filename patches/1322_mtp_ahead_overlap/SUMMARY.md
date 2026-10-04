# 1322_mtp_ahead_overlap

**Status:** untested
**Plan item:** FMTP03

## What it does

With `BIGCHERRY_MTP_AHEAD=1` (greedy slots, draft context with partial seq_rm):

- **Ahead draft during verify:** in `decode()`, after the target `llama_process()` submit and before
  `llama_synchronize()`, the MTP drafter replays the round's draft as a forced front (1321) and continues it into a
  tail (bonus-token prediction + next front) on the otherwise idle draft GPU. The draft KV it wrote is trimmed back to
  the checkpoint before `common_speculative_process()` reseeds, so the authoritative path is unchanged.
- **Promotion:** when the whole front was accepted and the target sampled the tail's first token, the rest of the
  tail becomes the next round's draft and the serial fresh draft (6.4-8.7 ms/round, FMTP01) is skipped. The round still
  checkpoints/trims like a fresh draft and the target verifies it, so greedy output is unchanged.
- Every 64 rounds: `BIGCHERRY_MTP_AHEAD rounds= ahead= ahead_tokens= promoted= promoted_tokens= ahead_us=`.

Requires 1321. Expected promotion rate ~0.37-0.42 of rounds (FMTP01 Gate 0 p_hit).

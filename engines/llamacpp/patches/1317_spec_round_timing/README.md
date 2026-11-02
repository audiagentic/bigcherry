# 1317_spec_round_timing

## Summary

Diagnostic only. `BIGCHERRY_SPEC_TIMING=1` logs one `BIGCHERRY_SPEC_TIMING` line per speculative round with the serial phase times: fresh draft, target submit (llama_process host time), target sync wait, draft-context catch-up (common_speculative_process), target sample-and-accept, plus drafted/accepted counts. No synchronization is added or moved.

## Outcome

Recorded state: `validated` ([patch.toml](patch.toml)); qualification and limitations: [SUMMARY.md](SUMMARY.md).

## Detail

## Promotion record

Promotion record (QFP18 lightweight tier, pin b11474, dev 2026-10-08).

- Build: `deploy-v4-diag`, compiled clean.
- Activation: `BIGCHERRY_DRAFT_TRACE=1 BIGCHERRY_SPEC_TIMING=1 BIGCHERRY_DRAFT_TIMING=1`; log contained
  **92 DRAFT_TRACE**, **23 SPEC_TIMING**, and **23 DRAFT_TIMING** lines.
- Correctness/neutrality: greedy text identical to production, md5 `a34a34c1f48d`.
- Errors: **0 error lines**.
- This patch's observed activation signal: **23 SPEC_TIMING**.

The diagnostic is dormant at its default settings and makes no performance claim.

# 1348_mtp_deferred_catchup

QFP31 implementation against llama.cpp b11474 (`b9acf138a1e2`).

The measured prompt path was ~354 ms per 512-token target chunk without the MTP draft and ~437 ms with it. Native MTP synchronizes the target, then performs ~83 ms of draft catch-up before the next target chunk is submitted. This patch preserves the target synchronization needed to copy NextN safely, but defers the draft decode: target k+1 is submitted before catch-up k is run.

The ownership mechanism is deliberately local to the MTP implementation. Two host snapshots retain the token metadata and target NextN rows. This avoids the generic scheduler/output lifetime machinery rejected by FMTP03/RV4223 and requires no thread.

## Switch

- Default: `BIGCHERRY_MTP_DEFERRED_CATCHUP=1`
- Control: `BIGCHERRY_MTP_DEFERRED_CATCHUP=0`
- Marker on first use: `BIGCHERRY_PATCH_HIT patch=1348_mtp_deferred_catchup`

## Required hardware validation

Compare default-on vs control on the production topology with identical prompts. Record prefill t/s, per-chunk target/MTP timing (1346 may be used separately as a diagnostic), generated-text hash, draft acceptance/decode, and any errors around cancellation/context shift. The expected gain is removal of most of the serialized ~83 ms catch-up from interior prompt chunks; the final catch-up remains serial before sampling.

## Not yet verified

No HIP build or hardware execution was available from the GitHub connector environment. Offline mechanics/composition results are recorded in the PR.

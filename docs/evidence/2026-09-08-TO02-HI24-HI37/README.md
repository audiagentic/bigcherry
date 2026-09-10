# TO02 / HI24 and HI37 — Brutus artifact verification

Observed 2026-09-08 UTC on `brutus` from the existing BC-native 27B record and
tuning artifacts. `bigcherry inventory hot-list` produced a 69-signature
call-weighted hot list, and `bigcherry inventory workload-check` computed the
workload/tuned-set digest and correctly reported `DIFFERENT` with zero covered
signatures. This is the intended advisory result for a mismatched workload,
not a safety failure.

The exact input paths and their source SHA-256 values are retained in the
remote command output and `SHA256SUMS`; the generated hot-list and command
logs are the compact review payload.

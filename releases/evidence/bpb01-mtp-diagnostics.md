# BPB01 MTP diagnostic promotion

Hardware evidence (Brutus, 2026-10-08, pin b11474).

Build `deploy-v4-diag` compiled clean. With
`BIGCHERRY_DRAFT_TRACE=1 BIGCHERRY_SPEC_TIMING=1 BIGCHERRY_DRAFT_TIMING=1` the log contained:

- 92 `DRAFT_TRACE` lines;
- 23 `SPEC_TIMING` lines;
- 23 `DRAFT_TIMING` lines;
- 0 error lines.

Greedy text was identical to production: md5 `a34a34c1f48d`.

Promotion uses the QFP18 lightweight tier: current-pin activation/neutrality evidence plus package mechanics and
patch-lint; no validation.toml/evidence/validation.json campaign is required.

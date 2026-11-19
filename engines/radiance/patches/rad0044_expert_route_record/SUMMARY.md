# rad0044_expert_route_record

**Status:** diagnostic
**Plan item:** VTP44

## Summary

`--expert-route-record FILE` writes the experts each token's router chose, for every routed trunk layer, when the run ends. A KLD run on a routed model cannot rank KV caches because any change in the arithmetic moves about 1.1% of the choices (FLASHNEXT-ROUTING.md); to score a cache alone the measured run has to take a reference run's choices. This is the record half; replay is a later patch. Manifest field `engine.tuning.expert_route_record` (contract amendment 21).

## Outcome

Qualified as a diagnostic tool (2026-11-20, record in `evidence/promotion.json`). Run on Flash-Next (hcbf16, TP2): `--expert-route-record` writes the experts each token chose (4,163 tokens x 48 layers x top-10) and the file's per-layer histogram equals the `--expert-stats` counts of the same run in all 48 layers (docs/evidence/2026-11-16-vtp44-generic-kv-layouts/FLASHNEXT-REPLAY.md, flashnext-replay/). Lab tool for scoring KV caches; it adds nothing unless the flag is given.

## Detail

- The recording rank (rank 0; every rank runs the full router) copies each routed trunk layer's reported ids, stream-ordered, into a staging buffer before the next layer's router overwrites the shared ids buffer, and appends them to a host table at the end of the pass. Routed layers are those with expert weights, so a mixed dense/MoE model records only its routed layers; their indices are in the header.
- Only one-sequence prefill passes without drafting are recorded, so rows are tokens in order. Every token position must continue the previous pass, or the pass restarts at 0 for the next document (the file lists the row where each document starts; a re-executed document is indistinguishable from a new one, so replay must check document lengths against its own run). A pass that reports no routing once recording has begun, a layer missing from a pass, or more than 2 GiB of ids marks the file incomplete with the first reason.
- Recorded passes (the step tape) are off, as for the calibration taps: a played pass would not re-run the copy.
- `--max-num-seqs` other than 1 is refused at start. A stale file at the path is removed at start. At shutdown the file is written to a per-process temporary name and renamed; a failed run, an incomplete record or a write failure fails the run (and an incomplete file carries no ids).
- Format: `RADROUTE1` line, one JSON header line (layers, routed, top_k, n_expert, rows, tp, max_tok, complete, why, doc_starts, doc_hashes (each document's FNV-1a 64 over its token ids, so a replay can refuse other prompts); model text with quote, backslash and control characters replaced), then int32 ids for each routed layer in turn.
- Review: a GPT review (9 findings) led to the routed-layer mask, the per-token position check, the zero-report check, the failed-run and I/O failure status, the stale-file removal, header escaping, the per-process temporary name and the size limit.

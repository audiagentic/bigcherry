# rad0044_expert_route_record

**Status:** untested
**Plan item:** VTP44

## Summary

`--expert-route-record FILE` writes the experts each token's router chose, for every routed trunk layer, when the run ends. A KLD run on a routed model cannot rank KV caches because any change in the arithmetic moves about 1.1% of the choices (FLASHNEXT-ROUTING.md); to score a cache alone the measured run has to take a reference run's choices. This is the record half; replay is a later patch. Manifest field `engine.tuning.expert_route_record` (contract amendment 21).

## Outcome

Not yet run. Expected on Flash-Next (hcbf16) over the 4,163-token run: a complete file of 4,163 rows x 48 layers x top-10, whose per-layer expert counts equal the `--expert-stats` counts of the same run.

## Design

- The recording rank (rank 0; every rank runs the full router) copies each routed trunk layer's reported ids, stream-ordered, into a staging buffer before the next layer's router overwrites the shared ids buffer, and appends them to a host table at the end of the pass.
- Only one-sequence prefill passes without drafting are recorded, so rows are tokens in order; positions must continue, or restart at 0 for the next document (the file lists the row where each document starts). Anything else keeps the first reason and marks the file incomplete.
- Recorded passes (the step tape) are off, as for the calibration taps: a played pass would not re-run the copy.
- `--max-num-seqs` other than 1 is refused at start. The file is written to a temporary name and renamed; after a step-loop error it is written incomplete, without ids.
- Format: `RADROUTE1` line, one JSON header line (layers, top_k, n_expert, rows, tp, max_tok, complete, why, doc_starts), then int32 ids layer by layer.

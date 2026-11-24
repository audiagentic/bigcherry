# rad0064 — uneven tensor-parallel ownership with a vocabulary-weighted head (VTP57)

**Status:** validated
**Plan item:** VTP57

## Summary

The uneven ownership split of the 27B layout (two XTX and one R9700) refused any container whose output head is stored as width classes: it read only a plain `output.weight`. The three-rank vocabulary-weighted head gives every rank the same number of rows per class (a multiple of 256 x 3, padded with dropped rows), so the classes shard by rows exactly as in the even split; only the vocabulary-sized tensors need the explicit span of the rank's vocabulary share: the token table and the id-to-row map. This patch declares those two spans instead of refusing, and leaves the plain-head path unchanged. Requires rad0019.

## Outcome

Smoke on the three-rank layout with the engine built from this patch: serve starts, the canary texts are byte-identical to the two-XTX run, 1 stream 29.85 ms/step, draft acceptance 0.958. The plain path is unchanged: IQ8R on qwen27b-tp3 gives identical KLD and top-1 on the stock 2020.8.7 release and on the swap root. The IQR container on three ranks: mean KLD 0.001906, top-1 96.82% (IQ8R 0.001511, 97.27%). Follow-up: it belongs in serving-core once validated.

## Detail

One anchored edit in the qwen35 architecture file (the ownership block of the vocabulary declare): with `tp_owned` and a weighted head it declares `rad_weight_shard_span` of the token table and of `output.vrow` over this rank's vocabulary share; with a plain head it declares the token table and `output.weight` spans exactly as before; with neither it refuses. The unit test applies the patch to the pinned source behind its requirement and compares the text before and after.

# 1210_rd26_bitidentical_decode_verify_standalone: Decode vs speculative-verify bit-identity, base-standalone hunks (RD26a)

**Status:** untested
**Group:** rdna-boosts
**Plan item:** RD26

## What it does

Ports the two hunks (MMVF decision for ne11<=8 in ggml-cuda.cu, and an n<=8 vs n<2 rejection fix in llamafile/sgemm.cpp) from a five-commit fork cluster that make decode (n_q=1) and speculative-verify (n_q=n_draft+1) batches produce bit-identical logits, restricted to the hunks whose pre-images anchor cleanly on the framework base alone.

## Why

Bit-identical decode/verify logits are a soundness precondition for speculative acceptance checks; the remaining three hunks of the cluster are deferred because their pre-images depend on code introduced by patches 1202 (RD04) and 1203 (RD05/06), so they will be added once those are benched and retained.

## Upstream / provenance

Ported from a five-commit stew675-rdna-boosts fork cluster (93510434f, b2655d381, d152888fc, plus RD26b commits 10b83d6b2/6cdf5aff9, https://github.com/stew675/llama.cpp). Not merged into ggml-org/llama.cpp master.

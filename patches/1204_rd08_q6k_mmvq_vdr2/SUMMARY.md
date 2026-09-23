# 1204_rd08_q6k_mmvq_vdr2: Q6_K mmvq VDR=2 decode kernel (RD08)

**Status:** rejected
**Plan item:** RD08

## What it does

Adds a vdr2 Q6_K vec_dot entry point that processes both 8-element chunks of a Q6_K dot product in one call, amortizing ql/qh/scales/d8 loads over 4 dp4a ops instead of 2, and switches get_vec_dot_q_cuda's Q6_K case to it. Also makes GGML_CUDA_OP_TIMING disable CUDA graph capture instead of aborting, plus adds decode-shaped perf test cases.

## Why

Decode is DRAM-bound, so halving loop iterations for the same row gives modest tg64 gains (fork: 23.35 -> 23.48 t/s d8192 on gfx1201); the fork reports the kernel is bit-identical to the VDR=1 path.

## Upstream / provenance

Ported from stew675-rdna-boosts fork commit 4591cc980 (https://github.com/stew675/llama.cpp). Not merged into ggml-org/llama.cpp master.

## DEMOTION (2026-09-23, PA40)

The sections above describe the original claim and are preserved unchanged.
The isolated RD08 campaign (retry-4, Brutus gfx1201 device 2, revision
`df1f91e3`, pin `28ff0958`/b10901) completed with full evidence and failed
its declared promotion threshold:

- backend-reference correctness (15 rows, worst error `2.598e-05` < `5e-4`)
  and activation PASS;
- decode effect `+0.204%`, CI95 `[+0.094%, +0.330%]`, required CI95 low
  `>= +0.3%` -> FAIL;
- prefill effect `-0.089%`, CI95 `[-0.162%, -0.007%]` (small regression).

Evidence: `campaign/producer-execution.json` SHA-256
`a02d50e61a2bcf9edbc5acdb7d8c4a013eb223fd89a923a1af4885d7234b4800`;
performance `1f7fcf63177d25dc60477e312ab83edf4bd4655be4eefbf4d8d5ae5e0dae144b`;
correctness `56d808bce774f0babff2a2c8e3b605094a830a82c6e934ec77c5607c531cbc86`;
activation `c0a172deda977e39c5f0a3043d504e0311d1290b9769744c3fd7df7ea756278f`.
Recorded in PA40's DoD matrix.

Decision (GPT lifecycle review `req_f34f50a25c6240fe`): the exact candidate
was conclusively measured and missed its threshold, so `rejected` is more
truthful than leaving it `untested` with a deferral. No threshold change; the
evidence is preserved. Any future VDR=2 attempt returns as a new identity with
fresh evidence. Kept in `[experiment.rd08-only]` for reproducibility.

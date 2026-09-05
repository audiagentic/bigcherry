# 1244: N-way fused-root internal allreduce

**Status:** untested
**Group:** gpu-collectives
**Plan item:** GP11

> DO NOT PROMOTE. A real completion request produces GARBAGE TEXT — an open
> correctness bug in the llama.cpp integration (see "Known defect" below).
> The throughput number recorded during the smoke test is not a result.

## What it does

Generalises patch 1001's internal allreduce beyond the hardcoded two-device
case to an N-way fused-root reduction: device[0] is the root, every other
device contributes into it, and the result is broadcast back. Gated narrowly —
F32 tensors only, and only for sizes already eligible for the chunked kernel.
Also removes a second hardcoded `n_backends == 2` assert in `ggml-cuda.cu`'s
comm dispatcher that the first fix missed.

Origin is local (bigcherry-original): this generalises patch 1001, which is
itself local, so there is no upstream commit to cite as provenance.

## Why

The existing internal allreduce refused to engage with more than two devices,
so any 3+ GPU configuration fell back to the host-staged path. The
`gp10-collective-harness` validated the fused-root design against synthetic
buffers before it was wired into production here.

## Known defect (blocking)

Real-hardware smoke test on 3 GPUs: no crash, the pipeline correctly reports
3 devices, and tg32 rose from 20.4 to 38.3 t/s. **But a real completion
request returns garbage text.** The harness's synthetic-buffer validation did
not catch this, which is itself the finding: buffer-level numeric agreement is
not sufficient evidence that a collective is correct inside real inference.

Until that is resolved, the throughput figure above must not be cited as a
measured gain — it is throughput from a run that produced wrong output.

## Requires

`1001_hip_internal_allreduce`.

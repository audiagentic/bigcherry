# 1244: N-way fused-root internal allreduce

**Status:** landed; qualification pending
**Group:** gpu-collectives
**Plan item:** GP11

> Do not promote yet. The production integration correctness bugs found during
> initial smoke testing were fixed and later harness/full-stack evidence is
> positive. GP11 still requires the remaining qualification gates below.

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

## Qualification status

The initial real-inference garbage-output result was traced to two integration
bugs: a missing per-block arrival offset and cross-device mapped-pointer alias
reuse. Both were fixed and verified. Subsequent evidence includes clean
high-repetition harness validation, real full-stack MTP correctness, alternate
root validation, and a positive N=3 decode result.

Patch metadata remains `state = "untested"` until GP11 completes its declared
qualification. Remaining gates are a controlled long soak, supported root /
topology / size coverage, provider and threshold telemetry, and controlled
comparison against RCCL and an unmodified baseline. Do not cite an unconditional
promotion or generalize the measured result outside the qualified envelope.

## Requires

`1001_hip_internal_allreduce`.

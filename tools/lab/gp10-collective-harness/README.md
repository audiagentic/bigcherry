# GP10: base-level collective transform harness

Fast, sub-llama.cpp iteration on candidate AllReduce transforms -- real
HIP/GPU correctness + latency numbers without a model load or a full
llama.cpp build cycle. See `docs/planning/active/gpu-collectives/GP10.md`.

This is the FAST, CHEAP, EARLY-ITERATION tier. A pass here is encouraging,
not sufficient, evidence -- a candidate transform still needs to go through
GP07 (RCCL-candidate qualification, if it touches RCCL) and GP08 (full-stack
OPTIMIZE comparison against required baselines) before it's a real result.

## Primitives

- `nway_star_allreduce.cpp` -- N-way generalization of patch
  `1001_hip_internal_allreduce`'s pairwise pinned-host chunked-kernel
  mechanism (`vendor/llama.cpp/ggml/src/ggml-cuda/allreduce.cu`). Each rank
  writes its contribution to its own pinned host slot, signals an arrival
  token, spins on every OTHER rank's token (generalizing the original's
  single-peer spin), then reads and sums all N-1 peer slots. This is GP11's
  first real go/no-go signal: does the pairwise mechanism's decode-latency
  win plausibly survive at N=3 before committing to a full ggml-cuda.cu
  implementation.

GP09's pinned-host bridge primitive (device-3's contribution crossing the
slow PCH-routed link once via pinned host memory into an RCCL-safe subset)
is not yet implemented here -- add it as a second small `.cpp` file in this
directory when GP09 is picked up, following the same pattern (no shared
plugin/callback ABI until a second real transform actually needs one, per
this item's own scope note).

## Build

```
hipcc -O3 -std=c++17 -o nway_star_allreduce nway_star_allreduce.cpp
```

## Run

```
./nway_star_allreduce --devices 0,1,2 --elements 4096,30720,2621440 --reps 20
```

`--devices` is a comma-separated list of HIP device ordinals (2..8 devices).
`--elements` is a comma-separated list of F32 element counts to sweep (GP04's
real production signature sizes are a good default: small ~30720, large
~2621440). `--reps` controls how many timed repetitions after one untimed
correctness-checked warmup rep.

Output is one line per element count: correctness (checked against an
independent CPU-computed reference sum, F32 accumulation, generous but real
epsilon) and median/p90 latency in microseconds (measured via HIP events
around each rank's kernel, not wall-clock-around-the-process).

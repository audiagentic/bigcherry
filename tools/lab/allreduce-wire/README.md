# allreduce-wire

Pristine llama.cpp b11233 (d77dd0806dc26fc418273ef99f88da11239ca41b, MIT) excerpts of the
HIP/CUDA internal AllReduce pipeline, committed so reviewers without vendor access can author
anchored patches for the host-staged compressed AllReduce wire (q8_0 / bf16 / f16 decoupled
from the p2p provider, PGC04/PGC07).

- vendor-b11233/allreduce.cu, allreduce.cuh: full files.
- vendor-b11233/ggml-cuda.cu.comm-950-1260.txt: ggml-cuda.cu lines 950-1260 (comm context,
  provider init/selection, internal try_allreduce dispatch).

Refresh on a pin bump: `git -C vendor/llama.cpp show <sha>:ggml/src/ggml-cuda/<file>`.

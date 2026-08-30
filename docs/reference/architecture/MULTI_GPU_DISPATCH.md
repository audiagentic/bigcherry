# Multi-GPU dispatch: layer split, tensor split, and reduction providers

What actually runs when you choose `-sm layer` vs `-sm tensor`, and which
reduction provider (RCCL / internal / META) handles tensor-split's
cross-device sum on a given hardware topology. Established through real
hardware investigation on Brutus (HI84, HI85, HI132, HI134) — see those
plan items for the full evidence trail.

## `-sm layer` (pipeline/layer split)

Model layers are partitioned sequentially across GPUs: each GPU owns a
contiguous block of layers, runs them fully, and passes activations to the
next GPU via a simple point-to-point copy. No partial results ever need to
be summed across devices, so this path **never constructs a `comm_ctx` at
all** — RCCL, internal AllReduce, and META are all irrelevant here.

Confirmed directly (HI134): a real `-sm layer` profiling run on a
heterogeneous 3-device set showed zero reduction-related activity and none
of the copy anomalies seen under `-sm tensor` on the same device set.

## `-sm tensor` (tensor/row split)

Individual matmuls are split *within* a layer across GPUs, so each GPU
computes a partial result that must be summed back together — this needs a
real cross-device allreduce on every matmul that's split this way.

Provider selection happens once, at backend-construction time, in
`ggml_backend_cuda_comm_init()` (`ggml/src/ggml-cuda/ggml-cuda.cu`),
controlled by `GGML_CUDA_ALLREDUCE` (default: unset, which tries RCCL first
on Linux):

1. **RCCL** — the first choice by default. Works **only when every
   participating device shares the same GPU architecture**. On Brutus, that
   means only the `{0,1}` dual-XTX pair (both gfx1100/RDNA3). Any
   architecture mismatch — even a single RDNA3+RDNA4 pair — reliably
   crashes RCCL's collective kernel dispatch (HI85's finding: the kernel
   binary compiled for one rank's gfx target cannot execute on another
   rank's device; a hard `SIGABRT`/"invalid device function" fault inside
   `ncclGroupEnd()`, not a catchable NCCL error code, and independent of
   `NCCL_ALGO`/`NCCL_PROTO` — this is a real upstream RCCL/ROCm limitation,
   reproduced identically on stock unpatched llama.cpp, not a BigCherry
   defect).

   **Fail-closed policy (patch 1225, HI85):** the comm-init guard compares
   every participating device's compute-capability identifier *before*
   calling `ncclCommInitAll`. On a mismatch, it does **not** silently
   substitute a different provider — it `GGML_ABORT`s with a clear
   diagnostic naming the architectures involved and the remediation options
   (force `GGML_HIP_REDUCE_PLAN=meta`, use `-sm layer`, or restrict
   tensor-split to a same-architecture subset). This guard is only ever
   reached via an explicit `-sm tensor` request (META's `comm_init` hook is
   exclusive to `SPLIT_MODE_TENSOR`), so reaching it always means the user
   explicitly asked for tensor-split — silently downgrading to a different
   reduction path there would change behavior/semantics behind the user's
   back, which this project treats as unacceptable.

2. **internal AllReduce** — a P2P-based path. Only supports exactly 2
   devices, and needs real peer access. Brutus has **no PCIe P2P bridge for
   any device pair, homogeneous or heterogeneous** (HI84), so this path
   declines too, on every topology tested.

3. **META (generic butterfly reduction)** — the actual fallback that works.
   Verified correct for every heterogeneous-architecture group and device
   count 2–4 on this hardware (HI18/HI84's correctness-evidence probe).
   Since there's no P2P anywhere on this box, every META transfer is
   **host-staged** (device→host→device), confirmed directly via real
   `rocprofv3` memory-copy traces (HI134): the reduction's `FOLD` /
   `BUTTERFLY` / `COPY_BACK` stages all show up as
   `HOST_TO_DEVICE`/`DEVICE_TO_HOST` copy pairs, never true device-to-device
   transfers.

### Which reduction provider actually runs, in practice, on this hardware

| Device group | RCCL | internal | META |
| --- | --- | --- | --- |
| `{0,1}` (dual XTX, same arch) | ✅ works | — | not needed |
| Any 2-device mixed-arch pair | ❌ fails closed (patch 1225) | ❌ no P2P | ✅ used |
| Any 3+ device group (any arch mix) | ❌ fails closed | ❌ only supports 2 devices | ✅ used |

So: **on Brutus, META is what's actually doing the reduction work any time
`-sm tensor` is used across anything other than the dual-XTX pair.**

## HI134's finding: META's cost is already near the hardware limit

A dedicated investigation (HI134, closed 2026-08-28) profiled META's real
per-reduction cost on the real heterogeneous topologies ({0,1,2},
{0,1,2,3}) using `bigcherry profile-campaign` plus purpose-built
attribution instrumentation (patch 1242, an optional trace hook in
`ggml-backend-meta.cpp`'s `allreduce_fallback`). Result: META's real
transfer traffic is small, entirely host-staged as expected (no P2P to
exploit), and scales linearly with the real reduction count — no
demonstrated implementation-level bottleneck. No tuning change was made or
justified.

A separate, real anomaly was found during that investigation
(`__amd_rocclr_copyBufferRectAligned`, a multi-millisecond intra-device
copy that appears whenever a 3rd heterogeneous-architecture device joins a
tensor-split group) — decisively shown to be **unrelated to META's
reduction transfers** (it's a same-device copy on the *other* GPUs, not the
newly-joined device, and its call count doesn't correlate with the real
reduction count). Tracked separately as HI135; do not assume a connection
to reduction/allreduce without new evidence.

## Practical guidance

- Production topology today is `{0,1}` (dual XTX) with `-sm tensor` — RCCL
  handles it natively, no META involvement.
- Any tensor-split deployment spanning different GPU architectures on this
  box will hard-abort under the default `GGML_HIP_REDUCE_PLAN=auto` /
  `GGML_CUDA_ALLREDUCE` settings unless META is forced explicitly
  (`GGML_HIP_REDUCE_PLAN=meta`), per patch 1225's fail-closed policy.
- `-sm layer` remains unaffected by any of this regardless of architecture
  mix, since it never invokes a reduction provider at all — it's the
  simplest path for a genuinely heterogeneous deployment if tensor-split's
  per-layer reduction isn't required.

See also: [PROFILING.md](../tooling/PROFILING.md) for how to reproduce
these measurements, and HI85/HI132/HI134/HI135 in
`docs/planning/` for the full evidence trail.

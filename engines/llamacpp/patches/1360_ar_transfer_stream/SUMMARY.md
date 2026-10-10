# 1360_ar_transfer_stream

**Status:** untested
**Plan item:** QFP49

## What it does

`BIGCHERRY_AR_TRANSFER_STREAM=1` splits the existing NCCL/RCCL AllReduce into a begin half and an end half.
Pre-work remains on each device's compute stream. Begin records an input event after that work, makes a dedicated
per-device transfer stream wait on it, enqueues the grouped AllReduce on the transfer streams, and records an output
event. End makes each compute stream wait on the output event and then enqueues the existing BF16-to-FP32 post-work.

The current `ggml_backend_cuda_comm_allreduce_nccl` entry point calls begin and end back to back. That preserves the
existing dependency chain while exposing the seam QFP49 needs to later run independent batch-B compute between
begin(A) and end(A). The flag is read once and defaults off; with it off, execution falls through to the pre-existing
AllReduce body unchanged.

Transfer streams and events are communication-context resources, not per-call resources. No host thread is added.

## Shared-state inventory

- F32 tensor, small path: the compute stream is the producer (including the zero-fill for non-compute ranks);
  `e_in` orders the transfer stream after that producer; RCCL reads/writes the tensor in place; `e_out` orders
  every later compute-stream consumer after the reduction.
- BF16 temporary, large path: the compute stream writes it via F32-to-BF16 or zero-fill; `e_in` orders RCCL after
  that write; RCCL reduces it in place on the transfer stream; `e_out` orders the compute stream before
  BF16-to-FP32. The pooled allocation lives in the begin/end state until end has enqueued the post-work, matching
  the lifetime rule of the original monolithic function.
- F32 tensor, large path: read by the compute-stream conversion before `e_in`, then written by the
  compute-stream BF16-to-FP32 conversion only after the `e_out` wait.
- Transfer streams: one non-blocking stream per device, created lazily once per communication context and destroyed
  by that context.
- Events: two reusable input/output slots per device, created with the transfer stream resources and destroyed by
  the communication context. Two slots leave room for a later QFP49 slice to have A and B reductions outstanding.
- Slot bookkeeping (`bc_ar_transfer_busy`, `bc_ar_transfer_next_slot`) is host-side state used by the same caller
  thread as the communication path. No worker or helper thread is introduced.

The safety chain for every buffer is therefore compute producer/pre-work -> `e_in` -> transfer-stream RCCL ->
`e_out` -> compute-stream consumer/post-work.

## Composition

The patch declares the AllReduce-neighbourhood dependencies requested by QFP49:
`0840_hybrid_allreduce_dispatch`, `1250_nro01_allreduce_q8_wire`, and `1291_ar_cpu_root`.
It also directly requires `0910_feature_sets` because the patch registers its environment flag there.

The `ar-transfer-stream` recipe experiment adds 1360 on top of the production source; dependency resolution supplies
the declared dependency closure. It is not in the production patch set.

## Validation required

This package is intentionally `untested`. Required qualification is: offline mechanics and patch-lint; marker
activation only with the flag on; bit-identical output against flag-off; no slowdown when begin/end are called back
to back; and hardware traces showing the RCCL exchange on the transfer stream before any staggered-batch work is
attempted.

## API assumptions to qualify on hardware

The pinned HIP shim exposes the CUDA-named stream/event calls used by this file, including
`cudaStreamWaitEvent -> hipStreamWaitEvent`. RCCL accepts a stream argument for collective enqueue. The enabled
path uses one stable dedicated transfer stream per device rather than alternating streams between calls. Hardware
qualification still has to prove that this RCCL/ROCm combination preserves ordering and permits the intended
compute/collective overlap without hidden serialization.

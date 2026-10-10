# 1360_ar_transfer_stream

QFP49 piece 1: put the RCCL exchange on one dedicated transfer stream per participating device, fenced from the
existing compute stream with events.

- State: `untested`
- Flag: `BIGCHERRY_AR_TRANSFER_STREAM=1` (default off, read once)
- Marker: `BIGCHERRY_PATCH_HIT patch=1360_ar_transfer_stream`
- Experiment: `ar-transfer-stream`
- Host threads added: none

With the flag enabled, `ggml_backend_cuda_comm_allreduce_nccl` calls
`bc_1360_ar_transfer_begin` and `bc_1360_ar_transfer_end` back to back. Begin leaves the reduction ordered after
the compute-stream pre-work; end restores the dependency from the reduction to compute-stream consumers and the
large-path BF16-to-FP32 conversion. The begin/end state owns the pooled BF16 temporaries until end.

The communication context owns one non-blocking transfer stream per device and two reusable input/output event slots
per device. No stream or event is created per AllReduce call. With the flag off, the original NCCL/RCCL body remains
the executed path.

This slice does not implement staggered half-batches, paired graph scheduling, or a second compute-buffer set.

## Qualification

Hardware-free mechanics should prove composition after the production set/dependency closure, idempotence,
fail-closed anchor drift, event ordering, preservation of the original flag-off body, and environment documentation.

Hardware qualification must then show:

1. the marker only in the enabled arm;
2. greedy/output identity against the disabled arm;
3. no throughput regression when begin/end are back to back;
4. RCCL runs on the dedicated transfer stream and the compute stream waits on the completion event;
5. no ordering failures across repeated reductions.

The pinned HIP compatibility header maps the CUDA-named stream/event APIs used here to HIP, including
`cudaStreamWaitEvent` to `hipStreamWaitEvent`. RCCL's collective API takes a stream argument; actual overlap and
any hidden serialization remain hardware questions for this untested package.

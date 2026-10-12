# 1360_ar_transfer_stream

**Status:** evaluated
**Plan item:** QFP49

## Shelved as evaluated (owner decision, 2026-10-12)

Default off and in no build. On its own the flag is neutral: Flash-Next prefill and decode with it on are inside the
run-to-run spread (the first reading of a 19% cost was a bimodal run, corrected). Its purpose was to let the cross-card
sum overlap the next half-batch of 1361, and 1361 never produced a correct real prompt (PR #147, closed unmerged).
The standalone probe (tools/lab/hip-probes) bounds what that overlap could hide at Flash-Next's ~44 kernels a step:
42% of the sum at 40 kernels, 13% at 80; and the b11552 census shows the cards 91-92% busy in a median step. Kept as
the record of the mechanism; not a candidate for promotion as it stands.

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

The patch hard-requires `0840_hybrid_allreduce_dispatch` and `1291_ar_cpu_root`, which shape the composed
communication context/provider path, and `0910_feature_sets` because 1360 registers its environment flag there.

`1250_nro01_allreduce_q8_wire` was also inspected as requested. Its `ggml-cuda.cu` edits are downstream of the
NCCL helper/anchors changed here and do not alter them, so 1360 does not depend on its output. A hard dependency would
also make 1360 unresolvable because 1250 currently requires rejected `1252_nro03_allreduce_p2p_provider`.

The `ar-transfer-stream` recipe lists 1360 alone: its requirements are in the production set, and an overlay may not
repeat a module of its base. 1360 is not in the production patch set.

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

## First hardware run (2026-10-11)

Flash-Next production profile, 24K (38,725-token prompt), build `b-ar1` (production + 1360), run `tsar`, ABBA on one
binary, A = flag unset, B = `BIGCHERRY_AR_TRANSFER_STREAM=1`:

| | Prefill t/s | Decode t/s | Greedy md5 |
|---|---|---|---|
| A (off) | 1424.1 / 1424.4 | 86.7 / 92.4 | a789c151 |
| B (on) | 1411.9 / 1438.3 | 90.0 / 86.0 | a789c151 |

The marker `BIGCHERRY_PATCH_HIT patch=1360_ar_transfer_stream` appears in both B passes and in neither A pass. Text
identical, speed unchanged within the run's spread: called back to back, the split sum is neutral, which is what
this slice has to show. It is an enabler for the paired schedule (QFP49), not a gain on its own.

Outside the engine (`tools/lab/hip-probes/rccl_stagger_probe.cpp`, run rs1): RCCL takes a stream other than the
compute stream on the same communicator, and with two half-batches staggered around it 84% of the sum's cost is hidden.

## Sequential-path cost observed in b-stg11

The earlier production-profile ABBA above was neutral, but the smaller real-library qualification path is not.
Flash-Next, ctx 65536, 4096 pseudo-random prompt tokens, `-b 512 -ub 256`, eight calls, no stagger pair:

- transfer stream off: 1093.7 t/s; `llama_decode` held the host 3591 / 3745 ms;
- `BIGCHERRY_AR_TRANSFER_STREAM=1`: 882.9 t/s, -19%, logits bit-identical; calls after the first are about
  470 ms rather than about 340 ms;
- trace per XTX: sum resident mean rises from 0.91 to 1.01 ms. Off has every kernel on the compute stream; on moves
  all 1528 sum kernels to the transfer stream, but both traces still have 0.0% sum+compute overlap.

Therefore the flag alone is a cost on a sequential path and must not be treated as a neutral optimization.
The 0.10 ms increase in measured sum-kernel residency explains only part of the loss. Event record/wait latency is
not included in kernel residency, so the remaining cost is in the producer-event -> transfer-stream wait -> RCCL ->
output-event -> compute-stream wait dependency/enqueue path and resulting idle/serialization; the current trace
cannot split those components further. The existing F32->BF16 pre-work and BF16->F32 post-work remain on the compute
stream, so they are not newly moved to the transfer stream.

There is no safe event to remove from the begin/end seam: the input event orders RCCL after its compute producer and
the output event orders the consumer/post-conversion after RCCL. Streams/events are already persistent and pooled,
so allocation is not the steady-state cost. A cheaper sequential path is the original monolithic RCCL path; the
transfer seam should be paid only when independent compute is actually inserted between begin and end. 1361 calls
the begin/end seam directly for its staggered reductions. Further reduction of the standalone-flag overhead would
need a ROCm API/host trace to distinguish event/enqueue latency from RCCL dedicated-stream behavior.

Later reading of the same configuration (QFP49): repeated runs of this small real-library path are bimodal with the
flag on or off, so the 19% above is one mode against the other, not a cost of the flag; the production-profile ABBA
stays the reference (neutral). The paragraph on where a seam cost would sit is kept for a later revisit.

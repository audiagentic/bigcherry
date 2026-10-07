---
id: QFP39
order: 39
plan: patching-qwen-flash-next
state: deprecated
created-at: '2026-10-07T00:39:52.781091+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# One-shot PCIe P2P AllReduce for small tensors: re-evaluate 1252 on the XTX pair

## Description

External report: direct peer traffic for small AllReduce tensors reduced a reported collective from ~57 us to ~9 us. Different hardware/provider/topology: hypothesis only.

BigCherry already has the correct first experiment: 1252 is an opt-in direct-P2P provider for a two-device internal HIP AllReduce, with a content-checked peer probe because API success was previously insufficient evidence on gfx1100. It is untested. 1275 separately experiments with fixed latency in the mapped-host provider and is only evaluated. 1277 can expose the production AllReduce size/provider population.

Do not write another provider until 1252 is actually tested. For the three-card Flash-Next topology, 1252's current two-participant path cannot by itself accelerate a collective spanning all three target devices; any N-device extension must be capability-driven and remain in 1252 because it is the same direct-P2P provider.

## Steps

1. Gate 0A: use 1277 on both production models to census AllReduce size/call populations at decode and prefill.
2. Gate 0B: probe the complete ordered peer-capability matrix on the physical target devices, including actual content transfer:
   - XTX↔XTX;
   - each XTX↔R9700;
   - sidecar only if it ever participates in the same collective.
3. First hardware experiment: current 1252, unchanged, on the 27B two-XTX collective.
4. Compare against pristine/current provider and the best separately qualified 1275 setting.
5. If and only if 1252 wins and three-card small AllReduce remains material, design/qualify the N-device extension **inside 1252**.
6. Fully separated ABBA on the topology for which each path is eligible.

## Detailed Solution & Technical Design

### A. Two-participant qualification: use 1252 as written

No new package.

Eligibility remains:

- HIP backend;
- direct-P2P flag enabled;
- collective has exactly two participants;
- bidirectional peer-access API probe succeeds;
- byte-checked transfer probe succeeds;
- tensor fits the provider's direct-copy scratch limit/type contract.

Do not hard-code ROCm0/1. The current pipeline's participant list determines the device pair, so this algorithm remains valid for any two participating devices.

For the 27B two-XTX model, measure whether one source-current peer copy in each direction plus local add beats the current mapped-host/copy provider at the actual decode tensor sizes.

### B. Small-tensor crossover

1252 currently chooses P2P whenever its provider is enabled/available. That is too broad for promotion without a crossover.

If hardware shows a size-dependent win, improve **patch 1252** with:

`BIGCHERRY_AR_P2P_MAX_BYTES=<N>`, default 0/disabled during qualification.

The direct provider is selected only when:

- all existing P2P qualification passes;
- `nbytes <= max_bytes`;
- normal internal AllReduce is otherwise eligible.

Above the threshold, use the current provider unchanged. Derive N from 1277 + a size sweep, not from the external 9 us result.

Do not combine 1275 slot-sync geometry into the P2P path. 1275 remains an independent control arm.

### C. Three-or-more participant extension

The production Flash-Next target normally has three target devices. Do not pretend the two-device 1252 path covers it.

Only pursue an N-device extension after:

1. all ordered pairs participating in the collective pass the same byte-checked peer probe;
2. 1277 shows small three-device AllReduce is a material decode cost;
3. a standalone microbenchmark shows direct peer traffic wins despite the R9700 PCIe x4 link.

This remains an **improvement to patch 1252**, not a new package.

A safe first generic design is explicit destination scratch, not remote atomics:

- allocate N source slots in each destination device's P2P scratch for the qualified maximum small tensor;
- each source uses its own source-current peer stream to copy its contribution to every other participant's source slot;
- local contribution occupies the local slot;
- destination compute waits for all source-copy completion events and performs one deterministic local reduction across source slots in participant-list order;
- write the full reduced tensor to the destination's normal output;
- reuse the existing slot/event pool for lifetime protection.

This is O(N²) peer bytes and is intended only for very small tensors. It is topology-agnostic in correctness but should fail closed when the all-pairs peer clique is unavailable or a measured threshold says it loses.

Do **not** implement remote floating atomics or unordered peer accumulation: they make reduction order and completion difficult to prove. Do not assume a ring/path based on device ordinal.

A later reduce-scatter/allgather slice protocol is only justified if this simple N-way small-tensor path proves peer latency is worthwhile but O(N²) traffic limits the three-card result. That would require its own plan review before code.

### Synchronization contract

The provider must preserve the existing AllReduce call contract:

- application-stream writes to each source complete before peer transfer;
- peer transfers complete before destination reduction reads scratch;
- destination reduction completes before downstream graph work sees output;
- scratch slot is not reused until every dependent transfer/reduction has retired.

No device-wide `cudaDeviceSynchronize()` in the steady path. The startup content probe may synchronize because it runs once.

Cross-device event waits must be hardware-validated; if HIP rejects or mishandles them on a pair, fail closed rather than inserting a per-call host synchronization that would erase the latency objective.

### Data types and reduction order

Use the same source/destination/wire type selected by the current internal AllReduce. P2P changes transport, not precision policy.

For N=2, preserve the existing deterministic local+peer add order. For N>2, reduce source slots in fixed participant-list order. Expect normal floating-point equivalence to current provider rather than guaranteed bit identity if the provider's reduction tree/order differs.

No q4 KV or KV datatype change is involved.

### Meta split / arbitrary topology

The provider receives the participant list from the collective; it must not infer it from model type, tensor split percentages, attention split, or physical ordinals.

Thus:
- two-participant collectives may use qualified 1252 on any working pair;
- N-participant collectives use the generic extension only when the full participant clique qualifies;
- otherwise current provider remains authoritative.

The MTP sidecar is irrelevant unless it is actually in the same collective; do not include it merely because it is visible to ROCm.

## Code Samples & Guidance

Existing 1252 anchors in `ggml/src/ggml-cuda/allreduce.cu`:

- `ggml_cuda_ar_pipeline` fields around `p2p_enabled`;
- `ggml_cuda_ar_p2p_probe()`;
- `ggml_cuda_ar_pipeline_init()` peer-access setup;
- `ggml_cuda_ar_allreduce_p2p_impl<T_src,T_dst>()`;
- provider selection around the existing call to `ggml_cuda_ar_allreduce_copy_impl`.

If a size gate is needed, parse it once during pipeline init and store it in the pipeline. Do not read the environment per collective.

Existing flag remains `GGML_CUDA_AR_P2P`; add only `BIGCHERRY_AR_P2P_MAX_BYTES` for qualification if a threshold is proven necessary. Default off/0.

Activation marker should include participants/size once per representative bucket without assuming ordinals, e.g.:

`BIGCHERRY_PATCH_HIT patch=1252_nro03 path=allreduce_p2p n_devices=<N> bytes=<B>`.

1275 flags remain independent:
- `BIGCHERRY_AR_SLOT_SYNC`;
- `BIGCHERRY_AR_SMALL_BLOCKS`;
- `BIGCHERRY_AR_SMALL_THREADS`.

## Files

No new package for the direct-P2P mechanism.

If qualification requires improvements, update:

- `patches/1252_nro03_allreduce_p2p_provider/patch.py`;
- its `README.md`/`SUMMARY.md`;
- its existing/new package test under `tools/tests/patch/`.

If only the mapped-host baseline wins, any change stays in `patches/1275_ar_small_latency`.

Use `patches/1277_ar_size_trace` unchanged for lab census.

Production source touched by 1252 remains `ggml/src/ggml-cuda/allreduce.cu`.

## Validation

### Offline/mechanics

- Patch-lint and exact/idempotent mechanics.
- Flag off preserves current provider.
- Capability matrix/probe test records every ordered participant pair.
- Provider dispatch tests:
  - N=1/no collective;
  - N=2 qualified/unqualified;
  - N=3 full clique and one missing directed edge;
  - size below/equal/above threshold.
- Reduction probe over F32/F16/BF16 wire/output combinations currently supported by the internal provider.
- Odd element counts and sizes around scratch/chunk boundaries.

### Activation and microbenchmark

Use 1277 for real size census.

For each relevant pair/topology, sweep representative small sizes and record end-to-end collective latency including waits/events, not only copy-engine duration.

Activation must prove:
- exact provider selected;
- participant count;
- bytes;
- fallback reason when peer qualification fails.

The existing 1252 startup content probe is mandatory. API success alone is never acceptance evidence.

### Hardware ABBA

Use `tools/lab/flash-next/queue-env-ab.sh` with complete process separation.

#### 27B dual XTX

A: production/current provider, 1252 off.
B: 1252 on with any proven small-size threshold.

Also run a separate 1275 arm before choosing the best baseline, but do not combine 1252+1275 until isolated.

Record:
- per-AllReduce latency by size;
- collectives/token;
- decode t/s at 8K and 24K/32K;
- prefill sanity;
- greedy output identity/equivalence.

#### Flash-Next three-card

Do not run a “1252 on” ABBA as evidence unless the actual collective dispatch reports an eligible path. For current two-device-only 1252, expected activation may be zero.

Only after an N-device extension passes its probe should A/B run at 8K/24K/~98K. Include the R9700 x4 pair timings and total decode critical path.

### Equivalence

Transport-only N=2 should normally preserve the same arithmetic order; require greedy identity and direct tensor correctness.

If N>2 fixed-order reduction differs from the current provider's reduction tree, state the equivalence explicitly: normal AllReduce numeric tolerance plus greedy target identity. Never accept nondeterministic output.

No multi-session contract campaign.

## Effort & Risk

Two-XTX 1252 qualification: S.
Size crossover gate: S.
Generic N-device extension: M/H.

Risk:
- current 1252 qualification: medium because peer correctness has already been problematic on gfx1100;
- N-device extension: high integration/synchronization risk, especially mixed gfx1100/gfx1201 and PCIe x4.

Expected gain:
- 27B two-XTX: medium potential if P2P probe passes and real decode tensors are in the small-latency regime;
- Flash-Next three-card: zero from current 1252; low-to-medium only if a safe generic path wins despite the R9700 link.

This ranks below exact compute-kernel work for the main three-card target until capability/latency probes are positive.

## Standards

- Reuse/improve 1252; no duplicate direct-P2P package.
- Content-check every peer direction; API capability is insufficient.
- No fixed card count or ordinal in correctness logic.
- Specialized N=2 path is conditionally valid, with generic fallback for any other participant count.
- No host/device-wide sync in steady-state qualification path.
- Deterministic reduction; no floating remote atomics.
- No q4 KV.
- No legacy/back-compat shim.
- Default off during qualification.
- External latency is a hypothesis.

## Acceptance Criteria

- 1277 establishes actual AllReduce size populations for both models.
- The complete relevant peer matrix is recorded with byte-checked transfer evidence.
- Current 1252 is hardware-tested on the two-XTX model before any redesign.
- Direct P2P beats the best current/1275 baseline at real decode sizes, or 1252 remains unpromoted.
- Three-card claims require an actual N-device implementation and activation; two-XTX evidence is not extrapolated.
- Correctness/equivalence and greedy target output pass.
- Fully separated ABBA shows a repeatable end-to-end gain on every topology promoted.

## Notes

Execution order: tenth. First priority is a cheap 1252/1277 hardware qualification on the dual-XTX model. Do not spend on the N-device extension unless the two-device result and three-card peer matrix are both favorable.

2026-10-07 REJECTED for this machine after a re-test (owner: "i dont think p2p works on this pc but you can test again before proceeding"). The GP11 diagnostics of 2026-09-04 (tools/lab/gp10-collective-harness/p2p-diagnostics) were rebuilt with the current ROCm (/mnt/vault/tmp/bc-rocm hipcc) and run on the two RX 7900 XTX (HIP_VISIBLE_DEVICES=0,1), outputs in /mnt/data/bigcherry-work/runs/p2p-retest-20261007: p2p_diag - every kernel access to peer memory ends in 'an illegal memory access was encountered'; p2p_spin - the same, a peer write is never seen; p2p_d2d - hipDeviceEnablePeerAccess returns 'invalid device ordinal' in both directions (the hipMemcpyPeer / hipMemcpy D2D copies that follow are correct, i.e. the runtime stages them itself). A one-shot AllReduce needs a kernel writing into a peer's memory, which this board/runtime does not provide. Nothing to build; 1252 stays untested and unselected. Reopen only on a different board or a ROCm release that changes peer access.

## What we already have

### b11402 / internal CUDA-HIP AllReduce

- `ggml/src/ggml-cuda/allreduce.cu` owns the internal multi-device AllReduce pipeline used by the CUDA/HIP backend.
- The pipeline already has:
  - per-device application streams/events;
  - mapped-host/small reduction machinery;
  - copy-engine/device scratch path;
  - slot/event lifecycle and typed reduction kernels.
- Meta remains responsible for deciding where an AllReduce is required; the provider only implements the collective and must not change Meta split/delayed-branch semantics.

### 1252 direct P2P provider

`patches/1252_nro03_allreduce_p2p_provider`:

- status: **untested**;
- flag: `GGML_CUDA_AR_P2P=1`, default off;
- currently gated to a two-device pipeline;
- probes `cudaDeviceCanAccessPeer` both directions and enables peer access;
- then performs an actual byte-checked bidirectional copy probe at 4 KiB, ~64 KiB, 1 MiB and 4 MiB using asymmetric nonzero patterns;
- uses source-current `cudaMemcpyPeerAsync` because earlier gfx1100 destination-current copies could report success while producing bad data;
- has per-direction source-device streams/events;
- source-pushes one device's tensor into peer scratch, then locally adds peer data into each destination;
- emits `BIGCHERRY_PATCH_HIT patch=1252_nro03 path=allreduce_p2p_source_push`.

This is the first thing to qualify on the two XTXs. Do not replace its content check with an API capability check.

### 1275 small-latency provider tuning

`patches/1275_ar_small_latency`:

- status: **evaluated**, not validated/promoted;
- keeps the mapped-host provider;
- can skip pool-wrap host event waits for a single-chunk small reduction;
- independently tunes active blocks (1/2/4/8) and threads (128/256);
- leaves the fixed arrival-ring allocation contract unchanged;
- defaults preserve existing behavior.

It addresses host-provider fixed overhead, not direct peer traffic. It is a useful baseline against 1252, not a prerequisite.

### 1277 size/provider trace

`patches/1277_ar_size_trace` is a lab-only diagnostic:

`BIGCHERRY_AR_SIZE_TRACE=<n>`

logs bytes, `ne0/ne1`, type, current provider and switch threshold for the first N AllReduces. Use it to establish the actual decode/prefill tensor-size populations before choosing a small-P2P threshold.

Finding: QFP39 is **not covered in production** because 1252 is untested. No new mechanism should be created before qualifying it.

## Change Log

- 2026-10-07T00:39:52.781091+00:00 (created-by): Created by agent
- 2026-10-07: grounded at b11402 and 1252/1275/1277; made current 1252 the first experiment, added size-crossover guidance and a topology-agnostic deterministic N-device extension gate.
- 2026-10-07T06:40:43.078327+00:00 (updated-by): Updated: section:notes
- 2026-10-07T06:40:46.637192+00:00 (state-transition): State: pending → deprecated

# Findings — things worth telling people outside this project

This is not a bigcherry changelog; that's the ledger
(`docs/releases/CURRENT_RELEASE_LEDGER.ndjson`) and it already covers our own
code. This file is for the opposite case: something discovered *while*
working on bigcherry that is about llama.cpp, ROCm/HIP, a specific model, or
hardware — genuinely notable independent of bigcherry, and worth a GitHub
issue, a forum post, or a line in release notes elsewhere.

Three shapes of entry:

- **Kernel/upstream bug** — a real correctness or crash bug in llama.cpp or
  ROCm/HIP, found via bigcherry's tooling but not caused by it.
- **Exceptional tune** — a result surprising enough that it's worth telling
  people about on its own (a huge win, a counterintuitive loser, a
  hardware-specific cliff).
- **Notable interaction** — a real but non-obvious behavior at a boundary
  bigcherry doesn't own (git/hosting, SMB/CIFS, a driver quirk) that others
  hitting the same combination would want to know about.

If you're not sure something belongs here, ask: "would this be useful to
someone who has never heard of bigcherry?" If yes, it's a candidate.

## Adding an entry

Newest entry at the top of the log, under its category heading. Copy the
template below. Write **Summary** and **Evidence** as if they were about to
be pasted into a GitHub issue on someone else's repo — no bigcherry jargon,
no internal file paths in the prose (put those in **Internal context**
instead, which stays out of anything published). Fill in **Environment**
completely enough that a stranger could reproduce it: GPU, ROCm/HIP version,
compiler, llama.cpp revision.

```markdown
### <Title> — <YYYY-MM-DD>

**Status:** internal-only | drafted | reported-upstream (<link>) | published (<link>)

**Summary:** <1-3 sentences, stands alone>

**Environment:** <GPU(s), ROCm/HIP version, compiler, llama.cpp revision/tag, OS>

**Evidence:** <exact repro steps / commands / log excerpts — enough for a
stranger to reproduce without asking a follow-up question>

**Internal context:** <ledger event id, plan item, run directory — anything
that helps *us* trace it back, kept separate from the publishable text above>
```

Update **Status** in place as it moves — don't duplicate the entry.

---

## Kernel / upstream bugs

### Default `hipStreamWaitValue32` armed on an idle stream makes kernels on another stream ~2.2x slower (gfx1100) — 2026-10-25

**Status:** internal-only (candidate to add to ROCm/TheRock#9297)

**Summary:** With ROCm's default implementation of `hipStreamWaitValue32` (the `__amd_rocclr_streamOpsWait` blit
kernel, `GPU_STREAMOPS_CP_WAIT=0`), a single unsatisfied wait armed on one otherwise idle stream makes a
latency-sensitive kernel on a different stream of the same GPU run about 2.2x longer (64 -> 140 us), with identical
ISA, identical clocks (also with `power_dpm_force_performance_level=high`) and no overlapping kernel in the trace.
Replacing that wait with any of: nothing, a cross-stream `hipStreamWaitEvent` (an AQL BARRIER_AND), or a one-wave
`s_sleep` wait kernel leaves the kernel at 65-67 us. It does not depend on the HW queue count
(`GPU_MAX_HW_QUEUES` 13/16/24/32 all slow). A plain bandwidth-bound streaming kernel does not show it; the affected
kernel is a decode GEMV doing scattered expert-weight reads. The default wait also keeps a kernel resident while
armed, so the GPU reads busy and polls host memory for as long as the wait is outstanding.

**Environment:** Linux 7.0.13 (amdgpu in-tree), ROCm 7.2.4 (HIP/ROCclr/ROCr 7.2.4), MES firmware 0x80 on the XTX. Helper GPUs: 2x Radeon RX 7900 XTX (gfx1100, PCIe 4.0 x8 each); engine GPU: Radeon AI PRO R9700 (gfx1201, PCIe 4.0 x4); no P2P between them; processes share a POSIX shm segment registered with hipHostRegister(Portable|Mapped).

**Evidence:** rocprofv3 `--kernel-trace --stats` on the helper process, one 512-token greedy decode, same binary,
only the idle extra stream's armed wait differs (`R3_MOE_WORKER_IDLE_WAIT` lab switch):
none 65.1 us / `hipStreamWaitValue32` 140.3 us / `s_sleep` wait kernel 67.3 us / sleep kernel on a third stream 66.1 us /
`hipStreamWaitEvent` barrier 67.4 us (gate/up GEMV p50); the down GEMV 35 -> 48 us; decode 96.6 -> 84.4 tok/s. The
same 2.2x appears when a second pre-queued lane class sits idle with its waits armed (8-row class alone 64 us,
8+16 134 us, 8+16+32 138 us). Related public reports: ROCm/TheRock#9297 (gfx1201, pending `hipStreamWaitEvent` on
other streams adds ~17 us a small kernel, no AMD root cause yet), vllm-project/vllm#58639 (side streams in pending
waits double the R9700 forward pass). No public report found for gfx11 with `hipStreamWaitValue32`; ROCm 10.0 release
notes mention no change to the stream-wait implementation.

**Internal context:** RAD20 (docs/planning/active/patching-radiance-rdna3/RAD20.md); runs
`${BC_WORK_ROOT}/runs/rad20-profile-{idlewait-10252147,classes-10252111,queues-10252039,largewait-10251824,noevent-10251753}`;
harness `tools/lab/radiance/rad20-profile.sh`; branch feat/rad20-blocking-lanes.

### Historical illegal-memory-access report for generated MMQ candidate (q6_K, gfx1100) — 2026-08-11

**Status:** internal-only

**Summary:** A historical run recorded a specific MMQ kernel configuration for
q6_K-quantized inputs failing with `an illegal memory access was encountered`
on an AMD RX 7900 XTX (gfx1100) under real MTP speculative-decoding draft
widths. The incident was reproduced deterministically three times across two
independent instrumentation approaches — same candidate, same failure, every
time. It has not been reproduced in the subsequent diagnostic runs, and the
surviving artifacts do not contain the complete canonical signature needed to
replay the original shape. Not observed under
narrower (non-speculative-decoding) workloads, which only exercise a smaller
range of matmul widths.

**Environment:** AMD RX 7900 XTX (gfx1100), ROCm 7.2.4, Clang 22.0.0
(`roc-7.2.4 26084`), llama.cpp `4801e3c567d5` (upstream tag `b10362`), model
Qwen3.5-9B (MTP variant, Q6_K), `--spec-type draft-mtp --spec-draft-n-max 5`.

**Evidence:**
```
ROCm error: an illegal memory access was encountered
  current device: -1, in function ggml_cuda_get_device at ggml-cuda.cu:142
  hipGetDevice(&id)
```
The historical run failed ~10-16s into generation, after dozens of other candidates measure
successfully — narrows the fault to one specific kernel launch, not general
device/driver instability (confirmed: identical hardware/model/context runs
cleanly moments before and after under a build that doesn't exercise this
candidate). Deterministic across repeated runs with `HIP_LAUNCH_BLOCKING=1`
forcing synchronous kernel completion, which rules out the fault belonging to
a different, asynchronously-queued launch.

Candidate identity at the point of the fault:
`mmq:q6_k:j112:fb0:t256:o2:i128:sram-q6_k:k256:sk0:v1` — an MMQ variant for
q6_K-quantized input, J-tile size 112, thread configuration t256/o2/i128,
shared-memory q6_k path.

**Internal context:** found via a live end-to-end build/tune/replay pipeline
test on the build server. Diagnostic tooling built to pin this down (an opt-in
`GGML_HIP_TUNE_TRACE_ATTEMPTS` journal event, see HI48) is now part of the
tuner permanently. Tracked for investigation and fix at
`docs/planning/active/external-fixes/EX02.md`. Quarantined from tuning
pending root cause — see EX02 for the exact mechanism and how to lift it.

**Investigation log, 2026-08-11:**

Checked whether this is already reported: no. Searched llama.cpp/ROCm issue
trackers for the crash signature and nearby symptoms (`WebSearch`); closest
hits (#21140 "device -1" in recurrent-state restore, #22052 RX 7900 XTX
memory fault via Oculink, #20839 invalid-device-function from an
architecture-spoofing issue) are all different triggers. This combination —
WMMA data-layout q6_K MMQ, J=112, driven by MTP speculative-decoding batch
widths — appears undocumented. Plausible reason: native dispatch's own
J-selection (`ggml_cuda_mmq_native_j_best` in `mmq.cu`) picks the *smallest*
J that covers the batch in one tile, so for MTP's narrow draft widths
(3-16) it would pick J=8, never J=112. Only an exhaustive tuning sweep
(bigcherry's own) forces J=112 against a batch this narrow — a genuinely
new code path, not a known-bad one.

Two specific hypotheses were formed, checked against real numbers/source, and
ruled out — recorded here so nobody re-derives and re-discards them:

1. **Shared-memory overflow via the CUDA/HIP opt-in gap.** `mmq.cuh`'s
   `CUDA_SET_SHARED_MEMORY_LIMIT` macro is a complete no-op on HIP builds
   (calls `cudaFuncSetAttribute` on CUDA, does nothing on HIP) — a real,
   confirmed platform gap in llama.cpp's own code. But direct measurement on
   the actual hardware (`hipGetDeviceProperties`) shows
   `sharedMemPerBlock == sharedMemPerBlockOptin == 65536` on this GPU/ROCm
   combination — there is no elevated ceiling being missed here. Computing
   the crashing config's real requirement by hand from `mmq_get_nbytes_shared`
   (I=128, J=112, Q6_K SRAM stride via the MMA data-layout branch, since
   gfx1100 has WMMA) gives **55,744 bytes against the 65,536-byte limit — about
   9.75 KB of headroom**, not an overflow. Ruled out with numbers, not
   plausibility.
2. **Y-tile global-memory over-read.** Hypothesis: the `src1_q8_1` staging
   buffer is sized for the real (small) batch, but the kernel unconditionally
   reads a full J=112-wide tile from it, over-reading past a small
   allocation for MTP's narrow widths. Checked `mmq.cu`'s own allocation
   (`ggml_cuda_mul_mat_q`, non-MoE path): it already adds
   `ggml_cuda_mmq_get_J_max(...) * sizeof(block_q8_1_mmq)` as explicit
   padding beyond the real data, specifically to make exactly this scenario
   safe — the `J_max` term bigcherry's own `ggml_hip_mmq_workspace` was
   built to mirror (see HI54's ledger entry). Upstream already anticipated
   and defends against this. Ruled out by reading the actual allocation.

Both are genuine platform/behavior facts worth knowing even though they are
not this bug's cause — the HIP no-op macro in particular is real and could
matter for a different, larger config that does exceed the default limit.

**Current status, 2026-08-14:** two exact-model build-server runs with the quarantine
disabled only through the opt-in diagnostic switch completed cleanly. The
candidate was measured 39 times in each run and the Q6_K MMQ family 526 times
in the full sweep; the second run was under `rocgdb`, with no device fault.
This is strong clean non-reproduction evidence, not proof that the historical
canonical dispatch signature was exercised. The separate historical
`/tmp/bc-tune-*` artifact is a Qwen3.5-0.8B run and contains no EX02 fault.

The attempt trace now persists the complete canonical signature JSON alongside
the digest when `GGML_HIP_TUNE_TRACE_ATTEMPTS` is enabled. A further diagnostic
replay is conditional on recovering the original canonical signature or digest
from historical provenance; candidate-only matches and reconstructed shapes do
not establish identity. If that identity becomes available, replay it through
the tuner measurement path under `HIP_LAUNCH_BLOCKING=1` and `rocgdb`, then
capture the actual faulting instruction/address inside the MMA vec_dot kernel
(`ggml_cuda_mmq_vec_dot_q6_K_q8_1_mma` in `mmq-vec-dot.cuh`). No generic GPU
sweep, quarantine change, or speculative kernel fix is justified by clean
non-reproduction runs.

The mechanism was validated on the rebuilt build-server vendor overlay with one real
Qwen3.5-0.8B completion: 1,395/1,395 durable attempt events contained a
parsed object-valued `signature_json`, including the complete extents, strides,
types, flags, operation, and refinement fields. The journal SHA-256 is
`d0eb2dc70bbe2ac1d0de1627ae09755d06fd05b749704d98ec80dcf0079ef9d3`. This is
instrumentation evidence, not a reproduction of the historical EX02 fault.

EX02 remains an open, deferred safety record with the exact gfx1100
q6_K/J=112/fallback=0 quarantine active. This deferral is non-blocking for the
mainline replay and tuning work; it is not a closure of the incident.

---

## Exceptional tunes

_(none logged yet)_

---

## Notable interactions

### ROCm stream and signal waits that cannot be used to block (ROCm 7.2.4) — 2026-10-25

**Status:** internal-only

**Summary:** Three ROCm behaviours that matter when one process must sleep until another process's GPU work is done:
(1) `GPU_STREAMOPS_CP_WAIT=1` switches `hipStreamWaitValue32` to a command-processor barrier-value packet, but that path
waits on a ROCclr memory-object signal and fails outright on a `hipHostRegister`ed POSIX shared-memory word
(`hipStreamWaitValue32` returns an error), so it cannot replace the blit wait for cross-process mailboxes.
(2) HSA IPC signals (`hsa_amd_signal_create(HSA_AMD_SIGNAL_IPC)` + `hsa_amd_ipc_signal_create` / `_attach`) carry no
driver event (`amd_signal_t.event_mailbox_ptr == 0`, `event_id == 0`) in either process, so
`hsa_signal_wait_scacquire(..., HSA_WAIT_STATE_BLOCKED)` on them spins a full core; they work only for GPU-side waits and
value checks. (3) On an ordinary (interrupt) signal, ROCr's `HSA_WAIT_STATE_BLOCKED` wait spins before it blocks: a waiter
woken every 150 us used 81% of a core, every 1 ms 19%. Waiting in the driver directly (as radiance's AQL backend does)
avoids the spin; a true interrupt wake then costs ~80 us end to end from another GPU's completion (p50; 30 us while
the waiter is still spinning).

**Environment:** Linux 7.0.13 (amdgpu in-tree), ROCm 7.2.4 (HIP/ROCclr/ROCr 7.2.4), MES firmware 0x80 on the XTX. Helper GPUs: 2x Radeon RX 7900 XTX (gfx1100, PCIe 4.0 x8 each); engine GPU: Radeon AI PRO R9700 (gfx1201, PCIe 4.0 x4); no P2P between them; processes share a POSIX shm segment registered with hipHostRegister(Portable|Mapped).

**Evidence:** standalone `tools/lab/radiance/ipc-signal-bench.cpp`: producer process on the R9700 submits BARRIER_AND
packets whose completion signal counts down from 2^40; consumer process on an XTX waits for the value to drop.
IPC signal, direct blocked wait: 5000/5000 delivered, no lost wakes, p50 25 us, p99 29 us, max 268 us, consumer CPU 100%.
Relay (ordinary interrupt signal waited in the producer process, futex to the consumer): consumer CPU 0.6-1.3%,
e2e p50 30 us (150 us interval) / 82 us (1 ms interval), max 425-517 us, relay thread CPU 81% / 19%.
`GPU_STREAMOPS_CP_WAIT=1` on the live helper: `lane 0 enqueue seq 1 failed rc=-5` at the first `hipStreamWaitValue32`.

**Internal context:** RAD20; run `${BC_WORK_ROOT}/runs/rad20-profile-cpwait-10252212`; GPT design reviews
req_bigcherry_3e23e104d8c3 (search), req_bigcherry_288a2ccd0399 (triage).

### `--split-mode row` fails outright on consumer RDNA3 (gfx1100) ROCm/HIP — 2026-08-12

**Status:** internal-only

**Summary:** `--split-mode row` (tensor-parallel split with cross-GPU
reduction, the mode that would exercise RCCL) fails immediately at model-load
time on two AMD RX 7900 XTX (gfx1100) GPUs, before any tensor-split ratio or
memory sizing is even attempted: `device ROCm0 does not support split
buffers`. `--split-mode layer` (pipeline-parallel, no cross-device buffer
sharing) works normally on the same hardware/model/build. This looks like a
real, hardware-class limitation, not a bigcherry-specific regression: `row`
split needs the backend to allocate GPU buffers spanning multiple devices
(peer-accessible "split buffers"), a capability data-center ROCm parts
typically have and consumer RDNA cards typically do not.

**Environment:** 2x AMD RX 7900 XTX (gfx1100), ROCm 7.2.4, Clang 22.0.0
(roc-7.2.4 26084), llama.cpp `4801e3c567d5` (tag b10362, `bigcherry-replay`
build), Qwopus3.6-27B-v2-MTP Q8_0.

**Evidence:**
```
0.00.374.130 E llama_model_load: error loading model: device ROCm0 does not support split buffers
0.00.374.150 E llama_model_load_from_file_impl: failed to load model
```
Reproducible immediately (fails in <1s, before weight loading starts) with
`--split-mode row --tensor-split 1,1` on `HIP_VISIBLE_DEVICES=0,1`. The
identical command with `--split-mode layer` instead loads and serves
normally.

**Internal context:** found while attempting the requested dual-XTX
MTP-5/RCCL test (`docs/planning` autonomous testing sweep, 2026-08-12).
Proceeded with `--split-mode layer` instead -- see the corresponding
tuning-runs artifact directory `sweep-qwen27b-mtp5-dualxtx-20260812` for the
actual run. Worth checking whether this is a known llama.cpp/ROCm
limitation upstream, or specific to this GPU generation/ROCm version
combination, before assuming "row + RCCL" is unusable across all AMD
hardware this project targets (gfx1201/RDNA4 or gfx1030/RDNA2 might behave
differently -- not yet tested).

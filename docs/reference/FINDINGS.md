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

### Illegal memory access in generated MMQ candidate (q6_K, gfx1100) — 2026-08-11

**Status:** internal-only

**Summary:** A specific MMQ kernel configuration for q6_K-quantized inputs
crashes with `an illegal memory access was encountered` on an AMD RX 7900 XTX
(gfx1100) under real MTP speculative-decoding draft widths. Reproduced
deterministically three times across two independent instrumentation
approaches — same candidate, same failure, every time. Not observed under
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
Crashes ~10-16s into generation, after dozens of other candidates measure
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
test on brutus. Diagnostic tooling built to pin this down (an opt-in
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

**Next step, not yet done:** attach `rocgdb` to a live reproduction to get the
actual faulting instruction/address inside the MMA vec_dot kernel
(`ggml_cuda_mmq_vec_dot_q6_K_q8_1_mma` in `mmq-vec-dot.cuh`), rather than
continuing to guess from source reading — two plausible-looking hypotheses
have already been wrong, and a third guess isn't worth proposing as a "fix"
without proof.

---

## Exceptional tunes

_(none logged yet)_

---

## Notable interactions

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

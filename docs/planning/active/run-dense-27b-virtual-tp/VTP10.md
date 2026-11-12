---
id: VTP10
order: 9
plan: run-dense-27b-virtual-tp
state: in_progress
created-at: '2026-11-10T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: S
---

# Real 27B W8A8 end-to-end TP2 baseline and IQ6R calibration blocker

> Archived 2026-11-12 from branch `exp/xtx27-prefill-opt` (git tag `archive/exp-xtx27-prefill-opt`), which never merged. Its lab programs (tools/lab/radiance/xtx27_*), tests and early patch drafts are in that tag. Superseded by VTP14 (native Radiance TP3, completed 2026-11-12), which kept these programs as reference oracles only.

## Scope

Live Radiance 1.17.0 on Brutus under an exclusive GPU lease, using the
pinned build at BC_WORK_ROOT/builds/f17f21fda44979e9abe84f13e96b8896/08b5dd81b4874390bf0cbf489db0d7df.
Two RX 7900 XTX gfx1100 cards (inventory 0,1), TP2, pinned-host all-reduce
(--p2p off), fp8 attention KV. R9700 gfx1201 was not used for
W8A8 serving baseline. All scratch receipts are out of the worktree.

No calibrated native-IQ6R Qwen3.8-27B artifact or complete RADGRAM2
Hessians were found locally. This work DOES NOT present a full-model
Q6-versus-Q8 comparison: the Q6 numeric microkernels and dispatch were
qualified in VTP07/VTP09, not a quantized Q6 checkpoint's quality.

## Real W8A8 serving without drafter: exact versus wht6

Native file: BC_MODEL_ROOT/radiance/bigcherry/qwen3.8-27b-w8a8/qwen3.8-27b-w8a8.rad
whole-file SHA-256 6bbf6e4a3dbf5634babdd5659b27336a11c372cdfa641802f2d451fb3233d619.
Existing BigCherry engine-bench lifecycle at
tools/lab/radiance/xtx27_prefill.py.
Context8192, one stream, batch2048, two repetitions and 64 decoded
tokens per corpus depth, same native model, fp8 KV. The collective
wire is the only changed flag. Results are client-observed prefill
tokens / HTTP time to first text, NOT GPU-only kernel time.

| Approx prompt depth | Exact prefill tok/s | wht6 prefill tok/s | Gain | Exact decode tok/s | wht6 decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 | 1393.09 | 1774.64 | 1.27x | 36.61 | 36.41 |
| 2048 | 1429.31 | 1856.73 | 1.30x | 35.83 | 35.70 |
| 6144 | 1402.13 | 1843.65 | 1.31x | 33.97 | 33.74 |

Scratch evidence: BC_WORK_ROOT/xtx27-prefill/vtp10-w8-engine-20261110/
with immutable provenance, server logs, raw per-request metrics and
summary.json. Both arms passed and shut down. wht6 is lossy;
the quality delta was NOT remeasured.

## Real W8A8 + DFlash2 concurrent serving

Native file: BC_MODEL_ROOT/radiance/bigcherry/qwen3.8-27b-w8a8/qwen3.8-27b-w8a8-df2.rad,
whole SHA-256 5a9af2dfaf3150e59cfc0001f8fe4f33549b201c6be19700c09c9e9209cc2701.
Both arms use the SAME model, two XTX, TP2, wht6, pinned host
all-reduce, fp8 KV, 8192 context, max sequences 6 and batch 2048.
The only changed setting is speculative decoding off versus auto.
BigCherry stream-bench sends tokenizer-calibrated ~512-token prompts,
128 completion tokens, 2 repetitions and seed17. Rates are finite
batch completion tokens per client-observed makespan.

| Streams | No speculation aggregate tok/s | DFlash2 auto aggregate tok/s | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 33.50 | 83.91 | 2.50x |
| 2 | 61.47 | 134.47 | 2.19x |
| 4 | 105.38 | 176.59 | 1.68x |
| 6 | 138.69 | 165.88 | 1.20x |

All sixteen rounds returned server token usage and passed; both
servers shut down. Auto step cost grows from 55.53 ms / 16.08
tokens per step at four streams to 86.43 ms / 22.95 tokens per
step at six; four outperformed six in this short workload,
not necessarily with longer outputs.

Scratch: BC_WORK_ROOT/xtx27-prefill/vtp10-w8-df2-concurrency-20261110/
with campaign.json, two stream-bench.json records, startup/server logs
and ROCm snapshots. Historical 205.6 tok/s at four streams from
2026-11-04 uses a DIFFERENT workload: not a matched speed regression.

### Repeatability/quality limitation

Every matched prompt has a text SHA-256. Between no-spec and auto,
only 8 of 26 matched outputs are byte-identical. Comparing each
arm's own two repetitions by stream: no-spec 9/13 identical and
auto-spec 4/13 identical. Single-stream outputs agree. This is
NOT proof of speculative corruption; greedy output can diverge
due to batch placement, numerical order and scheduling. Equally,
no bit-identical concurrency claim is justified. Required next
gate: fixed-batch/position controls, greedy replay, numerical
KLD/logprob comparison, with an acceptance floor measured at
speculation OFF.

## Original BF16 IQ6R calibration prerequisite

Original BF16 checkpoint is 18 complete safetensors shards and
55,562,855,904 tensor bytes. Two independent full-shard SHA-256
receipts agree (BC_WORK_ROOT/xtx27-prefill/quant-shape/
bf16-audit-fullsha-20261110-v2.json).
The native engine has a qwen35_bf16 architecture plugin and
--calib-corpus, --calib-dir, --calib-layers options. Pinned
Radiance docs/TOOLS.md says 64 dense-layer full-input Hessians
need multiple bounded ranges; the host accumulator approaches
786 MB per layer.

A tiny 2-document, 185-token, layer-0 smoke corpus was used only
to validate plumbing; it is NOT suitable for actual IQ6R GPTQ.
No full model was converted, no quality threshold passed.

- v1: dual-XTX startup correctly rejects vision placement because
  --mm-max-patches 0 declares no encoder.
- v2: dual-XTX TP2 layer offload with 16-GiB host pool, context
  2048/batch512: SIGSEGV exit -11 after weights load; no Hessians.
- v3: dual-XTX without offload: startup cleanly refuses insufficient
  VRAM, zero KV budget.
- v4: dual-XTX TP2 layer offload, context512/batch128: repeat
  SIGSEGV -11, no Hessians.
- v5: R9700 trial was misconfigured with BOTH ROCR_VISIBLE_DEVICES=2
  and HIP_VISIBLE_DEVICES=2; therefore selected CPU host backend,
  and nonexistent gram directory made writes fail. It is NOT
  usable GPU calibration evidence.
- v6: correct R9700 gfx1201-only TP1, 32-GiB host pool, layer
  offload, context512/batch128: 50.10 GiB original model weights
  loaded (about 30.03 GiB on GPU and 17.70 GiB host) then
  SIGSEGV -11 during the run, zero Hessians. Gram dir existed.

Per-run exact command flags, logs and process exit receipts are in
BC_WORK_ROOT/xtx27-prefill/vtp10-bf16-calibration-smoke-20261110-vX.

### Debugger: concrete failure stack

A bounded GDB run reproduced the R9700 failure and captured:

    SIGSEGV in rad_store_f32 (libradiance libavx.so)
    avx_gdn_conv_prep_v2._omp_fn (libavx.so)
    libgomp OpenMP worker

Full trace and args at BC_WORK_ROOT/xtx27-prefill/
vtp10-bf16-gdb-20261110/{gdb.log,receipt.json}.
Thus the immediate observed crash is in the CPU AVX GDN convolution
prepare fallback, not in IQ6R kernels and not gfx1100-specific.
The reason for the invalid destination and the selection of an
AVX kernel remains to be established; the stack alone does not
identify a safe source-code fix.

## Restricted GPU plugin experiment: AVX crash avoidance is not sufficient

R9700 TP1 replay with --kernels libr4d,libref (no libavx loaded),
original BF16, model-layer 0 only, 185 corpus tokens, 512 context,
128 batch, same offload and host-pool configuration, successfully
loaded the 50.10-GiB text checkpoint and did NOT immediately crash
in the previous AVX kernel. However, it completed ZERO of two
documents after more than 210 seconds, used approximately 17 CPU
cores while the R9700 showed only ~3% GPU use, and repeatedly
reported the active step waiting on a futex with KFD event threads
waiting. This is a nonfunctional calibration path, not a PASS.
The exclusively owned process was interrupted by SIGINT; receipt
shows returncode -2 and ZERO Hessians. The machine is clean.
Logs/receipt: BC_WORK_ROOT/xtx27-prefill/
vtp10-bf16-calibration-smoke-20261110-v7/.

The missing GPU-native BF16 GDN path / CPU operand ownership and
step-scheduler dependencies require detailed instrumentation; do not
assume the AVX SIGSEGV itself is the only blocker. A solution must
prove correct GDN results, bounded memory and usable end-to-end
calibration speed on model-sized inputs.

## Single-threaded AVX discriminant

A further R9700-only replay of the v6 native BF16 calibration
retained libavx and forced OMP_NUM_THREADS=1 with OMP_DYNAMIC=FALSE.
This also failed **SIGSEGV (-11)** after original-BF16 load, before
any Hessian output; isolated receipt/log at
BC_WORK_ROOT/xtx27-prefill/vtp10-bf16-calibration-smoke-20261110-v8/.
Single-threaded OpenMP therefore does not remove the crash in this
configuration, and a simple multithread race is not established as
its cause. The crash PC from the bounded GDB trace remains in the
libavx gdn_conv_prep write path. Audit device/host destination
provenance, shape/stride and lifetime around that path before changing
the kernel. The unmodified pinned upstream mirror remains read-only.

## Release decisions

PASS: actual W8A8 TP2 engine prefill/decode under both exact and
wht6 wire; W8A8+DFlash2 1/2/4/6 streams at measured FP8 KV
residency. This is a control baseline for future IQ6R.

NOT PASS: any full-model IQ6R performance or BF16 quality claim;
any three-physical-card virtual TP4 claim; any concurrent
bit-identical-output claim.

NEXT: debug original BF16 calibration kernel routing and GDN
host-buffer correctness, produce a valid layer-0 Hessian smoke,
then design 64-range representative calibration and VHEAD preparation.
IQ6R-V4 GPTQ requires full, representative source Hessians, matched
BF16 quality oracle and a real end-to-end model result.

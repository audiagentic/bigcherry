---
id: PGC10
order: 0
plan: patching-gpu-collectives
state: pending
created-at: '2026-09-30T05:20:19.155660+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# 3-GPU (2x XTX + R9700) AllReduce: adaptive root3/RCCL hybrid for models that need three cards

> Current disposition (2026-10-09): root3/1276 remains untested; QFP01/1291 already owns the validated 3-rank CPU-root production alternative. See authoritative audit below. Historical steps are retained for provenance, not as an active queue.

## Description

Goal (owner): run larger models / larger variants that need three cards (2x 7900 XTX gfx1100 + R9700 gfx1201, 80 GB VRAM) as fast as possible, combining RCCL with BigCherry's host-pipeline optimisations. Baseline measured on Qwen3.8-27B Q8_0 across 3 GPUs (ab-3g-allreduce, 6 balanced rounds): RCCL tg512 91.8 / pp4096 1098; 1244 root3 decode +2.0% (tg512) / +3.4% (tg2048) vs RCCL but prefill -68% (root3 has no large-message path, same as `none`). So the target is the same hybrid as dual XTX: root3/host for small (decode) reductions, RCCL for large (prefill).

## Steps

1. Extend 0840 adaptive to N=3: small -> 1244 root3 internal path, large -> RCCL (0840 currently assumes n_devices == 2 for its internal side).
2. Apply the decode-latency fixes from the Patch B review (1275: slot-sync host->stream, size-adaptive small-kernel geometry, later fused residual) to the root3 path as well as N=2.
3. Uneven split for mixed cards (R9700 32 GB vs XTX 24 GB, different bandwidth): sweep --tensor-split.
4. Target model: unsloth Qwen3.8-27B BF16 (~55 GB, needs three cards; /mnt/data/llm-models/qwen3.8-27b/gguf/unsloth/BF16/); later a 60-75 GB MoE.
5. Balanced A/Bs: RCCL vs root3 vs adaptive(N=3), then with 1275 switches.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Balanced server A/B on the 3-GPU 27B BF16 config with CI95 excluding zero; prefill must not regress vs RCCL; decode must beat RCCL. Heterogeneous-arch RCCL safety via 1225 guard. vLLM (radiance-vllm) must be stopped for R9700 tests and restarted after.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

The 3-GPU Q8_0 27B runs slower than the dual-XTX host path (tg512 95.5), so Q8_0 27B stays on two cards; this item is for models that do not fit on two. 1244 forces F32 wire for N=3; 1272 codec extension to root3 is Patch C territory (gfx1201 fp8 sender).

2026-09-30 design review with dev-gpt-agent (req_ba6aa3f60e0e4f92) agreed: decode AllReduce is fixed-latency bound (per small AR: one 8x256 mapped-host kernel per GPU for ~10 KiB bf16, ev.ker records, two host hipEventSynchronize in acquire_slot from AR #3; AR outside the device graph). Patch B (pipelined copy-engine wire) dropped. 1275_ar_small_latency: BIGCHERRY_AR_SLOT_SYNC=host|stream|none (none primary; proven safe for 2-slot ring incl. pinned-host reuse and root3 as long as collectives are not skipped/reordered and each AR is single-chunk), BIGCHERRY_AR_SMALL_BLOCKS {1,2,4,8} + _THREADS {128,256}, trace host_enqueue_us/slot_wait_us, marker with n_devices/blocks/threads/slot_sync; covers 1244 root3. 1276: 0840 adaptive composes with 1244 for N=3 (small->root3, large->RCCL) and BIGCHERRY_AR_ROOT3_ROOT=0|1|2 (default 0; choose root from measurements, not architecture). Authoring requested (req after ba6aa3f6).

2026-10-01 1275 A/B on the dual-XTX host path (27B Q8_0 MTP, 6 balanced rounds, ab-27b-ar-small): BIGCHERRY_AR_SLOT_SYNC=none vs pristine tg512 +0.04% (CI -0.16..+0.23), tg2048 -0.01%, pp4096 -0.24% -> neutral; BIGCHERRY_AR_SMALL_BLOCKS=1 vs pristine tg512 -8.56%, tg2048 -8.47%, pp4096 -2.39% -> strongly worse. MTP acceptance identical (82.22%). Conclusion: the fixed 8x256 small-AR grid is not oversized and host slot syncs are not the decode bottleneck; the design review's fixed-latency hypothesis for these two knobs is refuted. 1275 stays untested (no gain); next question is whether MORE than 8 blocks helps (needs the arrival-ring layout widened).

2026-10-04 status: realised for Flash-Next by patch 1291_ar_cpu_root (--allreduce cpu-root): CPU-root one-shot AR for <= 64 KiB f32 messages on 3 GPUs (2x XTX + R9700), RCCL above; +5-7% decode vs RCCL, prefill unchanged; the large-message host path lost to RCCL and is off. Detail and follow-ups in patching-qwen-flash-next QFP01. Remaining scope here: generalising the adaptive root/RCCL policy to other models.

## 2026-10-09 authoritative audit: N=3 root3 is experimental; CPU-root already owns the deployed fast path

**Disposition:** Keep PGC10 as the N=3 *alternative-provider qualification* owner, not a second production AllReduce dispatcher. Patch 1276 remains `untested` and **not promoted**. QFP01/validated patch 1291 already supplies the measured three-rank exact-F32 CPU-root small-message path and RCCL fallback; do not reimplement its worker, epochs, graph lifetime, or provider selection. GP11/1244 owns the root3 kernel/protocol. PGC09 owns the validated dual-gfx1100 adaptive default; PGC12 owns phase-aware routing. The historical Steps above are superseded by this gate; no new experiment is queued by this audit.

### Source-verified admission and control path (b11474)

- `0860_allreduce_provider_cli/patch.py`: adaptive switch default is 96 KiB. `0840_hybrid_allreduce_dispatch/patch.py::ggml_backend_cuda_comm_try_allreduce_hybrid` compares **logical F32 tensor bytes** using `reduction_bytes < switch_bytes`, with RCCL fallback. Its `auto` admission is restricted to two gfx1100 participants; **N=3 does not become automatic** merely by applying 1276.
- `1244_gp11_internal_allreduce_nway_root/patch.py::ggml_cuda_ar_allreduce_root3` adds three-rank root/leaf mapped-host aliases, arrival slots and kernel events; its N=3 branch admits **F32 only** and independently rejects `input_nbytes >= p->copy_threshold` (stock chunk/copy crossover). `1276_ar_adaptive_nway/patch.py` only selects root 0/1/2 at pipeline initialization and rewrites rank-indexed aliases/events; it does not change `auto` admission or introduce a new N=3 fast transport.
- `1291_ar_cpu_root/patch.py::ggml_backend_cuda_comm_try_allreduce_cpu_root` admits contiguous, equal-length F32 tensors up to **65,536 bytes** by default; a per-communicator CPU worker sums pinned mapped rank slots and GPU producers/consumers use device-advanced epochs. Larger/noneligible calls use RCCL (optional large CPU-root is off by default). The CPU-root comm context owns its worker, pinned buffers, teardown drain and graph-replay epochs. No root3 patch is needed for that path.
- These are **different provider selections**. Under explicit `adaptive` with 1244+1276 and the default 96 KiB threshold, 10/20/64/80 KiB F32 messages prefer root3 if its own admission passes; 120 KiB and larger prefer RCCL. Under explicit `cpu-root`, 10/20/64 KiB F32 messages use CPU-root, but 80/120 KiB use RCCL. Thus the 80 KiB prompt-tail shape enters the experimental root3 path yet avoids CPU-root. PGC09 previously measured a serious *different* host-provider prompt-tail regression; **do not infer root3's 80 KiB latency from it**. The overlap shows why size-only promotion without phase evidence is unsafe.
- N=3 {two gfx1100 + one gfx1201} is RCCL-admissible only after the existing 1225 per-device native-host-atomic check. The chipset-routed gfx1030 auxiliary card must not be silently included in RCCL; no P2P is assumed. `BIGCHERRY_AR_ROOT3_ROOT` is a **logical rank**, not a physical PCI bus index. For root r, leaves are (r+1)%3 and (r+2)%3; all mapped aliases and arrival pointers must be resolved in the consuming GPU's address space.

### Existing measured evidence, not a new qualification

- PGC10 historical 3-rank 27B Q8_0 six-round RCCL baseline: tg512 91.8 t/s, pp4096 1098 t/s; root3 +2.0% tg512 / +3.4% tg2048, but **-68% prefill** when root3 was used without a large-message RCCL crossover. These results are older than the deployed CPU-root implementation and are not a direct comparison with it.
- GP11 root3 microbenchmark at 30,720 F32 elements: ~65 us versus RCCL ~87.8 us (different message size and benchmark from Flash-Next). 1275's dual-XTX slot-sync `none` was neutral and 1-block geometry regressed ~8.5% decode; do not repeat those settings without new mechanism evidence.
- QFP01/1291 three-rank Flash-Next: stream-ordered 10 KiB microbench ~13.8 us CPU-root versus ~33.1 us RCCL; actual ABBA decode **+6.4% plain / +5.1% MTP**, greedy identical. Existing profile evidence shows prefill large-message CPU-root lost to RCCL and was disabled. These numbers are **not** root3-versus-CPU-root matched measurements.
- The no-overlap microbench upper bound of 96 x (33.1-13.8) us = **1.85 ms/token** is theoretical launch/collective savings, not measured E2E gain; in-model arrival skew and overlap can dominate. Never pool BF16 wire experiments with exact-F32 comparisons.

### Implementation-ready next decision, without new infrastructure

1. **Cheap offline gate (now):** reuse `tools/tests/patch/test_1276_ar_adaptive_nway.py` (already checks patch composition, idempotence, exact rank aliases and fail-closed anchors). Add a table-driven **host-only** admission/size/phase fixture for {10,20,64,80,120 KiB,1 MiB}, root {0,1,2}, inactive shards, 96 KiB switch and 64 KiB CPU-root threshold. Assert `auto` stays N=2-only, RCCL admission is checked, and fallback remains available. No new dispatcher/telemetry schema.
2. **Only if a real three-card workload is limited by CPU-root** (CPU worker saturation, contention, or observed 64-96 KiB phase-specific RCCL latency): collect actual per-request provider **completion** and logical byte count, prefill/decode/MTP-verify phase, root, rank arrival skew, CPU worker busy time, GPU wait, graph capture/replay identity, transfer bytes, and `ggml_backend_cuda_comm_try_allreduce_*` selection. Reuse existing QFP01/PGC12 tracing and lab runner; no synthetic marker-only attribution. Do not touch queued Flash-Next/1357/1358 experiments.
3. If step 2 proves a material root3 opportunity, run one **isolated** explicit-provider A/B/C on the exact 2xXTX+R9700 topology: RCCL control; validated `--allreduce cpu-root` (1291) control; `--allreduce adaptive --allreduce-switch-bytes 98304` + 1244/1276 subject. Include the 80 KiB prompt-tail control, 10/20 KiB decode, 120 KiB MTP verify, 1 MiB+ prefill, long context, ubatch 16/128/512, root 0/1/2, and multi-request same-process/graph replay. Match model, quant, device order, tensor split, exact F32 wire, RCCL linkage and production flags. Verify `HIP_VISIBLE_DEVICES` mapping and 1225 admission; isolate the lab from other owners.
4. **Correctness gate:** finite full-vocabulary logits, KLD/top-token/greedy comparison to exact-F32 RCCL, MTP draft/acceptance and expected rank contributions (including zeroed inactive shards); no missing collectives, stale slot generations, graph use-after-free, rank skew deadlock or cross-request contamination. Do not require F32 bit identity if summation order differs, but reject any unexplained trajectory or acceptance divergence.
5. **Promotion:** >=4 independent sessions, >=10 interleaved ABBA pairs each; CI95-low >=3% E2E decode versus **CPU-root** with CI95-low >=0% versus RCCL, prefill CI95-low >=-0.5% versus RCCL, no >1% other-control regression, correctness above and bounded CPU/VRAM/RSS. Keep default `auto` unchanged until these gates pass. **Terminal:** if step 2 finds no CPU-root bottleneck, or subject fails any gate, close 1276/PGC10 as superseded for this topology and retain QFP01/1291 + RCCL. Do not create a second policy cache or root-placement solver.

### Upstream and other-engine disposition

- [llama.cpp #27825](https://github.com/ggml-org/llama.cpp/pull/27825) merged 2026-09-15: HIP enables the stock **two-GPU** pinned-host AllReduce; it does not supply root3 or replace 1291. [#29793](https://github.com/ggml-org/llama.cpp/pull/29793) merged 2026-10-01: inactive shards must be filled rather than scaled from potentially NaN/Inf memory; preserve this correctness baseline.
- [vLLM `cuda_communicator.py`](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/cuda_communicator.py) uses capability/world-size gates and an ROCm QuickReduce fallback. [QuickAllReduce](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/quick_all_reduce.py) supports 2/4/8, **not 3**, and requires its own connectivity/quantization conditions. [SGLang #31117](https://github.com/sgl-project/sglang/issues/31117) documents a concurrent-stream custom-collective deadlock; use as a multi-stream safety warning, not as an AMD benchmark. None provides a directly portable, qualified no-P2P three-RDNA-card implementation.

**This audit's validation:** source-static/host-model checks (22 conditions; one initially mis-specified source-string assertion was corrected and independently passed on recheck). No repository pytest, HIP build, GPU run, benchmark, or deployment performed. Evidence/decision owner is PGC10; BCOP97 is only the action ledger.

## Change Log

- 2026-09-30T05:20:19.155660+00:00 (created-by): Created by agent
- 2026-09-30T06:29:06.022444+00:00 (updated-by): Updated: section:notes
- 2026-09-30T14:44:36.752908+00:00 (updated-by): Updated: section:notes
- 2026-10-03T15:21:51.445491+00:00 (updated-by): Updated: section:notes

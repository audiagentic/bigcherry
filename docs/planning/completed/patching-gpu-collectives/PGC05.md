---
id: PGC05
order: 0
plan: patching-gpu-collectives
state: superseded
created-at: '2026-09-29T09:34:02.512413+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: S
---

# Adaptive AllReduce threshold CLI: implemented; bucket table retired

## Description

## 2026-10-13 terminal disposition (BCOP183)

**Done.** The original `--allreduce-switch-bytes` request was implemented by 0860/0840 on 2026-09-30. Do not add a second size-bucket dispatcher or Q8 threshold. This closes duplicate planning; it does not introduce a code change or measured speedup.

### Exact implementation and ownership

- `engines/llamacpp/patches/0860_allreduce_provider_cli/patch.py`: `ADAPTIVE_SWITCH_BYTES_DEFAULT = 96 << 10`, CLI parser `common_parse_allreduce_switch_bytes`, and `common_apply_allreduce_config` applied once after argument parsing through `ggml_backend_comm_set_config`. The process-global value is an initialization input, not a per-call policy table.
- `engines/llamacpp/patches/0840_hybrid_allreduce_dispatch/patch.py::ggml_backend_cuda_comm_init_hybrid`: snapshot to `comm_ctx->adaptive_switch_bytes`. `ggml_backend_cuda_comm_try_allreduce_hybrid`: host exact-F32 for `ggml_nbytes(tensors[0]) < switch_bytes`, RCCL at or above it when available. Equality selects RCCL. Missing RCCL falls back to available internal host. An attempted RCCL failure is not replayed on host after potential partial execution.
- Auto-adaptive admission: two distinct physical gfx1100 HIP devices and RCCL eligibility; other devices retain the base provider. Explicit `adaptive` and startup `BIGCHERRY_PATCH_TRACE` are not proof of actual per-reduction provider completion. PGC12 owns correlated route/transition evidence.
- `tools/tests/patch/test_0860_allreduce_provider_cli.py` checks CLI option, post-parse application and idempotence; `test_0840_adaptive_provider.py` checks auto admission and per-context snapshot. `PGC04` in the 0860 SUMMARY is historical patch provenance, not a second current runtime owner.
- The old `Q8 threshold` is not implemented as a generic adaptive crossover. `--allreduce-wire q8` is a restricted provider/codec selection; **PGC11** owns precision and MTP acceptance. No new Q8 threshold is justified.

### First-party evidence and limits

Validated `t-0840e-gfx1100-s1..s4`, dual gfx1100 exact-F32 96 KiB: **tg128 +3.87%..+3.98%**, **pp512 +0.02%..+0.04%**, bit-identical. Historical 1 MiB plain pp1024 regressed approximately **4.2%**, including a four-token prompt tail of 128 reductions around 80 KiB; its additional latency remains unattributed. Decode around 20 KiB, tail around 80 KiB, MTP verify around 120 KiB and large prefill are compatible with one monotone 96 KiB threshold. This is **not proof** that all models have a monotone crossover. Historical BF16 MTP gains must not be pooled with exact-F32 sessions; PGC11 observed BF16 acceptance loss. No new GPU measurement in this audit.

### Consolidation and conditional reopen gate

**PGC09** alone owns production provider/configuration (0860/0840). **PGC13** owns a bounded *offline* comparison of existing 64/96/128 KiB scalar settings, after **PGC12** proves actual provider-completion/first-transition cost and >=3% exposed E2E opportunity. **PGC11** owns wire, **PGC14** RCCL; **QFP01/1291** validated N=3 CPU-root; **PGC10/1276** experimental root3. Do not combine N=2/N=3, mixed architectures, P2P, PCIe, wire or driver contexts.

Reopen a size-bucket table only on matched source/build/model, physical GPU UUID/order, ROCm/RCCL, graph/phase, wire and provider-completion traces showing **two disjoint size intervals with reversed optimal providers after transition cost**. First run a host-only selector fixture. If the optimistic E2E ceiling is below 3%, close without hardware. Otherwise require four independent order-balanced paired sessions (>=10 ABBA rounds each), pp256/512/1024/4096, tg128/512, long context, MTP verify widths, multi-request same-process and graph replay, full-vocabulary logits/KLD/greedy and acceptance, no skipped reductions/hangs, <=0.5% prefill regression. Promote only CI95-low >=3% E2E over the validated 96 KiB baseline, under PGC09 ownership.

### Upstream and other engines

[llama.cpp #27825](https://github.com/ggml-org/llama.cpp/pull/27825) merged 2026-09-15, enabling HIP two-GPU host-staged AllReduce using mapped host memory and `__builtin_amdgcn_s_sleep`; it does not supply the BigCherry adaptive CLI. [vLLM](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/cuda_communicator.py) and [SGLang](https://github.com/sgl-project/sglang/blob/main/test/manual/test_quick_allreduce.py) gate QuickAllReduce to different ROCm hardware (MI300/gfx94*), not validated on gfx1100/gfx1201 or no-P2P dev.

### Audit evidence

Actual-source static checks **15/15**; source-shaped host threshold/availability fixture **11/11**. No repository pytest, HIP build, GPU test or new performance gain. The original PGC05 1254/1261/1241/1206/1245/1205/1263 activation tasks belong to their individual patch owners and are not adaptive-provider prerequisites.

## Change Log

- 2026-09-29: initial threshold and bucket proposal.
- 2026-09-30: threshold CLI implemented by 0860/0840.
- 2026-10-13: BCOP183 closes duplicate bucket/Q8 threshold ownership.
- 2026-10-23: state 'done' (invalid) corrected to superseded

## Steps


## Detailed Solution & Technical Design


## Code Samples & Guidance


## Files


## Validation


## Effort & Risk


## Standards


## Acceptance Criteria


## Notes


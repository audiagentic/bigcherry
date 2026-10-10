---
id: BCOP124
order: 124
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-10T21:05:00+00:00'
created-by: agent
priority: P2
work: S
---

# PKC03: gate collective evidence identity on completed provider, not configured route

## Discovery / disposition

The EC05 `ContractEvidenceRef` and contract hash identify the experiment, not the completed AllReduce provider. The 0840 hybrid dispatcher writes `provider_name` before provider execution; its internal failure can fall through to RCCL/Meta. Requested `--allreduce`, predicted `prefer_internal` and process environment cannot prove completed work. PKC03 records the representability/negative-evidence decision and remains pending until PGC12's existing trace proves the missing route. No new collective registry, telemetry subsystem, dispatcher or allocator.

## Ownership / independent-work exclusion

PKC03 owns the decision only. PGC12 owns completed-provider/phase/ubatch evidence; PGC09 threshold, PGC10 root3, PGC11 wire, PHA03 physical topology, EC05 contract evidence, RRVP01/02 future stack identity. Last PKC03 semantic update 2026-09-10; 2026-10-08 was line-ending normalization. No PKC03 patch/PR/hardware queue in the preceding 12 hours. Recent QFP17/1330, QFP41/1356, QFP48/49, QFP50/1359, 1357 and Radiance are excluded and untouched. BCOP114–123 did not audit PKC03's identity boundary.

## Bounded next action / terminal gate

Use existing PGC12 and EC05 test surfaces: distinguish internal success, rejected internal then RCCL success, rejected RCCL then Meta fallback, changed root/wire, missing/foreign/out-of-order receipts and true zero-call negatives under identical contract hashes. Bind completed route, effective config and physical device set to the measured process and request; missing or predicted-only evidence is INVALID. Consume existing PGC09/PGC10 evidence only when complete; no new queue. Close PKC03 if the existing sidecar plus attached PGC12 artifact proves every identity dimension; otherwise assign a specific gap to its existing owner. No throughput claim or new patch.

## Evidence

Pinned `contract.py::ContractEvidenceRef`, `bundle.py::ALLOWED_ENV`, 0840/0860 `patch.py`, PGC12 and PGC09; [llama.cpp AllReduce](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/allreduce.cu), [vLLM P2P RFC](https://github.com/vllm-project/vllm/issues/51513), [SGLang provider routing](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/distributed/parallel_state.py). 9/9 source-static checks and 4/4 illustrative host fixtures passed. No pytest, build, GPU benchmark, new measured gain, or implementation.

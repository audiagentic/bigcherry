---
id: BCOP103
order: 103
plan: patching-bc-optimizations
state: done
created-at: '2026-10-10T00:07:18+00:00'
created-by: agent
priority: P2
work: S
---

# PRBE112: gate rejected RD42 against native shared-expert MMVQ

## Discovery and disposition

Pinned b11474 `ggml_cuda_try_fuse` requires an empty concurrent-event map for native routed/shared MMVQ; rejected 1215 inserts an RD42 event, while stock QKV graph-opt can also create events. Source-level exclusion is proven; real-model matcher activation and regression causality are not. The later b11126 four-session rejection (including dense-control regression) overrides early positive README evidence. Keep 1215 rejected/default-off.

## Ownership and recent work

PRBE112 owns only a bounded new RD42 decision. PRBE35/1216 independently owns QKV join correctness; its old PASS was on top of 1215. QFP35 owns native fusion; PRBE113 owns host-exclusive profiling. Recent Radiance, 1334/1347, 1355/1356/1358 and queued 1357 capabilities were excluded and untouched. No new patch, scheduler, allocator, registry or queue.

## Gate / terminal outcome

Compare stock b11474 `GRAPH_OPT=0/1` with actual fused launches, event count, dense control and memory on single gfx1100. Close if no native-fusion-ineligible MoE window, dense regression >1%, or theoretical E2E ceiling <3%. Only then consider isolated default-off RD42 (never 1215), requiring full-vocab/MTP, multi-request/graph replay and four-session CI95-low >=3% E2E over best native control.

## Evidence

Pinned `ggml-cuda.cu` matcher/fusion/graph-opt; 1215/1216 patch metadata and stored validation; upstream #16991/#26574, AMD fork #36, SGLang dual-stream, vLLM ROCm #38665/#48111. **18/18 source-static/host-admission assertions passed**; no pytest, compilation, patch composition or GPU benchmark.

---
id: BCOP114
order: 114
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-10T11:24:00+00:00'
created-by: agent
priority: P1
work: S
---

# PHC03: gate no-P2P Meta D=3 on copy-path evidence

## Discovery / disposition

RU01's D=3 Meta SIGSEGV is real, cause unproven. Pinned b11474 has unguarded cross-device peer-copy calls in both CUDA async and buffer-copy paths unless built with `GGML_CUDA_NO_PEER_COPY`. D=3 fold/copy-back adds two cross-device edges versus D=2. Generic synchronized host staging already exists. Existing 1224 probe now accepts D=2..4 despite D=2-only README. Technical design and bounded A/B decision tree live in **PHC03**, not this ledger.

## Ownership / active-work exclusion

PHC03: copy safety. RU01/HI84: hardware qualification. 1224: native probe. 1242: Meta trace. PGC12: provider evidence. PGC14: RCCL separate. PHC01/Flash-Next remains protected and untouched. Last independent PHC03 update 2026-09-10; no associated patch, PR, benchmark or queued work in preceding 12 hours. Excluded active Radiance gfx1100 FP8/MXFP4/two-XTX and Flash-Next router/QSA/1330/1334/1347/1357, QFP35/36/41/43 and DFlash/MTP. No recent BCOP104–113 slice repeated.

## Terminal gate

Compare fresh isolated default build A versus `-DGGML_CUDA_NO_PEER_COPY=ON` build B with exact binary and ordinal receipts. If B passes and A fails, retain safe host fallback and only investigate an edge-proven minimal gate. If B also fails, abandon peer-copy hypothesis and trace Meta ordering/scratch. Require original D=3 groups, reversed order, all-rank CPU-double oracle, multi-request and graph replay before D=4. Reject any false success, unverified path or output mismatch. Performance work is optional and must satisfy matched CI95-low >=3% E2E/no >1% control regression. No new transport, cache, scheduler or duplicate harness.

## Evidence

Pinned llama.cpp b11474 CUDA/Meta/backend/CMake source; RU01/HI84; 1224 and 1242; [upstream #21648](https://github.com/ggml-org/llama.cpp/issues/21648); vLLM custom AllReduce; SGLang PCIe-IPC. Disposable host D=2/3/4 route fixtures passed; pinned source static checks 12/12. No build, pytest, GPU benchmark, measured gain or published implementation.

---
id: BCOP175
order: 175
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-13T02:07:23Z'
created-by: agent
priority: P2
work: S
---

# PRBE35: reconcile standalone join guard and bound current-pin evidence

## Discovery / change

At b11552, upstream still has native concurrent-region fork/join plus `ggml_cuda_try_fuse`, but no 1216 join cap; both 1216 anchors match once. `1216/patch.toml` is standalone/untested, while legacy `patch.py::REQUIRES`, producer TOML comment, README and PRBE35 old steps still assume rejected 1215 or an absent central runner. The patch-local producer and validation manifest **already exist**. The marker fires whenever the cap is entered, even if no fusion would cross; it is not a causal abort-prevention proof. All 22 1216 evidence records target b11126 (8 historically eligible), not current b11552. No new speedup or GPU correctness result.

## Authoritative owner / work already acting

PRBE35 owns 1216 graph-join correctness and the existing patch-local producer. PRBE112 owns rejected 1215/RD39–42 and its negative four-session results; do not reopen or inherit its historical positive throughput. PVPS15 owns generic activation evidence. Current RAD01/Radiance engine/campaign, 1340, and release work in the preceding 12 hours is excluded and untouched. No independent 1216/PRBE35 change, PR or queued hardware lane was found in that interval. No new scheduler, cache, dispatcher or producer is authorized.

## Bounded action / terminal disposition

First run host-only patch-composition, manifest-versus-legacy-metadata, contract-prerequisite scoping, marker-no-crossing, and synthetic join-skip fixtures. Reconcile stale implementation metadata only in a separate tested patch edit. If a native concurrent region is reachable at b11552 without 1215, run the existing control=validated BC vs subject=+1216 graph-opt-on producer with fixed model/device identity, full-vocab/MTP parity, >=20 multi-request/capture-replay safety runs and >=4 independent paired sessions/architecture with >=10 rounds, <=1% control regression. If no native region activates, close as dormant/no-op without benchmarking; if correctness or controls fail, retain default-off. Do not claim that a marker alone proves a prevented abort.

## Evidence

Pinned llama.cpp b11552 `ggml-cuda.cu::ggml_cuda_graph_evaluate_and_capture/ggml_cuda_try_fuse`; BigCherry `1216/patch.py`, `patch.toml`, `validation/producer.py`, `validation/producer.toml`, `patch/registry.py::_packaged_descriptor`, `experiment/contract.py::parse_contract`; upstream AMD-Ecosystem PR #71 and SGLang breakable graph docs. This audit ran 15/15 actual-source static assertions and 6/6 synthetic host assertions; no repository pytest, patch application, HIP build, GPU execution or new benchmark.


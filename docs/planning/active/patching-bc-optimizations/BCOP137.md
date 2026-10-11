---
id: BCOP137
order: 137
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-11T11:13:10+00:00'
created-by: agent
priority: P3
work: S
---

# MET06: qualify upstream host-weight layer pipeline before runtime slicing

## Discovery / disposition

[llama.cpp #29963](https://github.com/ggml-org/llama.cpp/pull/29963) is now open/ready for review, with a new 2026-10-10 [gfx906 HIP comparison](https://github.com/ggml-org/llama.cpp/issues/29963#issuecomment-6103252758). Its actual mechanism is layer-split, split-owned transient host-expert upload, one user-input copy per backend with input events, and worst-case graph re-reservation. It **does not** enable BigCherry's production tensor split. Its host-weight spread check also admits the zero-host-weight case when an unrelated tensor override exists; require a positive host-byte/copy admission check. The 16x/64x ID heuristic may upload whole expert tensors on large batches. No new BigCherry gain is established.

## Authoritative ownership and exclusions

MET06 owns the standard-GGUF runtime-slicing *decision* and upstream HIP layer-pipeline gate; MET01 owns residency policy, MET07 owns 1337/1338 cache qualification, MET11 owns offline/load-time permutation, MET05 owns auxiliary transport and MET12 alone owns any later predictive prefetch. #29963's existing scheduler/fit implementation is the preferred comparator, not a new BigCherry scheduler. Excluded recent b11552/0300/1006/1346/1359 pin/patch work, QFP41/1356 graph race, QFP48 step/MMQ census, QFP49/1360/1361 cross-card overlap, PA48/PA49 pin composition. None of those files or queued lanes was changed.

MET06's last independent change was 2026-10-05; MET12 last material plan creation 2026-10-08. Repository commits, active branches, open PRs and queued-lane references were checked for the preceding 12 hours; no independent MET06/MET12 host-weight pipeline experiment or submission was identified. This is a distinct, dormant source qualification, not a repeat of the recent MoE cache BCOP audit.

## Bounded action / terminal gate

Offline: validate layer-split/async/events/host-byte admission, 16x/64x whole-copy transition, graph re-reservation, input/weight copy ownership and no-host-weight negative case. Hardware only after active scheduler work stops: matched 2x XTX layer-split host-expert A/B with identical placement, CPU_Mapped versus pinned-host controls, explicit nonzero upload bytes, no-P2P transfers, repeat-request/graph/ubatch/MTP correctness and matched equal-VRAM whole-layer reference. Then consider R9700 third rank. Require >=4 sessions x >=10 ABBA pairs, CI95-low >=3% end-to-end PP, <=1% TG/control regression and exact expert/logit/greedy correctness. If no HIP benefit, reject upstream port; if coarse placement residual is <5% PP/TG and <1 GiB at equal throughput, retire MET06 runtime slicing. Do not introduce another loader/cache/prefetch scheduler without these gates.

## Evidence

Pinned b11552 `src/llama-context.cpp`, `ggml/src/ggml-backend.cpp`; upstream #29963 changed `common/fit.cpp`, scheduler, context; SGLang OffloaderV2 and vLLM UVAOffloader source. Local MET01 measured R9700 x4 H2D ~6.7 GB/s and host-expert pp ub512/1024/2048/4096 = 93.9/159.0/265.8/420.6 t/s. External MI50 32768 PP 122.47 -> 312.46 t/s is not transferable. 12/12 deterministic source-derived host assertions passed; no HIP build, repository pytest, GPU execution, storage experiment or BigCherry benchmark was run.


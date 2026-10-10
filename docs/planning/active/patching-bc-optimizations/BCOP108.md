---
id: BCOP108
order: 108
plan: patching-bc-optimizations
state: done
created-at: '2026-10-10T05:09:38+00:00'
created-by: agent
priority: P2
work: S
---

# PNRO02: retire infeasible 1250 composition; gate exact-F32 residual epilogue

## Discovery / disposition

The PNRO02 opening plan incorrectly says the matcher/API are missing, while 1250's source implements them. However 1250 **requires rejected 1252 P2P** and untested 1272; its fused API accepts **Q8_0 only**, not exact F32. `GGML_CUDA_AR_FUSED_RESIDUAL=0` still enables the presence-only gate. BPB01 already records the package as superseded/infeasible. Preserve its evaluated metadata and historical evidence, but **do not run or promote 1250** on the present no-P2P topology. No new patch was authored.

## Owner / dependencies / protected work

PNRO02 alone owns AR+MIRRORED residual epilogue semantics. PGC09/PGC12 own exact-F32 provider/telemetry, QFP01/1291 owns production CPU-root result write, 1272 owns Q8 codec; PNRO03/PNRO18/1252 remain terminal. PGC16's partial+partial AR-count reduction is distinct. No second transport, graph pass, selector or config. PNRO02 last independent commit 2026-10-03 15:22 UTC; 1250 BPB01 review 2026-10-08, both outside 12 hours. Active Radiance, QFP35/36/41/43, Flash-Next 1330/1334/1347/1355/1356/1357/1358 were excluded and untouched.

## Next gate / terminal conditions

Read-only production graph census + exclusive ADD wall: close `NO_MATCH` if no safe rank-identical, one-consumer AR→mirrored ADD; close `NO_CRITICAL_PATH` if ADD <2.913% E2E wall (even zero-cost ADD then cannot yield 3% speedup). Otherwise host fixture with negative shape/rank/alias/partial-enqueue/graph cases, then one default-off exact-F32 existing-provider finish. Promote only after >=4 sessions, >=10 paired ABBA rounds, full-vocab/greedy/MTP and graph/multi-request correctness, CI95-low >=3% E2E, <=1% controls. Fail closed on uncertainty; no P2P rerun.

## Evidence

1250 `patch.py`/metadata, 1252 rejected metadata/PNRO03, 1272 finish kernels, 1250 mechanics tests, BPB01. AITER #6270 open residual-identical-on-all-ranks epilogue (actual diff inspected), vLLM fusion design/#59483, llama.cpp #27825. 12 source-static checks and 12 disposable host-admission fixtures passed; 32/32 synthetic Q8-vs-F32 values differed (not GPU). No repository pytest, build, hardware or model benchmark.

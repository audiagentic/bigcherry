# 1206_rd13_mul_mat_add_view_fusion: Fuse mul_mat + add through a view (reshape) node (RD13)

**Status:** untested
**Plan item:** RD13

> Current 2026-10-09: patch remains `untested`/not promoted. The September 21 producer now implements real paired positive/control tg128 lanes, trace probes and full-vocab backend reference. The generic performance CLI is forbidden only to avoid duplicate evidence. Old BLOCKED records remain historical; b11474 requalification is outstanding. PRBE12/BCOP80 own the gate.

## What it does

Extends the existing mul_mat+add fusion in ggml_cuda_try_fuse to accept one RESHAPE or qualified zero-offset contiguous VIEW node between the matmul and the add (using ggml_can_fuse_subgraph, verifying the view's src[0] is the matmul), instead of only matching an add directly after the matmul.

## Why

SSM models (e.g. qwen35moe) insert a reshape view between the output projection and the residual add, so the existing fusion never fired for them and every layer ran a separate add kernel.

## Current acceptance disposition

The bound `validation.toml` has six required checks. `validation/producer.py::run` emits `promotion_lane_effects`, `promotion_trigger_evidence`, `contract_correctness_results`, `rd13-performance.json` and subject/control trace logs; `producer.toml` skips duplicate generic trace probes and forbids the generic benchmark CLI. Historical evidence (gfx1100 b11126 four sessions +0.66/+0.53/+0.47/+0.65%, controls flat, full-vocab identical) is not a current-pin qualification. gfx1201 historical sessions are noisy/inconclusive; gfx1030 lacks a performance series. Keep default-off/untested until the frozen contract's four independent sessions, >=10 paired rounds/session, CI95-low >0% and <=1% control regression are met per architecture. PRBE39 VIEW/overlap hardening already resides in 1206.

## Upstream / provenance

Ported from stew675-rdna-boosts fork commit 0153d580d (<https://github.com/stew675/llama.cpp>). Not merged into ggml-org/llama.cpp master.

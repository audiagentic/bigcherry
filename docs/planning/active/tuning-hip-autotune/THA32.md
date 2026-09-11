---
id: THA32
order: 32
plan: tuning-hip-autotune
state: pending
created-at: '2026-09-11T22:57:08.866764+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Qwen3.8 Flash-Next incremental QSA cache A/B (#28699)

## Description

Evaluate upstream PR #28699's incremental pooled QSA cache for Qwen3.8 Flash-Next, with special attention to long-context decode and no-P2P multi-GPU behavior.

## Steps

1. Resolve PR ancestry/head and current-pin applicability before any local patching. 2. Build a clean A/B using the same source/build except for the QSA cache toggle or change. 3. Test MTP n-max 2/4, q8_0 KV, depths 16K/64K/128K/192K, single R9700, then 2x XTX tensor split. 4. Require bit-identical greedy output and capture TG/PP, cache identity, device placement, and memory. 5. Decide baseline adoption, local patch qualification, or deferment based on exact evidence.

## Detailed Solution & Technical Design

This is an experiment/qualification item, not permission to backport. The shared report says #28699 incrementally caches completed QSA block summaries, uses per-device cache allocation, reports roughly +9% deep decode in submitted results, and exposes LLAMA_QSA_NO_POOLED_CACHE=1. Verify all claims against the actual PR and current pin. Reuse existing experiment contracts, paired A/B, correctness, build identity, and replay evidence; do not create a QSA-specific promotion path.

## Code Samples & Guidance



## Files

tools/bigcherry/experiment/**; tools/bigcherry/campaign/**; tools/bigcherry/tuning/**; patches/** only if ancestry proves the change absent; tools/tests/**; docs/evidence/<run-id>/

## Validation

Upstream ancestry/current-pin audit; paired same-build A/B; deterministic output parity; long-context TG effect; single- and multi-GPU device-locality evidence; no-P2P topology; cache hit/miss and replay identity; statistical effect evidence with controls.

## Effort & Risk



## Standards



## Acceptance Criteria

A current-pin evidence package establishes whether #28699 is applicable and beneficial for the target Qwen3.8 Flash-Next workloads. No local patch is promoted without exact source/build identity, correctness, and replay-safe evidence; an upstream-equivalent change is not duplicated.

## Notes

Provenance: shared ChatGPT conversation, 12 Sep 2026, '#28699 — Qwen3.8 Flash-Next long-context decode optimization'; source https://github.com/ggml-org/llama.cpp/pull/28699. Existing PNRO12/NRO13 concern Qwen4exp gather-based QSA and is not this change; this is a distinct incremental pooled-cache experiment.

## Change Log

- 2026-09-11T22:57:08.866764+00:00 (created-by): Created by agent

## Ledger-events

- chg_20260911_225756_added-six-provenance-rich-buil_2622
- 2026-09-11T22:57:56.710779+00:00 (updated-by): Updated: section:ledger-events

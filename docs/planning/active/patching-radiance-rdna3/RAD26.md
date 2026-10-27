---
id: RAD26
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-10-27T02:15:24.188732+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# moe-remote: large-lane GPU-to-CPU request wake (host callback / graph host node / HSA interrupt) harness

## Description

The large lane yield-spins on its request word during bursts because the engine publishes from GPU code, which cannot wake a futex. Measure candidate wake paths in a two-process harness without model load: hipLaunchHostFunc -> eventfd, hipGraphAddHostNode -> eventfd, HSA interrupt signal, against the current spin. Large requests are few (prefill steps of 256+ rows), so tens of microseconds may be acceptable; per-request small-lane callbacks are rejected (engine stream stall per MoE layer).

## Steps

1. Harness: publish->wake latency p50/p99, engine stream stall, CPU use, AQL packets (barrier check for graph host nodes).
2. If one is cheap: implement for the large lane only; A/B prefill and mixed workloads.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Harness numbers recorded in the RAD20 research record; end-to-end prefill within noise before adoption.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

HSA IPC blocked waits already measured at 100% CPU (FINDINGS); interrupt relay spun before blocking. Do not repeat unchanged.

## Change Log

- 2026-10-27T02:15:24.188732+00:00 (created-by): Created by agent

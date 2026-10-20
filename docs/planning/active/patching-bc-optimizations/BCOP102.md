---
id: BCOP102
order: 102
plan: patching-bc-optimizations
state: done
created-at: '2026-10-09T23:06:00+00:00'
created-by: agent
priority: P2
work: S
---

# PNRO18: retire obsolete P2P probe-hardening

## Disposition

PNRO18 is superseded by PNRO03/1252 rejection on 2026-10-08 (b11474, 2026-10-07 hardware fault). 1252's startup probe verifies fresh buffers at four sizes, but not the production dev_tmp and cross-device event ring. Stored validation shows bit-identical fallback output with failed candidate activation and failed contract verdict. Keep 1252 rejected and default-off; do not queue a new P2P campaign.

## Ownership and gate

PNRO03 owns the rejected provider; PNRO18 is terminal. PGC09/PGC12 own production host/RCCL routing. No independent P2P work within 12 hours; Radiance, QFP 1334/1347/1355/1356/1358 and router work excluded. Reopen only for a new supported topology with positive provider completion, actual production-buffer/event correctness, graph replay, full-vocab parity, crash-free multi-request and matched E2E benefit. No new selector or allocator.

## Evidence

1252 patch.py/patch.toml/evidence/validation.json, PNRO03/PNRO18, test_p2p_copy_correctness.py, ab-27b-p2p.json, upstream llama.cpp #27825/#21648, vLLM ROCm custom AllReduce. Ten static checks and sixteen disposable fail-closed fixtures passed; no GPU tests or build.

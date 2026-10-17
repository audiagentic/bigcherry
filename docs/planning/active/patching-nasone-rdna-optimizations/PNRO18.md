---
id: PNRO18
order: 0
plan: patching-nasone-rdna-optimizations
state: done
created-at: '2026-09-25T23:16:44.258298+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: S
---

# 1252 P2P probe hardening — retired with rejected provider

**Terminal disposition (2026-10-18; BCOP292): DONE / SUPERSEDED.** This item proposed expanding `ggml_cuda_ar_p2p_probe` to check every async error and `copy_bytes`/chunk threshold edge on production streams and scratch. The exact production path, expected test matrix, correctness fence and conditional reopening criteria now belong **only to PNRO03**. Patch 1252 was already rejected on 2026-10-08 after the 2026-10-07 Brutus fault; its experiment was removed. Hardening an unselected, rejected transport cannot establish performance and must not be independently queued.

The 1252 probe currently tests 4 KiB, 64 KiB+256, 1 MiB and 4 MiB, both directions, twice. It uses temporary buffers rather than production `dev_tmp` and does not cover threshold±1; these are source-verified coverage gaps, **not** demonstrated causes of the hardware fault. The permanent `hipMemcpyDefault` byte-verification fixture is adjacent coverage, not an async-P2P substitute.

No implementation, patch metadata, hardware queue or other owner's plan is changed. See `PNRO03.md` for conditional reopening, existing test reuse and acceptance/rejection thresholds. Historical steps remain in Git history.

**Validation:** host-only seven-case admission/edge model passed; Python compilation passed. No HIP build, repository tests, GPU execution or new measured performance.

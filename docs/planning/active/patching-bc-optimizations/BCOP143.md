---
id: BCOP143
order: 143
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-11T17:07:58Z'
created-by: agent
priority: P2
work: S
---

# QFP23: deterministic runtime auto-profile admission

## Discovery / disposition

Validated 0910 already ships INI profiles. C uses unsorted directory iteration and first matching `arch`; Python checks only within-profile arch duplicates. `flashnext.ini` binds `qwen4exp` to a three-GPU-specific placement. C external-file validation and early-read timing are not fully qualified. Retain explicit profiles; do not create a second loader, selector or scheduler.

## Ownership and exclusion

QFP23/0910 owns admission, identity and flag timing; QFP18 owns ENV_DOCS. QFP/MET/MMQ/MTP owners retain tuning. Last independent QFP23 work Oct 5 and profile edit Oct 10 11:49 UTC, outside 12 hours. Recent BCOP136–142 are distinct. Protected PA49/b11552, MET07, QFP41, QFP48, QFP49 and PGC17 were not edited. Upstream llama.cpp #17859/#26118 already cover CLI/model presets.

## Evidence and terminal gate

Eight source-static checks and reversed-order host fixture passed; no repository pytest, C/HIP build or GPU measurement. Extend existing Python/C tests to reject duplicate cross-file arch, invalid external flags and unmatched auto, preserving rollback/explicit overrides. Require model/device/read-phase attestation and external profile digest. If explicit/auto parity is unproven, keep auto unqualified and explicit profiles available. No new performance claim or hardware job. Technical plan: QFP23.

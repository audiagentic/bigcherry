---
id: PEF03
order: 0
plan: patching-external-fixes
state: pending
created-at: '2026-09-23T14:16:04.207572+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P3
work: M
---

# Triage b11126 sources-check findings (fork history rewrites, incomplete mainline checks)

## Description

Raised by the b10901 -> b11126 pin bump (log: artifacts/logs/manual/bump-b11126-sources-check.log). No confirmed merged-upstream finding; live-patch absorption is enforced separately by patch-rebase-check (it caught 1001, now superseded). Provenance hygiene findings:
- mrlordcat-rdna-lab: moved 133 commits; RD69/71/72/73/76 tracked commits not found by title; mainline cherry check timed out (1800s).
- nasone-rdna-optimizations: moved; NRO06 tracked commit not found by title.
- amd-ecosystem-llama-cpp: moved; 14 tracked commits (RD39-44, RD49/50, RD30, RD97/98, untagged) not found by title; mainline cherry timed out.
- stew675-rdna-boosts: tip unrelated to active head; fetch hit a network reset, so its 33 'drifted' lines are most likely artifacts -- rerun this source alone first.
- joursbleu-llama-cpp: rebased (RD54, evidence-only) -- new snapshot needed.
- davetha-llama-cpp: unchanged.

## Steps

1. Rerun `bigcherry sources check --timeout 7200` (stew675 alone if the CLI gains a per-source filter) to separate network artifacts from real drift.
2. For each fork that rewrote history, take a new snapshot and map tracked commits by patch-id, not title; update config/external-sources.toml.
3. Only for tracked commits backing a live, non-superseded patch: check whether the fork changed content that should be pulled forward.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Non-blocking for the bump by policy: conflicts and upstream absorption are the hard gates.

## Change Log

- 2026-09-23T14:16:04.207572+00:00 (created-by): Created by agent

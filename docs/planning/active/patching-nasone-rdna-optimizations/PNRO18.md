---
id: PNRO18
order: 0
plan: patching-nasone-rdna-optimizations
state: pending
created-at: '2026-09-25T23:16:44.258298+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: S
---

# 1252 P2P probe hardening (GPT review)

## Description

GPT review req_4d131e8b7c1c452d: keep the empirical byte-verified probe (hipDeviceCanAccessPeer establishes capability, not correctness; AMD documents PCIe P2P as platform-dependent), and harden it: check cudaDeviceCanAccessPeer both directions before enabling (already), clear/check errors after every step, synchronize before compare, probe through the exact production copy path (same streams, same dev_tmp ownership), and include sizes straddling every runtime dispatch threshold (copy_threshold, chunk sizes), not only 4 KiB..4 MiB.

## Steps

1. Probe uses p->p2p_stream[src] and dev_tmp-sized buffers.
2. Sizes: each threshold -1, =, +1 element, plus max copy_bytes.
3. cudaGetLastError check after each call; fail closed.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Static package test; hardware P2P arm on 2x gfx1100 per TESTING.md.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

## Change Log

- 2026-09-25T23:16:44.258298+00:00 (created-by): Created by agent

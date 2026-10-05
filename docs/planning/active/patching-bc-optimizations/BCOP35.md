---
id: BCOP35
order: 35
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-05T21:49:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P3
work: S
---

# Gate context-parallel long-context attention

## Description

Action/disposition ledger for determining whether distributed attention/KV work is viable on BigCherry's no-P2P topology. Existing Flash Attention, sparse-attention, KV/cache and RPL01 owners remain authoritative. Do not create a context-parallel runtime until communication feasibility is demonstrated.

## Actions

1. Derive communication lower bounds for sequence/KV/head partition candidates separately for prefill and decode using actual target model dimensions.
2. Apply measured host-mediated/collective transport costs from PHA03/related topology evidence and compare against removable attention time at 32K/64K/128K/256K-class contexts.
3. Reject candidates whose physical lower bound cannot support >=5% end-to-end gain.
4. For a surviving candidate, create only an offline/synthetic correctness-cost prototype first; no persistent runtime implementation until exact attention semantics and crossover are demonstrated.
5. Route any proven implementation into the existing attention/KV/backend scheduler ownership, with RPL01 comparing placement cost.
6. Record terminal disposition: `infeasible-no-P2P`, `upstream-mechanism-watch`, `prototype-justified`, or `existing-sparse-FA-sufficient`.

## Gate

No NVLink/P2P assumptions. Correctness precedes timing. Failed prototypes are removed. Existing sparse FA gains must be treated as the control because they reduce the remaining benefit available to context parallelism.

## Related

RPL01; PHA03; QFP/Flash-Attention and KV-cache owners; BCOP32.

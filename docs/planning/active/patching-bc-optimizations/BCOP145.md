---
id: BCOP145
order: 145
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-11T19:12:47Z'
created-by: agent
priority: P3
work: S
---

# RNX08: gate cold PLE row acquisition after upstream prefetch

## Discovery / change

Pinned b11552 already contains merged llama.cpp #29599: lazy host-backed PLE rows are page-prefetched before a **single I32 index upload per ubatch** and native `GET_ROWS`. Historical "~23/token H2D" counts indices, not transfers; aggregate 4.7 s H2D does not identify PLE. Upstream direct-reader #28136 and successor #29030 both closed unmerged. No duplicate int6 gather, new PLE cache, or speculative 1309 patch is justified.

## Owner / independent work / protection

RNX08 is the authoritative PLE gate; native llama.cpp owns page prefetch and gather; QFP17 owns prefill attribution, RNX09 integration, 1004 generic norm+RoPE. RNX08 last independent update 2026-10-03 15:24 UTC; no independent PLE patch, PR, benchmark or queue found in the preceding 12 hours across inspected active branches. Distinct from BCOP136–144. Protected PA49/b11552, QFP41, QFP48, QFP49, PGC17, MTP/1359 and the recently changed shared RNX README were not edited.

## Unresolved bounded action / terminal

First record deployed GGUF PLE type/row geometry/lazy host backing and cold-versus-warm PLE-only major faults, page residency, `GET_ROWS`/prefetch wall and per-ubatch index bytes. Use diverse/repeated prompt controls; existing prefetch is baseline. **Close without a port** if inactive, <5% critical wall, <3% Amdahl E2E ceiling, or no cold gap. Only if admitted, prototype upstream #29030's packed-row per-context reader default-off with owned staging, row-count and graph-lifetime proofs; require exact rows/logits/greedy/MTP, four sessions and >=10 paired ABBA rounds, cold TTFT CI95-low >=3%, warm/decode CI95-high <=1%. Otherwise retain #29599. No queue or new implementation this run. Technical design and links: RNX08.

## Validation

14/14 pinned-source static assertions; six synthetic page-interval checks. Synthetic 11,776 random 90-B rows: 12,048 distinct 4-KiB pages (46.56x potential page footprint versus useful packed bytes); 128-row hot set: three pages. **Not** model measurements. No C++/HIP build, repository pytest, GPU benchmark or new BigCherry speedup.

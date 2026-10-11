---
id: BCOP88
order: 88
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-09T09:07:50+00:00'
created-by: agent
priority: P2
work: S
---

# PRBE112: retire duplicated RD39/40 and gate MoE-only concurrency rework

## Discovery / disposition

Rejected 1215's current b11474 patch has no active cuBLAS handle/matmul-stream rewrite: upstream #26574 already owns per-stream handles/workspaces. The remaining RD41 patch replaces **all native QKV interleaving** with a shared max-footprint scratch buffer even on dense graphs, while RD42's named shared-expert fork cannot activate on dense. Eight first-party b11126 contract sessions (four gfx1100/four gfx1201) show MoE -2.84..-3.30%/-8.05..-8.18% and dense -1.59..-2.21%/-6.83..-6.86%. Generic scratch is the minimal unproved cause candidate; do not present it as measured attribution.

## Ownership / protected activity

PRBE112 is authoritative for the bounded rework/retire decision. PRBE33/34 are historical, PRBE35/1216 owns independent join safety, and rejected 1215 remains immutable. QFP35/36/41, 1357/1358 queues, Meta split cache, MTP, Radiance, engine/lab and patch-ablation work were active within 12h and are not modified. The selected PRBE112 mechanism had no independent 12h commit, submission, plan edit or queued hardware lane; recent BCOP78-87 audit slices concern other mechanisms.

## Bounded next action / terminal

First use static patch composition and a host graph/lifetime fixture to test native QKV unchanged plus MoE-only auxiliary event, including scratch growth/recapture and missing/overlapping join cases. If native code already suffices or alias-free replay cannot be proven, close PRBE112 without new patch. Only a safe narrow variant may proceed to gfx1100 then gfx1201 A/B against native+1216, with exact activation, full-vocab parity, four sessions and >=10 paired rounds/session, CI95-low >0%, <=1% dense regression. Do not schedule over existing lab queue. Host/source assertions 12/12 passed; no build, GPU, rocprof or new benchmark ran.

## References

PRBE112; completed PRBE33/34; PRBE35/1216; 1215 patch.py, SUMMARY.md and evidence/validation.json; config/experiment-contracts.toml RD39-42; llama.cpp #16991/#26574; AMD-Ecosystem/llama.cpp #36; vLLM #48111 and moe_runner.py.

---
id: BCOP159
order: 159
plan: patching-bc-optimizations
state: done
created-at: '2026-10-12T09:08:15+00:00'
created-by: agent
priority: P1
work: S
---

# PNRO11: close target-wide output replication

PNRO11's proposed patch 1264 duplicates QFP43/1286's existing draft-local borrowed-tensor copy. Close PNRO11 without implementing 1264. QFP43 retains ownership of untested 1286 and its correctness qualification.

The current 1286 output-scale selector is conditional on the copied output weight rather than the copied scale. A 12-case host model exposed two mixed-residency pointer mismatches; no GPU failure was reproduced. Distinguish this from DFlash2 Meta-split TOP_K, which 1286 does not solve. Upstream llama.cpp #27858 proposes a CPU selector but remains open.

Next gate: per-tensor pointer and backend-accessibility fixtures, then one-time transfer/VRAM accounting and same-process logits, MTP, acceptance and graph checks if demand remains. No new scheduler, allocator, patch or hardware queue. Active Radiance, QFP48, QFP15 and release work were excluded. Source-static 13/13; host 12 cases; no build/GPU run.

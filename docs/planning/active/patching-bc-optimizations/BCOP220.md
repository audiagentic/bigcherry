---
id: BCOP220
order: 220
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-15T11:42:17+11:00'
created-by: agent
priority: P1
work: S
---

# PGC15: fence tiled AllReduce precision and partial-fallback safety

**Finding:** PGC15's token-tiled compute/AllReduce overlap is a documentation and design candidate, not an implemented or qualified patch. At the pinned b11552 RCCL path, provider precision is selected from the whole reduction's element count. Recomputing that decision for each tile can silently change arithmetic: a 768 x 512 reduction is above the 131072-element boundary, while a 768 x 128 tile is below it and would select a different path. The tile implementation must preserve the original reduction's precision policy for every range.

**Owner/consolidation:** PGC15 remains authoritative for producer tiling, range AllReduce, stream/event lifetime and eligibility proof. PGC12 owns phase/provider provenance; PGC14 owns RCCL transport/SHM; PGC09/11 own provider and wire precision; QFP17 owns Flash-Next prefill acceptance. Do not add a second scheduler, allocator, dispatcher, provider registry or telemetry framework.

**Eligibility:** The audit used `feat/rad03-gpu-lanes` at independent HEAD `6786b9d477e3d725a2ede2a7c5624786f06d8674` (2026-10-15 10:40:50 +11:00). PGC15 had no same-capability repository update, open PR or queued experiment in the preceding 12 hours. Active Radiance/RAD02-RAD06, MEN10, release automation and other GPU/helper work were excluded as concurrent independent work; they were not modified. The named proposal files were not available in the local checkout, Brutus `/mnt/data`, or the configured workspace search paths.

**Fail-closed gate:** Complete preflight for every planned range before submitting tile zero. Snapshot the whole-reduction F32/BF16 policy before partitioning. If preflight fails, use the untouched whole path. If a range operation fails after submission, surface an execution error and stop; never fall back to a whole-tensor retry after a partial reduction. Keep provider-owned scratch, streams and events alive until completion fences prove that all queued ranges are finished.

**Qualification gate:** First prove a real attention-output producer tile `i+1` overlaps range AllReduce tile `i` on the actual three-rank Brutus configuration. Run host/direct precision-boundary and admission tests, then same-build flag-off versus tile128/256 ABBA at representative prefill depths with logits/top-k and decode controls. Reject implementation or promotion if the provider cannot preserve whole-reduction precision, if temporal overlap is not observed, if any lifetime/fallback case is unsafe, or if the exposed end-to-end opportunity does not meet PGC15's existing threshold.

**Status:** No production patch, hardware benchmark, or performance gain is claimed. This ledger records the correctness fence for a future PGC15 phase-1 qualification; it does not authorize implementation or queue a hardware run.

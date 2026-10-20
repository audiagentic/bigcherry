---
id: BCOP157
order: 157
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-12T07:13:00+00:00'
created-by: agent
priority: P1
work: S
---

# PNRO09/1260: correct the capacity validation producer

The producer passes a CMake target name where an architecture is required, refers to source trees it has not materialized, and replays historical hardware evidence. The fixture creates zero-length views. PNRO09 owns repair; no new validator or allocator. The historical real MTP workload fits stock headroom 16, so no promotion is justified.

Correct source/build identity, view geometry, capacity failure classification and fresh real-graph positive/control evidence. Account for extra host metadata and require a stock-fails/patched-passes workload. Otherwise retire 1260. Keep MSM03, QFP53 and active Radiance work untouched. Host source-static checks: 15/15; no build or GPU run.

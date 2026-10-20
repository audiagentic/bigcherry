---
id: BCOP156
order: 156
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-12T06:14:00+00:00'
created-by: agent
priority: P2
work: S
---

# PRBE02/1266: use native WMMA combine guard; reproduce before restoring inter-tile sync

## Discovery / change

PRBE02 still says PRBE110 must extract RD05, but isolated 1266 already exists and has four b11126 gfx1201 correctness/activation sessions without performance qualification. The independently active Radiance branch pins b11474; main pins b11552. Merged upstream llama.cpp #30103 (2026-10-10) changed the native combine guard, invalidating 1266's literal k00 anchor on b11552. The separate inter-`kbc` barrier is not present upstream, but no failure was reproduced. AMD WMMA does not admit head sizes >256, so the historical 320/512/576 positive matrix is ineligible. No new GPU measurement.

## Owner / protected work

PRBE02 owns RD05 correctness; 1266 remains historical default-off. PRBE110 owns Q6_K/1267, not a second RD05 extraction; PA39/1203 retains rejected evidence. Reuse native FA selector, backend-op and trace producers. 12-hour check passed for PRBE02/1266; active Radiance RR08/MXFP4/GDN/paged prefill/all-reduce, PA44/PA49, QFP15/1306 and QFP53/1362 were excluded. No protected implementation or queued hardware lane changed. This is distinct from BCOP146-155.

## Bounded action / terminal disposition

Use b11552 native #30103 as correctness baseline; first prove an eligible gfx1201 WMMA route and an actual repeated-`kbc` mismatch. No failure after bounded alternating-shape/graph/repeated-request tests => retire 1266 and complete PRBE02 without a patch. Reproduced failure => isolate only the uncovered inter-tile barrier, preserve #30103, require fail-before/pass-after, full logits/MTP/graph safety and <=1% unrelated control regression before any new patch identity. Do not queue a performance-only RD05 campaign.

## References

PRBE02; PRBE110; 1266; 1203; llama.cpp #30103, #27870, #28102, #28867; b11474/b11552 `fattn-mma-f16.cuh` and `fattn.cu`. Host-only selector fixture: 69 assertions; no repository test or GPU run.
